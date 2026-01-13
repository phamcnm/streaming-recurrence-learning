import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RealLRUBPTT(nn.Module):
    def __init__(self, in_features, out_features, state_features, rmin=0.0, rmax=1.0, mode='sequential', epsilon: float = 0.03, max_steps: int = 4, use_internal_flag=False):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.state_features = state_features
        self.use_internal_flag = use_internal_flag
        self.mode = mode
        if self.mode in ['pondernet', 'act']:
            self.halting_head = True
        else:
            self.halting_head = False
        if self.mode != 'sequential':
            self.epsilon = epsilon
            self.max_steps = max_steps
        
        # Diagonal Lambda parameter (decay factor) via log-parameterization
        u = torch.rand(state_features)
        self.lambda_log = nn.Parameter(torch.log(-0.5*torch.log(u*(rmax+rmin)*(rmax-rmin) + rmin**2)))
        
        # B and C parameters for hidden state transformation
        if use_internal_flag:
            in_dim = in_features + 1
        else:
            in_dim = in_features
        self.B = nn.Parameter(torch.randn(state_features, in_dim) / math.sqrt(in_dim))
        self.C = nn.Parameter(torch.randn(out_features, state_features) / math.sqrt(state_features))
        if self.halting_head:
            self.H = nn.Parameter(torch.randn(1, state_features) / math.sqrt(state_features)) # halting head

    def reset_state(self, rnn_state, done):
        return rnn_state * (1.0 - done).unsqueeze(-1)

    def _single_forward_step(self, input_t, hidden=None):
        """
        Perform a single forward step (one pondering step).
        
        Args:
            input_t: [batch, in_features]
            hidden: [batch, state_features] or None
            
        Returns:
            tuple: (output, new_hidden, lambda_t)
                output: [batch, out_features]
                new_hidden: [batch, state_features]
                halt_head
        """
        batch_size = input_t.shape[0]
        
        if hidden is None:
            state = torch.zeros(batch_size, self.state_features, device=input_t.device, dtype=input_t.dtype)
        else:
            state = hidden
        
        # Get lambda and gamma values
        lamda = torch.exp(-torch.exp(self.lambda_log))
        gamma = torch.sqrt(1 - lamda**2 + 1e-7)
        
        # Process single timestep
        if self.use_internal_flag:
            flag_t = input_t.new_ones(batch_size, 1)
            x_t_with_flag = torch.cat([input_t, flag_t], dim=-1)
            b_step = F.linear(x_t_with_flag, self.B)
        else:
            b_step = F.linear(input_t, self.B)
        
        new_state = lamda * state + gamma * b_step
        output = F.linear(new_state, self.C)
        
        if self.halting_head: # Compute halting probability
            halt = F.linear(new_state, self.H)
            halt_sigmoid = torch.sigmoid(halt)  # [batch, 1]
            return output, new_state, halt_sigmoid
        else:
            return output, new_state, None

    def forward_sequential(self, input, hidden=None, done=None, inner_loops=1, rnn_burnin=0.0):
        """
        Forward pass for RealLRUBPTT with batch support.
        
        Args:
            input: Input tensor of shape (seq_len, batch_size, features)
            hidden: Optional state tensor or None
            done: Optional done flags of shape (seq_len, batch_size) - resets state when True
            inner_loops: Number of warm-up iterations
            burnin: Beginning portion of the sequence to just get an rnn start state without tracking gradients
            
        Returns:
            output: Output tensor of shape (seq_len, batch_size, out_features)
            hidden_final: Final state tensor of shape (batch_size, state_features)
            aux_data: None (for compatibility with other modes)
        """
        seq_len, batch_size, _ = input.shape
        assert 0 <= rnn_burnin <= 1
        burnin_steps = int(rnn_burnin * seq_len)
        
        # Initialize state with batch dimension
        if hidden is None:
            state = torch.zeros(batch_size, self.state_features, device=input.device, dtype=input.dtype)
        else:
            state = hidden
        
        # Get lambda and gamma values
        lamda = torch.exp(-torch.exp(self.lambda_log))
        gamma = torch.sqrt(1 - lamda**2 + 1e-7)
        
        output = torch.empty(seq_len, batch_size, self.out_features, device=input.device, dtype=input.dtype)
        grad_enabled = torch.is_grad_enabled()
        for i in range(seq_len):
            x_t = input[i]
            if done is not None and done[i].any():
                state = self.reset_state(state, done[i])

            for loop_idx in range(inner_loops):
                is_last = (loop_idx == inner_loops - 1)

                # Toggle gradient: only last pass tracks gradients, and only after burnin period
                # torch.set_grad_enabled(is_last and i >= burnin_steps)
                with torch.set_grad_enabled(grad_enabled and is_last and i >= burnin_steps):
                    if self.use_internal_flag:
                        # 0 during warmup, 1 for last pass
                        flag_val = 1.0 if is_last else 0.0
                        flag_t = x_t.new_full((batch_size, 1), flag_val)
                        x_in = torch.cat([x_t, flag_t], dim=-1)
                        b_step = F.linear(x_in, self.B)
                    else:
                        b_step = F.linear(x_t, self.B)

                    # State update
                    state = lamda * state + gamma * b_step

            output[i] = F.linear(state, self.C)

        return output, state, inner_loops
    
    def forward_sequential_convergence(self, input, hidden=None, done=None, one_step_backpass=True, inner_loops: int = 1, rnn_burnin=0.0):
        seq_len, batch_size, _ = input.shape
        device, dtype = input.device, input.dtype
        
        if hidden is None:
            hidden = torch.zeros(batch_size, self.state_features, device=device, dtype=dtype)
        
        sampled_indices = []
        outputs = []
        
        for t in range(seq_len):
            x_t = input[t]
            if done is not None and done[t].any():
                hidden = self.reset_state(hidden, done[t])
            hidden_t = hidden
            converged = torch.zeros(batch_size, dtype=torch.bool, device=device)
            steps_taken = torch.full((batch_size,), self.max_steps - 1, dtype=torch.long, device=device)
            final_output = torch.zeros(batch_size, self.out_features, device=device, dtype=dtype)
            final_hidden = torch.zeros(batch_size, self.state_features, device=device, dtype=dtype)
            
            for n in range(self.max_steps):
                is_last_step = (n == self.max_steps - 1)
                converged_detached = converged.clone()
                
                if one_step_backpass and not is_last_step:
                    # Check if will converge (no grad)
                    with torch.no_grad():
                        pred_check, new_hidden_check, _ = self._single_forward_step(x_t, hidden_t)
                        change = torch.norm(new_hidden_check - hidden_t, dim=-1)
                        will_converge = (change <= self.epsilon) & ~converged
                    
                    # Build gradients only for elements that will converge
                    if will_converge.any():
                        # Do gradient pass (for all batch - simpler than splitting)
                        pred, new_hidden, _ = self._single_forward_step(x_t, hidden_t)
                        
                        # Save converged results (WITH gradients)
                        final_output[will_converge] = pred[will_converge]
                        final_hidden[will_converge] = new_hidden[will_converge]
                        steps_taken[will_converge] = n
                        converged |= will_converge
                        
                        # Detach for non-converged elements to save memory
                        hidden_t = torch.where(
                            converged_detached.unsqueeze(-1),
                            new_hidden.detach(),
                            new_hidden
                        )
                    else:
                        # Nothing converging: use no-grad result
                        pred, new_hidden = pred_check, new_hidden_check
                        hidden_t = new_hidden
                else:
                    # Normal mode OR last step: always compute with gradients
                    pred, new_hidden, _ = self._single_forward_step(x_t, hidden_t)
                    
                    # Check convergence
                    with torch.no_grad():
                        change = torch.norm(new_hidden - hidden_t, dim=-1)
                        newly_converged = (change <= self.epsilon) & ~converged
                    
                    # Save converged results (WITH gradients)
                    if newly_converged.any():
                        final_output[newly_converged] = pred[newly_converged]
                        final_hidden[newly_converged] = new_hidden[newly_converged]
                        steps_taken[newly_converged] = n
                        converged |= newly_converged
                    
                    hidden_t = new_hidden
                
                if converged.all():
                    break
            
            # Use final results for non-converged elements
            final_output = torch.where(converged.unsqueeze(-1), final_output, pred)
            final_hidden = torch.where(converged.unsqueeze(-1), final_hidden, new_hidden)
            
            sampled_indices.append(steps_taken+1)
            outputs.append(final_output)
            hidden = final_hidden
        
        sampled_indices = torch.stack(sampled_indices, dim=0)
        return torch.stack(outputs, dim=0), hidden, (None, None, sampled_indices)
    
    # def forward_sequential_pondernet(self, input, hidden=None, done=None):
    #     """
    #     Forward pass with PonderNet.
        
    #     Args:
    #         input: Input tensor of shape (seq_len, batch_size, features)
    #         hidden: Optional state tensor or None
    #         epsilon: Threshold for halting (remaining probability mass)
            
    #     Returns:
    #         output: Output tensor of shape (seq_len, batch_size, out_features)
    #         hidden_final: Final state tensor of shape (batch_size, state_features)
    #         aux_data: Tuple of (all_predictions, all_p, sampled_indices) for loss computation
    #     """
    #     seq_len, batch_size, _ = input.shape
        
    #     if hidden is None:
    #         hidden = torch.zeros(batch_size, self.state_features, device=input.device, dtype=input.dtype)
        
    #     all_predictions = []
    #     all_p_halts = []
    #     sampled_indices = []
    #     outputs = []
        
    #     for t in range(seq_len):
    #         x_t = input[t]
    #         if done is not None and done[t].any():
    #             hidden = self.reset_state(hidden, done[t])
    #         hidden_t = hidden
    #         preds_t = []
    #         p_unconditional_t = []
    #         states_t = []
            
    #         p_total = torch.zeros(batch_size, 1, device=input.device, dtype=input.dtype)
    #         p_continue = torch.ones(batch_size, 1, device=input.device, dtype=input.dtype)
            
    #         for n in range(self.max_steps):
    #             pred, new_hidden, lambda_n = self._single_forward_step(x_t, hidden_t)
                
    #             preds_t.append(pred)
    #             states_t.append(new_hidden)
                
    #             # Compute unconditional probability: p(halt at n) = lambda_n * p_continue_n-1
    #             p_n = lambda_n * p_continue
                
    #             # Check which batch elements have reached threshold
    #             will_exceed = (p_total + p_n) >= (1.0 - self.epsilon)
                
    #             # For elements that exceed: assign remaining probability
    #             # For elements that don't: use p_n as computed
    #             p_n_adjusted = torch.where(will_exceed, 1.0 - p_total, p_n)
    #             p_unconditional_t.append(p_n_adjusted)
                
    #             # Update totals
    #             p_total = p_total + p_n_adjusted
    #             p_continue = torch.where(will_exceed, torch.zeros_like(p_continue), p_continue * (1 - lambda_n))
                
    #             hidden_t = new_hidden
                
    #             # Stop if all batch elements have stopped OR reached max steps
    #             if n == self.max_steps - 1 or torch.all(p_total >= 1.0 - self.epsilon):
    #                 break
            
    #         # Stack for this timestep
    #         preds_t = torch.stack(preds_t, dim=0)  # [num_ponder_steps, batch, out_features]
    #         p_unconditional_t = torch.stack(p_unconditional_t, dim=0)  # [num_ponder_steps, batch, 1]
    #         states_t = torch.stack(states_t, dim=0)  # [num_ponder_steps, batch, state_features]
            
    #         all_predictions.append(preds_t)
    #         all_p_halts.append(p_unconditional_t)
            
    #         # Sample halting step
    #         probs = p_unconditional_t.squeeze(-1).t()  # [batch, num_ponder_steps]
    #         sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch]
    #         sampled_indices.append(sampled_idx)
            
    #         # Use sampled hidden state and output
    #         batch_indices = torch.arange(batch_size, device=input.device)
    #         hidden = states_t[sampled_idx, batch_indices]  # [batch, state_features]
    #         output_t = preds_t[sampled_idx, batch_indices]  # [batch, out_features]
            
    #         outputs.append(output_t)
        
    #     outputs = torch.stack(outputs, dim=0)  # [seq_len, batch, out_features]
        
    #     return outputs, hidden, (all_predictions, all_p_halts, sampled_indices)

    def forward_sequential_act(self, input, hidden=None, done=None, inner_loops: int = 1, rnn_burnin=0.0):
        """
        Forward pass with Adaptive Computation Time.
        """
        seq_len, batch_size, _ = input.shape

        if hidden is None:
            hidden = torch.zeros(batch_size, self.state_features, device=input.device, dtype=input.dtype)

        all_predictions = []
        all_p_halts = []
        sampled_indices = []
        outputs = []
        ponder_costs = []

        for t in range(seq_len):
            x_t = input[t]
            if done is not None and done[t].any():
                hidden = self.reset_state(hidden, done[t])
            hidden_t = hidden
            preds_t = []
            weights_t = []
            states_t = []

            p_total = torch.zeros(batch_size, 1, device=input.device, dtype=input.dtype)
            still_running = torch.ones(batch_size, 1, device=input.device, dtype=torch.bool)

            for n in range(self.max_steps):
                pred, new_hidden, p_n = self._single_forward_step(x_t, hidden_t)

                preds_t.append(pred)
                states_t.append(new_hidden)

                p_n = torch.where(still_running, p_n, torch.zeros_like(p_n))

                will_exceed = (p_total + p_n) >= (1.0 - self.epsilon)
                will_halt_now = still_running & will_exceed
                is_last_step = (n == self.max_steps - 1)

                p_n_adjusted = torch.where(will_halt_now | (still_running & is_last_step), 1.0 - p_total, p_n)
                weights_t.append(p_n_adjusted)

                p_total = p_total + p_n_adjusted

                still_running_next = still_running & ~will_halt_now
                hidden_t = torch.where(still_running_next, new_hidden, hidden_t)
                still_running = still_running_next

                if not still_running.any():
                    break

            preds_t = torch.stack(preds_t, dim=0)
            weights_t = torch.stack(weights_t, dim=0)
            states_t = torch.stack(states_t, dim=0)

            all_predictions.append(preds_t)
            all_p_halts.append(weights_t)

            probs = weights_t.squeeze(-1).t()
            sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            sampled_indices.append(sampled_idx + 1)

            output_t = (preds_t * weights_t).sum(dim=0)
            hidden = (states_t * weights_t).sum(dim=0)

            step_idx = torch.arange(
                1,
                weights_t.shape[0] + 1,
                device=weights_t.device,
                dtype=weights_t.dtype,
            ).view(-1, 1, 1)
            ponder_cost_t = (weights_t * step_idx).sum(dim=0)  # [batch, 1]
            ponder_costs.append(ponder_cost_t.mean())

            outputs.append(output_t)

        outputs = torch.stack(outputs, dim=0)
        ponder_cost = torch.stack(ponder_costs).mean() if ponder_costs else torch.tensor(0.0, device=input.device)

        return outputs, hidden, (all_predictions, all_p_halts, sampled_indices, ponder_cost)
    
    def forward(self, x, hidden=None, **kwargs):
        """
        Returns:
            output: [seq_len, batch, out_features]
            hidden: hidden state
            aux_data: None for sequential, (all_predictions, all_p, sampled_indices)
        """
        if self.mode in ['sequential', 'ponder_hardcoded']:
            return self.forward_sequential(x, hidden, **kwargs)
        elif self.mode == 'convergence':
            return self.forward_sequential_convergence(x, hidden, **kwargs)
        # elif self.mode == 'pondernet':
        #     return self.forward_sequential_pondernet(x, hidden, **kwargs)
        elif self.mode == 'act':
            return self.forward_sequential_act(x, hidden, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
