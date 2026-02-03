import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GatedRealLRUBPTT(nn.Module):
    def __init__(self, in_features, out_features, state_features, rmin=0.9, rmax=0.999, mode='sequential', epsilon: float = 0.03, max_steps: int = 4):
        super().__init__()
        self.alpha = 3.0
        self.out_features = out_features
        self.state_features = state_features
        self.in_features = in_features
        self.mode = mode
        
        if self.mode in ['pondernet', 'act']:
            self.halting_head = True
        else:
            self.halting_head = False
        if self.mode != 'sequential':
            self.epsilon = epsilon
            self.max_steps = max_steps
        
        # Initialize decay parameters
        u1 = torch.rand(state_features)
        self.nu_log = nn.Parameter(torch.log(-0.5*torch.log(u1*(rmax+rmin)*(rmax-rmin) + rmin**2)))
        
        # Input projection matrices
        self.B = nn.Parameter(torch.randn(state_features, in_features) / math.sqrt(in_features))
        self.gate_W = nn.Parameter(torch.randn(state_features, in_features) / math.sqrt(in_features))
        self.V_nu = nn.Parameter(torch.randn(state_features, in_features) / math.sqrt(in_features))
        
        if self.halting_head:
            self.H = nn.Parameter(torch.randn(1, state_features) / math.sqrt(state_features))  # halting head

    def reset_state(self, rnn_state, done):
        return rnn_state * (1.0 - done).unsqueeze(-1)

    def _single_forward_step(self, input_t, hidden=None):
        """
        Perform a single forward step (one pondering step).
        
        Args:
            input_t: [batch, in_features]
            hidden: [batch, state_features] or None
            
        Returns:
            tuple: (output, new_hidden, halt_prob)
                output: [batch, out_features]
                new_hidden: [batch, state_features]
                halt_prob: [batch, 1] or None
        """
        batch_size = input_t.shape[0]
        
        if hidden is None:
            state = torch.zeros(batch_size, self.state_features, device=input_t.device, dtype=input_t.dtype)
        else:
            state = hidden
        
        # Compute gates and projections
        recurrence_gate = torch.sigmoid(torch.matmul(input_t, self.V_nu.t()))
        exp_nu_log = torch.exp(self.nu_log)
        alpha_nu = self.alpha * exp_nu_log
        lamda = torch.exp(-alpha_nu * (1 - recurrence_gate))
        gammas = (1 - lamda.pow(2) + 1e-7).sqrt_()
        
        # Compute input projections
        b_x = torch.matmul(input_t, self.B.t())
        gate_x = torch.matmul(input_t, self.gate_W.t())
        
        # Update state
        new_state = lamda * state + gammas * b_x * gate_x
        
        # Output is the state itself
        output = new_state
        
        if self.halting_head:  # Compute halting probability
            halt = F.linear(new_state, self.H)
            halt_sigmoid = torch.sigmoid(halt)  # [batch, 1]
            return output, new_state, halt_sigmoid
        else:
            return output, new_state, None

    def forward_sequential(self, input, hidden=None, done=None, inner_loops=1, rnn_burnin=0.0):
        """
        Forward pass for GatedRealLRUBPTT with batch support.
        
        Args:
            input: Input tensor of shape (seq_len, batch_size, features)
            hidden: Optional state tensor or None
            done: Optional done flags of shape (seq_len, batch_size) - resets state when True
            inner_loops: Number of warm-up iterations
            rnn_burnin: Beginning portion of the sequence to just get an rnn start state without tracking gradients
            
        Returns:
            output: Output tensor of shape (seq_len, batch_size, out_features)
            hidden_final: Final state tensor of shape (batch_size, state_features)
            aux_data: inner_loops (for compatibility with other modes)
        """
        seq_len, batch_size, _ = input.shape
        assert 0 <= rnn_burnin <= 1
        burnin_steps = int(rnn_burnin * seq_len)
        
        # Initialize state with batch dimension
        if hidden is None:
            state = torch.zeros(batch_size, self.state_features, device=input.device, dtype=input.dtype)
        else:
            state = hidden
        
        output = torch.empty(seq_len, batch_size, self.out_features, device=input.device, dtype=input.dtype)
        grad_enabled = torch.is_grad_enabled()
        
        for i in range(seq_len):
            # Get current input step (batch_size, features)
            x_t = input[i]
            if done is not None and done[i].any():
                state = self.reset_state(state, done[i])
            
            for loop_idx in range(inner_loops):
                is_last = (loop_idx == inner_loops - 1)
                
                # Toggle gradient: only last pass tracks gradients, and only after burnin period
                with torch.set_grad_enabled(grad_enabled and is_last and i >= burnin_steps):
                    # Compute gates and projections
                    recurrence_gate = torch.sigmoid(torch.matmul(x_t, self.V_nu.t()))
                    exp_nu_log = torch.exp(self.nu_log)
                    alpha_nu = self.alpha * exp_nu_log
                    lamda = torch.exp(-alpha_nu * (1 - recurrence_gate))
                    gammas = (1 - lamda.pow(2) + 1e-7).sqrt_()
                    
                    # Compute input projections
                    b_x = torch.matmul(x_t, self.B.t())
                    gate_x = torch.matmul(x_t, self.gate_W.t())
                    
                    # Update state
                    state = lamda * state + gammas * b_x * gate_x
            
            # Compute output
            output[i] = state
        
        # Return output and final state
        return output, state, inner_loops

    def forward_sequential_act(self, input, hidden=None, done=None, inner_loops: int = 1, rnn_burnin=0.0):
        """
        Forward pass with Adaptive Computation Time.
        
        Args:
            input: Input tensor of shape (seq_len, batch_size, features)
            hidden: Optional state tensor or None
            done: Optional done flags of shape (seq_len, batch_size) - resets state when True
            inner_loops: Number of warm-up iterations (unused in ACT mode)
            rnn_burnin: Beginning portion of the sequence to just get an rnn start state without tracking gradients (unused in ACT mode)
            
        Returns:
            output: Output tensor of shape (seq_len, batch_size, out_features)
            hidden_final: Final state tensor of shape (batch_size, state_features)
            aux_data: Tuple of (all_predictions, all_p_halts, sampled_indices) for loss computation
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
                
                # Update totals
                p_total = p_total + p_n_adjusted
                still_running_next = still_running & ~will_halt_now
                hidden_t = torch.where(still_running_next, new_hidden, hidden_t)
                still_running = still_running_next
                
                # Stop if all batch elements have stopped OR reached max steps
                if not still_running.any():
                    break
            
            # Stack for this timestep
            preds_t = torch.stack(preds_t, dim=0)  # [num_ponder_steps, batch, out_features]
            weights_t = torch.stack(weights_t, dim=0)  # [num_ponder_steps, batch, 1]
            states_t = torch.stack(states_t, dim=0)  # [num_ponder_steps, batch, state_features]
            
            all_predictions.append(preds_t)
            all_p_halts.append(weights_t)

            # Sample halting step
            probs = weights_t.squeeze(-1).t()  # [batch, num_ponder_steps]
            sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch]
            sampled_indices.append(sampled_idx+1)
            
            # Compute weighted average for both output and hidden state
            output_t = (preds_t * weights_t).sum(dim=0)  # [batch, out_features]
            hidden = (states_t * weights_t).sum(dim=0)  # [batch, state_features]

            step_idx = torch.arange(
                1,
                weights_t.shape[0] + 1,
                device=weights_t.device,
                dtype=weights_t.dtype,
            ).view(-1, 1, 1)
            ponder_cost_t = (weights_t * step_idx).sum(dim=0)  # [batch, 1]
            ponder_costs.append(ponder_cost_t.mean())

            outputs.append(output_t)
        
        outputs = torch.stack(outputs, dim=0)  # [seq_len, batch, out_features]
        ponder_cost = torch.stack(ponder_costs).mean() if ponder_costs else torch.tensor(0.0, device=input.device)

        return outputs, hidden, (all_predictions, all_p_halts, sampled_indices, ponder_cost)

    def forward(self, x, hidden=None, **kwargs):
        """
        Returns:
            output: [seq_len, batch, out_features]
            hidden: hidden state
            aux_data: None for sequential, (all_predictions, all_p_halts, sampled_indices) for ACT
        """
        if self.mode in ['sequential', 'ponder_hardcoded']:
            return self.forward_sequential(x, hidden, **kwargs)
        elif self.mode == 'act':
            return self.forward_sequential_act(x, hidden, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
