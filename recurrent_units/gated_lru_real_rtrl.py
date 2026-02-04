import torch, math
import torch.nn as nn
import torch.nn.functional as F

class GatedRealLRUFunction(torch.autograd.Function):
    """
    Implements a single time-step of a Gated diagonal RNN cell with RTRL.
    
    The recurrence is:
        r_gate = sigmoid(V_nu @ x_t)
        lambda = exp(-alpha * exp(nu_log) * (1 - r_gate))
        gamma = sqrt(1 - lambda^2 + epsilon)
        h_t = lambda * h_prev + gamma * (B @ x_t) * (gate_W @ x_t)
    """
    @staticmethod
    def forward(ctx, input_t, h_prev, nu_log, V_nu, B, gate_W, s_nu_log_prev, s_V_nu_prev, s_B_prev, s_gate_W_prev, alpha=8.0):
        # Calculate gating factor
        recurrence_input = F.linear(input_t, V_nu)
        recurrence_gate = torch.sigmoid(recurrence_input)
        
        # Calculate modulated decay rate with alpha factor (matching BPTT)
        exp_nu_log = torch.exp(nu_log)
        alpha_nu = alpha * exp_nu_log
        lamda = torch.exp(-alpha_nu * (1 - recurrence_gate))
        gamma = (1 - lamda.pow(2) + 1e-7).sqrt_()
        
        # Compute input gate modulation (matching BPTT)
        input_gate = F.linear(input_t, gate_W)  # gate_W @ x_t
        
        # Next hidden state with gated input
        Bx = F.linear(input_t, B)
        h_next = lamda * h_prev + gamma * Bx * input_gate
        
        # Derivatives for RTRL
        d_lamda_d_nu_log = -alpha * lamda * exp_nu_log * (1 - recurrence_gate)
        d_lamda_d_gate = alpha * lamda * exp_nu_log
        d_gate_d_input = recurrence_gate * (1 - recurrence_gate)
        
        # Update sensitivity for nu_log
        s_nu_log_next = h_prev * d_lamda_d_nu_log + lamda * s_nu_log_prev - (lamda / gamma) * Bx * input_gate * d_lamda_d_nu_log
        
        # Update sensitivity for V_nu (recurrence gate parameter)
        lamda_reshaped = lamda.unsqueeze(-1)  # [batch_size, state_features, 1]
        s_V_nu_next = lamda_reshaped * s_V_nu_prev
        
        # Add direct effect through the recurrence gate (outer product)
        gate_effect = d_lamda_d_gate * d_gate_d_input
        factor = (h_prev - (lamda / gamma) * Bx * input_gate) * gate_effect
        s_V_nu_next += factor.unsqueeze(2) * input_t.unsqueeze(1)
        
        # Update sensitivity for B (outer product)
        s_B_next = lamda_reshaped * s_B_prev
        s_B_next += (gamma * input_gate).unsqueeze(2) * input_t.unsqueeze(1)
        
        # Update sensitivity for gate_W (outer product)
        s_gate_W_next = lamda_reshaped * s_gate_W_prev
        factor2 = gamma * Bx
        s_gate_W_next += factor2.unsqueeze(2) * input_t.unsqueeze(1)
        
        # Save everything needed for backward - adding B, V_nu, and gate_W to saved tensors
        ctx.save_for_backward(h_prev, lamda, gamma, input_gate, Bx, recurrence_gate, 
                             s_nu_log_next, s_V_nu_next, s_B_next, s_gate_W_next, B, V_nu, gate_W)
        ctx.alpha = alpha
        ctx.exp_nu_log = exp_nu_log
        
        return h_next, s_nu_log_next, s_V_nu_next, s_B_next, s_gate_W_next

    @staticmethod
    def backward(ctx, grad_h_next, grad_s_nu_log_next, grad_s_V_nu_next, grad_s_B_next, grad_s_gate_W_next):
        # Retrieve saved tensors
        h_prev, lamda, gamma, input_gate, Bx, recurrence_gate, \
        s_nu_log, s_V_nu, s_B, s_gate_W, B, V_nu, gate_W = ctx.saved_tensors
        alpha = ctx.alpha
        exp_nu_log = ctx.exp_nu_log
        
        # Parameter gradients - ensure proper sum over batch dimension
        grad_nu_log = torch.sum(grad_h_next * s_nu_log, dim=0)
        grad_V_nu = torch.sum(grad_h_next.unsqueeze(-1) * s_V_nu, dim=0)
        grad_B = torch.sum(grad_h_next.unsqueeze(-1) * s_B, dim=0)
        grad_gate_W = torch.sum(grad_h_next.unsqueeze(-1) * s_gate_W, dim=0)
        
        grad_input_B = torch.matmul((grad_h_next * gamma * input_gate), B)
        d_recurrence_gate_d_input = recurrence_gate * (1 - recurrence_gate)
        d_lamda_d_recurrence_gate = alpha * exp_nu_log * lamda
        
        # Safely compute division
        safe_gamma = gamma.clamp(min=1e-6)
        d_h_d_recurrence = (h_prev - (lamda / safe_gamma) * Bx * input_gate) * d_lamda_d_recurrence_gate
        
        grad_input_V = torch.matmul((grad_h_next * d_h_d_recurrence * d_recurrence_gate_d_input), V_nu)
        grad_input_W = torch.matmul((grad_h_next * gamma * Bx), gate_W)
        
        # Combined input gradient
        grad_input_t = grad_input_B + grad_input_V + grad_input_W
        
        # No gradients needed for detached tensors
        grad_h_prev = None
        grad_s_nu_log_prev = None
        grad_s_V_nu_prev = None
        grad_s_B_prev = None
        grad_s_gate_W_prev = None
        grad_alpha = None
        
        return grad_input_t, grad_h_prev, grad_nu_log, grad_V_nu, grad_B, grad_gate_W, grad_s_nu_log_prev, grad_s_V_nu_prev, grad_s_B_prev, grad_s_gate_W_prev, grad_alpha

class GatedRealLRURTRL(nn.Module):
    """
    Gated diagonal RNN cell with RTRL (matching the BPTT implementation).
    
    The recurrence is:
        recurrence_gate = sigmoid(recurrence_gate @ x_t)
        lambda = exp(-alpha * exp(nu_log) * (1-recurrence_gate))
        gamma = sqrt(1 - lambda^2 + epsilon)
        h_t = lambda * h_{t-1} + gamma * (B * x_t) * (gate_W @ x_t)
    """
    def __init__(self, in_features, out_features, state_features, rmin=0.0, rmax=1.0, use_internal_flag=False, **kwargs):
        super().__init__()

        self.in_features = in_features
        self.state_features = state_features
        self.out_features = out_features
        self.alpha = 3.0
        self.use_internal_flag = use_internal_flag

        # Initialize parameters
        u1 = torch.rand(state_features)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin**2)))
        
        # Input dimensions depend on whether we use internal flag
        in_dim = in_features + 1 if use_internal_flag else in_features
        
        # Recurrence gate parameter (equivalent to recurrence_gate in BPTT)
        self.V_nu = nn.Parameter(torch.randn(state_features, in_dim) / math.sqrt(in_dim))
        
        # Input gate parameter (equivalent to gate_W in BPTT)
        self.gate_W = nn.Parameter(torch.randn(state_features, in_dim) / math.sqrt(in_dim))
        
        # Input-to-state and state-to-output matrices
        self.B = nn.Parameter(torch.randn(state_features, in_dim) / math.sqrt(in_dim))

        self.reset_rtrl_state()
        
    def reset_rtrl_state(self, batch_size: int = 1) -> None:
        """Resets hidden state and RTRL sensitivities to zero with batch dimension."""
        device = self.nu_log.device
        self.h = torch.zeros(batch_size, self.state_features, device=device)
        self.s_nu_log = torch.zeros(batch_size, self.state_features, device=device)
        self.s_V_nu = torch.zeros(batch_size, self.state_features, self.V_nu.size(1), device=device)
        self.s_B = torch.zeros(batch_size, self.state_features, self.B.size(1), device=device)
        self.s_gate_W = torch.zeros(batch_size, self.state_features, self.gate_W.size(1), device=device)
        self.batch_size = batch_size

    def forward_step(self, input_t: torch.Tensor, hidden=None, apply_change: bool = True, inner_loops: int = 1) -> torch.Tensor:
        """Process a single time step with optional inner loops (N-1 detached, 1 with grad).
           Expects input_t: [batch, in_features] and hidden: [batch, state_features]."""
        assert input_t.dim() == 2, "Input must be [batch, in_features]."
        batch = input_t.size(0)
        if hidden is None:
            self.reset_rtrl_state(batch_size=batch)
            h = self.h
        else:
            h = hidden

        # Choose local sensitivity tensors; avoid mutating module state when apply_change=False
        if apply_change:
            s_nu_log = self.s_nu_log
            s_V_nu = self.s_V_nu
            s_B = self.s_B
            s_gate_W = self.s_gate_W
        else:
            s_nu_log = self.s_nu_log.clone()
            s_V_nu = self.s_V_nu.clone()
            s_B = self.s_B.clone()
            s_gate_W = self.s_gate_W.clone()

        loops = max(1, int(inner_loops))

        # N-1 warmup passes without tracking gradients
        for _ in range(loops - 1):
            # Prepare input with flag for warmup
            if self.use_internal_flag:
                flag = input_t.new_zeros(input_t.size(0), 1)
                input_with_flag = torch.cat([input_t, flag], dim=-1)
            else:
                input_with_flag = input_t
                
            with torch.no_grad():
                h_next, s_nu_log_next, s_V_nu_next, s_B_next, s_gate_W_next = GatedRealLRUFunction.apply(
                    input_with_flag, h, self.nu_log, self.V_nu, self.B, self.gate_W,
                    s_nu_log, s_V_nu, s_B, s_gate_W,
                    self.alpha
                )
            # Update local (or module) state after detached pass
            if apply_change:
                self.h = h_next.detach()
                self.s_nu_log = s_nu_log_next.detach()
                self.s_V_nu = s_V_nu_next.detach()
                self.s_B = s_B_next.detach()
                self.s_gate_W = s_gate_W_next.detach()
                # Keep local references in sync
                s_nu_log, s_V_nu, s_B, s_gate_W = self.s_nu_log, self.s_V_nu, self.s_B, self.s_gate_W
                h = self.h
            else:
                h = h_next.detach()
                s_nu_log = s_nu_log_next.detach()
                s_V_nu = s_V_nu_next.detach()
                s_B = s_B_next.detach()
                s_gate_W = s_gate_W_next.detach()

        # Final pass tracked by autograd
        if self.use_internal_flag:
            flag = input_t.new_ones(input_t.size(0), 1)
            input_with_flag = torch.cat([input_t, flag], dim=-1)
        else:
            input_with_flag = input_t
            
        h_next, s_nu_log_next, s_V_nu_next, s_B_next, s_gate_W_next = GatedRealLRUFunction.apply(
            input_with_flag, h, self.nu_log, self.V_nu, self.B, self.gate_W,
            s_nu_log, s_V_nu, s_B, s_gate_W,
            self.alpha
        )

        if apply_change:
            self.h = h_next.detach()
            self.s_nu_log = s_nu_log_next.detach()
            self.s_V_nu = s_V_nu_next.detach()
            self.s_B = s_B_next.detach()
            self.s_gate_W = s_gate_W_next.detach()

        output = h_next
        return output, h_next

    def forward_sequential(self, x_sequence: torch.Tensor, hidden=None, apply_change=True, inner_loops: int = 1) -> torch.Tensor:
        """Process a sequence step by step with optional inner loops handled in forward_step.
           Expects x_sequence: [seq_len, batch, in_features]."""
        seq_len, batch_size, _ = x_sequence.shape
        
        outputs = []
        if hidden is None:
            self.reset_rtrl_state(batch_size)
            hidden = self.h
        
        for t in range(seq_len):
            x_t = x_sequence[t]
            out, hidden = self.forward_step(x_t, hidden, apply_change=apply_change, inner_loops=inner_loops)
            outputs.append(out)
        outputs = torch.stack(outputs, dim=0)
        return outputs, hidden, None

    def forward(self, x: torch.Tensor, hidden=None, apply_change=True, inner_loops: int = 1, **kwargs) -> torch.Tensor:
        return self.forward_sequential(x, hidden, apply_change=apply_change, inner_loops=inner_loops)