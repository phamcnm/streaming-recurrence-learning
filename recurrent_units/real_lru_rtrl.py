import torch, math
import torch.nn as nn
import torch.nn.functional as F

class RealLRUFunction(torch.autograd.Function):
    """
    Implements a single time-step of a diagonal RNN cell with RTRL and normalization.
    
    The recurrence is:
        h_t = λ ⊙ h_{t-1} + γ ⊙ (B * x_t')
    where x_t' is the pre-processed input and
        γ = sqrt(1 - λ² + 1e-7).
    """
    @staticmethod
    def forward(ctx, input_t, h_prev, lambda_log, B, s_lambda_prev, s_B_prev):
        # Calculate decay rates
        lamda = torch.exp(-torch.exp(lambda_log))
        gamma = torch.sqrt(1 - lamda**2 + 1e-7)
        
        # Next hidden state
        Bx = F.linear(input_t, B)
        h_next = lamda * h_prev + gamma * Bx
        
        # Derivatives for RTRL
        d_lamda_d_lambda_log = -lamda * torch.exp(lambda_log)
        
        # Update sensitivity for lambda_log (dh/dλ_log)
        s_lambda_next = h_prev * d_lamda_d_lambda_log + lamda * s_lambda_prev
        
        # Additional term for lambda_log: dh/dγ * dγ/dλ * dλ/dλ_log
        # dγ/dλ = -λ / γ
        # Add this term to properly account for the effect through γ
        dg_dl = -lamda / gamma
        s_lambda_next += Bx * dg_dl * d_lamda_d_lambda_log
        
        # Update sensitivity for B
        # Fix the dimension issue by ensuring gamma has batch dimension
        gamma_batch = gamma.expand_as(h_prev)  # Expand to match batch dimension
        lamda_reshaped = lamda.unsqueeze(-1)  # [batch_size, state_features, 1]
        
        # Use proper outer product handling for batch dimension
        gamma_expanded = gamma_batch.unsqueeze(-1)  # [batch_size, state_features, 1]
        input_expanded = input_t.unsqueeze(1)       # [batch_size, 1, in_features]
        s_B_direct = gamma_expanded * input_expanded  # [batch_size, state_features, in_features]
        s_B_next = lamda_reshaped * s_B_prev + s_B_direct
        
        # Save for backward
        ctx.save_for_backward(gamma_batch, s_lambda_next, s_B_next, B)
        
        return h_next, s_lambda_next, s_B_next

    @staticmethod
    def backward(ctx, grad_h_next, grad_s_lambda_next, grad_s_B_next):
        # Retrieve saved tensors
        gamma, s_lambda, s_B, B = ctx.saved_tensors
        
        # Parameter gradients - ensure proper sum over batch dimension
        grad_lambda_log = torch.sum(grad_h_next * s_lambda, dim=0)
        grad_B = torch.sum(grad_h_next.unsqueeze(-1) * s_B, dim=0)
        
        # Input gradient
        grad_input_t = torch.matmul((grad_h_next * gamma), B)
        
        # No gradients needed for detached tensors
        grad_h_prev = None
        grad_s_lambda_prev = None
        grad_s_B_prev = None
        
        return grad_input_t, grad_h_prev, grad_lambda_log, grad_B, grad_s_lambda_prev, grad_s_B_prev

class RealLRURTRL(nn.Module):
    """
    Diagonal RNN cell with RTRL (using normalization).
    
    The recurrence is:
        h_t = λ ⊙ h_{t-1} + γ ⊙ (B * x_t)
    with γ = sqrt(1-λ²).
    
    This implementation supports both sequential processing and parallel scan
    for efficient forward and backward passes on long sequences.
    """
    def __init__(self, in_features, out_features, state_features, rmin=0.0, rmax=1.0, use_internal_flag=False, **kwargs):
        super().__init__()

        self.in_features = in_features
        self.state_features = state_features
        self.out_features = out_features
        self.use_internal_flag = use_internal_flag

        # Initialize parameters
        u1 = torch.rand(state_features)
        self.lambda_log = nn.Parameter(torch.log(-0.5*torch.log(u1*(rmax+rmin)*(rmax-rmin) + rmin**2)))
        in_dim = in_features + 1 if use_internal_flag else in_features
        self.B = nn.Parameter(torch.randn(state_features, in_dim) / math.sqrt(in_dim))
        self.C = nn.Parameter(torch.randn(out_features, state_features) / math.sqrt(state_features))

        self.reset_rtrl_state()

    def reset_rtrl_state(self, batch_size: int = 1) -> None:
        """Resets hidden state and RTRL sensitivities to zero with batch dimension."""
        device = self.lambda_log.device
        self.h = torch.zeros(batch_size, self.state_features, device=device)
        self.s_lambda = torch.zeros(batch_size, self.state_features, device=device)
        self.s_B = torch.zeros(batch_size, self.state_features, self.B.size(1), device=device)
        self.batch_size = batch_size

    def forward_step(self, input_t: torch.Tensor, hidden=None, apply_change: bool = True, inner_loops: int = 1) -> torch.Tensor:
        """Process a single time step with optional inner loops.
           Expects input_t: [batch, in_features] and hidden: [batch, state_features]."""
        assert input_t.dim() == 2, "Input must be [batch, in_features]."
        if hidden is None:
            hidden = self.h

        # Choose sensitivity tensors; avoid mutating module state when apply_change=False
        if apply_change:
            s_lambda = self.s_lambda
            s_B = self.s_B
        else:
            s_lambda = self.s_lambda.clone()
            s_B = self.s_B.clone()

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
                h_next, s_lambda_next, s_B_next = RealLRUFunction.apply(
                    input_with_flag, hidden, self.lambda_log, self.B,
                    s_lambda, s_B
                )
            if apply_change:
                self.h = h_next.detach()
                self.s_lambda = s_lambda_next.detach()
                self.s_B = s_B_next.detach()
                hidden, s_lambda, s_B = self.h, self.s_lambda, self.s_B
            else:
                hidden = h_next.detach()
                s_lambda = s_lambda_next.detach()
                s_B = s_B_next.detach()

        # Final tracked pass
        if self.use_internal_flag:
            flag = input_t.new_ones(input_t.size(0), 1)
            input_with_flag = torch.cat([input_t, flag], dim=-1)
        else:
            input_with_flag = input_t
            
        h_next, s_lambda_next, s_B_next = RealLRUFunction.apply(
            input_with_flag, hidden, self.lambda_log, self.B,
            s_lambda, s_B
        )

        if apply_change:
            self.h = h_next.detach()
            self.s_lambda = s_lambda_next.detach()
            self.s_B = s_B_next.detach()

        # Compute output [batch, out_features]
        output = torch.matmul(h_next, self.C.t())

        return output, h_next

    def forward_sequential(self, x_sequence: torch.Tensor, hidden=None, apply_change: bool = True, inner_loops: int = 1) -> torch.Tensor:
        """Process a sequence step by step, handling batches appropriately.
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

    def forward(self, x: torch.Tensor, hidden=None, apply_change: bool = True, inner_loops: int = 1, **kwargs) -> torch.Tensor:
        return self.forward_sequential(x, hidden, apply_change=apply_change, inner_loops=inner_loops)