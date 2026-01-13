import torch, math
import torch.nn as nn
import torch.nn.functional as F

class GatedRealLRUBPTTMin(nn.Module):
    """
    Gated Linear Recurrent Unit
    Real-valued LRU with backpropagation through time
    
    r_gate = sigmoid(recurrence_gate @ x_t)
    λ = exp(-alpha * nu * (1 - r_gate))
    γ = sqrt(1 - λ^2 + epsilon)
    h_t = λ * h_{t-1} + γ * (B @ x_t) * (gate_W @ x_t)
    """
    def __init__(self, in_features, out_features, state_features, rmin=0.0, rmax=1.0, mode='sequential', use_internal_flag=True):
        super().__init__()
        self.alpha = 3.0
        self.out_features = out_features
        self.state_features = state_features
        self.in_features = in_features
        self.use_internal_flag = use_internal_flag

        # Initialize decay parameters
        u1 = torch.rand(state_features)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin**2)))
        # Input projection matrices
        in_dim = in_features + 1 if use_internal_flag else in_features
        self.B = nn.Parameter(torch.randn(state_features, in_dim) / math.sqrt(in_dim))
        self.gate_W = nn.Parameter(torch.randn(state_features, in_dim) / math.sqrt(in_dim))
        self.V_nu = nn.Parameter(torch.randn(state_features, in_dim) / math.sqrt(in_dim))

    def reset_state(self, batch_size: int = 1) -> None:
        pass

    def glru_unit(self, x_t, state):
        # Compute gates and projections
        recurrence_gate = torch.sigmoid(F.linear(x_t, self.V_nu))
        exp_nu_log = torch.exp(self.nu_log)
        alpha_nu = self.alpha * exp_nu_log
        lamda = torch.exp(-alpha_nu * (1 - recurrence_gate))
        gammas = (1 - lamda.pow(2) + 1e-7).sqrt_()

        # Compute input projections
        b_x = F.linear(x_t, self.B)
        gate_x = F.linear(x_t, self.gate_W)

        # Update state
        state = lamda * state + gammas * b_x * gate_x

        # Compute output
        output = state
        return output, state


    def forward_step(self, x_t, hidden=None, inner_loops: int = 1):

        batch_size, feature_dim = x_t.shape

        # Initialize state with batch dimension
        if hidden is None:
            state = torch.zeros(batch_size, self.state_features, device=x_t.device, dtype=x_t.dtype)
        else:
            state = hidden

        with torch.no_grad():
            for _ in range(inner_loops-1):
                x_inner = x_t
                if self.use_internal_flag:
                    flag = x_t.new_zeros(batch_size, 1)
                    x_inner = torch.cat([x_t, flag], dim=-1)
                _, state = self.glru_unit(x_inner, state)
        x_tracked = x_t
        if self.use_internal_flag:
            flag = x_t.new_ones(batch_size, 1)
            x_tracked = torch.cat([x_t, flag], dim=-1)
        output, state = self.glru_unit(x_tracked, state)
        return output, state

    def forward_sequential(self, input, hidden=None, inner_loops: int = 1):
        """
        Forward pass for the GatedLRUBPTT with batch support.
        
        Args:
            input: Input tensor of shape (seq_len, batch_size, features)
            hidden: Optional state tensor or None
            
        Returns:
            output: Output tensor of shape (seq_len, batch_size, out_features)
            hidden_final: Final state tensor of shape (batch_size, state_features)
        """
        seq_len, batch_size, _ = input.shape
        
        # Initialize state with batch dimension
        if hidden is None:
            state = torch.zeros(batch_size, self.state_features, device=input.device, dtype=input.dtype)
        else:
            state = hidden
        
        output = torch.empty(seq_len, batch_size, self.out_features, device=input.device, dtype=input.dtype)
        
        for i in range(seq_len):
            output[i], state = self.forward_step(input[i], state, inner_loops=inner_loops)
        # Return output and final state
        return output, state

    def forward(self, x, hidden=None, done=None, apply_change: bool = True, inner_loops: int = 1):
        outputs, h_final = self.forward_sequential(x, hidden, inner_loops)            
        return outputs, h_final, None