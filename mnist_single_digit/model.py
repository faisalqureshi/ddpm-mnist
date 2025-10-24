#
# model.py 
#

import math
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(t, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """
    Sinusoidal embeddings

    t: (B,) int 
    Returns (B, dim) sinusoidal embedding
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=device) / half)
    angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)         # (B, half)
    emb = torch.cat([angles.sin(), angles.cos()], dim=1)         # (B, 2*half)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

class TimeMLP(nn.Module):
    """
    Learnable time embeddings.  Uses an MLP with SiLU activation functions.
    """
    def __init__(self, in_dim=128, hidden=256, out_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
    
    def forward(self, t) -> torch.Tensor:  # t: (B,)
        pe = timestep_embedding(t, self.fc1.in_features)
        return F.silu(self.fc2(F.silu(self.fc1(pe))))
    
class FiLMCond(nn.Module):
    """
    Conditions on time embeddings.  Here time embeddings are used to compute \\gamma and \\beta, 
    which are subsequently used to modulate x
    
    Group Norm -> (1 + \\gamma) h + \\beta -> SiLU -> Conv
    """

    def __init__(self, C: int, time_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, C)
        self.proj = nn.Linear(time_dim, 2 * C)  
        self.conv = nn.Conv2d(C, C, kernel_size=3, padding=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W)
        Returns time conditioned x (B,C,H,W)
        """

        # Group Norm
        h = self.norm(x)
        # Compute \gamma and \beta from time embeddings
        gam, bet = self.proj(t_emb).chunk(2, dim=1)  # (B,C), (B,C)
        # (1 + \gamma) h + \beta
        h = (1 + gam)[:, :, None, None] * h + bet[:, :, None, None]
        # SiLU activation func
        h = F.silu(h)
        # Convolution
        return self.conv(h)
    
class Denoiser(nn.Module):
    """
    Predicts noise
    """

    def __init__(self, base_ch: int = 64, time_dim: int = 256):
        super().__init__()

        self.time = TimeMLP(in_dim=128, hidden=256, out_dim=time_dim)            
        self.in_conv = nn.Conv2d(1, base_ch, 3, padding=1)
        self.b1a = FiLMCond(base_ch, time_dim); 
        self.b1b = FiLMCond(base_ch, time_dim)
        self.b2a = FiLMCond(base_ch, time_dim); 
        self.b2b = FiLMCond(base_ch, time_dim)
        self.b3a = FiLMCond(base_ch, time_dim); 
        self.b3b = FiLMCond(base_ch, time_dim)
        self.b4a = FiLMCond(base_ch, time_dim); 
        self.b4b = FiLMCond(base_ch, time_dim)
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: x_t
        t: t
        Returns noise \\epsilon (x_t, t)
        """

        # Construct time embedding    
        e = self.time(t)
        # Compute h from x using initial convolution
        h = self.in_conv(x)
        # Residual layers
        r = h; h = self.b1a(h, e); h = self.b1b(h, e); h = h + r
        r = h; h = self.b2a(h, e); h = self.b2b(h, e); h = h + r
        r = h; h = self.b3a(h, e); h = self.b3b(h, e); h = h + r
        r = h; h = self.b4a(h, e); h = self.b4b(h, e); h = h + r
        # Group norm followed by SiLU activation function
        h = F.silu(self.out_norm(h))
        # Another convolution
        return self.out_conv(h)  # predict ε

def precompute_schedules(T: int, beta_start: float, beta_end: float, device: torch.device) -> Dict[str, torch.Tensor]:
    beta = torch.linspace(beta_start, beta_end, T, device=device)
    alpha = 1.0 - beta
    abar = torch.cumprod(alpha, dim=0)
    sqrt_alpha = alpha.sqrt()
    sqrt_abar = abar.sqrt()
    sqrt_one_minus_abar = (1.0 - abar).sqrt()
    abar_prev = torch.cat([torch.tensor([1.0], device=device), abar[:-1]], dim=0)
    posterior_var = (1.0 - abar_prev) / (1.0 - abar + 1e-12) * beta
    posterior_var[0] = 0.0
    return {
        "beta": beta, "alpha": alpha, "abar": abar,
        "sqrt_alpha": sqrt_alpha, "sqrt_abar": sqrt_abar,
        "sqrt_one_minus_abar": sqrt_one_minus_abar,
        "posterior_var": posterior_var,
    }

def q_sample(x0: torch.Tensor, t: torch.Tensor, sched: Dict[str, torch.Tensor]):
    s = sched["sqrt_abar"][t].view(-1, 1, 1, 1)
    r = sched["sqrt_one_minus_abar"][t].view(-1, 1, 1, 1)
    eps = torch.randn_like(x0)
    x_t = s * x0 + r * eps
    return x_t, eps

@torch.no_grad()
def generate_images(model: nn.Module, sched: Dict[str, torch.Tensor], n: int = 16, device: Optional[torch.device] = None):
    model.eval()
    device = device or next(model.parameters()).device
    x = torch.randn(n, 1, 28, 28, device=device)
    beta = sched["beta"]; sqrt_alpha = sched["sqrt_alpha"]; sqrt_one_minus_abar = sched["sqrt_one_minus_abar"]
    post_var = sched["posterior_var"]
    T = beta.numel()

    for ti in reversed(range(T)):
        t = torch.full((n,), ti, device=device, dtype=torch.long)
        eps_hat = model(x, t)
        mean = (x - (beta[ti] / (sqrt_one_minus_abar[ti] + 1e-12)) * eps_hat) / (sqrt_alpha[ti] + 1e-12)
        if ti > 0:
            x = mean + torch.sqrt(post_var[ti]) * torch.randn_like(x)
        else:
            x = mean
    return (x.clamp(-1, 1) + 1) / 2  # [0,1]