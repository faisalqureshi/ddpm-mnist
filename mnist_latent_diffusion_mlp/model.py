#
# model.py - Latent Diffusion Model for MNIST
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

class FiLMCondMLP(nn.Module):
    """
    FiLM conditioning for MLP layers (latent space diffusion)

    LayerNorm -> (1 + γ) * h + β -> SiLU -> Linear
    """
    def __init__(self, latent_dim: int, time_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.proj = nn.Linear(time_dim, 2 * latent_dim)  # for γ and β
        self.fc = nn.Linear(latent_dim, hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, latent_dim)
        t_emb: (B, time_dim)
        Returns: (B, hidden_dim)
        """
        # Layer Norm
        h = self.norm(x)
        # Compute γ and β from time embeddings
        gam, bet = self.proj(t_emb).chunk(2, dim=1)  # (B, latent_dim) each
        # (1 + γ) * h + β
        h = (1 + gam) * h + bet
        # SiLU activation
        h = F.silu(h)
        # Linear projection
        return self.fc(h)

class LatentDenoiser(nn.Module):
    """
    MLP-based denoiser for latent space diffusion
    Operates on 16-dimensional latent vectors
    """
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 256, time_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        # Time embedding
        self.time = TimeMLP(in_dim=128, hidden=256, out_dim=time_dim)

        # Input projection
        self.in_proj = nn.Linear(latent_dim, hidden_dim)

        # Residual blocks with FiLM conditioning
        self.b1a = FiLMCondMLP(hidden_dim, time_dim, hidden_dim)
        self.b1b = FiLMCondMLP(hidden_dim, time_dim, hidden_dim)
        self.b2a = FiLMCondMLP(hidden_dim, time_dim, hidden_dim)
        self.b2b = FiLMCondMLP(hidden_dim, time_dim, hidden_dim)
        self.b3a = FiLMCondMLP(hidden_dim, time_dim, hidden_dim)
        self.b3b = FiLMCondMLP(hidden_dim, time_dim, hidden_dim)
        self.b4a = FiLMCondMLP(hidden_dim, time_dim, hidden_dim)
        self.b4b = FiLMCondMLP(hidden_dim, time_dim, hidden_dim)

        # Output layers
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z: latent vector z_t (B, latent_dim)
        t: timestep (B,)
        Returns: predicted noise ε (B, latent_dim)
        """
        # Construct time embedding
        e = self.time(t)

        # Input projection
        h = F.silu(self.in_proj(z))

        # Residual blocks
        r = h; h = self.b1a(h, e); h = self.b1b(h, e); h = h + r
        r = h; h = self.b2a(h, e); h = self.b2b(h, e); h = h + r
        r = h; h = self.b3a(h, e); h = self.b3b(h, e); h = h + r
        r = h; h = self.b4a(h, e); h = self.b4b(h, e); h = h + r

        # Output projection
        h = F.silu(self.out_norm(h))
        return self.out_proj(h)  # predict ε

def precompute_schedules(T: int, beta_start: float, beta_end: float, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Precompute diffusion schedule coefficients
    """
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

def q_sample(z0: torch.Tensor, t: torch.Tensor, sched: Dict[str, torch.Tensor]):
    """
    Forward diffusion process: add noise to latents

    z0: clean latent (B, latent_dim)
    t: timestep (B,)
    Returns: (z_t, eps) where z_t is noisy latent and eps is the noise
    """
    s = sched["sqrt_abar"][t].view(-1, 1)  # (B, 1)
    r = sched["sqrt_one_minus_abar"][t].view(-1, 1)  # (B, 1)
    eps = torch.randn_like(z0)
    z_t = s * z0 + r * eps
    return z_t, eps

@torch.no_grad()
def generate_latents(model: nn.Module, sched: Dict[str, torch.Tensor],
                     n: int = 16, latent_dim: int = 16,
                     device: Optional[torch.device] = None):
    """
    Generate latent vectors using the reverse diffusion process

    Returns: (B, latent_dim) latent vectors
    """
    model.eval()
    device = device or next(model.parameters()).device

    # Start from pure noise
    z = torch.randn(n, latent_dim, device=device)

    beta = sched["beta"]
    sqrt_alpha = sched["sqrt_alpha"]
    sqrt_one_minus_abar = sched["sqrt_one_minus_abar"]
    post_var = sched["posterior_var"]
    T = beta.numel()

    # Reverse diffusion
    for ti in reversed(range(T)):
        t = torch.full((n,), ti, device=device, dtype=torch.long)
        eps_hat = model(z, t)
        mean = (z - (beta[ti] / (sqrt_one_minus_abar[ti] + 1e-12)) * eps_hat) / (sqrt_alpha[ti] + 1e-12)
        if ti > 0:
            z = mean + torch.sqrt(post_var[ti]) * torch.randn_like(z)
        else:
            z = mean

    return z

@torch.no_grad()
def generate_images(model: nn.Module, decoder: nn.Module, sched: Dict[str, torch.Tensor],
                    n: int = 16, latent_dim: int = 16,
                    device: Optional[torch.device] = None):
    """
    Generate images by first generating latents, then decoding them

    model: latent diffusion model
    decoder: autoencoder decoder
    Returns: (B, 1, 28, 28) images in [0, 1] range
    """
    # Generate latents
    z = generate_latents(model, sched, n, latent_dim, device)

    # Decode to images
    decoder.eval()
    images = decoder(z)  # (B, 1, 28, 28)

    # Clamp to [0, 1] (decoder uses Sigmoid, so already in [0,1])
    return images.clamp(0, 1)
