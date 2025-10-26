import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import math

# Set this True for [0,1] targets (Sigmoid head), False for [-1,1] (Tanh head)
USE_SIGMOID = True

class Encoder(nn.Module):
    """Conv encoder: (1,28,28) -> (latent_dim,) via 28→14→7→4 downsampling."""
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 7  -> 4
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                 # 128*4*4
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        x = self.conv(x)      # (B,128,4,4)
        z = self.fc(x)        # (B,latent_dim)
        return z

class Decoder(nn.Module):
    """
    Upsample+Conv decoder: 4 -> 7 -> 14 -> 28 using explicit target sizes.
    This avoids deconv artifacts and size off-by-ones.
    """
    def __init__(self, latent_dim: int = 16, use_sigmoid: bool = True):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        # simple conv blocks after each upsample
        self.conv4  = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.conv7  = nn.Sequential(nn.Conv2d(128,  64, 3, padding=1), nn.ReLU(inplace=True))
        self.conv14 = nn.Sequential(nn.Conv2d( 64,  32, 3, padding=1), nn.ReLU(inplace=True))
        self.conv28 = nn.Conv2d(32, 1, 3, padding=1)  # final logits
        self.conv28.is_last = True

        self.head = nn.Sigmoid() if use_sigmoid else nn.Tanh()

    def forward(self, z):
        B = z.size(0)
        x = self.fc(z).view(B, 128, 4, 4)        # (B,128,4,4)

        x = self.conv4(x)                         # keep (4,4)
        x = F.interpolate(x, size=(7, 7), mode="nearest")
        x = self.conv7(x)                         # (7,7)

        x = F.interpolate(x, size=(14, 14), mode="nearest")
        x = self.conv14(x)                        # (14,14)

        x = F.interpolate(x, size=(28, 28), mode="nearest")
        x = self.conv28(x)                        # (28,28) logits or pre-activation

        x = self.head(x)                          # [0,1] (Sigmoid) or [-1,1] (Tanh)
        return x

class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 16, use_sigmoid: bool = USE_SIGMOID):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, use_sigmoid)

    def _initalize_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            if getattr(m, "is_last", False):
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def initialize_weights(self):
        with torch.no_grad():
            self.apply(self._initalize_weights)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z