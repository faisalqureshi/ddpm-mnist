import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import math

# MLP AE: 784 -> ... -> 16 -> ... -> 784
class MLPEncoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),            # (B,1,28,28) -> (B,784)
            nn.Linear(784, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )
        self.name = 'mlp-ae'

    def forward(self, x):
        return self.net(x)           # (B,latent_dim)

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int = 16, use_sigmoid: bool = True):
        super().__init__()
        layers = [
            nn.Linear(latent_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256),        nn.ReLU(inplace=True),
            nn.Linear(256, 512),        nn.ReLU(inplace=True),
            nn.Linear(512, 784)
        ]
        self.core = nn.Sequential(*layers)
        self.head = nn.Sigmoid() if use_sigmoid else nn.Tanh()
        self.name = "ae-mlp"

    def forward(self, z):
        x = self.core(z)                  # (B,784)
        x = self.head(x)
        return x.view(z.size(0), 1, 28, 28)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 16, use_sigmoid: bool = True):
        super().__init__()
        self.encoder = MLPEncoder(latent_dim)
        self.decoder = MLPDecoder(latent_dim, use_sigmoid)

    def initialize_weights(self):
        pass

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z