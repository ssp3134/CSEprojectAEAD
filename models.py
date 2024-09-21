# Defines the Autoencoder neural network architecture using PyTorch.

# Key Components
# AEArch Class
# Purpose: Implements the Autoencoder architecture.
# Architecture:
# Encoder: Compresses the input data into a latent space.
# Encoded Space: A bottleneck layer representing the compressed data.
# Decoder: Reconstructs the data from the latent representation.
# Activation Functions: Uses SELU (Scaled Exponential Linear Unit) for non-linearity.
# Forward Method: Defines the forward pass through the network.

from torch import nn 

class AEArch(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window_size, 32),
            nn.SELU(),
            nn.Linear(32, 16),
            nn.SELU(),
            nn.Linear(16, 8),
            nn.SELU(),
            nn.Linear(8, 4),
            nn.SELU()
        )
        self.encoded_space = nn.Linear(4, 4)
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.SELU(),
            nn.Linear(8, 16),
            nn.SELU(),
            nn.Linear(16, 32),
            nn.SELU(),
            nn.Linear(32, window_size)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        enc = self.encoded_space(x)
        return self.decoder(enc)