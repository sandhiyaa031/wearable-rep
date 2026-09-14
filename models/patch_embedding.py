"""
MODULE: patch_embedding.py

WHAT:
    Converts a raw time-series window into a sequence of patch embeddings.

WHY:
    The Transformer operates on sequences of tokens. Continuous time-series
    data lacks natural "tokens" (like words in NLP). We create tokens by
    dividing the time axis into non-overlapping patches (length P), flattening
    the channels within each patch, and projecting them down to D dimensions.
    This also dramatically reduces the sequence length for the Transformer,
    making self-attention computationally cheaper (O(N_p^2) instead of O(T^2)).

MATH:
    Input x: shape (B, T, C)
    We want P timesteps per patch.
    1. Reshape x to (B, T/P, P, C)
    2. Flatten the last two dimensions to (B, T/P, P*C)
    3. Apply Linear layer: (P*C) -> D
    Output: (B, N_p, D) where N_p = T/P

TENSOR SHAPES (UCI HAR Example):
    Input:       (B, 128, 6)   (T=128 timesteps, C=6 channels)
    Patch size:  P=16
    Reshaped:    (B, 8, 16, 6)
    Flattened:   (B, 8, 96)    (16 * 6 = 96 raw features per patch)
    Projected:   (B, 8, 128)   (if Embedding dim D = 128)

VIVA ANSWER:
    "To feed IMU data into a Transformer, I divide each time-series window
    into non-overlapping patches — for instance, taking 128 timesteps and
    grouping them into 8 patches of 16 timesteps each. I then flatten the
    channels in each patch and use a linear layer to project these raw values
    into a 128-dimensional embedding space, creating the sequence of tokens."
"""

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        """
        Args:
            in_channels (int): Number of raw sensors (e.g. 6 for UCI, 9 for SisFall)
            patch_size (int): Number of timesteps per patch (e.g. P=16)
            embed_dim (int): Target embedding dimension (e.g. D=128)
        """
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Linear projection from raw patch features to embedding space
        self.proj = nn.Linear(patch_size * in_channels, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        Returns: (B, N_p, D)
        """
        B, T, C = x.shape
        
        # Ensure T is perfectly divisible by patch_size
        assert T % self.patch_size == 0, f"Sequence length {T} must be divisible by patch size {self.patch_size}."
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}."
        
        N_p = T // self.patch_size
        
        # 1. Reshape to (B, N_p, P, C)
        x = x.view(B, N_p, self.patch_size, C)
        
        # 2. Flatten P and C -> (B, N_p, P*C)
        x = x.view(B, N_p, self.patch_size * C)
        
        # 3. Project -> (B, N_p, D)
        x = self.proj(x)
        
        return x


if __name__ == "__main__":
    # Self-test: UCI HAR configuration
    B, T, C = 32, 128, 6
    P = 16
    D = 128
    
    model = PatchEmbedding(in_channels=C, patch_size=P, embed_dim=D)
    x_test = torch.randn(B, T, C)
    out = model(x_test)
    
    print("PATCH EMBEDDING TEST (UCI HAR scale)")
    print(f"  Input  : {tuple(x_test.shape)}")
    print(f"  Output : {tuple(out.shape)}")
    assert out.shape == (B, 8, 128), "Shape mismatch!"
    print("  Status : PASS")
