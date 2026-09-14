"""
MODULE: positional_encoding.py

WHAT:
    Injects sequence order information into the patch embeddings.

WHY:
    The Transformer's self-attention mechanism is permutation equivariant —
    it treats the input as an unordered set (a "bag of patches"). Without
    positional information, the model wouldn't know whether a patch occurred
    at the start or end of the movement. We add a positional encoding vector
    to each patch embedding so the attention mechanism knows its location in time.

MATH:
    Input x: shape (B, N_p, D)
    Pos_enc: Learnable parameter of shape (1, max_len, D)
    We slice Pos_enc to match the sequence length N_p, and add it to x.
    Output: x + Pos_enc[:, :N_p, :]
    
    Why Learnable instead of Sinusoidal?
    For relatively short, fixed-length patch sequences (e.g., 8 tokens for UCI,
    10 tokens for SisFall), learnable embeddings empirically perform at least
    as well as sinusoidal and are simpler to implement. The final locked design
    specifies Learnable.

TENSOR SHAPES:
    Input x      : (B, 8, 128)
    pos_embed    : (1, 8, 128)   (assuming max_len=8)
    Output       : (B, 8, 128)   (broadcast addition over batch dim)

VIVA ANSWER:
    "Self-attention naturally ignores sequence order. To fix this, I initialize
    a learnable parameter matrix containing a unique vector for every possible 
    patch position. I add this position vector directly to the corresponding
    patch embedding so the model can distinguish temporal relationships."
"""

import torch
import torch.nn as nn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, embed_dim: int):
        """
        Args:
            max_len (int): Maximum possible sequence length (e.g. 10 for SisFall)
            embed_dim (int): Embedding dimension (D=128)
        """
        super().__init__()
        # Learnable parameter matrix of shape (1, max_len, D)
        # Using 1 in the batch dimension allows seamless broadcasting
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        # Standard initialization for learnable embeddings (trunc normal)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N_p, D)
        Returns: (B, N_p, D)
        """
        B, N_p, D = x.shape
        
        assert N_p <= self.pos_embed.size(1), (
            f"Input sequence length {N_p} exceeds maximum {self.pos_embed.size(1)}"
        )
        assert D == self.pos_embed.size(2), (
            f"Input embedding dim {D} does not match pos_embed dim {self.pos_embed.size(2)}"
        )
        
        # Add positional embedding to x (broadcasting over batch dimension)
        # Slicing creates a view to the exact length of the current sequence
        return x + self.pos_embed[:, :N_p, :]


if __name__ == "__main__":
    # Self-test: SisFall configuration
    B, N_p, D = 32, 10, 128
    
    model = LearnablePositionalEncoding(max_len=10, embed_dim=D)
    x_test = torch.zeros(B, N_p, D)
    out = model(x_test)
    
    print("POSITIONAL ENCODING TEST (SisFall scale)")
    print(f"  Input  : {tuple(x_test.shape)}")
    print(f"  PosEmb parameter shape: {tuple(model.pos_embed.shape)}")
    print(f"  Output : {tuple(out.shape)}")
    assert out.shape == (B, N_p, D), "Shape mismatch!"
    
    # Check that broadcast worked correctly: all batches should have same pos enc
    assert torch.allclose(out[0], out[1]), "Broadcasting failed across batch!"
    print("  Status : PASS")
