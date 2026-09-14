"""
MODULE: transformer_block.py

WHAT:
    A single Transformer Encoder block using Pre-Layer Normalization architecture.

WHY:
    The original Transformer paper (Vaswani 2017) placed LayerNorm *after* the
    residual connection (Post-LN). Modern Transformers (like Vision Transformers)
    virtually all use Pre-LN (LayerNorm *before* the operation, adding the residual
    back un-normalized) because it provides significantly more stable gradients and
    removes the need for complex learning rate warm-up schemes.

MATH:
    Input x
    1. x_norm_1 = LayerNorm(x)
    2. attn_out = MHSA(x_norm_1)
    3. x = x + attn_out           (Residual 1)
    4. x_norm_2 = LayerNorm(x)
    5. ffn_out  = FFN(x_norm_2)
    6. x = x + ffn_out            (Residual 2)
    
    FFN = Linear(D, 2*D) -> GELU -> Dropout -> Linear(2*D, D) -> Dropout
    (Final locked design specifies 256 for FFN hidden dim when D=128, which is 2*D)

TENSOR SHAPES:
    Input x:     (B, N_p, D)
    LayerNorm 1: (B, N_p, D)
    MHSA out:    (B, N_p, D)
    Residual 1:  (B, N_p, D)
    LayerNorm 2: (B, N_p, D)
    FFN up:      (B, N_p, 2D)
    FFN down:    (B, N_p, D)
    Residual 2:  (B, N_p, D)

VIVA ANSWER:
    "I implemented the Transformer encoder block using a Pre-Layer Normalization
    architecture, which is standard for modern transformers like ViT to ensure
    gradient stability without heavy warmup. It consists of two sub-layers: a
    multi-head self attention layer and a feed-forward network, each preceded
    by layer norm and bypassed by a residual connection. For the FFN, I used
    an expansion factor of 2, expanding the 128 dimensions to 256, applying GELU,
    and projecting back."
"""

import torch
import torch.nn as nn
from models.attention import MultiHeadAttention

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_p: float = 0.1):
        super().__init__()
        
        # We need attention weights inside for visualization later
        self.mhsa = MultiHeadAttention(embed_dim, num_heads, dropout_p)
        self.ln_1 = nn.LayerNorm(embed_dim)
        
        # Feed Forward Network (expansion factor 2 as per locked design)
        ffn_hidden = embed_dim * 2
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(ffn_hidden, embed_dim),
            nn.Dropout(dropout_p)
        )
        self.ln_2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor):
        """
        x: (B, N_p, D)
        Returns:
            out: (B, N_p, D)
            attn_weights: (B, H, N_p, N_p) - from this block's MHSA layer
        """
        # Block 1: Pre-LN MHSA + Residual
        # ln_1 applied before mhsa, added to UN-normalized x
        norm_x = self.ln_1(x)
        attn_out, attn_weights = self.mhsa(norm_x)
        x = x + attn_out
        
        # Block 2: Pre-LN FFN + Residual
        norm_x2 = self.ln_2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + ffn_out
        
        return x, attn_weights

if __name__ == "__main__":
    # Self-test:
    B, N_p, D = 32, 8, 128
    block = TransformerEncoderBlock(embed_dim=D, num_heads=4)
    x_test = torch.randn(B, N_p, D)
    
    out, weights = block(x_test)
    
    print("TRANSFORMER BLOCK TEST")
    print(f"  Input        : {tuple(x_test.shape)}")
    print(f"  Output       : {tuple(out.shape)} (expect (32, 8, 128))")
    print(f"  Attn weights : {tuple(weights.shape)}")
    
    assert out.shape == (B, N_p, D), "Output shape mismatch!"
    print("  Status       : PASS")
