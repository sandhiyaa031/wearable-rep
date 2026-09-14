"""
MODULE: attention.py

WHAT:
    Implements Scaled Dot-Product Attention and Multi-Head Self Attention (MHSA)
    entirely from scratch, without using torch.nn.MultiheadAttention.

WHY:
    The core of the Transformer. Building it from scratch demonstrates deep
    theoretical ownership of the architecture for the viva, and allows us to
    log/visualise internal attention weights later (which is an optional task
    in the final locked design).

MATH:
    Scaled Dot-Product Attention:
        Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) ) @ V
    
    Multi-Head Attention:
        Instead of one large attention operation, we project the input into
        H smaller subspaces (heads). 
        d_k = D / H
        Output = Concat(head_1, ..., head_H) @ W_O

TENSOR SHAPES:
    Input x:      (B, N_p, D)
    Q, K, V projs:(B, N_p, D)
    Reshaped QKV: (B, N_p, H, d_k)
    Transposed:   (B, H, N_p, d_k)   ← required for bmm (batch matrix mult)
    Scores:       (B, H, N_p, N_p)   ← Q @ K^T
    Attn weights: (B, H, N_p, N_p)   ← softmax(scores)
    Context:      (B, H, N_p, d_k)   ← weights @ V
    Concat:       (B, N_p, D)
    Output:       (B, N_p, D)

VIVA ANSWER:
    "I implemented multi-head attention from scratch to prove I understand it.
     It works by linearly projecting the input sequence into Queries, Keys, and Values.
     I reshape these into H separate heads, compute the dot-product similarity between 
     Queries and Keys, scale it by the square root of the head dimension to prevent
     gradient vanishing in the softmax, and use those normalized weights to mix the Values."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_p: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        q, k, v: (B, H, N_p, d_k)
        Returns:
            context: (B, H, N_p, d_k)
            weights: (B, H, N_p, N_p)
        """
        d_k = q.size(-1)
        
        # 1. Dot product: (B, H, N_p, d_k) @ (B, H, d_k, N_p) -> (B, H, N_p, N_p)
        # transpose(-2, -1) swaps the last two dimensions (N_p and d_k)
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # 2. Scale
        scores = scores / math.sqrt(d_k)
        
        # 3. Softmax to get probability distribution over keys for each query
        attn_weights = F.softmax(scores, dim=-1)
        
        # 4. Dropout on attention weights (standard practice)
        attn_weights_dropped = self.dropout(attn_weights)
        
        # 5. Weighted sum of values: (B, H, N_p, N_p) @ (B, H, N_p, d_k) -> (B, H, N_p, d_k)
        context = torch.matmul(attn_weights_dropped, v)
        
        return context, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_p: float = 0.1):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        
        # Linear projections for Q, K, V
        # Defining them as separate layers makes the code easier to read
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.attention = ScaledDotProductAttention(dropout_p)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor):
        """
        x: (B, N_p, D)
        Returns:
            out: (B, N_p, D)
            attn_weights: (B, H, N_p, N_p) - useful for visualization later
        """
        B, N_p, D = x.shape
        
        # 1. Linear projections
        q = self.q_proj(x)  # (B, N_p, D)
        k = self.k_proj(x)  # (B, N_p, D)
        v = self.v_proj(x)  # (B, N_p, D)
        
        # 2. Reshape and transpose to separate heads
        # (B, N_p, D) -> (B, N_p, H, d_k) -> (B, H, N_p, d_k)
        q = q.view(B, N_p, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, N_p, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, N_p, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Apply Scaled Dot-Product Attention
        context, attn_weights = self.attention(q, k, v)
        
        # 4. Transpose and flatten heads back together
        # (B, H, N_p, d_k) -> transpose(1, 2) -> (B, N_p, H, d_k)
        # contiguous() is required before view/reshape if tensor in memory is non-contiguous
        # -> (B, N_p, H * d_k) == (B, N_p, D)
        context = context.transpose(1, 2).contiguous().view(B, N_p, self.embed_dim)
        
        # 5. Final linear projection and dropout
        out = self.out_proj(context)
        out = self.dropout(out)
        
        return out, attn_weights


if __name__ == "__main__":
    # Self-test:
    B, N_p, D = 32, 8, 128
    num_heads = 4
    
    mhsa = MultiHeadAttention(embed_dim=D, num_heads=num_heads)
    x_test = torch.randn(B, N_p, D)
    
    # Test forward pass in eval mode (no dropout randomness)
    mhsa.eval()
    with torch.no_grad():
        out, weights = mhsa(x_test)
    
    print("MULTI-HEAD ATTENTION TEST")
    print(f"  Input        : {tuple(x_test.shape)}")
    print(f"  Output       : {tuple(out.shape)}  (expect (32, 8, 128))")
    print(f"  Attn weights : {tuple(weights.shape)}  (expect (32, 4, 8, 8))")
    
    assert out.shape == (B, N_p, D), "Output shape mismatch!"
    assert weights.shape == (B, num_heads, N_p, N_p), "Weights shape mismatch!"
    
    # Crucial mathematical property: attention weights must sum to 1 over the last dimension (keys)
    sum_weights = weights.sum(dim=-1)
    is_sum_one = torch.allclose(sum_weights, torch.ones_like(sum_weights))
    print(f"  Weights sum to 1? : {is_sum_one}")
    assert is_sum_one, "Attention weights do not sum to 1!"
    
    print("  Status       : PASS")
