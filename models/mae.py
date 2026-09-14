"""
MODULE: mae.py

WHAT:
    The complete Masked Autoencoder (MAE) model.
    Handles the complex logic of randomly masking patches, passing only visible
    patches to the Encoder, un-shuffling and inserting [MASK] tokens, passing
    the full sequence to the Decoder, and computing the MSE reconstruction loss
    only on the masked patches.

WHY:
    MAE is dramatically more efficient than standard autoencoders because the
    Encoder only processes visible patches (e.g., 25% of the sequence). The
    Decoder, which is typically much lighter (e.g. 2 layers, dim 64), processes
    the full sequence. This allows us to train Transformers on small hardware
    while learning robust, high-quality representations for IMU data.

MATH / MASKING LOGIC:
    1. Patch + PosEmbed full sequence  -> (B, N_p, D)
    2. Generate random noise per sequence -> (B, N_p)
    3. argsort noise to get a shuffled index (B, N_p)
    4. Slice first len_keep elements to get indices of visible patches
    5. Slice remainder to get indices of masked patches
    6. gather() visible patches from (B, N_p, D) -> (B, len_keep, D)
    7. Pass to Encoder -> (B, len_keep, D)
    8. Generate (B, len_mask, D) [MASK] tokens + add positional encoding
    9. Concat encoded visible + MASK tokens
    10. scatter() them back into their original shuffled positions -> (B, N_p, D)
    11. Pass to Decoder -> (B, N_p, D_dec) -> Linear(D_dec, patch_dim)
    12. Compute MSE between predictions and input patches, gathering only the 
        masked indices.

TENSOR SHAPES (UCI HAR, 75% masking):
    Input:        (B, 128, 6)
    Patched:      (B, 8, 128)
    Visible:      (B, 2, 128)  [8 * 0.25 = 2]
    Encoded:      (B, 2, 128)
    Mask tokens:  (B, 6, 128)  [8 * 0.75 = 6]
    Unshuffled:   (B, 8, 128)
    Decoded:      (B, 8, 64)
    Reconstruct:  (B, 8, 96)   [96 = 16 timesteps * 6 channels]
    Loss target:  (B, 6, 96)   [Gathered only masked positions]

VIVA ANSWER:
    "The core of my MAE implementation is the masking algorithm. I randomly
    mask 75% of the patches and pass ONLY the 25% visible patches through the 
    heavy 4-layer encoder. Before the 2-layer decoder, I re-insert learnable 
    [MASK] tokens at the exact original positions using `torch.scatter`, and add
    positional encodings again so the decoder knows which patches need to be 
    reconstructed. Loss is computed purely on the masked patches, forcing 
    the encoder to learn the underlying semantics of human motion, not just copy data."
"""

import torch
import torch.nn as nn
from models.transformer_encoder import TransformerEncoder
from models.transformer_block import TransformerEncoderBlock

class MAE(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 patch_size: int, 
                 max_len: int,
                 # Encoder params
                 enc_embed_dim: int = 128, 
                 enc_heads: int = 4, 
                 enc_layers: int = 4,
                 # Decoder params (lightweight)
                 dec_embed_dim: int = 64,  
                 dec_heads: int = 2, 
                 dec_layers: int = 2,
                 dropout_p: float = 0.1):
        super().__init__()
        
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_dim = patch_size * in_channels
        
        # ─── ENCODER ───
        self.encoder = TransformerEncoder(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=enc_embed_dim,
            num_heads=enc_heads,
            num_layers=enc_layers,
            max_len=max_len,
            dropout_p=dropout_p
        )
        
        # ─── ENCODER TO DECODER PROJECTION ───
        # Decoder often has a smaller dimension than encoder
        self.enc_to_dec = nn.Linear(enc_embed_dim, dec_embed_dim)
        
        # ─── DECODER ───
        # Mask token (learnable parameter, 1 for all masked patches)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Decoder positional encoding (we re-add position before decoding)
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, max_len, dec_embed_dim))
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)
        
        self.decoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(dec_embed_dim, dec_heads, dropout_p)
            for _ in range(dec_layers)
        ])
        
        self.dec_norm = nn.LayerNorm(dec_embed_dim)
        self.pred_head = nn.Linear(dec_embed_dim, self.patch_dim)
        
    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        """
        x: (B, N_p, D) already patched and projected representations
        Returns:
            x_vis:   (B, len_keep, D)   - patches to keep
            mask:    (B, N_p)           - binary mask (1 = masked, 0 = visible)
            ids_restore: (B, N_p)       - indices to restore original order
        """
        B, N_p, D = x.shape
        len_keep = int(N_p * (1 - mask_ratio))
        
        # Random noise to sort by
        noise = torch.rand(B, N_p, device=x.device)
        
        # Sort noise to get shuffled indices
        ids_shuffle = torch.argsort(noise, dim=1)           # (B, N_p)
        ids_restore = torch.argsort(ids_shuffle, dim=1)     # (B, N_p)
        
        # Split into kept (first len_keep) and masked (remainder)
        ids_keep = ids_shuffle[:, :len_keep]                # (B, len_keep)
        
        # Gather the visible patches
        # ids_keep.unsqueeze(-1).expand(-1, -1, D) -> (B, len_keep, D)
        gather_indices = ids_keep.unsqueeze(-1).expand(-1, -1, D)
        x_vis = torch.gather(x, dim=1, index=gather_indices) # (B, len_keep, D)
        
        # Generate the boolean mask for loss calculation (1 = masked, 0 = kept)
        mask = torch.ones([B, N_p], device=x.device)
        mask[:, :len_keep] = 0
        
        # Unshuffle the mask to original positions so we know WHICH patches are masked
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_vis, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float):
        """
        Applies patch embed, pos embed, random masking, then encoder blocks.
        """
        # 1. Patch & Pos Embed (on the full sequence)
        x = self.encoder.patch_embed(x)      # (B, N_p, D_enc)
        x = self.encoder.pos_embed(x)        # (B, N_p, D_enc)
        
        # 2. Random masking (keep only visible)
        x_vis, mask, ids_restore = self.random_masking(x, mask_ratio) # (B, len_keep, D_enc)
        
        # 3. Apply Encoder blocks (we use the encoder's internal loops but skip patch embed)
        encoded_vis = x_vis
        for block in self.encoder.blocks:
            encoded_vis, _ = block(encoded_vis)
        encoded_vis = self.encoder.norm(encoded_vis)
        
        return encoded_vis, mask, ids_restore

    def forward_decoder(self, x_vis: torch.Tensor, ids_restore: torch.Tensor):
        """
        Projects visible, injects MASK tokens, unshuffles, adds pos embed, runs decoder.
        """
        B, len_keep, D_enc = x_vis.shape
        N_p = ids_restore.shape[1]
        
        # 1. Project to decoder dim
        x_vis = self.enc_to_dec(x_vis)  # (B, len_keep, D_dec)
        D_dec = x_vis.shape[-1]
        
        # 2. Create MASK tokens for the missing patches
        mask_tokens = self.mask_token.expand(B, N_p - len_keep, -1) # (B, len_mask, D_dec)
        
        # 3. Concatenate visible + mask exactly like [kept | masked]
        x_full = torch.cat([x_vis, mask_tokens], dim=1)             # (B, N_p, D_dec)
        
        # 4. Unshuffle to original positions (scatter)
        # gather takes data from x_full according to ids_restore
        gather_indices = ids_restore.unsqueeze(-1).expand(-1, -1, D_dec)
        x_unshuffled = torch.gather(x_full, dim=1, index=gather_indices) # (B, N_p, D_dec)
        
        # 5. Add decoder positional encoding
        x_unshuffled = x_unshuffled + self.dec_pos_embed[:, :N_p, :]
        
        # 6. Apply Decoder blocks
        x = x_unshuffled
        for block in self.decoder_blocks:
            x, _ = block(x)
        x = self.dec_norm(x)
        
        # 7. Predict patches
        predictions = self.pred_head(x)   # (B, N_p, patch_dim)
        
        return predictions
        
    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        """
        Computes MSE loss on masked patches only.
        imgs: (B, T, C)
        pred: (B, N_p, P*C)
        mask: (B, N_p)  1 is masked, 0 is visible
        """
        # Create target by patching input manually
        B, T, C = imgs.shape
        P = self.patch_size
        N_p = T // P
        
        target = imgs.view(B, N_p, P*C)  # (B, N_p, patch_dim)
        
        # MSE per patch element
        loss = (pred - target) ** 2      # (B, N_p, patch_dim)
        loss = loss.mean(dim=-1)         # (B, N_p)
        
        # Take mean over masked patches only
        # mask is 1 for masked patches, 0 otherwise
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.75):
        """
        Full MAE pipeline.
        Returns:
            loss: scalar torch.Tensor containing MSE loss on masked patches
            pred: (B, N_p, patch_dim) the reconstructed patches
            mask: (B, N_p) boolean mask of which patches were masked
        """
        encoded_vis, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(encoded_vis, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        
        return loss, pred, mask


if __name__ == "__main__":
    # Self-test (UCI HAR configuration)
    B, T, C = 32, 128, 6
    P = 16
    D = 128
    
    mae = MAE(
        in_channels=C, patch_size=P, max_len=8,
        enc_embed_dim=D, enc_heads=4, enc_layers=4,
        dec_embed_dim=64, dec_heads=2, dec_layers=2
    )
    
    x_test = torch.randn(B, T, C)
    
    # Forward pass
    loss, pred, mask = mae(x_test, mask_ratio=0.75)
    
    print("MAE FULL STACK TEST")
    print(f"  Input        : {tuple(x_test.shape)}")
    print(f"  Predictions  : {tuple(pred.shape)}  (expect (32, 8, 96))")
    print(f"  Mask         : {tuple(mask.shape)}   (expect (32, 8))")
    print(f"  Loss         : {loss.item():.4f}")
    
    total_params = sum(p.numel() for p in mae.parameters())
    print(f"  Total Params : {total_params:,d}")
    
    # Verify exactly 75% was masked
    # 8 patches * 0.75 = 6 patches masked
    mask_count = mask[0].sum().item()
    print(f"  Mask count   : {mask_count} / 8 patches masked")
    assert mask_count == 6, "Mask ratio calculation failed!"
    
    print("  Status       : PASS")
