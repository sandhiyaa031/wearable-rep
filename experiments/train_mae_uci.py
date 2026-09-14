"""
MODULE: train_mae_uci.py

WHAT:
    Pre-trains the Masked Autoencoder on the UCI HAR unlabeled training corpus.

WHY:
    This is Phase 3: learning representations without labels. The encoder will
    learn the underlying semantic structure of human IMU movement by solving
    the pretext task of imputing missing patches (75% masked). We save the
    pretrained encoder weights to be loaded during downstream tasks.

PROTOCOL (From locked design):
    Dataset        : UCI HAR train subjects (7,352 windows, unlabeled)
    Optimizer      : AdamW (lr=1e-4, weight_decay=0.05)
    Batch size     : 64
    Epochs         : 400 (we will run 50 for this script, save best via val loss)
    Masking ratio  : 0.75
    Saved artifact : encoder weights

VIVA ANSWER:
    "I trained the MAE purely on the UCI HAR training subjects without looking
    at any labels. I masked exactly 75% of every input sequence. The model used 
    an AdamW optimizer to minimize Mean Squared Error only on the masked tokens,
    preventing the encoder from simply acting as an identity function. After
    pretraining, I extracted the encoder weights for the downstream tasks and
    discarded the decoder."
"""

import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
from pathlib import Path

from preprocessing.uci_har_loader import build_uci_har_dataloaders
from models.mae import MAE

# Config
BATCH_SIZE = 64
EPOCHS = 100  # shortened from 400 for practical run time
LR = 1e-4
WEIGHT_DECAY = 0.05
MASK_RATIO = 0.75
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHKPT_DIR = Path("checkpoints")
CHKPT_DIR.mkdir(exist_ok=True)
OUT_DIR = Path("inspection_results/mae_reconstructions")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_reconstructions(mae, x_true, epoch, num_samples=4):
    """
    Plots the true, masked, and reconstructed signal for the Accel X channel.
    x_true: (B, 128, 6)
    """
    mae.eval()
    with torch.no_grad():
        x_true_device = x_true.to(DEVICE)
        loss, pred, mask = mae(x_true_device, mask_ratio=MASK_RATIO)
        
        # pred is (B, 8, 96). We need to reshape back to (B, 128, 6)
        B, N_p, patch_dim = pred.shape
        P = 16
        C = 6
        
        # Build the full reconstructed sequence
        # We start with the TRUE sequence, and overwrite the MASKED patches with PRED
        
        # pred: (B, N_p, P*C) -> (B, N_p, P, C) -> (B, N_p*P, C)
        pred_full = pred.view(B, N_p, P, C).view(B, N_p * P, C).cpu()
        
        # mask is (B, 8). 1 means masked.
        # Expand mask to point level: (B, N_p) -> (B, N_p, P) -> view (B, N_p*P)
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, P).reshape(B, N_p * P).cpu()
        
        composite = x_true.clone()
        # For every item in batch and time, if masked, use pred_full
        for b in range(B):
            for t in range(128):
                if mask_expanded[b, t] == 1:
                    composite[b, t, :] = pred_full[b, t, :]
                    
    # Plotting Accel X (channel 0)
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2*num_samples))
    if num_samples == 1: axes = [axes]
    
    for i in range(min(num_samples, B)):
        ax = axes[i]
        
        # Plot true signal (gray)
        ax.plot(x_true[i, :, 0].numpy(), label="True (Acc_X)", color="gray", alpha=0.5, linewidth=2)
        
        # Highlight visible patches in blue, masked predicted in red
        time_axis = torch.arange(128)
        
        # Where mask == 0 (visible)
        vis_idx = time_axis[mask_expanded[i] == 0]
        # Where mask == 1 (masked geometry imputed)
        msk_idx = time_axis[mask_expanded[i] == 1]
        
        # Overlay the composite
        # We plot in segments to get the colors right
        for p in range(8):
            start = p * 16
            end = start + 16
            if mask[i, p] == 0:
                ax.plot(range(start, end), composite[i, start:end, 0].numpy(), color="blue", linewidth=1.5)
            else:
                ax.plot(range(start, end), composite[i, start:end, 0].numpy(), color="red", linewidth=1.5, linestyle="--")
                
        if i == 0:
            import matplotlib.lines as mlines
            gray_line = mlines.Line2D([], [], color='gray', label='Original Ground Truth')
            blue_line = mlines.Line2D([], [], color='blue', label='Visible Patch (Encoder Input)')
            red_line  = mlines.Line2D([], [], color='red', linestyle='--', label='Masked Imputation (Decoder Output)')
            ax.legend(handles=[gray_line, blue_line, red_line], loc="upper right", fontsize=8)
            
        ax.set_title(f"Sample {i+1} - Validation Reconstruction")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"reconstruction_epoch_{epoch}.png", dpi=150)
    plt.close()


def main():
    print(f"Pre-training MAE on {DEVICE}")
    
    print("\n[1] Building data loaders...")
    loaders = build_uci_har_dataloaders(batch_size=BATCH_SIZE, num_workers=0)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    
    print("\n[2] Building MAE model...")
    # 128x6 -> P=16 -> 8 tokens -> Enc=128 (4L), Dec=64 (2L)
    model = MAE(
        in_channels=6, patch_size=16, max_len=8,
        enc_embed_dim=128, enc_heads=4, enc_layers=4,
        dec_embed_dim=64, dec_heads=2, dec_layers=2
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    print("\n[3] Starting MAE pretraining (Unlabeled Data Only)...")
    best_val_loss = float('inf')
    
    # Grab a fixed batch for visualization
    fixed_x, _ = next(iter(val_loader))
    
    for epoch in range(1, EPOCHS + 1):
        # -- TRAIN --
        model.train()
        train_loss = 0.0
        
        start_t = time.time()
        for x, _ in train_loader:  # ignore labels
            x = x.to(DEVICE)
            
            optimizer.zero_grad()
            loss, _, _ = model(x, mask_ratio=MASK_RATIO)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # -- VAL --
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(DEVICE)
                loss, _, _ = model(x, mask_ratio=MASK_RATIO)
                val_loss += loss.item() * x.size(0)
                
        val_loss /= len(val_loader.dataset)
        epoch_t = time.time() - start_t
        
        print(f"Epoch {epoch:03d}/{EPOCHS} [{epoch_t:.1f}s] | "
              f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
              
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save JUST the encoder portion. That is all we need for inference/transfer!
            torch.save(model.encoder.state_dict(), CHKPT_DIR / "mae_encoder_best.pt")
            
        # Visualize every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            plot_reconstructions(model, fixed_x[:4].clone(), epoch)
            
    print(f"\n[4] MAE Pretraining Complete. Best Val MSE: {best_val_loss:.4f}")
    print(f"    Encoder weights saved to {CHKPT_DIR / 'mae_encoder_best.pt'}")
    print(f"    Reconstruction plots saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
