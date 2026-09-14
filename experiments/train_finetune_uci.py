"""
MODULE: train_finetune_uci.py

WHAT:
    Executes the UCI HAR Label Efficiency downstream experiment.
    Compares:
        1. Supervised from scratch
        2. Frozen Linear Probe (load MAE, freeze encoder, train head only)
        3. End-to-End Fine-tune (load MAE, train encoder + head)
    Across fractions {25%, 50%, 100%} using 5 random seeds.

WHY:
    Tests the core hypothesis: does MAE pretraining on a small dataset (5.2h) 
    generate a strong prior that reduces the need for labeled data on a downstream 
    task? We test this under a strict subject-independent protocol so the model 
    truly must generalize to unseen subjects.

PROTOCOL:
    Frozen probe:   LR=1e-2, 50 epochs (Adam)
    Fine-tune:      LR=1e-4, 100 epochs (AdamW)
    Supervised:     LR=1e-3, 100 epochs (Adam)
    Fractions:      0.25, 0.50, 1.00
    Seeds:          42, 43, 44, 45, 46
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

from preprocessing.uci_har_loader import build_uci_har_dataloaders, build_uci_label_efficiency_loader
from models.supervised_transformer import SupervisedTransformer

# Config
BATCH_SIZE = 64
EPOCHS_PROBE = 20
EPOCHS_FINETUNE = 30
EPOCHS_SUPERVISED = 30
FRACTIONS = [0.25, 0.50, 1.00]
SEEDS = [42, 43, 44]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAE_WEIGHTS = Path("checkpoints/mae_encoder_best.pt")

def build_model(mode="scratch"):
    # Target architecture
    model = SupervisedTransformer(
        in_channels=6, patch_size=16, embed_dim=128,
        num_heads=4, num_layers=4, max_len=8, num_classes=6, dropout_p=0.1
    ).to(DEVICE)
    
    if mode in ["probe", "finetune"]:
        if not MAE_WEIGHTS.exists():
            raise FileNotFoundError(f"Missing MAE weights at {MAE_WEIGHTS}. Pretrain first.")
            
        print(f"    -> Loading MAE weights from {MAE_WEIGHTS}")
        state = torch.load(MAE_WEIGHTS, map_location=DEVICE, weights_only=True)
        # Load into encoder part
        model.encoder.load_state_dict(state, strict=True)
        
        if mode == "probe":
            # Freeze the encoder
            for param in model.encoder.parameters():
                param.requires_grad = False
                
    return model

def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    f1 = f1_score(all_targets, all_preds, average='macro')
    return f1

def train_run(model, train_loader, val_loader, test_loader, mode):
    criterion = nn.CrossEntropyLoss()
    
    if mode == "probe":
        optimizer = optim.Adam(model.head.parameters(), lr=1e-2)
        epochs = EPOCHS_PROBE
    elif mode == "finetune":
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        epochs = EPOCHS_FINETUNE
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        epochs = EPOCHS_SUPERVISED
        
    best_val_f1 = 0.0
    best_test_val = 0.0 # Standard protocol: report test score of the model with best val score
    
    print(f"    -> Training {mode.upper()} for {epochs} epochs...")
    
    # We will just train rapidly without printing every epoch to keep logs clean
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
        val_f1 = evaluate_model(model, val_loader)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_test_val = evaluate_model(model, test_loader)
            
    return best_test_val


def main():
    print(f"UCI HAR LABEL EFFICIENCY EXPERIMENT")
    print(f"Device: {DEVICE}")
    print(f"Fractions: {FRACTIONS}")
    print(f"Seeds: {SEEDS}\n")
    
    # 1. Base dataloaders to get test set and norm stats
    loaders = build_uci_har_dataloaders(batch_size=BATCH_SIZE, return_stats=True)
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    norm_mean, norm_std = loaders["norm_stats"]
    
    results = {}
    
    for frac in FRACTIONS:
        print(f"\n==================================================")
        print(f" EVALUATING LABEL FRACTION: {frac*100:.0f}%")
        print(f"==================================================")
        
        results[frac] = {"scratch": [], "probe": [], "finetune": []}
        
        for seed in SEEDS:
            print(f"\n  [Seed {seed}] ─────────")
            
            # 2. Build training loader for this fraction + seed
            if frac == 1.0:
                # With 100%, we use all train subjects (1-17), but we can still 
                # rebuild the loader to ensure different shuffle/seeds
                train_loader = build_uci_label_efficiency_loader(1.0, norm_mean, norm_std, BATCH_SIZE, seed=seed)
            else:
                train_loader = build_uci_label_efficiency_loader(frac, norm_mean, norm_std, BATCH_SIZE, seed=seed)
                
            # 3. Supervised from scratch
            model = build_model("scratch")
            f1 = train_run(model, train_loader, val_loader, test_loader, "scratch")
            print(f"    [SCRATCH] Test F1 = {f1:.4f}")
            results[frac]["scratch"].append(f1)
            
            # 4. Linear Probe
            model = build_model("probe")
            f1 = train_run(model, train_loader, val_loader, test_loader, "probe")
            print(f"    [PROBE]   Test F1 = {f1:.4f}")
            results[frac]["probe"].append(f1)
            
            # 5. Fine-tune
            model = build_model("finetune")
            f1 = train_run(model, train_loader, val_loader, test_loader, "finetune")
            print(f"    [FINETUNE]Test F1 = {f1:.4f}")
            results[frac]["finetune"].append(f1)
            
    # 6. Final Report
    print("\n\n##################################################")
    print(" FINAL RESULTS: TEST MACRO F1 (Mean ± Std)")
    print("##################################################")
    print(f"{'Fraction':<10} | {'Scratch':<15} | {'Probe':<15} | {'Fine-Tune':<15}")
    print("-" * 65)
    
    for frac in FRACTIONS:
        s_mean, s_std = np.mean(results[frac]["scratch"]), np.std(results[frac]["scratch"])
        p_mean, p_std = np.mean(results[frac]["probe"]), np.std(results[frac]["probe"])
        f_mean, f_std = np.mean(results[frac]["finetune"]), np.std(results[frac]["finetune"])
        
        print(f"{frac*100:3.0f}%      | "
              f"{s_mean:.4f} \u00B1 {s_std:.4f} | "
              f"{p_mean:.4f} \u00B1 {p_std:.4f} | "
              f"{f_mean:.4f} \u00B1 {f_std:.4f}")
              
    # Save Results Dictionary
    out_path = Path("inspection_results/label_efficiency_results.pt")
    torch.save(results, out_path)
    print(f"\nRaw results tensor saved to {out_path}")

if __name__ == "__main__":
    main()
