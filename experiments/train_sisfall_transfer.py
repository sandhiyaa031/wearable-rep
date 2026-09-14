"""
MODULE: train_sisfall_transfer.py

WHAT:
    Executes the Fall Detection (SisFall) cross-domain downstream experiment.
    Compares:
        1. Supervised baseline (trained from scratch)
        2. MAE Fine-tuned (pretrained MAE weights loaded in blocks, randomly 
           initialized SisFall-specific patch projection, then entire model 
           fine-tuned end-to-end with labels).

WHY:
    Tests the cross-domain transferability hypothesis from the design spec:
    "Does pretraining on 50Hz waist-mounted locotomotion (UCI HAR) provide a 
     measurably better initialization for 100Hz belt-mounted fall detection 
     (SisFall)?"
     
MATH / SHAPE ALIGNMENT:
    Since SisFall has 9 channels (instead of 6) and uses P=20 patches (instead 
    of P=16), the `patch_embed.proj` layer shapes do not match the frozen MAE 
    weights (180->128 vs 96->128). We specifically filter out the patch_embed 
    weights when loading the checkpoint. The TransformerEncoderBlocks map 128->128
    regardless of how the sequence was generated, so they transfer perfectly.

PROTOCOL:
    Dataset:        SisFall (25 train, 7 val, 6 test subjects)
    Windowing:      2 seconds @ 100Hz (200 timesteps) -> 10 patches of P=20
    Labels:         Recording-level Option A (0=ADL, 1=Fall)
    Metrics:        F1-Fall (Primary), Recall-Fall, Precision-Fall (Secondary)
    Seeds:          5 seeds for robustness
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from preprocessing.sisfall_loader import build_sisfall_dataloaders
from models.supervised_transformer import SupervisedTransformer

# Config
BATCH_SIZE = 64
EPOCHS = 5
SEEDS = [42, 43, 44]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAE_WEIGHTS = Path("checkpoints/mae_encoder_best.pt")

def build_model(mode="scratch"):
    """
    SisFall config: in_channels=9, patch_size=20 (so 200/20 = 10 max_len), 
                    embed=128, classes=2.
    """
    model = SupervisedTransformer(
        in_channels=9, patch_size=20, embed_dim=128,
        num_heads=4, num_layers=4, max_len=10, num_classes=2, dropout_p=0.1
    ).to(DEVICE)
    
    if mode == "finetune":
        if not MAE_WEIGHTS.exists():
            raise FileNotFoundError(f"Missing MAE weights at {MAE_WEIGHTS}. Pretrain first.")
            
        print(f"    -> Loading MAE weights from {MAE_WEIGHTS}")
        # Load weights, ignore patch embedding layer since channels/patch_size differ
        state = torch.load(MAE_WEIGHTS, map_location=DEVICE, weights_only=True)
        filtered_state = {
            k: v for k, v in state.items() 
            if 'patch_embed' not in k and 'pos_embed' not in k
        }
        # Notice we also don't load pos_embed because max_len is 10 not 8.
        # It takes ~2 epochs for the random pos_embed to align with the transferred blocks.
        
        missing, unexpected = model.encoder.load_state_dict(filtered_state, strict=False)
        # Expected missing: patch_embed.proj.weight, patch_embed.proj.bias, pos_embed.pos_embed
        
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
            
    # Binary classification, positive class = 1 (Fall)
    f1 = f1_score(all_targets, all_preds, pos_label=1)
    rec = recall_score(all_targets, all_preds, pos_label=1)
    prec = precision_score(all_targets, all_preds, pos_label=1)
    return {"f1": f1, "recall": rec, "precision": prec}

def train_run(model, train_loader, val_loader, test_loader, mode):
    # SisFall has severe class imbalance (~60% ADL, 40% Fall windows, but 
    # more like 80/20 in true free-living). Here we use standard CrossEntropy
    # but we measure F1 to capture imbalance handling.
    criterion = nn.CrossEntropyLoss()
    
    # We use AdamW 1e-4 for fine-tuning pretrained, 1e-3 for scratch
    lr = 1e-4 if mode == "finetune" else 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05) if mode == "finetune" else optim.Adam(model.parameters(), lr=lr)
        
    best_val_f1 = 0.0
    best_test_metrics = {}
    
    print(f"    -> Training {mode.upper()} for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
        val_metrics = evaluate_model(model, val_loader)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_test_metrics = evaluate_model(model, test_loader)
            
    return best_test_metrics


def main():
    print(f"SISFALL CROSS-DOMAIN TRANSFER EXPERIMENT")
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}\n")
    
    # 1. Base dataloaders
    loaders = build_sisfall_dataloaders(batch_size=BATCH_SIZE)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    
    results = {"scratch": {"f1": [], "recall": [], "precision": []},
               "finetune": {"f1": [], "recall": [], "precision": []}}
    
    for seed in SEEDS:
        print(f"\n==================================================")
        print(f" EVALUATING SEED: {seed}")
        print(f"==================================================")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 2. Supervised from scratch
        model = build_model("scratch")
        metrics = train_run(model, train_loader, val_loader, test_loader, "scratch")
        print(f"    [SCRATCH]  Test F1-Fall: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}")
        for k in metrics: results["scratch"][k].append(metrics[k])
        
        # 3. Fine-tune MAE weights
        model = build_model("finetune")
        metrics = train_run(model, train_loader, val_loader, test_loader, "finetune")
        print(f"    [FINETUNE] Test F1-Fall: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}")
        for k in metrics: results["finetune"][k].append(metrics[k])
        
            
    # 4. Final Report
    print("\n\n##################################################")
    print(" FINAL RESULTS: SISFALL TEST METRICS (Mean ± Std)")
    print("##################################################")
    print(f"{'Metric':<12} | {'Scratch Baseline':<20} | {'MAE Transfer':<20}")
    print("-" * 60)
    
    for k in ["f1", "recall", "precision"]:
        s_mean, s_std = np.mean(results["scratch"][k]), np.std(results["scratch"][k])
        f_mean, f_std = np.mean(results["finetune"][k]), np.std(results["finetune"][k])
        
        print(f"{k.capitalize():<12} | "
              f"{s_mean:.4f} \u00B1 {s_std:.4f} | "
              f"{f_mean:.4f} \u00B1 {f_std:.4f}")
              
    # Save Results Dictionary
    out_path = Path("inspection_results/sisfall_transfer_results.pt")
    torch.save(results, out_path)
    print(f"\nRaw results tensor saved to {out_path}")

if __name__ == "__main__":
    main()
