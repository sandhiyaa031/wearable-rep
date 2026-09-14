"""
MODULE: train_supervised_uci.py

WHAT:
    Trains the supervised Transformer baseline on the UCI HAR dataset from scratch.

WHY:
    Validates that our custom Transformer architecture and data loaders are
    working correctly before we move on to MAE pretraining (Phase 3). If the
    supervised model converges and learns, the representation foundation is solid.

PROTOCOL (From locked design):
    Supervised baseline: Train TransformerEncoder + Head from scratch
    Optimizer: Adam, LR=1e-3
    Epochs: 100 with Early Stopping on Validation F1
    Metrics: Macro F1, Accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

from preprocessing.uci_har_loader import build_uci_har_dataloaders
from models.supervised_transformer import SupervisedTransformer

# Configuration
BATCH_SIZE = 64
EPOCHS = 20  # Fast self-test. Proper run is 100.
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Training on {DEVICE}")
    
    # 1. Load Data
    print("\n[1] building data loaders...")
    loaders = build_uci_har_dataloaders(batch_size=BATCH_SIZE, num_workers=0)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    
    # 2. Build Model
    print("\n[2] Building model...")
    # shape: 128x6 -> P=16 -> 8 tokens -> 128 D
    model = SupervisedTransformer(
        in_channels=6, patch_size=16, embed_dim=128,
        num_heads=4, num_layers=4, max_len=8, num_classes=6, dropout_p=0.1
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("\n[3] Starting training...")
    best_val_f1 = 0.0
    
    for epoch in range(EPOCHS):
        # -- TRAIN --
        model.train()
        train_loss = 0.0
        
        start_t = time.time()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.apply_gradients = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # -- VAL --
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        val_acc = accuracy_score(all_targets, all_preds)
        
        epoch_t = time.time() - start_t
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} [{epoch_t:.1f}s] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")
              
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            
    print(f"\n[4] Training Complete. Best Val F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    main()
