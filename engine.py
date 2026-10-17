"""
Shared dataloader-building and train/eval-loop logic, used by both
train.py (single model) and compare.py (loops over every registered
architecture). Keeping this in one place means a fix or tweak here applies
identically to every model in the comparison.
"""
import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import config
from data.dataset import LIDDataset, FeatureNormalize
from utils import count_parameters, save_checkpoint


def build_dataloaders(train_dir, val_dir, feature_type, batch_size, num_workers,
                       label_to_idx=None):
    """
    Builds train/val datasets + loaders for a given feature_type.
    Raw waveforms are already peak-normalized in preprocessing, so no
    per-channel normalizer is fit for feature_type == "raw"; MFCC/mel
    features get a global mean/std normalizer fit on the training set.
    """
    raw_train = LIDDataset(train_dir, feature_type=feature_type, label_to_idx=label_to_idx)
    effective_label_map = raw_train.label_to_idx

    if feature_type == "raw":
        normalizer = None
    else:
        raw_loader = DataLoader(raw_train, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)
        normalizer = FeatureNormalize().fit(raw_loader)

    train_dataset = LIDDataset(train_dir, feature_type=feature_type,
                                transform=normalizer, label_to_idx=effective_label_map)
    val_dataset = LIDDataset(val_dir, feature_type=feature_type,
                              transform=normalizer, label_to_idx=effective_label_map)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    return train_dataset, val_dataset, train_loader, val_loader, normalizer


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def train_single_model(model, train_loader, val_loader, epochs, lr, patience, device,
                        checkpoint_path, log_csv_path, extra_checkpoint_fields=None):
    """
    Runs the full train/validate/early-stop loop for one model, logging
    per-epoch loss/accuracy to a CSV (for convergence plots) and saving the
    best checkpoint. Returns (best_val_loss, best_val_acc, param_info).
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                   factor=config.LR_SCHEDULER_FACTOR,
                                   patience=config.LR_SCHEDULER_PATIENCE)

    param_info = count_parameters(model)
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"  Epoch {epoch}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, current_lr])
            f.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                extra = {"val_acc": val_acc, "epoch": epoch, "param_info": param_info}
                if extra_checkpoint_fields:
                    extra.update(extra_checkpoint_fields)
                save_checkpoint(model, checkpoint_path, extra=extra)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("  Early stopping triggered.")
                    break

    return best_val_loss, best_val_acc, param_info
