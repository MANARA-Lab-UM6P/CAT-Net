#!/usr/bin/env python3
"""
Train CAT‑Net on synthetic overlapping speech datasets (GRID/RAVDESS).

This script loads pre‑extracted features (e.g. log gammatonegrams,
Wav2Vec 2.0, HuBERT or WavLM) for each recording condition (clean,
reverberant, noise types at multiple SNRs) and trains a CAT‑Net model
with early stopping based on validation loss.  The best model for each
condition is saved along with evaluation metrics on the held‑out test set.

You can specify which conditions to train on via the ``--conditions``
argument.  If omitted, the script trains on all available leaf
directories (directories containing ``train.npz``).  Model and
training hyperparameters are exposed as command‑line options.

Usage
-----

```bash
python scripts/train_catnet_synth.py \
    --features_root <path/to/extracted/features> \
    --results_root <path/to/save/models> \
    --model_name CATNet_grid \
    --conditions clean/no_reverb,clean/reverberated,babble/0/no_reverb,babble/5/no_reverb
```

See ``--help`` for a full list of arguments.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import CAT_Net from the top‑level models package
from models import CAT_Net


###############################################################################
# Dataset and data loading
###############################################################################

class OSDDataset(Dataset):
    """Dataset for overlapping speech detection from precomputed features."""
    def __init__(self, root: Path, split: str) -> None:
        self.samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        root = Path(root)
        for npz_path in root.rglob(f"{split}.npz"):
            try:
                data = np.load(npz_path)
                feats = data['features']  # (N, L, C)
                labels = data['labels']   # (N, L)
                masks = data['mask']      # (N, L)
                nseq = feats.shape[0]
                for i in range(nseq):
                    self.samples.append((feats[i], labels[i], masks[i]))
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat, label, mask = self.samples[idx]
        return (torch.from_numpy(feat).float(),
                torch.from_numpy(label).float(),
                torch.from_numpy(mask).float())


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feats, labels, masks = zip(*batch)
    feats = torch.stack(feats, dim=0)
    labels = torch.stack(labels, dim=0)
    masks = torch.stack(masks, dim=0)
    return feats, labels, masks


###############################################################################
# Training and evaluation helpers
###############################################################################

def train_epoch(model: CAT_Net, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.BCEWithLogitsLoss, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_frames = 0
    tp = fp = fn = tn = 0
    for feats, labels, masks in tqdm(dataloader, desc="Training", leave=True):
        feats = feats.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        # permute to (batch, channels, time)
        feats = feats.permute(0, 2, 1)
        optimizer.zero_grad()
        outputs = model(feats)
        loss_raw = criterion(outputs, labels)
        loss = (loss_raw * masks).sum() / masks.sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = (outputs >= 0).float()
        labels_flat = labels.view(-1)
        preds_flat = preds.view(-1)
        masks_flat = masks.view(-1)
        valid_idx = masks_flat == 1
        labels_valid = labels_flat[valid_idx]
        preds_valid = preds_flat[valid_idx]
        n_frames = valid_idx.sum().item()
        total_frames += n_frames
        tp += ((preds_valid == 1) & (labels_valid == 1)).sum().item()
        tn += ((preds_valid == 0) & (labels_valid == 0)).sum().item()
        fp += ((preds_valid == 1) & (labels_valid == 0)).sum().item()
        fn += ((preds_valid == 0) & (labels_valid == 1)).sum().item()
    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = (tp + tn) / max(total_frames, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    return avg_loss, metrics


@torch.no_grad()
def validate_epoch(model: CAT_Net, dataloader: DataLoader,
                   criterion: nn.BCEWithLogitsLoss, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_frames = 0
    tp = fp = fn = tn = 0
    for feats, labels, masks in tqdm(dataloader, desc="Validating", leave=True):
        feats = feats.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        feats = feats.permute(0, 2, 1)
        outputs = model(feats)
        loss_raw = criterion(outputs, labels)
        loss = (loss_raw * masks).sum() / masks.sum()
        total_loss += loss.item()
        preds = (outputs >= 0).float()
        labels_flat = labels.view(-1)
        preds_flat = preds.view(-1)
        masks_flat = masks.view(-1)
        valid_idx = masks_flat == 1
        labels_valid = labels_flat[valid_idx]
        preds_valid = preds_flat[valid_idx]
        n_frames = valid_idx.sum().item()
        total_frames += n_frames
        tp += ((preds_valid == 1) & (labels_valid == 1)).sum().item()
        tn += ((preds_valid == 0) & (labels_valid == 0)).sum().item()
        fp += ((preds_valid == 1) & (labels_valid == 0)).sum().item()
        fn += ((preds_valid == 0) & (labels_valid == 1)).sum().item()
    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = (tp + tn) / max(total_frames, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    return avg_loss, metrics


@torch.no_grad()
def evaluate_test_set(model: CAT_Net, condition_dir: Path,
                      criterion: nn.BCEWithLogitsLoss, device: torch.device,
                      batch_size: int) -> Dict[str, float]:
    test_files = list(Path(condition_dir).rglob("test.npz"))
    if not test_files:
        return {"loss": float('nan'), "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    total_loss = 0.0
    total_frames = 0
    total_tp = total_fp = total_fn = total_tn = 0
    num_valid = 0
    for test_file in test_files:
        try:
            data = np.load(test_file, allow_pickle=False)
            feats = data['features']
            labels = data['labels']
            masks = data['mask']
        except Exception:
            continue
        class _F(Dataset):
            def __len__(self_inner): return feats.shape[0]
            def __getitem__(self_inner, idx):
                return (torch.from_numpy(feats[idx]).float(),
                        torch.from_numpy(labels[idx]).float(),
                        torch.from_numpy(masks[idx]).float())
        loader = DataLoader(_F(), batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=False)
        file_loss = 0.0
        file_frames = 0
        tp = fp = fn = tn = 0
        for f_feats, f_labels, f_masks in loader:
            f_feats = f_feats.to(device)
            f_labels = f_labels.to(device)
            f_masks = f_masks.to(device)
            f_feats = f_feats.permute(0, 2, 1)
            outputs = model(f_feats)
            loss_raw = criterion(outputs, f_labels)
            loss = (loss_raw * f_masks).sum() / f_masks.sum()
            file_loss += loss.item()
            preds = (outputs >= 0).float()
            labels_flat = f_labels.view(-1)
            preds_flat = preds.view(-1)
            masks_flat = f_masks.view(-1)
            valid_idx = masks_flat == 1
            labels_valid = labels_flat[valid_idx]
            preds_valid = preds_flat[valid_idx]
            n_frames = valid_idx.sum().item()
            file_frames += n_frames
            tp += ((preds_valid == 1) & (labels_valid == 1)).sum().item()
            tn += ((preds_valid == 0) & (labels_valid == 0)).sum().item()
            fp += ((preds_valid == 1) & (labels_valid == 0)).sum().item()
            fn += ((preds_valid == 0) & (labels_valid == 1)).sum().item()
        if file_frames > 0:
            num_valid += 1
            total_loss += file_loss / max(len(loader), 1)
            total_frames += file_frames
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
    if num_valid == 0:
        return {"loss": float('nan'), "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    avg_loss = total_loss / num_valid
    accuracy = (total_tp + total_tn) / max(total_frames, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"loss": avg_loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


###############################################################################
# Utilities to find and filter condition directories
###############################################################################

def find_leaf_dirs(features_root: Path) -> List[Path]:
    """Return a sorted list of leaf directories that contain train.npz files."""
    leaf_dirs: List[Path] = []
    for path in features_root.rglob("train.npz"):
        leaf_dirs.append(path.parent)
    return sorted(leaf_dirs)


def filter_condition_dirs(all_dirs: List[Path], features_root: Path, allowed: Optional[List[str]]) -> List[Path]:
    """Filter condition directories according to a list of allowed relative paths."""
    if allowed is None:
        return all_dirs
    allowed_set = {Path(cond) for cond in allowed}
    selected: List[Path] = []
    for d in all_dirs:
        rel = d.relative_to(features_root)
        if rel in allowed_set:
            selected.append(d)
    return selected


###############################################################################
# Main training loop over selected conditions
###############################################################################

def train_condition(condition_dir: Path, device: torch.device, model_kwargs: Dict,
                    train_kwargs: Dict, batch_size: int) -> Tuple[float, Optional[Dict[str, torch.Tensor]], Dict[str, float]]:
    """Train a CAT‑Net on a single condition directory and return best model and test metrics."""
    # prepare datasets
    train_dataset = OSDDataset(condition_dir, 'train')
    val_dataset = OSDDataset(condition_dir, 'val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=False)
    # model
    model = CAT_Net(**model_kwargs).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.RAdam(model.parameters(), lr=train_kwargs['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    best_val_loss = float('inf')
    best_state: Optional[Dict[str, torch.Tensor]] = None
    no_improve = 0
    min_delta = 1e-8
    for epoch in range(1, train_kwargs['epochs'] + 1):
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f}, Acc: {train_metrics['accuracy']:.4f}, "
            f"Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_loss:.6f}, Acc: {val_metrics['accuracy']:.4f}, "
            f"Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}"
        )
        if val_loss < best_val_loss - min_delta:
            best_val_loss = float(val_loss)
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= train_kwargs['patience']:
            print(f"Early stopping on validation loss after {train_kwargs['patience']} epochs without improvement.")
            break
    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_test_set(model, condition_dir, criterion, device, batch_size=batch_size)
    return best_val_loss, best_state, test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CAT‑Net on synthetic OSD datasets.")
    parser.add_argument('--features_root', type=str, required=True,
                        help='Root directory containing aggregated feature files (train/val/test npz).')
    parser.add_argument('--results_root', type=str, required=True,
                        help='Root directory where models and metrics will be saved.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model (used as subdirectory under results_root).')
    parser.add_argument('--conditions', type=str, default=None,
                        help='Comma‑separated list of relative condition paths to train on (e.g. "clean/no_reverb,clean/reverberated").  If omitted, train on all conditions.')
    # model hyperparameters
    parser.add_argument('--in_channels', type=int, default=80, help='Number of input feature channels.')
    parser.add_argument('--bn_channels', type=int, default=128, help='Bottleneck channel dimension.')
    parser.add_argument('--hid_channels', type=int, default=512, help='Hidden channel dimension inside conv blocks.')
    parser.add_argument('--n_blocks', type=int, default=5, help='Number of convolutional blocks per repeat.')
    parser.add_argument('--n_repeats', type=int, default=3, help='Number of repeats of the block stack.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size of the depthwise convolution.')
    parser.add_argument('--norm_type', type=str, default='gLN', choices=['gLN', 'bN'], help="Normalisation type ('gLN' or 'bN').")
    parser.add_argument('--out_classes', type=int, default=1, help='Number of output classes (1 for binary classification).')
    # training hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Mini‑batch size.')
    parser.add_argument('--lr', type=float, default=4e-3, help='Initial learning rate.')
    parser.add_argument('--patience', type=int, default=6, help='Patience for early stopping based on validation loss.')
    parser.add_argument('--device', type=str, default=None, help='Device to train on (e.g. "cuda" or "cpu").  If unspecified, CUDA is used if available.')
    args = parser.parse_args()
    features_root = Path(args.features_root).resolve()
    results_root = Path(args.results_root).resolve()
    model_name = args.model_name
    if not features_root.is_dir():
        raise FileNotFoundError(f"Features root not found: {features_root}")
    # Determine device
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training CAT‑Net on device: {device}")
    # Find all condition directories (leaf dirs containing train.npz)
    all_cond_dirs = find_leaf_dirs(features_root)
    # Parse conditions
    if args.conditions:
        allowed = [c.strip() for c in args.conditions.split(',') if c.strip()]
    else:
        allowed = None
    cond_dirs = filter_condition_dirs(all_cond_dirs, features_root, allowed)
    if not cond_dirs:
        print(f"No matching condition directories found under {features_root}")
        return
    print("Selected condition directories (relative to features_root):")
    for d in cond_dirs:
        print("  -", d.relative_to(features_root))
    # Prepare model kwargs and training kwargs
    model_kwargs = dict(
        in_chan=args.in_channels,
        bn_chan=args.bn_channels,
        hid_chan=args.hid_channels,
        n_blocks=args.n_blocks,
        n_repeats=args.n_repeats,
        kernel_size=args.kernel_size,
        norm_type=args.norm_type,
        out_classes=args.out_classes
    )
    train_kwargs = dict(
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience
    )
    batch_size = args.batch_size
    # Train on each condition
    for cond_dir in cond_dirs:
        rel_key = cond_dir.relative_to(features_root)
        result_dir = results_root / model_name / rel_key
        result_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Training condition {rel_key} ===")
        best_val_loss, best_state, test_metrics = train_condition(cond_dir, device, model_kwargs, train_kwargs, batch_size)
        # Save best model
        if best_state is not None:
            torch.save(best_state, result_dir / 'best_model.pt')
        # Save test results
        with open(result_dir / 'test_results.txt', 'w') as f:
            f.write(
                f"Best Val Loss: {best_val_loss:.6f}\n"
                f"Test Loss: {test_metrics['loss']:.6f}\n"
                f"Accuracy: {test_metrics['accuracy']:.4f}\n"
                f"Precision: {test_metrics['precision']:.4f}\n"
                f"Recall: {test_metrics['recall']:.4f}\n"
                f"F1: {test_metrics['f1']:.4f}\n"
            )
        print(f"Results saved to {result_dir}")


if __name__ == '__main__':
    main()