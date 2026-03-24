#!/usr/bin/env python3
"""
Train CAT‑Net on the AMI Meeting Corpus.

This script trains CAT‑Net on continuous acoustic features extracted from
the AMI corpus.  It supports training with multiple random seeds and
reports mean/standard deviation of validation and test metrics across
seeds.  During training the model is evaluated on the validation set
using sliding windows (3 s windows with 50 % overlap by default), and
the best model (by validation F1) for each seed is saved.

Usage
-----

```bash
python scripts/train_catnet_ami.py \
    --features_root <path/to/ami/features> \
    --results_root <path/to/save/models> \
    --model_name CATNet_AMI \
    --seeds 0 1 2 3 4
```

Additional options include model hyperparameters, training hyperparameters
and evaluation window parameters.  See ``--help`` for details.
"""

from __future__ import annotations

import argparse
import os
import random
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
# Dataset definitions
###############################################################################

class AMITrainDataset(Dataset):
    """Training dataset: uses continuous features, slices into windows with overlap."""
    def __init__(self, features_root: Path, split: str, window_frames: int, stride_frames: int) -> None:
        super().__init__()
        self.features_root = Path(features_root)
        self.split = split
        split_dir = self.features_root / split
        if not split_dir.is_dir():
            raise RuntimeError(f"Split directory not found: {split_dir}")
        # Store the window and stride on the instance for later use
        self.window_frames = window_frames
        self.stride_frames = stride_frames
        self.meetings: List[Dict[str, np.ndarray]] = []
        self.indices: List[Tuple[int, int]] = []  # (meeting_idx, start_frame)
        npz_paths = sorted(split_dir.glob("*.npz"))
        if not npz_paths:
            raise RuntimeError(f"No NPZ files found in {split_dir}")
        for m_idx, npz_path in enumerate(npz_paths):
            data = np.load(npz_path, allow_pickle=False)
            feats = data["features"].astype(np.float32)  # (T, D)
            labels = data["labels"].astype(np.float32)   # (T,)
            if "mask" in data:
                mask = data["mask"].astype(bool)         # (T,)
            else:
                mask = np.ones_like(labels, dtype=bool)
            self.meetings.append({"features": feats, "labels": labels, "mask": mask})
            T = feats.shape[0]
            start = 0
            # slide with stride and stop when the window fits entirely within the meeting
            while start + self.window_frames <= T:
                self.indices.append((m_idx, start))
                start += self.stride_frames
        if not self.indices:
            raise RuntimeError(f"No training windows found in {split_dir}")
        print(f"[AMITrainDataset] Split '{split}': {len(self.meetings)} meetings, {len(self.indices)} windows of {self.window_frames} frames.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m_idx, start = self.indices[idx]
        meeting = self.meetings[m_idx]
        # slice continuous features according to the configured window length
        feats = meeting["features"][start:start + self.window_frames]   # (L, C)
        labels = meeting["labels"][start:start + self.window_frames]    # (L,)
        mask = meeting["mask"][start:start + self.window_frames]        # (L,)
        return (torch.from_numpy(feats).float(),
                torch.from_numpy(labels).float(),
                torch.from_numpy(mask.astype(np.float32)))


def collate_fn_train(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feats, labels, masks = zip(*batch)
    feats = torch.stack(feats, dim=0)
    labels = torch.stack(labels, dim=0)
    masks = torch.stack(masks, dim=0)
    return feats, labels, masks


###############################################################################
# Sliding‑window inference for evaluation
###############################################################################

def sliding_window_predict_meeting(model: nn.Module, features: np.ndarray, device: torch.device,
                                   win_frames: int, stride_frames: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    T, C = features.shape
    L = win_frames
    if T < L:
        return np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.int32)
    starts: List[int] = list(range(0, T - L + 1, stride_frames))
    # Optionally ensure coverage of tail
    last_start = starts[-1]
    if last_start + L < T:
        extra_start = T - L
        if extra_start > last_start:
            starts.append(extra_start)
    n_windows = len(starts)
    logits_sum = np.zeros(T, dtype=np.float64)
    counts = np.zeros(T, dtype=np.int32)
    with torch.no_grad():
        for b_start in range(0, n_windows, batch_size):
            b_end = min(b_start + batch_size, n_windows)
            B = b_end - b_start
            batch_feats = np.empty((B, C, L), dtype=np.float32)
            for i, s in enumerate(starts[b_start:b_end]):
                seg = features[s:s + L]
                batch_feats[i] = seg.T
            batch_tensor = torch.from_numpy(batch_feats).to(device)
            logits_batch = model(batch_tensor)
            logits_np = logits_batch.cpu().numpy()
            for i, s in enumerate(starts[b_start:b_end]):
                logits_win = logits_np[i]
                logits_sum[s:s + L] += logits_win
                counts[s:s + L] += 1
    logits_avg = np.zeros(T, dtype=np.float32)
    nonzero = counts > 0
    logits_avg[nonzero] = (logits_sum[nonzero] / counts[nonzero]).astype(np.float32)
    return logits_avg, counts


@torch.no_grad()
def evaluate_split(model: nn.Module, features_root: Path, split: str, device: torch.device,
                   criterion: nn.BCEWithLogitsLoss, win_frames: int, stride_frames: int, batch_size: int) -> Dict[str, float]:
    model.eval()
    split_dir = features_root / split
    npz_paths = sorted(split_dir.glob("*.npz"))
    if not npz_paths:
        return {"loss": float("nan"), "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    total_loss_sum = 0.0
    total_frames = 0
    tp = fp = fn = tn = 0
    for npz_path in tqdm(npz_paths, desc=f"Evaluating {split}", leave=True):
        data = np.load(npz_path, allow_pickle=False)
        feats = data["features"].astype(np.float32)   # (T, C)
        labels = data["labels"].astype(np.float32)    # (T,)
        if "mask" in data:
            mask = data["mask"].astype(bool)
        else:
            mask = np.ones_like(labels, dtype=bool)
        logits_avg, counts = sliding_window_predict_meeting(model, feats, device,
                                                             win_frames, stride_frames, batch_size)
        valid = (counts > 0) & mask
        if not np.any(valid):
            continue
        logits_valid = torch.from_numpy(logits_avg[valid])
        labels_valid = torch.from_numpy(labels[valid])
        loss_raw = criterion(logits_valid, labels_valid)
        if loss_raw.ndim > 0:
            loss_sum = loss_raw.sum().item()
            n_frames = labels_valid.numel()
        else:
            loss_sum = float(loss_raw.item())
            n_frames = 1
        total_loss_sum += loss_sum
        total_frames += n_frames
        preds = (logits_valid >= 0).float()
        tp += ((preds == 1) & (labels_valid == 1)).sum().item()
        tn += ((preds == 0) & (labels_valid == 0)).sum().item()
        fp += ((preds == 1) & (labels_valid == 0)).sum().item()
        fn += ((preds == 0) & (labels_valid == 1)).sum().item()
    if total_frames == 0:
        return {"loss": float("nan"), "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    avg_loss = total_loss_sum / total_frames
    accuracy = (tp + tn) / max(total_frames, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"loss": avg_loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


###############################################################################
# Main training routine for AMI
###############################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description="Train CAT‑Net on the AMI Meeting Corpus.")
    parser.add_argument('--features_root', type=str, required=True,
                        help='Root directory containing continuous AMI features (train/val/test splits).')
    parser.add_argument('--results_root', type=str, required=True,
                        help='Root directory where models and metrics will be saved.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model (used as subdirectory under results_root).')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0],
                        help='List of random seeds to run (default: 0).')
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
    parser.add_argument('--patience', type=int, default=6, help='Patience for early stopping.')
    parser.add_argument('--device', type=str, default=None, help='Device to train on (e.g. "cuda" or "cpu").  If unspecified, CUDA is used if available.')
    # evaluation window parameters
    parser.add_argument('--train_window_secs', type=float, default=5.0, help='Duration of training windows in seconds (default: 5.0).')
    parser.add_argument('--eval_window_secs', type=float, default=3.0, help='Duration of evaluation windows in seconds (default: 3.0).')
    parser.add_argument('--hop_time', type=float, default=0.010, help='Hop time of features in seconds (default: 0.010).')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Batch size for evaluation windows.')
    args = parser.parse_args()
    features_root = Path(args.features_root).resolve()
    results_root = Path(args.results_root).resolve()
    model_name = args.model_name
    if not features_root.is_dir():
        raise FileNotFoundError(f"Features root not found: {features_root}")
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Training CAT‑Net on device: {device}")
    # compute frames per second from hop time
    frames_per_second = int(round(1.0 / args.hop_time))
    train_window_frames = int(round(args.train_window_secs * frames_per_second))
    train_stride_frames = train_window_frames // 2  # 50% overlap
    eval_window_frames = int(round(args.eval_window_secs * frames_per_second))
    eval_stride_frames = eval_window_frames // 2
    # To store results across seeds
    all_seed_results: List[Dict[str, Dict[str, float]]] = []
    metric_names = ["loss", "accuracy", "precision", "recall", "f1"]
    val_metrics_all: Dict[str, List[float]] = {m: [] for m in metric_names}
    test_metrics_all: Dict[str, List[float]] = {m: [] for m in metric_names}
    # Prepare result directory
    results_dir = results_root / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        print("\n" + "=" * 70)
        print(f"[INFO] Starting training for seed {seed}")
        print("=" * 70)
        # set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # model
        base_model = CAT_Net(
            in_chan=args.in_channels,
            bn_chan=args.bn_channels,
            hid_chan=args.hid_channels,
            n_blocks=args.n_blocks,
            n_repeats=args.n_repeats,
            kernel_size=args.kernel_size,
            norm_type=args.norm_type,
            out_classes=args.out_classes
        ).to(device)
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"[INFO] Using DataParallel with {torch.cuda.device_count()} GPUs")
            model: nn.Module = nn.DataParallel(base_model)
        else:
            model = base_model
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=False)
        # training dataset
        train_dataset = AMITrainDataset(features_root, split="train", window_frames=train_window_frames, stride_frames=train_stride_frames)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn_train, drop_last=False)
        # track best F1 per seed
        best_val_loss = float("inf")
        best_val_f1 = 0.0
        best_val_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        best_state: Optional[Dict[str, torch.Tensor]] = None
        no_improve = 0
        min_delta = 1e-8
        for epoch in range(1, args.epochs + 1):
            print(f"\n===== Seed {seed} | Epoch {epoch:03d} / {args.epochs} =====")
            train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate_split(model, features_root, split="val", device=device,
                                         criterion=criterion, win_frames=eval_window_frames,
                                         stride_frames=eval_stride_frames, batch_size=args.eval_batch_size)
            val_loss = val_metrics["loss"]
            scheduler.step(val_loss)
            print(
                f"Seed {seed} | Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.6f}, Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Train Prec: {train_metrics['precision']:.4f}, Train Rec: {train_metrics['recall']:.4f}, Train F1: {train_metrics['f1']:.4f} | "
                f"Val Loss: {val_loss:.6f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val Prec: {val_metrics['precision']:.4f}, Val Rec: {val_metrics['recall']:.4f}, Val F1: {val_metrics['f1']:.4f}"
            )
            current_f1 = float(val_metrics["f1"])
            if current_f1 > best_val_f1 + min_delta:
                best_val_f1 = current_f1
                best_val_loss = float(val_loss)
                best_val_metrics = dict(val_metrics)
                if isinstance(model, nn.DataParallel):
                    best_state = {k: v.cpu() for k, v in model.module.state_dict().items()}
                else:
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= args.patience:
                print(f"[INFO] Early stopping for seed {seed} after {args.patience} epochs without F1 improvement.")
                break
        # load best model state
        if best_state is not None:
            base_model.load_state_dict(best_state)
            if device.type == "cuda" and torch.cuda.device_count() > 1:
                model = nn.DataParallel(base_model.to(device))
            else:
                model = base_model.to(device)
            # save best model for this seed
            model_path = results_dir / f"best_model_seed{seed}.pt"
            torch.save(best_state, model_path)
            print(f"[INFO] Best model for seed {seed} saved to {model_path}")
        else:
            print(f"[WARN] No best_state found for seed {seed}.")
            best_val_loss = float("nan")
        # evaluate on test set
        test_metrics = evaluate_split(model, features_root, split="test", device=device,
                                      criterion=criterion, win_frames=eval_window_frames,
                                      stride_frames=eval_stride_frames, batch_size=args.eval_batch_size)
        print(
            f"\n[SUMMARY Seed {seed}]\n"
            f"Best Val Loss (at best F1): {best_val_loss:.6f}\n"
            f"Val:  Acc={best_val_metrics['accuracy']:.4f}, Prec={best_val_metrics['precision']:.4f}, Rec={best_val_metrics['recall']:.4f}, F1={best_val_metrics['f1']:.4f}\n"
            f"Test: Loss={test_metrics['loss']:.6f}, Acc={test_metrics['accuracy']:.4f}, Prec={test_metrics['precision']:.4f}, Rec={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}\n"
        )
        # store per‑seed results
        seed_result = {
            "seed": seed,
            "best_val_loss": best_val_loss,
            "val": dict(best_val_metrics),
            "test": dict(test_metrics),
        }
        all_seed_results.append(seed_result)
        # accumulate for aggregation
        val_metrics_all["loss"].append(best_val_loss)
        for k in ["accuracy", "precision", "recall", "f1"]:
            val_metrics_all[k].append(best_val_metrics[k])
        test_metrics_all["loss"].append(test_metrics["loss"])
        for k in ["accuracy", "precision", "recall", "f1"]:
            test_metrics_all[k].append(test_metrics[k])
    # write summary results
    results_path = results_dir / "results.txt"
    def mean_std(x: List[float]) -> Tuple[float, float]:
        arr = np.array(x, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))
    with open(results_path, "w") as f:
        f.write("Per‑seed results (validation at best F1 and test):\n\n")
        for res in all_seed_results:
            seed = res["seed"]
            bv_loss = res["best_val_loss"]
            v = res["val"]
            t = res["test"]
            f.write(f"Seed {seed}:\n")
            f.write(f"  Val Loss (at best F1): {bv_loss:.6f}\n")
            f.write(f"  Val Accuracy:  {v['accuracy']:.4f}\n")
            f.write(f"  Val Precision: {v['precision']:.4f}\n")
            f.write(f"  Val Recall:    {v['recall']:.4f}\n")
            f.write(f"  Val F1:        {v['f1']:.4f}\n")
            f.write(f"  Test Loss:     {t['loss']:.6f}\n")
            f.write(f"  Test Accuracy: {t['accuracy']:.4f}\n")
            f.write(f"  Test Precision:{t['precision']:.4f}\n")
            f.write(f"  Test Recall:   {t['recall']:.4f}\n")
            f.write(f"  Test F1:       {t['f1']:.4f}\n")
            f.write("\n")
        f.write("============================================================\n")
        f.write("Averages over all seeds (mean ± std):\n\n")
        # Validation
        f.write("Validation (at best F1):\n")
        for m in metric_names:
            mu, sd = mean_std(val_metrics_all[m])
            if m == "loss":
                f.write(f"  Val {m.capitalize()}: {mu:.6f} ± {sd:.6f}\n")
            else:
                f.write(f"  Val {m.capitalize()}: {mu:.4f} ± {sd:.4f}\n")
        f.write("\n")
        # Test
        f.write("Test:\n")
        for m in metric_names:
            mu, sd = mean_std(test_metrics_all[m])
            if m == "loss":
                f.write(f"  Test {m.capitalize()}: {mu:.6f} ± {sd:.6f}\n")
            else:
                f.write(f"  Test {m.capitalize()}: {mu:.4f} ± {sd:.4f}\n")
    print(f"[INFO] All results saved to {results_path}")


if __name__ == '__main__':
    main()