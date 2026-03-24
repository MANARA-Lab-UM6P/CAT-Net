#!/usr/bin/env python3
"""
Evaluate CAT‑Net models on multiple overlap ratios and recording conditions.

Given a set of trained CAT‑Net models (one per recording condition) saved
under ``results_root/model_name/<condition>/best_model.pt``, this script
loads each model, iterates over a list of overlap‑ratio feature folders
(each containing ``test.npz`` files for each condition), and reports
frame‑level evaluation metrics (loss, accuracy, precision, recall, F1).

The script writes a separate results file for each combination of
condition and overlap ratio into the specified ``eval_root`` directory.
It also prints a summary table to the console.

Usage
-----

```bash
python scripts/evaluate_catnet.py \
    --test_root <path/to/features_per_overlap_ratio> \
    --results_root <path/to/saved/models> \
    --eval_root <path/to/save/evaluation> \
    --model_name CATNet_grid \
    --conditions clean/no_reverb,clean/reverberated,babble/0/no_reverb,babble/5/no_reverb \
    --overlap_folders features_gammatone_ov_60p,features_gammatone_ov_70p
```

If ``--conditions`` is omitted the script attempts to infer all conditions
based on the directory structure under ``results_root/model_name``.  If
``--overlap_folders`` is omitted it infers folders under ``test_root``.
See ``--help`` for additional options.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import CAT_Net from the top‑level models package
from models import CAT_Net


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feats, labels, masks = zip(*batch)
    return torch.stack(feats), torch.stack(labels), torch.stack(masks)


class NPZDataset(Dataset):
    def __init__(self, npz_path: Path):
        data = np.load(npz_path, allow_pickle=False)
        self.features = data["features"]  # (N, L, C)
        self.labels = data["labels"]    # (N, L)
        self.mask = data["mask"]       # (N, L)
        assert self.features.ndim == 3
        assert self.labels.ndim == 2
        assert self.mask.shape == self.labels.shape
    def __len__(self) -> int:
        return len(self.features)
    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.features[idx]).float(),
            torch.from_numpy(self.labels[idx]).float(),
            torch.from_numpy(self.mask[idx]).float(),
        )


@torch.no_grad()
def evaluate(model: CAT_Net, loader: DataLoader, device: torch.device) -> Dict[str, float | int]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    total_loss = 0.0
    total_batches = 0
    total_frames = 0
    tp = fp = fn = tn = 0
    for feats, labels, masks in loader:
        feats, labels, masks = feats.to(device), labels.to(device), masks.to(device)
        feats = feats.permute(0, 2, 1)  # (B, C, L)
        logits = model(feats)
        loss = (criterion(logits, labels) * masks).sum() / masks.sum()
        total_loss += loss.item()
        total_batches += 1
        preds = (logits >= 0).float()
        valid = masks.view(-1) == 1
        lv = labels.view(-1)[valid]
        pv = preds.view(-1)[valid]
        total_frames += len(lv)
        tp += ((pv == 1) & (lv == 1)).sum().item()
        tn += ((pv == 0) & (lv == 0)).sum().item()
        fp += ((pv == 1) & (lv == 0)).sum().item()
        fn += ((pv == 0) & (lv == 1)).sum().item()
    if total_frames == 0:
        return {
            "loss": float("nan"),
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "frames": 0,
        }
    acc = (tp + tn) / total_frames
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {
        "loss": total_loss / max(total_batches, 1),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "frames": total_frames,
    }


def infer_model_name(weights_path: Path) -> str:
    parts = list(weights_path.parts)
    if "results" in parts:
        i = parts.index("results")
        if i + 1 < len(parts):
            return parts[i + 1]
    return weights_path.parent.name


def ratio_from_folder(folder: str) -> str:
    m = re.search(r"ov_([0-9]+p)", folder)
    return m.group(1) if m else folder


def write_results(out_dir: Path, metrics: Dict[str, float | int], ratio: str,
                  src_npz: Path, model_name: str, condition: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    condition_safe = condition.replace("/", "_")
    fpath = out_dir / f"results_{condition_safe}.txt"
    with fpath.open("w") as f:
        f.write(f"Model    : {model_name}\n")
        f.write(f"Condition: {condition}\n")
        f.write(f"Test set : {ratio}\n")
        f.write(f"Source   : {src_npz}\n\n")
        f.write(f"Loss     : {metrics['loss']:.6f}\n")
        f.write(f"Accuracy : {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall   : {metrics['recall']:.4f}\n")
        f.write(f"F1       : {metrics['f1']:.4f}\n")
        f.write(
            f"(tp={metrics['tp']}, fp={metrics['fp']}, "
            f"fn={metrics['fn']}, tn={metrics['tn']}, frames={metrics['frames']})\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CAT‑Net models on multiple overlap ratios and conditions.")
    parser.add_argument('--test_root', type=str, required=True,
                        help='Root directory containing features per overlap ratio (e.g. features_gammatone_ov_* folders).')
    parser.add_argument('--results_root', type=str, required=True,
                        help='Root directory where trained models are stored.')
    parser.add_argument('--eval_root', type=str, required=True,
                        help='Directory to save evaluation results.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model under results_root to evaluate (e.g. CATNet_grid).')
    parser.add_argument('--conditions', type=str, default=None,
                        help='Comma‑separated list of conditions to evaluate (relative paths under the model_name directory).  If omitted, infer conditions automatically.')
    parser.add_argument('--overlap_folders', type=str, default=None,
                        help='Comma‑separated list of overlap‑ratio folders to evaluate (subdirectories of test_root).  If omitted, infer overlap folders automatically.')
    # Model hyperparameters (must match training)
    parser.add_argument('--in_channels', type=int, default=80, help='Number of input feature channels.')
    parser.add_argument('--bn_channels', type=int, default=128, help='Bottleneck channel dimension.')
    parser.add_argument('--hid_channels', type=int, default=512, help='Hidden channel dimension inside conv blocks.')
    parser.add_argument('--n_blocks', type=int, default=5, help='Number of convolutional blocks per repeat.')
    parser.add_argument('--n_repeats', type=int, default=3, help='Number of repeats of the block stack.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size of the depthwise convolution.')
    parser.add_argument('--norm_type', type=str, default='gLN', choices=['gLN', 'bN'], help="Normalisation type ('gLN' or 'bN').")
    parser.add_argument('--out_classes', type=int, default=1, help='Number of output classes (1 for binary classification).')
    parser.add_argument('--device', type=str, default=None, help='Device for evaluation (e.g. "cuda" or "cpu").  If unspecified, CUDA is used if available.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation.')
    args = parser.parse_args()
    test_root = Path(args.test_root).resolve()
    results_root = Path(args.results_root).resolve()
    eval_root = Path(args.eval_root).resolve()
    model_name = args.model_name
    if not test_root.is_dir():
        raise FileNotFoundError(f"Test root not found: {test_root}")
    model_root = results_root / model_name
    if not model_root.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_root}")
    # Determine device
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Determine conditions
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(',') if c.strip()]
    else:
        # infer conditions: list directories under model_root containing best_model.pt
        conditions = []
        for cond_dir in model_root.rglob('best_model.pt'):
            rel = cond_dir.parent.relative_to(model_root)
            conditions.append(rel.as_posix())
        conditions = sorted(set(conditions))
    if not conditions:
        print("No conditions found for evaluation.")
        return
    # Determine overlap folders
    if args.overlap_folders:
        overlap_folders = [f.strip() for f in args.overlap_folders.split(',') if f.strip()]
    else:
        # infer all subdirectories under test_root
        overlap_folders = [d.name for d in test_root.iterdir() if d.is_dir()]
    if not overlap_folders:
        print("No overlap ratio folders found for evaluation.")
        return
    # Model hyperparameters
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
    # Evaluate each condition across overlap ratios
    for condition in conditions:
        condition_path = Path(condition)
        weights_path = model_root / condition_path / 'best_model.pt'
        print("\n========================================")
        print(f"Condition      : {condition}")
        print(f"Model weights  : {weights_path}")
        if not weights_path.is_file():
            print(f"[SKIP] Missing weights: {weights_path}")
            continue
        # build model
        model = CAT_Net(**model_kwargs).to(device)
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=True)
        summary_rows = []
        for folder_name in overlap_folders:
            ratio = ratio_from_folder(folder_name)
            # path to test.npz for this overlap and condition
            test_npz = test_root / folder_name / condition_path / 'test.npz'
            if not test_npz.is_file():
                print(f"[SKIP] {test_npz} not found")
                continue
            print(f"\n--- Evaluating {ratio} for condition {condition} ---")
            ds = NPZDataset(test_npz)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=False)
            metrics = evaluate(model, dl, device)
            out_dir = eval_root / model_name / ratio
            write_results(out_dir, metrics, ratio, test_npz, model_name, condition)
            summary_rows.append((ratio, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]))
        # print summary
        if summary_rows:
            print(f"\n====== Summary ({model_name}, condition={condition}) ======")
            header = f"{'ratio':>6}  {'acc':>7}  {'prec':>7}  {'rec':>7}  {'f1':>7}"
            print(header)
            print("-" * len(header))
            for ratio, acc, prec, rec, f1 in sorted(summary_rows, key=lambda x: int(re.sub(r'[^0-9]', '', x[0]) or 0)):
                print(f"{ratio:>6}  {acc:7.4f}  {prec:7.4f}  {rec:7.4f}  {f1:7.4f}")


if __name__ == '__main__':
    main()