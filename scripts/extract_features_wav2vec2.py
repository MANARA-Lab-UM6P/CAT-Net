#!/usr/bin/env python3
"""
Extract Wav2Vec 2.0 self‑supervised features for overlapping speech detection.

This script loads mixtures from a synthetic OSD dataset, resamples them to
16 kHz, runs a pretrained Wav2Vec 2.0 (base) model from `torchaudio` to
obtain frame‑level embeddings, segments the embeddings into overlapping
windows, constructs binary labels for each frame based on overlap
annotations, and saves the result in compressed ``.npz`` files.

The output directory mirrors the input directory structure and contains
``train.npz``, ``val.npz`` and ``test.npz`` files per condition.  Each
``.npz`` file contains three arrays: ``features`` (shape ``(N, seq_len, D)``),
``labels`` (shape ``(N, seq_len)``) and ``mask`` (boolean mask indicating
valid frames).

Usage
-----

```bash
python scripts/extract_features_wav2vec2.py \
    --input_root <path/to/generated_osd_dataset> \
    --output_root <path/to/save/ssl_features>
```

You can specify which conditions to process via `--conditions` (comma
separated).  See `--help` for additional options.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


###############################################################################
# Helper functions
###############################################################################

def read_overlap_annotation(txt_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Read overlap annotation from ``*_start_end.txt``."""
    try:
        with open(txt_path, "r") as f:
            vals = f.read().strip().split()
        if len(vals) >= 2:
            start = int(vals[0])
            end = int(vals[1])
            if end < start:
                start, end = end, start
            return start, end
    except Exception:
        pass
    return None, None


def compute_labels(num_frames: int, orig_len: int, ov_start: Optional[int], ov_end: Optional[int],
                   frame_stride_samples: int, frame_window_samples: int) -> np.ndarray:
    """Compute a binary label vector for each embedding frame.

    A frame is labelled as overlapping if any part of its receptive field
    (specified by ``frame_stride_samples`` and ``frame_window_samples``) intersects
    the annotated overlap region.
    """
    labels = np.zeros(num_frames, dtype=np.int64)
    if ov_start is None or ov_end is None or ov_end <= ov_start or orig_len <= 0:
        return labels
    for i in range(num_frames):
        frame_start = i * frame_stride_samples
        frame_end = frame_start + frame_window_samples
        if frame_start >= orig_len:
            break
        if max(frame_start, ov_start) < min(frame_end, ov_end):
            labels[i] = 1
    return labels


def segment_sequences(frames: np.ndarray, labels: np.ndarray, seq_len: int, seq_stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment frames and labels into overlapping sequences with reflective padding."""
    T, D = frames.shape
    seq_starts = list(range(0, max(T - seq_len, 0) + 1, seq_stride))
    if not seq_starts or (seq_starts[-1] + seq_len < T):
        seq_starts.append(max(T - seq_len, 0))
    sequences = []
    seq_labs = []
    masks = []
    for start in seq_starts:
        end = start + seq_len
        if end <= T:
            seq_frames = frames[start:end]
            seq_labels = labels[start:end]
            mask = np.ones(seq_len, dtype=bool)
        else:
            tail = frames[start:T]
            pad_len = end - T
            if T >= 2:
                refl_idx = np.arange(T - 2, max(T - 2 - pad_len, -1), -1)[:pad_len]
                pad = frames[refl_idx]
                pad_labels = labels[refl_idx]
            else:
                pad = np.repeat(tail[-1:], pad_len, axis=0)
                pad_labels = np.repeat(labels[-1:], pad_len, axis=0)
            seq_frames = np.concatenate([tail, pad], axis=0)
            seq_labels = np.concatenate([labels[start:T], pad_labels], axis=0)
            mask = np.zeros(seq_len, dtype=bool)
            mask[:seq_len - pad_len] = True
        sequences.append(seq_frames)
        seq_labs.append(seq_labels)
        masks.append(mask)
    return np.stack(sequences, axis=0), np.stack(seq_labs, axis=0), np.stack(masks, axis=0)


def find_leaf_dirs(dataset_root: Path) -> List[Path]:
    """Locate leaf condition directories containing split subdirectories."""
    leaf_dirs: List[Path] = []
    for root, dirs, files in os.walk(dataset_root):
        path = Path(root)
        parts = path.parts
        if any(part in ['train', 'val', 'test'] for part in parts):
            leaf = path.parent
            if leaf not in leaf_dirs:
                leaf_dirs.append(leaf)
    return sorted(set(leaf_dirs))


def process_split(model: torchaudio.models.Wav2Vec2Model, split_dir: Path, sample_rate: int,
                  frame_stride_samples: int, frame_window_samples: int,
                  seq_len: int, seq_stride: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process all mixtures in a given split directory."""
    wav_paths = sorted([p for p in split_dir.rglob('*.wav')])
    if not wav_paths:
        return (np.empty((0, 0, 0), dtype=np.float32),
                np.empty((0, 0), dtype=np.int64),
                np.empty((0, 0), dtype=bool))
    sequences_list: List[np.ndarray] = []
    labels_seq_list: List[np.ndarray] = []
    masks_seq_list: List[np.ndarray] = []
    iterator = wav_paths
    for wav_path in tqdm(iterator, desc=f"{split_dir.name}", unit="file", leave=False):
        ann_path = wav_path.with_name(wav_path.stem + '_start_end.txt')
        ov_start, ov_end = read_overlap_annotation(ann_path)
        # Load audio at original sample rate
        wav_orig, sr_orig = torchaudio.load(str(wav_path))
        # Downmix to mono if necessary
        if wav_orig.dim() > 1 and wav_orig.size(0) > 1:
            wav_orig = wav_orig.mean(dim=0, keepdim=True)
        # Resample waveform to target SR if necessary
        if sr_orig != sample_rate:
            wav = torchaudio.functional.resample(wav_orig, orig_freq=sr_orig, new_freq=sample_rate)
            sr = sample_rate
        else:
            wav = wav_orig
            sr = sr_orig
        orig_len = wav.size(1)
        wav_tensor = wav.squeeze(0).to(device)
        wav_tensor = wav_tensor.unsqueeze(0)
        lengths = torch.tensor([orig_len], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs, out_lengths = model(wav_tensor, lengths)
            feats = outputs[0].cpu().numpy()
            num_frames = int(out_lengths[0].item())
            feats = feats[:num_frames]
        labels = compute_labels(num_frames, orig_len, ov_start, ov_end,
                               frame_stride_samples, frame_window_samples)
        seq_feats, seq_labels, seq_mask = segment_sequences(feats, labels,
                                                           seq_len=seq_len, seq_stride=seq_stride)
        for sf, sl, sm in zip(seq_feats, seq_labels, seq_mask):
            sequences_list.append(sf)
            labels_seq_list.append(sl)
            masks_seq_list.append(sm)
    if not sequences_list:
        return (np.empty((0, 0, 0), dtype=np.float32),
                np.empty((0, 0), dtype=np.int64),
                np.empty((0, 0), dtype=bool))
    features_all = np.stack(sequences_list, axis=0).astype(np.float32)
    labels_all = np.stack(labels_seq_list, axis=0).astype(np.int64)
    mask_all = np.stack(masks_seq_list, axis=0).astype(bool)
    return features_all, labels_all, mask_all


###############################################################################
# Main extraction routine
###############################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Wav2Vec 2.0 features for OSD datasets.")
    parser.add_argument('--input_root', type=str, required=True,
                        help='Root of the generated overlapping speech dataset.')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Output directory for the extracted features.')
    parser.add_argument('--conditions', type=str, default=None,
                        help='Comma‑separated list of condition names to process.  If omitted, all conditions are processed.')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Sampling rate expected by the Wav2Vec 2.0 model (default: 16000).')
    parser.add_argument('--seq_len', type=int, default=50,
                        help='Number of frames per output sequence.')
    parser.add_argument('--seq_stride', type=int, default=25,
                        help='Stride in frames between sequences.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device for extraction (e.g. "cuda" or "cpu").  If unspecified, CUDA is used if available.')
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input dataset root not found: {input_root}")
    # Determine device
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Extracting Wav2Vec 2.0 features on {device}…")
    # Load pretrained Wav2Vec 2.0 base model
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model().to(device)
    model.eval()
    # Frame alignment constants (approximate, based on the base model)
    frame_stride_samples = int(0.02 * args.sample_rate)  # ~20 ms → 320 samples @16 kHz
    frame_window_samples = int(0.025 * args.sample_rate)  # ~25 ms → 400 samples @16 kHz
    # Identify leaf condition directories
    leaf_dirs = find_leaf_dirs(input_root)
    # Filter by selected conditions if provided
    if args.conditions:
        selected_set = set(c.strip() for c in args.conditions.split(',') if c.strip())
        original_count = len(leaf_dirs)
        leaf_dirs = [leaf for leaf in leaf_dirs if leaf.relative_to(input_root).parts and leaf.relative_to(input_root).parts[0] in selected_set]
        print(f"Selected conditions: {sorted(selected_set)} -> {len(leaf_dirs)} of {original_count} leaf dirs")
    if not leaf_dirs:
        print(f"No leaf condition directories found under {input_root}")
        return
    for leaf in leaf_dirs:
        rel_leaf = leaf.relative_to(input_root)
        for split in ['train', 'val', 'test']:
            split_dir = leaf / split
            if not split_dir.exists():
                continue
            print(f"Processing {rel_leaf}/{split}…")
            feats, labs, mask = process_split(model, split_dir, args.sample_rate,
                                              frame_stride_samples, frame_window_samples,
                                              args.seq_len, args.seq_stride, device)
            # Create output directory
            out_dir = output_root / rel_leaf
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{split}.npz"
            # Save aggregated arrays
            np.savez_compressed(out_path, features=feats, labels=labs, mask=mask)
            print(f"Saved {out_path}")
    print("Finished extracting Wav2Vec 2.0 features.")


if __name__ == '__main__':
    main()