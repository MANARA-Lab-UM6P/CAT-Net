#!/usr/bin/env python3
"""
Extract log‑gammatonegram features for CAT‑Net.

This script traverses an overlapping speech detection (OSD) dataset and
computes log gammatone filterbank energies for each mixture at the
specified sampling rate.  It then segments the features into overlapping
sequences (with reflective padding for the final window), generates
frame‑level binary labels from the corresponding ``*_start_end.txt``
annotation files, and saves the result into compressed ``.npz`` files.

The resulting ``.npz`` files contain:

* ``features`` – tensor of shape ``(N, seq_len, channels)``
* ``labels``   – tensor of shape ``(N, seq_len)`` (0 = single speaker, 1 = overlap)
* ``mask``     – boolean mask of shape ``(N, seq_len)`` indicating which frames
                 are original (True) and which are padding (False)

Each leaf condition directory (e.g. ``clean/no_reverb``) produces
``train.npz``, ``val.npz`` and ``test.npz`` files in the output directory.

Usage
-----

```bash
python scripts/extract_features_gammatone.py \
    --input_root <path/to/generated_osd_dataset> \
    --output_root <path/to/save/features> \
    --expected_sr 8000
```

You can optionally specify the set of conditions to process using
``--conditions`` (comma‑separated).  If omitted, all conditions in
``input_root`` are processed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from multiprocessing import Pool
from typing import List, Tuple, Optional

import numpy as np
from scipy.io import wavfile
from gammatone import gtgram as ext_gtgram
from tqdm import tqdm


def read_overlap_annotation(txt_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Read overlap annotation from ``*_start_end.txt`` (sample indices)."""
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


def load_wav_strict_sr(path: Path, required_sr: int) -> Tuple[int, np.ndarray]:
    """Load a WAV file as mono float32.  Raises if sampling rate differs."""
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Convert to float32 in [-1, 1]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / max(1.0, float(np.iinfo(data.dtype).max))
    else:
        data = data.astype(np.float32)
    if sr != required_sr:
        raise ValueError(f"{path} has sampling rate {sr} Hz, expected {required_sr} Hz.")
    return sr, data


def frames_labels_from_samples(n_frames: int, win_samps: int, hop_samps: int,
                               ov_start: Optional[int], ov_end: Optional[int]) -> np.ndarray:
    """Generate frame‑level binary labels from overlap sample indices."""
    labels = np.zeros(n_frames, dtype=np.int64)
    if ov_start is None or ov_end is None or ov_end <= ov_start:
        return labels
    starts = np.arange(n_frames) * hop_samps
    ends = starts + win_samps
    inter = np.minimum(ends, ov_end) - np.maximum(starts, ov_start)
    labels[inter > 0] = 1
    return labels


def segment_sequences(frames: np.ndarray, labels: np.ndarray, seq_len: int, seq_stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment frames and labels into sequences with reflective padding."""
    T, F = frames.shape
    seq_starts = list(range(0, max(T - seq_len, 0) + 1, seq_stride))
    if not seq_starts or (seq_starts[-1] + seq_len < T):
        seq_starts.append(max(T - seq_len, 0))
    seqs, labs, masks = [], [], []
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
        seqs.append(seq_frames)
        labs.append(seq_labels)
        masks.append(mask)
    return np.stack(seqs, axis=0), np.stack(labs, axis=0), np.stack(masks, axis=0)


def worker_extract(wav_path: Path, ann_path: Path, temp_dir: Path,
                   expected_sr: int, window_time: float, hop_time: float,
                   channels: int, f_min: float, seq_len: int, seq_stride: int,
                   eps: float) -> Tuple[bool, str]:
    """Compute features for a single WAV and save to a temporary file."""
    try:
        sr, signal = load_wav_strict_sr(wav_path, expected_sr)
        # gammatonegram: (channels, T) -> transpose to (T, channels)
        gtfb = ext_gtgram.gtgram(signal, sr, window_time, hop_time, channels, f_min)
        frames = np.log(gtfb + eps).T
        # labels from annotation (sample indices at same sr)
        ov_start, ov_end = read_overlap_annotation(ann_path)
        win_samps = int(round(window_time * sr))
        hop_samps = int(round(hop_time * sr))
        labels = frames_labels_from_samples(frames.shape[0], win_samps, hop_samps, ov_start, ov_end)
        # segment
        seqs, seq_labels, seq_mask = segment_sequences(frames, labels, seq_len=seq_len, seq_stride=seq_stride)
        # temp output name is md5 of path for uniqueness
        h = hashlib.md5(str(wav_path).encode('utf-8')).hexdigest()
        out_path = temp_dir / f"{h}.npz"
        np.savez_compressed(out_path, features=seqs.astype(np.float32), labels=seq_labels.astype(np.int64), mask=seq_mask.astype(bool))
        return True, str(out_path)
    except Exception as e:
        return False, f"{wav_path}: {e}"


def concat_temp_npzs(temp_dir: Path, out_file: Path) -> None:
    """Concatenate arrays from temporary files and write to out_file."""
    files = sorted(p for p in temp_dir.glob("*.npz"))
    Xs, ys, ms = [], [], []
    for f in files:
        with np.load(f) as d:
            Xs.append(d['features'])
            ys.append(d['labels'])
            ms.append(d['mask'])
    if Xs:
        X_all = np.concatenate(Xs, axis=0)
        y_all = np.concatenate(ys, axis=0)
        m_all = np.concatenate(ms, axis=0)
        np.savez_compressed(out_file, features=X_all, labels=y_all, mask=m_all)


def find_leaf_dirs(dataset_root: Path) -> List[Path]:
    """Locate leaf condition directories containing split subdirectories."""
    leaf_dirs: List[Path] = []
    for root, dirs, _ in os.walk(dataset_root):
        if any(d in ['train', 'val', 'test'] for d in dirs):
            leaf_dirs.append(Path(root))
    return sorted(set(leaf_dirs))


def process_condition(condition_dir: Path, output_dir: Path, expected_sr: int,
                      window_time: float, hop_time: float, channels: int,
                      f_min: float, seq_len: int, seq_stride: int, eps: float,
                      num_workers: int, chunk_size: int) -> None:
    """Process a single condition directory and write train/val/test npz files."""
    # ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = condition_dir / split
        if not split_dir.is_dir():
            continue
        wavs = sorted(split_dir.rglob("*.wav"))
        if not wavs:
            continue
        temp_dir = output_dir / f".tmp_{split}"
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        tasks: List[Tuple[Path, Path, Path, int, float, float, int, float, int, int, float]] = []
        for wav_path in wavs:
            ann_path = wav_path.with_name(wav_path.stem + "_start_end.txt")
            tasks.append((wav_path, ann_path, temp_dir, expected_sr,
                          window_time, hop_time, channels, f_min,
                          seq_len, seq_stride, eps))
        print(f"[GAMMATONE] {condition_dir.relative_to(dataset_root)}/{split}: {len(tasks)} files. Extracting...")
        failures = 0
        with Pool(processes=num_workers) as pool:
            for ok, msg in tqdm(pool.imap_unordered(lambda a: worker_extract(*a), tasks, chunksize=chunk_size),
                                total=len(tasks), desc=f"[GAMMATONE] {condition_dir.relative_to(dataset_root)}/{split}"):
                if not ok:
                    failures += 1
                    print("Error:", msg)
        # concatenate temp npz → final split file
        out_file = output_dir / f"{split}.npz"
        concat_temp_npzs(temp_dir, out_file)
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"[GAMMATONE] Saved {out_file} (failures: {failures})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract log‑gammatonegram features for CAT‑Net.")
    parser.add_argument('--input_root', type=str, required=True,
                        help='Root directory of the generated OSD dataset.')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Output directory for the extracted features.')
    parser.add_argument('--conditions', type=str, default=None,
                        help='Comma‑separated list of condition names to process (e.g. "clean,babble"). If omitted, all conditions are processed.')
    parser.add_argument('--expected_sr', type=int, default=8000,
                        help='Expected sampling rate of the input WAV files.')
    parser.add_argument('--window_time', type=float, default=0.025,
                        help='Frame length in seconds (e.g. 0.025 for 25 ms).')
    parser.add_argument('--hop_time', type=float, default=0.010,
                        help='Frame hop in seconds (e.g. 0.010 for 10 ms).')
    parser.add_argument('--channels', type=int, default=80,
                        help='Number of gammatone channels.')
    parser.add_argument('--f_min', type=float, default=50.0,
                        help='Minimum frequency of the filterbank in Hz.')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='Number of frames per output sequence.')
    parser.add_argument('--seq_stride', type=int, default=50,
                        help='Stride between sequences in frames.')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Small constant to avoid log(0).')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for parallel extraction.')
    parser.add_argument('--chunk_size', type=int, default=16,
                        help='Chunk size for multiprocessing.')
    args = parser.parse_args()

    dataset_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Input dataset root not found: {dataset_root}")

    # Determine which condition roots to scan
    if args.conditions:
        cond_names = [c.strip() for c in args.conditions.split(',') if c.strip()]
        cond_dirs = [dataset_root / cond for cond in cond_names]
    else:
        cond_dirs = [p for p in dataset_root.iterdir() if p.is_dir()]

    if not cond_dirs:
        print(f"No conditions found under {dataset_root}")
        return

    for cond_dir in cond_dirs:
        if not cond_dir.exists() or not cond_dir.is_dir():
            continue
        # Identify leaf dirs that directly contain any of the split folders
        leaf_dirs: List[Path] = []
        for root, dirs, _ in os.walk(cond_dir):
            if any(s in dirs for s in ['train', 'val', 'test']):
                leaf_dirs.append(Path(root))
        leaf_dirs = sorted(set(leaf_dirs))
        for leaf in leaf_dirs:
            rel_leaf = leaf.relative_to(dataset_root)
            out_dir = output_root / rel_leaf
            process_condition(leaf, out_dir, args.expected_sr,
                              args.window_time, args.hop_time, args.channels,
                              args.f_min, args.seq_len, args.seq_stride,
                              args.eps, args.num_workers, args.chunk_size)


if __name__ == '__main__':
    main()