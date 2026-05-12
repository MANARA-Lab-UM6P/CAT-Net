#!/usr/bin/env python3
"""
Generate synthetic overlapping speech detection datasets.

This script reads a YAML configuration file describing the locations of
the source speech corpora, the MUSAN noise dataset, and room impulse
responses (RIRs), along with various generation parameters.  It then
creates synthetic mixtures of speech under different acoustic conditions
and writes them into a directory hierarchy suitable for training and
evaluating overlapping speech detection systems.  Both the GRID and
RAVDESS corpora are supported.

The configuration file defines:

* Paths to the speech corpora (``grid_dir``, ``ravdess_dir``).
* Paths to noise and RIR datasets.
* Output directory for the generated mixtures.
* Duration (in hours) to generate for each split (train/val/test).
* SIR and SNR ranges.
* Which noise types and room types to include.
* Whether to generate reverberated mixtures.
* Gender‑aware speaker splits with fixed male/female distributions.

An example configuration is provided at ``configs/dataset_config_example.yaml``.
See that file for a full list of options.

Usage
-----

From the repository root:

```bash
python scripts/generate_dataset.py --config configs/dataset_config_example.yaml
```

The generator uses only standard Python modules plus NumPy, SciPy and
PyYAML.  It does not depend on any GPU libraries or external command
line tools.  Nevertheless, generating large datasets can take many
hours; run it on a machine with sufficient CPU and disk resources.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Iterable, Optional

import yaml
import numpy as np
from scipy.io import wavfile
from scipy import signal as sg
from tqdm import tqdm


###############################################################################
# Utility functions (copied from the original dataset generator)
###############################################################################

def load_audio(path: str | Path, target_sr: int) -> np.ndarray:
    """Load an audio file, optionally resampling to ``target_sr``."""
    path = Path(path)
    sr, data = wavfile.read(path)
    # convert to float32
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
    else:
        data = data.astype(np.float32)
    # downmix to mono if necessary
    if data.ndim > 1:
        data = data.mean(axis=1)
    # resample if needed
    if sr != target_sr:
        num_samples = int(round(len(data) * float(target_sr) / sr))
        if len(data) == 0:
            return np.array([], dtype=np.float32)
        data = sg.resample(data, num_samples).astype(np.float32)
    return data


def write_audio(path: str | Path, signal: np.ndarray, sr: int, write_int16: bool = True) -> None:
    """Write a mono audio signal to a WAV file."""
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    if write_int16:
        max_val = np.max(np.abs(signal)) if len(signal) > 0 else 0.0
        if max_val > 0:
            norm = signal / max(max_val, 1.0)
        else:
            norm = signal
        wavfile.write(str(path), sr, (norm * 32767.0).astype(np.int16))
    else:
        wavfile.write(str(path), sr, signal.astype(np.float32))


def compute_scaling_factor(signal: np.ndarray, reference: np.ndarray, desired_db: float) -> float:
    """Compute scaling factor so that signal is mixed at desired dB wrt reference."""
    ref_power = np.mean(reference ** 2) + 1e-12
    sig_power = np.mean(signal ** 2) + 1e-12
    desired_ratio = 10.0 ** (desired_db / 10.0)
    alpha = np.sqrt(ref_power / (sig_power * desired_ratio))
    return alpha


def save_noisy_variant(args: Tuple[np.ndarray, np.ndarray, int, Path, int, bool, str]) -> Tuple[str, int]:
    """Add noise at a specific SNR and write the result to disk (for multiprocessing)."""
    signal, noise_signal, snr_db, audio_path, sample_rate, write_int16, cond_key = args
    noisy_mixture = add_noise(signal, noise_signal, snr_db)
    write_audio(audio_path, noisy_mixture, sample_rate, write_int16=write_int16)
    return cond_key, len(noisy_mixture)


def mix_two_signals(target: np.ndarray, interferer: np.ndarray,
                    sir_db: float, min_overlap_ratio: float) -> Tuple[np.ndarray, int, int]:
    """Mix two signals at a specified SIR and random overlap position."""
    len_t = len(target)
    len_i = len(interferer)
    if len_t == 0 or len_i == 0:
        mixture = target + interferer
        return mixture, 0, 0
    min_overlap_samples = int(np.ceil(min_overlap_ratio * len_t))
    max_start = max(0, len_t - min_overlap_samples)
    if max_start > 0:
        start_offset = random.randint(0, max_start)
    else:
        start_offset = 0
    alpha = compute_scaling_factor(interferer, target, sir_db)
    interferer_scaled = interferer * alpha
    mixture_length = max(len_t, start_offset + len_i)
    mixture = np.zeros(mixture_length, dtype=np.float32)
    mixture[:len_t] += target
    mixture[start_offset:start_offset + len_i] += interferer_scaled
    overlap_start = start_offset
    overlap_end = min(len_t, start_offset + len_i)
    return mixture, overlap_start, overlap_end


def add_noise(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add scaled noise to a signal at a given SNR."""
    if len(noise) < len(signal):
        reps = int(np.ceil(len(signal) / len(noise)))
        noise = np.tile(noise, reps)
    if len(noise) > len(signal):
        start = random.randint(0, len(noise) - len(signal))
        noise_seg = noise[start:start + len(signal)]
    else:
        noise_seg = noise[:len(signal)]
    sig_power = np.mean(signal ** 2) + 1e-12
    noise_power = np.mean(noise_seg ** 2) + 1e-12
    desired_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    beta = np.sqrt(desired_noise_power / noise_power)
    noisy = signal + beta * noise_seg
    return noisy


def convolve_rir(signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve a signal with a room impulse response and truncate to input length."""
    if len(signal) == 0 or len(rir) == 0:
        return signal
    conv = sg.fftconvolve(signal, rir, mode='full')
    return conv[:len(signal)]


def index_corpus(corpus_root: str | Path) -> Dict[str, List[str]]:
    """Index all utterance files in a speech corpus by speaker ID."""
    corpus_root = Path(corpus_root)
    speakers: Dict[str, List[str]] = {}
    if not corpus_root.exists():
        return speakers
    for entry in sorted(corpus_root.iterdir()):
        if not entry.is_dir():
            continue
        spk_id = entry.name
        wavs: List[str] = []
        for wav_path in entry.rglob('*.wav'):
            wavs.append(str(wav_path))
        if wavs:
            speakers[spk_id] = wavs
    return speakers


def index_noise(noise_root: str | Path, noise_types: Iterable[str]) -> Dict[str, Dict[str, List[str]]]:
    """Index noise files by type and partition (train/test)."""
    noise_root = Path(noise_root)
    noise_index: Dict[str, Dict[str, List[str]]] = {}
    for ntype in noise_types:
        type_dir = noise_root / ntype
        part_dict: Dict[str, List[str]] = {'train': [], 'test': []}
        for part in ['train', 'test']:
            part_dir = type_dir / part
            if not part_dir.exists():
                continue
            wavs = [str(p) for p in part_dir.rglob('*.wav')]
            part_dict[part] = wavs
        noise_index[ntype] = part_dict
    return noise_index


def index_rirs(rir_root: str | Path, room_types: Iterable[str]) -> Dict[str, Dict[str, List[str]]]:
    """Index RIR files by room type and partition (train/test)."""
    rir_root = Path(rir_root)
    rir_index: Dict[str, Dict[str, List[str]]] = {}
    for room in room_types:
        room_dir = rir_root / room
        part_dict: Dict[str, List[str]] = {'train': [], 'test': []}
        for part in ['train', 'test']:
            part_dir = room_dir / part
            if not part_dir.exists():
                continue
            wavs = [str(p) for p in part_dir.rglob('*.wav')]
            part_dict[part] = wavs
        rir_index[room] = part_dict
    return rir_index


###############################################################################
# Speaker splitting with gender constraints
###############################################################################

def split_speakers_with_gender(
    speakers: List[str],
    male_set: set,
    corpus_name: str,
    seed: Optional[int] = None
) -> Dict[str, List[str]]:
    """Partition speakers into train/val/test with fixed total and male counts.

    For GRID and RAVDESS corpora the target distributions of total speakers and
    number of males per split are fixed to match the experiments in the paper.
    For other corpora a simple random 80/10/10 split is used.
    """
    # Target distributions: total speakers and number of males per split
    target_counts: Dict[str, Dict[str, Dict[str, int]]] = {
        'grid': {
            'train': {'total': 23, 'male': 13},
            'val':   {'total':  4, 'male':  2},
            'test':  {'total':  7, 'male':  3},
        },
        'ravdess': {
            'train': {'total': 15, 'male':  3},
            'val':   {'total':  4, 'male':  2},
            'test':  {'total':  5, 'male':  2},
        },
    }

    # If we don't have a constraint for this corpus, fall back to generic split
    if corpus_name not in target_counts:
        rng = random.Random(seed)
        spks = list(speakers)
        rng.shuffle(spks)
        n = len(spks)
        n_test = max(1, int(round(n * 0.2)))
        n_val = max(1, int(round(n * 0.1)))
        n_test = min(n_test, n)
        n_val = min(n_val, n - n_test)
        test_spk = spks[:n_test]
        val_spk = spks[n_test:n_test + n_val]
        train_spk = spks[n_test + n_val:]
        return {'train': train_spk, 'val': val_spk, 'test': test_spk}

    # Gender-aware constrained split
    rng = random.Random(seed if seed is not None else 0)

    male_list = [s for s in speakers if s in male_set]
    female_list = [s for s in speakers if s not in male_set]

    rng.shuffle(male_list)
    rng.shuffle(female_list)

    splits: Dict[str, List[str]] = {'train': [], 'val': [], 'test': []}

    remaining_males = male_list
    remaining_females = female_list

    for split_name in ['train', 'val', 'test']:
        tgt = target_counts[corpus_name][split_name]
        total_needed = tgt['total']
        male_needed = tgt['male']
        female_needed = total_needed - male_needed

        if len(remaining_males) < male_needed or len(remaining_females) < female_needed:
            raise ValueError(
                f"Not enough male/female speakers to satisfy constraints for "
                f"{corpus_name} {split_name}: need {male_needed} M, "
                f"{female_needed} F; have {len(remaining_males)} M, "
                f"{len(remaining_females)} F."
            )

        selected_males = remaining_males[:male_needed]
        selected_females = remaining_females[:female_needed]

        splits[split_name] = selected_males + selected_females

        remaining_males = remaining_males[male_needed:]
        remaining_females = remaining_females[female_needed:]

    return splits


###############################################################################
# Dataset generation functions
###############################################################################

def generate_mixtures_for_split(split_name: str, speakers: List[str], utterances: Dict[str, List[str]],
                                duration_target: float, sample_rate: int, sir_range: Tuple[float, float],
                                min_overlap_ratio: float, noise_index: Dict[str, Dict[str, List[str]]],
                                rir_index: Dict[str, Dict[str, List[str]]],
                                config: Dict, output_root: Path,
                                corpus_name: str,
                                gender_map: Dict[str, str],
                                stats: Dict[str, Dict[str, Dict[str, int]]]) -> None:
    """Generate mixtures for a single corpus and split.

    The generator operates by sampling pairs of utterances, mixing them at a
    random SIR, optionally convolving with a room impulse response and
    adding noise at various SNRs.  Mixtures are saved as WAV files with
    corresponding overlap annotations in a structured directory.
    """
    target_samples_per_cond = int(duration_target * 3600.0 * sample_rate)
    mixture_counter = 0

    noise_types = config.get('noise_types', [])
    snr_levels: List[int] = config.get('snr_levels', [])
    room_types = config.get('rir_room_types', [])
    write_int16 = bool(config.get('write_int16', True))
    sir_min, sir_max = sir_range
    num_workers = int(config.get('num_workers', -1))

    # Preindex noise lists for this split
    noise_lists: Dict[str, List[str]] = {}
    for ntype in noise_types:
        part_list = noise_index.get(ntype, {}).get('train' if split_name != 'test' else 'test', [])
        noise_lists[ntype] = part_list

    # Preindex RIR lists for this split
    rir_lists: Dict[str, List[str]] = {}
    for rtype in room_types:
        part_list = rir_index.get(rtype, {}).get('train' if split_name != 'test' else 'test', [])
        rir_lists[rtype] = part_list

    enabled_conditions = set(config.get('enabled_conditions', ['clean'] + noise_types))
    generate_reverberated = bool(config.get('generate_reverberated', True))

    cond_keys: List[str] = []
    if 'clean' in enabled_conditions:
        cond_keys.append('clean/no_reverb')
        if generate_reverberated and any(rir_lists.values()):
            cond_keys.append('clean/reverberated')
    for ntype in noise_types:
        if ntype not in enabled_conditions:
            continue
        for snr_db in snr_levels:
            cond_keys.append(f"{ntype}/{snr_db}/no_reverb")
            if generate_reverberated and any(rir_lists.values()):
                cond_keys.append(f"{ntype}/{snr_db}/reverberated")

    cond_samples = {ck: 0 for ck in cond_keys}
    total_target = target_samples_per_cond * len(cond_keys) if cond_keys else 0
    pbar = tqdm(total=total_target, desc=f"{corpus_name}-{split_name}", unit='samples', leave=False)

    while True:
        if all(val >= target_samples_per_cond for val in cond_samples.values()):
            break
        if len(speakers) < 2:
            break

        spk1, spk2 = random.sample(speakers, 2)
        utt1 = random.choice(utterances[spk1])
        utt2 = random.choice(utterances[spk2])
        sig1 = load_audio(utt1, sample_rate)
        sig2 = load_audio(utt2, sample_rate)

        sir_db = random.uniform(sir_min, sir_max)

        reverberated_signals: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if room_types:
            for rtype in room_types:
                rirs = rir_lists.get(rtype, [])
                if not rirs:
                    continue
                rir_paths = random.sample(rirs, 2) if len(rirs) >= 2 else [rirs[0], rirs[0]]
                rir1 = load_audio(rir_paths[0], sample_rate)
                rir2 = load_audio(rir_paths[1], sample_rate)
                if np.max(np.abs(rir1)) > 0:
                    rir1 = rir1 / np.max(np.abs(rir1))
                if np.max(np.abs(rir2)) > 0:
                    rir2 = rir2 / np.max(np.abs(rir2))
                sig1_rev = convolve_rir(sig1, rir1)
                sig2_rev = convolve_rir(sig2, rir2)
                reverberated_signals[rtype] = (sig1_rev, sig2_rev)

        g1 = gender_map.get(spk1, 'F')
        g2 = gender_map.get(spk2, 'F')
        if g1 == 'M' and g2 == 'M':
            gender_pair = 'mm'
        elif g1 == 'F' and g2 == 'F':
            gender_pair = 'ff'
        else:
            gender_pair = 'mf'

        if corpus_name not in stats:
            stats[corpus_name] = {}
        if split_name not in stats[corpus_name]:
            stats[corpus_name][split_name] = {'mm': 0, 'ff': 0, 'mf': 0}
        stats[corpus_name][split_name][gender_pair] += 1

        mixture, ov_start, ov_end = mix_two_signals(sig1, sig2, sir_db, min_overlap_ratio)
        # always save clean/no_reverb mixture
        cond = 'clean/no_reverb'
        if cond in cond_samples and cond_samples[cond] < target_samples_per_cond:
            mix_id = mixture_counter
            out_path = output_root / 'clean' / 'no_reverb' / split_name
            audio_path = out_path / f"{mix_id}.wav"
            ann_path = out_path / f"{mix_id}_start_end.txt"
            write_audio(audio_path, mixture, sample_rate, write_int16=write_int16)
            os.makedirs(out_path, exist_ok=True)
            with open(ann_path, 'w') as f:
                f.write(f"{ov_start}\n{ov_end}\n")
            cond_samples[cond] += len(mixture)
            pbar.update(len(mixture))
            mixture_counter += 1

        # save clean/reverberated mixtures
        for rtype, (s1_rev, s2_rev) in reverberated_signals.items():
            mixture_r, ov_start_r, ov_end_r = mix_two_signals(s1_rev, s2_rev, sir_db, min_overlap_ratio)
            cond = 'clean/reverberated'
            if cond in cond_samples and cond_samples[cond] < target_samples_per_cond:
                mix_id = mixture_counter
                out_path = output_root / 'clean' / 'reverberated' / split_name
                audio_path = out_path / f"{mix_id}.wav"
                ann_path = out_path / f"{mix_id}_start_end.txt"
                write_audio(audio_path, mixture_r, sample_rate, write_int16=write_int16)
                os.makedirs(out_path, exist_ok=True)
                with open(ann_path, 'w') as f:
                    f.write(f"{ov_start_r}\n{ov_end_r}\n")
                cond_samples[cond] += len(mixture_r)
                pbar.update(len(mixture_r))
                mixture_counter += 1

        # prepare tasks for noisy and noisy‑reverberated mixtures
        task_args: List[Tuple[np.ndarray, np.ndarray, int, Path, int, bool, str]] = []
        for ntype in noise_types:
            if ntype not in enabled_conditions:
                continue
            noise_files = noise_lists.get(ntype, [])
            if not noise_files:
                continue
            noise_path = random.choice(noise_files)
            noise_signal = load_audio(noise_path, sample_rate)
            for snr_db in snr_levels:
                # noisy clean mixture
                cond = f"{ntype}/{snr_db}/no_reverb"
                if cond in cond_samples and cond_samples[cond] < target_samples_per_cond:
                    mix_id = mixture_counter
                    out_path = output_root / ntype / f"{snr_db}" / 'no_reverb' / split_name
                    audio_path = out_path / f"{mix_id}.wav"
                    ann_path = out_path / f"{mix_id}_start_end.txt"
                    os.makedirs(out_path, exist_ok=True)
                    with open(ann_path, 'w') as f:
                        f.write(f"{ov_start}\n{ov_end}\n")
                    task_args.append((mixture, noise_signal, snr_db, audio_path, sample_rate, write_int16, cond))
                    mixture_counter += 1

                # noisy reverberated mixture
                if generate_reverberated and reverberated_signals:
                    for rtype, (s1_rev, s2_rev) in reverberated_signals.items():
                        cond = f"{ntype}/{snr_db}/reverberated"
                        if cond in cond_samples and cond_samples[cond] < target_samples_per_cond:
                            mixture_r, ov_start_r, ov_end_r = mix_two_signals(s1_rev, s2_rev, sir_db, min_overlap_ratio)
                            mix_id = mixture_counter
                            out_path = output_root / ntype / f"{snr_db}" / 'reverberated' / split_name
                            audio_path = out_path / f"{mix_id}.wav"
                            ann_path = out_path / f"{mix_id}_start_end.txt"
                            os.makedirs(out_path, exist_ok=True)
                            with open(ann_path, 'w') as f:
                                f.write(f"{ov_start_r}\n{ov_end_r}\n")
                            task_args.append((mixture_r, noise_signal, snr_db, audio_path, sample_rate, write_int16, cond))
                            mixture_counter += 1

        # run noise addition in parallel
        if task_args:
            n_procs = num_workers if num_workers and num_workers > 0 else cpu_count()
            with Pool(processes=n_procs) as pool:
                for cond_key, length in pool.imap_unordered(save_noisy_variant, task_args):
                    if cond_key in cond_samples:
                        cond_samples[cond_key] += length
                        pbar.update(length)

    pbar.close()


def generate_dataset(config: Dict) -> None:
    """Top-level function to generate OSD datasets from the given configuration."""
    random_seed = config.get('random_seed', None)
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    sample_rate: int = config.get('sample_rate', 16000)

    corpora: Dict[str, Dict[str, List[str]]] = {}
    if 'grid_dir' in config and config['grid_dir']:
        corpora['grid'] = index_corpus(config['grid_dir'])
    if 'ravdess_dir' in config and config['ravdess_dir']:
        corpora['ravdess'] = index_corpus(config['ravdess_dir'])
    if not corpora:
        raise ValueError("No speech corpora found. Please set 'grid_dir' or 'ravdess_dir' in the configuration.")

    noise_index = index_noise(config['musan_dir'], config.get('noise_types', []))
    rir_index = index_rirs(config['rir_dir'], config.get('rir_room_types', []))

    durations = config.get('duration_hours', {'train': 20, 'val': 3, 'test': 2})
    sir_range = tuple(config.get('sir_range', [0, 5]))
    min_overlap_ratio = float(config.get('min_overlap_ratio', 0.35))
    output_root = Path(config.get('output_dir', 'OSD_dataset'))
    separate = bool(config.get('separate_corpora', True))

    stats: Dict[str, Dict[str, Dict[str, int]]] = {}
    speaker_stats: Dict[str, Dict[str, Dict[str, int]]] = {}

    for corpus_name, utts_by_spk in corpora.items():
        if not utts_by_spk:
            continue

        speaker_ids = sorted(utts_by_spk.keys())

        male_cfg = config.get('male_speakers', {})
        male_list = list(male_cfg.get(corpus_name, []))
        male_set = set(str(m) for m in male_list)

        # Gender-aware, constrained split
        splits = split_speakers_with_gender(
            speakers=speaker_ids,
            male_set=male_set,
            corpus_name=corpus_name,
            seed=random_seed,
        )

        corpus_root = output_root / corpus_name if separate else output_root

        gender_map = {spk: ('M' if spk in male_set else 'F') for spk in utts_by_spk.keys()}

        speaker_stats[corpus_name] = {}
        for split_name in ['train', 'val', 'test']:
            spk_list = splits.get(split_name, [])
            male_count = sum(1 for spk in spk_list if spk in male_set)
            female_count = len(spk_list) - male_count
            speaker_stats[corpus_name][split_name] = {
                'total': len(spk_list),
                'male': male_count,
                'female': female_count,
            }

        for split_name in ['train', 'val', 'test']:
            duration_hours = durations.get(split_name, 0)
            if duration_hours <= 0:
                continue
            spk_list = splits[split_name]
            generate_mixtures_for_split(split_name, spk_list, utts_by_spk,
                                        duration_hours, sample_rate, sir_range,
                                        min_overlap_ratio, noise_index, rir_index,
                                        config, corpus_root, corpus_name,
                                        gender_map, stats)

    # Write statistics CSV
    out_root = Path(config.get('output_dir', 'OSD_dataset'))
    csv_path = out_root / 'generation_stats.csv'
    rows = []
    for corpus_name in stats:
        for split_name in stats[corpus_name]:
            mm = stats[corpus_name][split_name].get('mm', 0)
            ff = stats[corpus_name][split_name].get('ff', 0)
            mf = stats[corpus_name][split_name].get('mf', 0)
            spk_info = speaker_stats.get(corpus_name, {}).get(split_name, {})
            total_spk = spk_info.get('total', 0)
            male_spk = spk_info.get('male', 0)
            female_spk = spk_info.get('female', 0)
            rows.append([corpus_name, split_name, total_spk, male_spk, female_spk, mm, ff, mf])
    os.makedirs(out_root, exist_ok=True)
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Corpus', 'Split', 'Total Speakers', 'Male Speakers', 'Female Speakers',
                         'Male-Male Mixtures', 'Female-Female Mixtures', 'Male-Female Mixtures'])
        for row in rows:
            writer.writerow(row)


###############################################################################
# CLI entry point
###############################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate overlapping speech detection datasets.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file.')
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    generate_dataset(config)


if __name__ == '__main__':
    main()