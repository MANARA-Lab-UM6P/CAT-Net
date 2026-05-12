# CAT‑Net: Channel and Self‑Attention TCN for Robust Frame‑Level Overlapping Speech Detection

This repository contains an open source implementation of **CAT‑Net**, the neural network architecture presented in the following paper:

> **Yassin Terraf and Youssef Iraqi**, “CAT‑Net: A Channel and Self‑Attention TCN for Robust Frame‑Level Overlapping Speech Detection,” *IEEE Transactions on Audio, Speech and Language Processing*, vol. 34, pp. 1184–1199, 2026【697161440799525†L138-L147】.

CAT‑Net combines *channel‑wise attention* with a *self‑attention temporal convolutional network* (SA‑TCN) to detect overlapping speech at the frame level.  It was designed to remain effective under noisy and reverberant conditions while retaining a lightweight model size, making it suitable for integration into real‑world speech processing systems such as speaker diarization.

## Repository structure

```
catnet/
├── models/
│   ├── __init__.py            # Re‑exports CAT_Net and baseline classifiers
│   ├── cat_net.py            # Implementation of the CAT_Net architecture
│   ├── lstm_classifier.py    # Unidirectional LSTM classifier
│   ├── crnn_classifier.py    # Convolutional Recurrent Neural Network (CRNN) classifier
│   ├── tcn_transformer_classifier.py  # TCN with Transformer encoder classifier
│   └── bbcnn.py              # Block‑Based CNN classifier
├── scripts/
│   ├── generate_dataset.py    # Generate synthetic overlapping speech datasets (GRID and RAVDESS)
│   ├── extract_features_gammatone.py   # Extract log‑gammatonegram features
│   ├── extract_features_wav2vec2.py    # Extract Wav2Vec 2.0 embeddings
│   ├── extract_features_hubert.py      # Extract HuBERT embeddings
│   ├── extract_features_wavlm.py       # Extract WavLM embeddings
│   ├── train_catnet_synth.py           # Train CAT‑Net on synthetic OSD datasets (GRID/RAVDESS)
│   ├── train_catnet_ami.py             # Train CAT‑Net on the AMI Meeting corpus
│   └── evaluate_catnet.py              # Evaluate saved CAT‑Net models on multiple overlap ratios
├── configs/
│   └── dataset_config_example.yaml     # Example YAML configuration for dataset generation
├── requirements.txt                # Python dependencies
├── setup.py                       # Optional: install models as a package
├── LICENSE                         # License for this codebase
└── README.md                       # You are here
```

Each `scripts/` file can be run as a standalone program.  They accept command‑line arguments rather than hard‑coded paths, enabling you to use the scripts with your own datasets and directory structures.  See the individual scripts for usage details.

## Installation

This repository requires Python 3.8 or later.  We recommend using a virtual environment (e.g. venv or conda) before installing dependencies.

```bash
git clone <this‑repo>
cd catnet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements include PyTorch and torchaudio for neural network training and SSL feature extraction, SciPy and NumPy for signal processing, and tqdm for progress bars.

## Dataset generation

To train CAT‑Net on synthetic overlapping speech data you must first generate the datasets.  The dataset generator is based on the script originally used for the paper and accepts a YAML configuration file describing where to find the clean speech corpora, the MUSAN noise dataset, and room impulse responses.  An example configuration file is provided under `configs/dataset_config_example.yaml`.  You should edit the paths in this file to match your local environment.

```bash
python scripts/generate_dataset.py --config configs/dataset_config_example.yaml
```

The script will create a directory hierarchy under the `output_dir` defined in your YAML file.  Each mixture is saved as a WAV file along with a `_start_end.txt` annotation listing the start and end samples of the overlapping region.  This structure is compatible with the feature extraction scripts.

### Source corpora

CAT‑Net was trained on two corpora:

* **GRID** – a neutral speech corpus consisting of 34 speakers with 1 000 utterances each.
* **RAVDESS** – an emotional speech corpus with 24 actors speaking in different emotional states.

The dataset generator supports both.  Set `grid_dir` and `ravdess_dir` in the YAML file to point at your local copies of these corpora.  The script can also pool both corpora together if `separate_corpora` is set to `false`.

### Noise and reverberation

The MUSAN dataset provides additive noise.  Configure which noise types to use (babble, music, ambient noise, etc.) and the SNR levels in the YAML file.  For reverberation, provide a directory of room impulse responses.  The script will randomly convolve the clean utterances with an RIR to generate reverberant speech.  Combined noisy‑reverberant conditions can also be generated.

### Speaker splits

Speakers are split into training/validation/testing partitions with fixed gender distributions (for GRID and RAVDESS) to match the paper.  You can customise the list of male speakers in the YAML file under `male_speakers`.

## Feature extraction

Once the synthetic mixtures are created you must extract frame‑level features.  CAT‑Net can operate on several types of acoustic features:

* **Log‑gammatonegrams** – used by default in the paper.  Extract these features with:

```bash
python scripts/extract_features_gammatone.py \
    --input_root <path/to/generated_dataset> \
    --output_root <path/to/save/features>
```

* **Self‑supervised embeddings** – Wav2Vec 2.0, HuBERT, and WavLM.  Use the corresponding scripts under `scripts/` to extract these features.  Each script accepts `--input_root` and `--output_root` arguments.  These scripts require torchaudio with the appropriate pretrained bundles available.

All feature extraction scripts segment the continuous feature streams into overlapping sequences (100 frames with 50 % overlap) and save them in `.npz` files containing arrays `features`, `labels`, and `mask` for each split (train/val/test).  These files can be consumed directly by the training scripts.

## Training

### Synthetic datasets (GRID/RAVDESS)

To train CAT‑Net on the synthetic datasets you generated, run:

```bash
python scripts/train_catnet_synth.py \
    --features_root <path/to/extracted/features> \
    --results_root <path/to/save/models> \
    --model_name CATNet_grid \
    --epochs 100 --batch_size 256
```

By default the script trains separate models for each selected acoustic condition (clean, reverberant, and selected noise conditions).  You can specify which conditions to include via `--conditions`, or leave it empty to train on all available conditions.  The hyperparameters (number of blocks, repeats, hidden dimensions, learning rate, etc.) are exposed as command‑line options.

### AMI Meeting corpus

To train CAT‑Net on the AMI Meeting corpus using continuous log‑mel or SSL features, use the AMI training script:

```bash
python scripts/train_catnet_ami.py \
    --features_root <path/to/ami/features> \
    --results_root <path/to/save/ami/models> \
    --seeds 0 1 2 3 4
```

This script trains separate models for multiple random seeds and reports mean and standard deviation of the metrics across seeds.  The evaluation is performed using sliding windows as in the paper.

## Evaluation

The `evaluate_catnet.py` script evaluates trained models on multiple overlap ratios and conditions.  Suppose you trained CAT‑Net on features extracted with a 45 % global overlap ratio and saved the results in `<results_root>/CATNet_grid`; you can evaluate it on other overlap ratios (e.g. 60 %, 70 %, 80 %, 90 %) like so:

```bash
python scripts/evaluate_catnet.py \
    --test_root <path/to/features_per_overlap_ratio> \
    --results_root <path/to/saved/models> \
    --eval_root <path/to/save/evaluation> \
    --model_name CATNet_grid \
    --conditions clean/no_reverb clean/reverberated babble/0/no_reverb babble/5/no_reverb \
    --overlap_folders features_gammatone_ov_60p features_gammatone_ov_70p features_gammatone_ov_80p features_gammatone_ov_90p
```

The script loads the appropriate model weights for each condition and writes evaluation results into separate text files under the specified `eval_root` directory.

## Citation

If you use this codebase or reproduce results from our work, please cite the original paper:

```bibtex
@ARTICLE{11371619,
  author={Terraf, Yassin and Iraqi, Youssef},
  journal={IEEE Transactions on Audio, Speech and Language Processing}, 
  title={CAT-Net: A Channel and Self-Attention TCN for Robust Frame-Level Overlapping Speech Detection}, 
  year={2026},
  volume={34},
  number={},
  pages={1184-1199},
  keywords={Noise;Feature extraction;Acoustics;Voice activity detection;Spectrogram;Reverberation;Convolutional neural networks;Noise measurement;Transformers;Training;Overlapping speech detection;channel attention;temporal convolutional networks;self-attention;acoustic feature extraction;deep learning;noise robustness},
  doi={10.1109/TASLPRO.2026.3661413}
}
```

## License

This project is licensed under the MIT License (see [`LICENSE`](LICENSE) for details).  You are free to use, modify and distribute this software as long as you include the original copyright notice and license.