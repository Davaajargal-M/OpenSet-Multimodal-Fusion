# Open-Set Multimodal Fusion for HSI--LiDAR Classification

This repository contains a PyTorch implementation of an open-set multimodal fusion framework for hyperspectral image (HSI) and LiDAR classification. The code supports Houston2013, MUUFL, and Augsburg datasets with strict open-set recognition (OSR) evaluation.

## Main features

- HSI + LiDAR patch-based data loading
- Known-only training and validation for strict OSR
- Unknown classes are used only at test time by default
- Uncertainty-aware multimodal gate
- Feedback-based iterative fusion
- Monotonic confidence head
- EVT/Weibull post-hoc calibrator fitted on known validation samples only
- Multi-seed reporting with mean and standard deviation

## Repository structure

```text
.
├── train.py
├── dataset_fixed_multi.py
├── openset_multimodal_fusion_change_signal_conbination.py
├── requirements.txt
├── environment.yml
├── configs/
├── scripts/
├── data/
└── results/
```

## Installation

### Option 1: pip

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version from the official PyTorch installation page.

### Option 2: conda

```bash
conda env create -f environment.yml
conda activate openset-mm-fusion
```

## Dataset preparation

Place dataset files under `data/<dataset_name>/` or pass an absolute path through `--data_root`.

Expected file names are listed in `data/README.md`.

## Example commands

### Houston2013

```bash
python train.py   --dataset houston2013   --data_root "data/houston2013"   --unknown_class 6 13 14 15   --samples_per_class 20   --val_unknown_samples 0   --epochs 200   --seeds 0 1 2 3 4
```

### MUUFL

```bash
python train.py   --dataset muufl   --data_root "data/muufl"   --unknown_class 7 8 9 10 11   --samples_per_class 20   --val_unknown_samples 0   --epochs 200   --seeds 0 1 2 3 4
```

### Augsburg

```bash
python train.py   --dataset augsburg   --data_root "data/augsburg"   --unknown_class 4 5   --samples_per_class 20   --val_unknown_samples 0   --epochs 200   --seeds 0 1 2 3 4
```

## Strict OSR protocol

The default command uses `--val_unknown_samples 0`, meaning:

- Train: known classes only
- Validation: known classes only
- Test: known + unknown classes
- EVT threshold: fitted from known validation samples only

This avoids using unknown samples for model selection or calibration.

## Outputs

Results are saved under:

```text
results/<dataset>/class<unknown_ids>/
```

Important output files include:

- `history.json`
- `val_posthoc_metrics.json`
- `test_posthoc_metrics.json`
- `seed_final_summary.json`
- `all_seed_results.csv`
- `seed_summary_results.csv`
- `summary_across_seeds.json`
- `checkpoints/full_model_best.pth`
- `checkpoints/posthoc_calibrator.pt`

## Citation note

When using this repository in a paper, describe the protocol clearly: known-only train/validation, unknown-only use at test time, and per-seed independent checkpoint and calibrator selection.
