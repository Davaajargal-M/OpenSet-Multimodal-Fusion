# Cleanup Report

This repository was cleaned for GitHub release and reproducible research use.

## Cleaned items

- Removed external `dataset.py` fallback dependency.
- Fixed `load_dataset()` so it uses the built-in Houston2013, MUUFL, and Augsburg loaders only.
- Removed the undefined `base` reference in `dataset_fixed_multi.py`.
- Removed duplicate pre-loop dataloader construction in `train.py`.
- Removed unused imports from dataset, model, and training scripts.
- Removed unused CLI arguments for inactive calibrator modes.
- Removed unused helper function `safe_item()`.
- Fixed seed output paths to avoid duplicated nested paths.
- Removed `__pycache__` and compiled binary files from the release archive.
- Verified Python syntax compilation for all Python files.

## Main entry point

```bash
python train.py --dataset houston2013 --data_root data/houston2013 --unknown_class 6 13 14 15 --samples_per_class 20 --val_unknown_samples 0 --epochs 200 --seeds 0 1 2 3 4
```
