# dataset_fixed_multi.py
"""
Dataset Loader for Open-Set Multimodal Classification
Supports: Houston2013, MUUFL, Augsburg

Features:
- HSI + LiDAR data loading
- Open-set split (known/unknown classes)
- Patch extraction
- Data augmentation
- Normalization
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import List, Tuple
import h5py
import os
import scipy.io as scio


class MultiModalOpenSetDataset(Dataset):
    """Base dataset class for multimodal open-set classification"""
    
    def __init__(
        self,
        hsi_data: np.ndarray,
        lidar_data: np.ndarray,
        labels: np.ndarray,
        known_classes: List[int],
        patch_size: int = 15,
        split: str = 'train',
        augment: bool = True,
        normalize: bool = True
    ):
        self.hsi_data = hsi_data
        self.lidar_data = lidar_data
        self.labels = labels
        self.gt = self.labels
        self.known_classes = set(known_classes)
        self.known_classes_list = sorted(list(known_classes))
        self.patch_size = patch_size
        self.split = split
        self.augment = augment and (split == 'train')
        
        # Get dimensions
        self.h, self.w, self.hsi_channels = hsi_data.shape
        self.lidar_channels = lidar_data.shape[2] if lidar_data.ndim == 3 else 1
        
        # Normalize
        if normalize:
            self.hsi_data = self._normalize(self.hsi_data)
            self.lidar_data = self._normalize(self.lidar_data)
        
        # Samples will be set by create_dataloaders (after __init__ returns)
        self.samples = []
        self._split_name = split
        self.known_classes_list_ref = known_classes  # for deferred print
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1]"""
        data_min = np.min(data, axis=(0, 1), keepdims=True)
        data_max = np.max(data, axis=(0, 1), keepdims=True)
        data_range = data_max - data_min
        data_range[data_range == 0] = 1
        return (data - data_min) / data_range
    
    def _extract_samples(self) -> List[Tuple[int, int, bool]]:
        """Extract valid sample locations"""
        samples = []
        margin = self.patch_size // 2
        
        for i in range(margin, self.h - margin):
            for j in range(margin, self.w - margin):
                ##label = self.labels[i, j]
                label = self.gt[i, j]
                if label == 0:  # Skip background
                    continue
                is_known = label in self.known_classes
                samples.append((i, j, is_known))
        
        return samples
    
    def _extract_patch(self, data: np.ndarray, row: int, col: int) -> np.ndarray:
        """Extract spatial patch"""
        margin = self.patch_size // 2
        patch = data[row-margin:row+margin+1, col-margin:col+margin+1, :]
        return patch
    
    def _augment_patch(self, hsi_patch: np.ndarray, lidar_patch: np.ndarray):
        """Apply augmentation"""
        if np.random.random() > 0.5:
            hsi_patch = np.flip(hsi_patch, axis=0).copy()
            lidar_patch = np.flip(lidar_patch, axis=0).copy()
        if np.random.random() > 0.5:
            hsi_patch = np.flip(hsi_patch, axis=1).copy()
            lidar_patch = np.flip(lidar_patch, axis=1).copy()
        
        k = np.random.randint(0, 4)
        if k > 0:
            hsi_patch = np.rot90(hsi_patch, k, axes=(0, 1)).copy()
            lidar_patch = np.rot90(lidar_patch, k, axes=(0, 1)).copy()
        
        return hsi_patch, lidar_patch
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        row, col, is_known = self.samples[idx]
        
        # Extract patches
        hsi_patch = self._extract_patch(self.hsi_data, row, col)
        lidar_patch = self._extract_patch(self.lidar_data, row, col)
        
        # Augment
        if self.augment:
            hsi_patch, lidar_patch = self._augment_patch(hsi_patch, lidar_patch)
        
        # Get original label
        original_label = self.labels[row, col]
        
        # REMAP LABEL to [0, num_known_classes-1]
        if is_known:
            # Find the index in the sorted known classes list
            label = self.known_classes_list.index(original_label)
        else:
            # Unknown samples - assign label 0 (will be ignored by loss)
            label = 0
        
        # To tensor
        hsi_patch = torch.from_numpy(hsi_patch).permute(2, 0, 1).float()
        lidar_patch = torch.from_numpy(lidar_patch).permute(2, 0, 1).float()
        label = torch.tensor(label, dtype=torch.long)
        is_known = torch.tensor(is_known, dtype=torch.bool)
        
        return hsi_patch, lidar_patch, label, is_known

class Houston2013Dataset:
    """
    Houston2013 Dataset Loader
    
    Dataset structure:
    - HSI: 144 channels
    - LiDAR: 1 channel (DSM)
    - Classes: 15 (0=background, 1-15=classes)
    """
    
    def __init__(self, data_root: str, patch_size: int = 15):
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        
        # Load data
        print("Loading Houston2013 dataset...")
        self.hsi_data, self.lidar_data, self.labels = self._load_data()
        self.num_classes = int(np.max(self.labels))

        # 1. Ensure we have a 2D labels array
        # (Try different attribute names your code might use)
        if hasattr(self, 'gt'):
            labels_2d = self.gt
        elif hasattr(self, 'labels') and isinstance(self.labels, np.ndarray):
            if self.labels.ndim == 2:
                labels_2d = self.labels
            else:
                # Already flattened, need to reshape
                h, w = self.hsi.shape[:2]
                labels_2d = self.labels.reshape(h, w)
        elif hasattr(self, 'y'):
            labels_2d = self.y if self.y.ndim == 2 else self.y.reshape(self.hsi.shape[:2])
        else:
            raise AttributeError("Cannot find labels in dataset!")
        
        # 2. Generate spatial coordinates
        h, w = labels_2d.shape
        rows, cols = np.meshgrid(range(h), range(w), indexing='ij')
        coordinates = np.stack([rows.flatten(), cols.flatten()], axis=1)
        
        # 3. Flatten all data
        labels_flat = labels_2d.flatten()
        hsi_flat = self.hsi_data.reshape(-1, self.hsi_data.shape[-1])  # ✅ Changed hsi to hsi_data
        lidar_flat = self.lidar_data.reshape(-1, self.lidar_data.shape[-1])  # ✅ Changed lidar to lidar_data
        
        # 4. Remove background (label == 0)
        valid_mask = labels_flat > 0
        
        # 5. Store as attributes (this is what create_dataloaders needs)
        self.coordinates = coordinates[valid_mask]
        self.labels = labels_flat[valid_mask]
        self.hsi_flat = hsi_flat[valid_mask]
        self.lidar_flat = lidar_flat[valid_mask]
        
        # Also keep the 2D version for reference
        self.gt = labels_2d
        
        print(f"  Processed dataset:")
        print(f"    Valid pixels: {len(self.labels)}")
        print(f"    Coordinates: {self.coordinates.shape}")
        print(f"    HSI flat: {self.hsi_flat.shape}")
        print(f"    LiDAR flat: {self.lidar_flat.shape}")
        print(f"    Unique classes: {np.unique(self.labels)}")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load Houston2013 data"""
        # Load HSI
        hsi_path = self.data_root / 'houston_hsi.mat'
        hsi_mat = sio.loadmat(hsi_path)
        hsi_data = hsi_mat['houston_hsi'].astype(np.float32)
        
        # Load LiDAR
        lidar_path = self.data_root / 'houston_lidar.mat'
        lidar_mat = sio.loadmat(lidar_path)
        lidar_data = lidar_mat['houston_lidar'].astype(np.float32)
        
        # Add channel dimension to LiDAR if needed
        if lidar_data.ndim == 2:
            lidar_data = lidar_data[:, :, np.newaxis]
        
        # Load labels
        label_path = self.data_root / 'houston_gt.mat'
        label_mat = sio.loadmat(label_path)
        labels = label_mat['houston_gt'].astype(np.int64)
        
        print(f"  HSI shape: {hsi_data.shape}")
        print(f"  LiDAR shape: {lidar_data.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Classes: {np.unique(labels)}")
        
        return hsi_data, lidar_data, labels
    
    def create_dataloaders(self, unknown_class, batch_size, num_workers=0, 
                      train_ratio=0.2, val_ratio=0.1, samples_per_class=None, val_unknown_samples=0):
        """
        Create dataloaders for open-set learning
        
        Args:
            unknown_class: Which class to treat as unknown
            batch_size: Batch size
            num_workers: Number of workers
            train_ratio: Training ratio (ignored if samples_per_class is set)
            val_ratio: Validation ratio
            samples_per_class: FIXED number of samples per class (e.g., 50) - NEW!
        """
        # Determine known classes (all except unknown_class and background)
        all_classes = np.unique(self.labels)
        all_classes = all_classes[all_classes > 0]  # Remove background
        #known_classes = [c for c in all_classes if c != unknown_class]
        if isinstance(unknown_class, (list, tuple, set)):
            unknown_set = set(unknown_class)
        else:
            unknown_set = {unknown_class}

        known_classes = [c for c in all_classes if c not in unknown_set]
        
        #print(f"\n  Creating datasets (unknown class: {unknown_class})")
        print(f"Creating datasets (unknown classes: {sorted(list(unknown_set))})")
        print(f"  Known classes: {known_classes}")
        
        # Get all valid pixel indices
        #h, w = self.labels.shape
        h, w = self.gt.shape
        margin = self.patch_size // 2
        all_indices = []
        
        for i in range(margin, h - margin):
            for j in range(margin, w - margin):
                #label = self.labels[i, j]
                label = self.gt[i, j]
                if label == 0:  # Skip background
                    continue
                all_indices.append((i, j, label))
        
        # ─────────────────────────────────────────
        # patch-safe unknown statistics
        # ─────────────────────────────────────────

        if isinstance(unknown_class, (list, tuple, set)):
            unknown_set = set(unknown_class)
        else:
            unknown_set = {unknown_class}

        print("\n[DEBUG] Patch-safe class distribution:")

        # per-class count
        class_counts = {}
        for _, _, label in all_indices:
            class_counts[label] = class_counts.get(label, 0) + 1

        for uc in sorted(unknown_set):
            cnt = class_counts.get(uc, 0)
            print(f"  Unknown class {uc}: {cnt}")

        total_unknown = sum(class_counts.get(uc, 0) for uc in unknown_set)
        print(f"  Total unknown (patch-safe): {total_unknown}")

        total_all = sum(class_counts.values())
        print(f"  Total patch-safe pixels: {total_all}\n")

        # Separate known and unknown indices
        known_indices = [(i, j, label) for i, j, label in all_indices if label in known_classes]
        #unknown_indices = [(i, j, label) for i, j, label in all_indices if label == unknown_class]
        unknown_indices = [(i, j, label) for i, j, label in all_indices if label in unknown_set]

        # ========== FIXED SAMPLE COUNT PROTOCOL ==========
        # Train and val each get their own independent fixed budget.
        # This prevents val size collapsing to n_train * val_ratio (e.g. 50*0.15=7/class).
        # Standard in HyLiOSR and similar work: 50 train + 50 val + remaining → test.
        VAL_PER_CLASS = samples_per_class if samples_per_class is not None else max(1, int(50 * val_ratio))

        train_indices = []
        val_indices = []
        test_known_indices = []

        if samples_per_class is not None:
            print(f"  Using FIXED {samples_per_class} train + {VAL_PER_CLASS} val samples per class")
        else:
            print(f"  Using {train_ratio*100:.0f}% training data per class")

        for cls in known_classes:
            cls_indices = [(i, j) for i, j, label in known_indices if label == cls]
            np.random.shuffle(cls_indices)

            if samples_per_class is not None:
                n_train = min(samples_per_class, len(cls_indices))
                n_val   = min(VAL_PER_CLASS, len(cls_indices) - n_train)
            else:
                n_total = len(cls_indices)
                n_train = int(n_total * train_ratio)
                n_val   = int(n_total * val_ratio)

            train_indices.extend([(i, j, cls, True) for i, j in cls_indices[:n_train]])
            val_indices.extend([(i, j, cls, True)   for i, j in cls_indices[n_train:n_train+n_val]])
            test_known_indices.extend([(i, j, cls, True) for i, j in cls_indices[n_train+n_val:]])
        
        # Split unknown samples into val AND test so AUROC can be computed on val
        # Standard protocol: unknowns are NOT seen during training,
        # but must appear in val for model selection and test for final evaluation.
        unknown_indices_list = list(unknown_indices)
        np.random.shuffle(unknown_indices_list)
        # Бүх unknown → test (val-д огт хуваарилахгүй)
        #unknown_test = [(i, j, unknown_class, False) for i, j, _ in unknown_indices_list]
        unknown_test = [(i, j, label, False) for i, j, label in unknown_indices_list]
        # val_unknown_samples>0 тохиолдолд val-д нэмж болно
        if val_unknown_samples > 0:
            n_unknown_val = min(val_unknown_samples, len(unknown_indices_list))
            #val_indices  = val_indices + [(i, j, unknown_class, False) for i, j, _ in unknown_indices_list[:n_unknown_val]]
            val_indices = val_indices + [(i, j, label, False) for i, j, label in unknown_indices_list[:n_unknown_val]]
            #unknown_test = [(i, j, unknown_class, False) for i, j, _ in unknown_indices_list[n_unknown_val:]]
            unknown_test = [(i, j, label, False) for i, j, label in unknown_indices_list[n_unknown_val:]]
            print(f"  Val unknown count: {n_unknown_val}")
        else:
            print(f"  Val after filtering: known={len([x for x in val_indices if x[3]])}, unknown=0")
        test_indices = test_known_indices + unknown_test
        
        # Create datasets using the PATCH-BASED MultiModalOpenSetDataset
        #self.labels,
        train_dataset = MultiModalOpenSetDataset( 
            self.hsi_data, self.lidar_data, self.gt,
            known_classes=known_classes,
            patch_size=self.patch_size,
            split='train',
            augment=True,
            normalize=True
        )
        train_dataset.samples = [(i, j, is_known) for i, j, _, is_known in train_indices]
        
        #self.labels,
        val_dataset = MultiModalOpenSetDataset(
            self.hsi_data, self.lidar_data, self.gt,
            known_classes=known_classes,
            patch_size=self.patch_size,
            split='val',
            augment=False,
            normalize=True
        )
        val_dataset.samples = [(i, j, is_known) for i, j, _, is_known in val_indices]
        
        #self.labels,
        test_dataset = MultiModalOpenSetDataset(
            self.hsi_data, self.lidar_data, self.gt,
            known_classes=known_classes,
            patch_size=self.patch_size,
            split='test',
            augment=False,
            normalize=True
        )
        test_dataset.samples = [(i, j, is_known) for i, j, _, is_known in test_indices]
        
        # ── Print correct sample counts (samples assigned above) ─────────────
        for _name, _ds in [('TRAIN', train_dataset), ('VAL', val_dataset), ('TEST', test_dataset)]:
            _nk = sum(1 for _, _, ik in _ds.samples if ik)
            _nu = sum(1 for _, _, ik in _ds.samples if not ik)
            print(f"  {_name}: {len(_ds.samples)} samples  (known={_nk}, unknown={_nu})")

        # ── Create dataloaders (guard against empty train) ───────────────────
        if len(train_dataset) > 0:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=True
            )
        else:
            # train_ratio=0.0 case — return val_loader as placeholder
            # (caller should discard it; only val_loader and test_loader matter)
            print("  INFO: train split is empty (train_ratio=0.0) — skipping train DataLoader")
            train_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True
            )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        return train_loader, val_loader, test_loader

class MUUFLDataset:
    """
    MUUFL Gulfport Dataset Loader
    
    Dataset structure:
    - HSI: 64 channels (after atmospheric correction)
    - LiDAR: 2 channels (elevation + intensity) or 1 channel
    - Classes: 11 (0=background, 1-11=classes)
    """
    
    def __init__(self, data_root, patch_size=15):
        self.patch_size = patch_size
        self.data_root = data_root
        
        # Load HSI
        hsi_path = os.path.join(data_root, 'muufl_hsi.mat')
        if os.path.exists(hsi_path):
            print(f"  Found HSI: muufl_hsi.mat")
            
            hsi_data = scio.loadmat(hsi_path)
            hsi = hsi_data['hsi']
            print(f"    Loaded HSI with key: 'hsi'")
        else:
            raise FileNotFoundError(f"HSI file not found: {hsi_path}")
        
        # Load LiDAR
        lidar_path = os.path.join(data_root, 'muufl_lidar.mat')
        if os.path.exists(lidar_path):
            print(f"  Found LiDAR: muufl_lidar.mat")
            lidar_data = scio.loadmat(lidar_path)
            lidar = lidar_data['lidar']
            print(f"    Loaded elevation with key: 'lidar'")
            lidar = lidar[:, :, 0:1]  # Use only elevation (first channel)
            print(f"    Using elevation only (channel 0)")
        else:
            raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")
        
        # Load Ground Truth
        gt_path = os.path.join(data_root, 'muufl_gt.mat')
        if os.path.exists(gt_path):
            print(f"  Found GT: muufl_gt.mat")
            gt_data = scio.loadmat(gt_path)
            
            # ✅ FIX: Extract labels from the loaded data
            labels = gt_data['gt']  # This line was missing or wrong!
            print(f"    Loaded GT with key: 'gt'")
        else:
            raise FileNotFoundError(f"GT file not found: {gt_path}")
        
        # Store data
        self.hsi_data = hsi
        self.lidar_data = lidar
        self.labels = labels  
        self.gt = self.labels 
        
        print(f"  HSI shape: {self.hsi_data.shape}")
        print(f"  LiDAR shape: {self.lidar_data.shape}")
        print(f"  Labels shape: {self.labels.shape}")
        print(f"  Unique classes: {np.unique(self.labels)}")  
        
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load MUUFL data with flexible file naming"""
        
        # ===== LOAD HSI =====
        # Try different possible HSI file names
        hsi_paths = [
            self.data_root / 'muufl_hsi.mat',
            self.data_root / 'MUUFL_HSI.mat',
            self.data_root / 'hsi.mat',
            self.data_root / 'HSI.mat',
            self.data_root / 'muufl_gulfport_hsi.mat'
        ]
        
        hsi_data = None
        for hsi_path in hsi_paths:
            if hsi_path.exists():
                print(f"  Found HSI: {hsi_path.name}")
                try:
                    hsi_mat = sio.loadmat(hsi_path)
                    # Try different possible key names
                    for key in ['hsi', 'HSI', 'data', 'muufl_hsi', 'hyperspectral']:
                        if key in hsi_mat and not key.startswith('__'):
                            hsi_data = hsi_mat[key].astype(np.float32)
                            print(f"    Loaded HSI with key: '{key}'")
                            break
                    if hsi_data is not None:
                        break
                except Exception as e:
                    print(f"    Failed to load {hsi_path.name}: {e}")
                    continue
        
        if hsi_data is None:
            raise FileNotFoundError(f"Could not find HSI file in {self.data_root}. Tried: {[p.name for p in hsi_paths]}")
        
        # ===== LOAD LiDAR =====
        # Try different possible LiDAR file names
        lidar_paths = [
            self.data_root / 'muufl_lidar.mat',
            self.data_root / 'MUUFL_LiDAR.mat',
            self.data_root / 'lidar.mat',
            self.data_root / 'LiDAR.mat',
            self.data_root / 'muufl_gulfport_lidar.mat',
            self.data_root / 'dsm.mat',
            self.data_root / 'DSM.mat'
        ]
        
        lidar_data = None
        for lidar_path in lidar_paths:
            if lidar_path.exists():
                print(f"  Found LiDAR: {lidar_path.name}")
                try:
                    lidar_mat = sio.loadmat(lidar_path)
                    for key in ['lidar', 'LiDAR', 'elevation', 'dem']:
                        if key in lidar_mat and not key.startswith('__'):
                            lidar_data = lidar_mat[key].astype(np.float32)
                            print(f"    Loaded elevation with key: '{key}'")
                            
                            # FIX: If 2 channels, use only first channel (elevation)
                            if lidar_data.ndim == 3 and lidar_data.shape[2] == 2:
                                print(f"    Using elevation only (channel 0)")
                                lidar_data = lidar_data[:, :, 0:1]  # Keep first channel only
                            
                            break
                    if lidar_data is not None:
                        break
                except Exception as e:
                    print(f"    Failed to load: {e}")
                    continue
        
        # Ensure 3D
        if lidar_data is not None and lidar_data.ndim == 2:
            lidar_data = lidar_data[:, :, np.newaxis]
        
        if lidar_data is None:
            raise FileNotFoundError(f"Could not find LiDAR file in {self.data_root}. Tried: {[p.name for p in lidar_paths]}")
        
        # ===== LOAD LABELS =====
        # Try different possible label file names
        label_paths = [
            self.data_root / 'muufl_gt.mat',
            self.data_root / 'MUUFL_gt.mat',
            self.data_root / 'gt.mat',
            self.data_root / 'GT.mat',
            self.data_root / 'labels.mat',
            self.data_root / 'muufl_gulfport_gt.mat'
        ]
        
        labels = None
        for label_path in label_paths:
            if label_path.exists():
                print(f"  Found GT: {label_path.name}")
                try:
                    label_mat = sio.loadmat(label_path)
                    # Try different possible key names
                    for key in ['gt', 'GT', 'labels', 'muufl_gt', 'MUUFL_gt', 'ground_truth']:
                        if key in label_mat and not key.startswith('__'):
                            labels = label_mat[key].astype(np.int64)
                            print(f"    Loaded GT with key: '{key}'")
                            break
                    if labels is not None:
                        break
                except Exception as e:
                    print(f"    Failed to load {label_path.name}: {e}")
                    continue
        
        if labels is None:
            raise FileNotFoundError(f"Could not find GT file in {self.data_root}. Tried: {[p.name for p in label_paths]}")
        
        # ===== VERIFY DIMENSIONS =====
        print(f"  HSI shape: {hsi_data.shape}")
        print(f"  LiDAR shape: {lidar_data.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        # Check dimensions match
        if hsi_data.shape[:2] != lidar_data.shape[:2] or hsi_data.shape[:2] != labels.shape:
            print(f"  ⚠️  Warning: Dimension mismatch detected!")
            print(f"    HSI: {hsi_data.shape[:2]}")
            print(f"    LiDAR: {lidar_data.shape[:2]}")
            print(f"    Labels: {labels.shape}")
            
            # Try to resize to match
            from scipy.ndimage import zoom
            target_shape = labels.shape
            
            if hsi_data.shape[:2] != target_shape:
                print(f"    Resizing HSI to match labels...")
                zoom_factors = (target_shape[0] / hsi_data.shape[0], 
                               target_shape[1] / hsi_data.shape[1], 1)
                hsi_data = zoom(hsi_data, zoom_factors, order=1)
            
            if lidar_data.shape[:2] != target_shape:
                print(f"    Resizing LiDAR to match labels...")
                zoom_factors = (target_shape[0] / lidar_data.shape[0], 
                               target_shape[1] / lidar_data.shape[1], 1)
                lidar_data = zoom(lidar_data, zoom_factors, order=1)
            
            print(f"    After resize:")
            print(f"      HSI: {hsi_data.shape}")
            print(f"      LiDAR: {lidar_data.shape}")
        
        print(f"  Unique classes: {np.unique(labels)}")
        
        return hsi_data, lidar_data, labels
    
    def create_dataloaders(self, unknown_class, batch_size, num_workers=0, 
                      train_ratio=0.2, val_ratio=0.1, samples_per_class=None, val_unknown_samples=0):
        """
        Create dataloaders for open-set learning
        
        Args:
            unknown_class: Which class to treat as unknown
            batch_size: Batch size
            num_workers: Number of workers
            train_ratio: Training ratio (ignored if samples_per_class is set)
            val_ratio: Validation ratio
            samples_per_class: FIXED number of samples per class (e.g., 50) - NEW!
        """
        # Determine known classes (all except unknown_class and background)
        all_classes = np.unique(self.labels)
        all_classes = all_classes[all_classes > 0]  # Remove background
        if isinstance(unknown_class, (list, tuple, set)):
            unknown_set = set(unknown_class)
        else:
            unknown_set = {unknown_class}

        known_classes = [c for c in all_classes if c not in unknown_set]
        
        #print(f"\n  Creating datasets (unknown class: {unknown_class})")
        print(f"Creating datasets (unknown classes: {sorted(list(unknown_set))})")
        print(f"  Known classes: {known_classes}")
        
        # Get all valid pixel indices
        #h, w = self.labels.shape
        h, w = self.gt.shape
        margin = self.patch_size // 2
        all_indices = []
        
        for i in range(margin, h - margin):
            for j in range(margin, w - margin):
                #label = self.labels[i, j]
                label = self.gt[i, j]
                if label == 0:  # Skip background
                    continue
                all_indices.append((i, j, label))
        
        # Separate known and unknown indices
        known_indices = [(i, j, label) for i, j, label in all_indices if label in known_classes]
        unknown_indices = [(i, j, label) for i, j, label in all_indices if label in unknown_set ] #label == unknown_class

        # ========== FIXED SAMPLE COUNT PROTOCOL ==========
        # Train and val each get their own independent fixed budget.
        # This prevents val size collapsing to n_train * val_ratio (e.g. 50*0.15=7/class).
        # Standard in HyLiOSR and similar work: 50 train + 50 val + remaining → test.
        VAL_PER_CLASS = samples_per_class if samples_per_class is not None else max(1, int(50 * val_ratio))

        train_indices = []
        val_indices = []
        test_known_indices = []

        if samples_per_class is not None:
            print(f"  Using FIXED {samples_per_class} train + {VAL_PER_CLASS} val samples per class")
        else:
            print(f"  Using {train_ratio*100:.0f}% training data per class")

        for cls in known_classes:
            cls_indices = [(i, j) for i, j, label in known_indices if label == cls]
            np.random.shuffle(cls_indices)

            if samples_per_class is not None:
                n_train = min(samples_per_class, len(cls_indices))
                n_val   = min(VAL_PER_CLASS, len(cls_indices) - n_train)
            else:
                n_total = len(cls_indices)
                n_train = int(n_total * train_ratio)
                n_val   = int(n_total * val_ratio)

            train_indices.extend([(i, j, cls, True) for i, j in cls_indices[:n_train]])
            val_indices.extend([(i, j, cls, True)   for i, j in cls_indices[n_train:n_train+n_val]])
            test_known_indices.extend([(i, j, cls, True) for i, j in cls_indices[n_train+n_val:]])
        
        # Split unknown samples into val AND test so AUROC can be computed on val
        # Standard protocol: unknowns are NOT seen during training,
        # but must appear in val for model selection and test for final evaluation.
        unknown_indices_list = list(unknown_indices)
        np.random.shuffle(unknown_indices_list)
        # Бүх unknown → test (val-д огт хуваарилахгүй)
        #unknown_test = [(i, j, unknown_class, False) for i, j, _ in unknown_indices_list]
        unknown_test = [(i, j, label, False) for i, j, label in unknown_indices_list]
        # val_unknown_samples>0 тохиолдолд val-д нэмж болно
        if val_unknown_samples > 0:
            n_unknown_val = min(val_unknown_samples, len(unknown_indices_list))
            #val_indices  = val_indices + [(i, j, unknown_class, False) for i, j, _ in unknown_indices_list[:n_unknown_val]]
            #unknown_test = [(i, j, unknown_class, False) for i, j, _ in unknown_indices_list[n_unknown_val:]]
            val_indices.extend([(i, j, label, False) for i, j, label in unknown_indices_list[:n_unknown_val]])
            unknown_test = [(i, j, label, False) for i, j, label in unknown_indices_list[n_unknown_val:]]
            print(f"  Val unknown count: {n_unknown_val}")
        else:
            print(f"  Val after filtering: known={len([x for x in val_indices if x[3]])}, unknown=0")
        test_indices = test_known_indices + unknown_test
        
        # Create datasets using the PATCH-BASED MultiModalOpenSetDataset
        #self.labels,
        train_dataset = MultiModalOpenSetDataset( 
            self.hsi_data, self.lidar_data, self.gt,
            known_classes=known_classes,
            patch_size=self.patch_size,
            split='train',
            augment=True,
            normalize=True
        )
        train_dataset.samples = [(i, j, is_known) for i, j, _, is_known in train_indices]
        
        #self.labels,
        val_dataset = MultiModalOpenSetDataset(
            self.hsi_data, self.lidar_data, self.gt,
            known_classes=known_classes,
            patch_size=self.patch_size,
            split='val',
            augment=False,
            normalize=True
        )
        val_dataset.samples = [(i, j, is_known) for i, j, _, is_known in val_indices]
        
        #self.labels,
        test_dataset = MultiModalOpenSetDataset(
            self.hsi_data, self.lidar_data, self.gt,
            known_classes=known_classes,
            patch_size=self.patch_size,
            split='test',
            augment=False,
            normalize=True
        )
        test_dataset.samples = [(i, j, is_known) for i, j, _, is_known in test_indices]
        
        # ── Print correct sample counts (samples assigned above) ─────────────
        for _name, _ds in [('TRAIN', train_dataset), ('VAL', val_dataset), ('TEST', test_dataset)]:
            _nk = sum(1 for _, _, ik in _ds.samples if ik)
            _nu = sum(1 for _, _, ik in _ds.samples if not ik)
            print(f"  {_name}: {len(_ds.samples)} samples  (known={_nk}, unknown={_nu})")

        # ── Create dataloaders (guard against empty train) ───────────────────
        if len(train_dataset) > 0:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=True
            )
        else:
            # train_ratio=0.0 case — return val_loader as placeholder
            # (caller should discard it; only val_loader and test_loader matter)
            print("  INFO: train split is empty (train_ratio=0.0) — skipping train DataLoader")
            train_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True
            )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        return train_loader, val_loader, test_loader


class AugsburgDataset:
    """
    Augsburg Dataset Loader
    
    Dataset structure:
    - HSI: 180 channels
    - LiDAR: 1 channel (DSM)
    - Classes: 7 (0=background, 1-7=classes)
    """
    
    def __init__(self, data_root: str, patch_size: int = 15):
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        
        print("Loading Augsburg dataset...")
        self.hsi_data, self.lidar_data, self.labels = self._load_data()
        self.num_classes = int(np.max(self.labels))

        # 1. Ensure we have a 2D labels array
        # (Try different attribute names your code might use)
        if hasattr(self, 'gt'):
            labels_2d = self.gt
        elif hasattr(self, 'labels') and isinstance(self.labels, np.ndarray):
            if self.labels.ndim == 2:
                labels_2d = self.labels
            else:
                # Already flattened, need to reshape
                h, w = self.hsi.shape[:2]
                labels_2d = self.labels.reshape(h, w)
        elif hasattr(self, 'y'):
            labels_2d = self.y if self.y.ndim == 2 else self.y.reshape(self.hsi.shape[:2])
        else:
            raise AttributeError("Cannot find labels in dataset!")
        
        # 2. Generate spatial coordinates
        h, w = labels_2d.shape
        rows, cols = np.meshgrid(range(h), range(w), indexing='ij')
        coordinates = np.stack([rows.flatten(), cols.flatten()], axis=1)
        
        # 3. Flatten all data
        labels_flat = labels_2d.flatten()
        hsi_flat = self.hsi_data.reshape(-1, self.hsi_data.shape[-1])  # ✅ Changed hsi to hsi_data
        lidar_flat = self.lidar_data.reshape(-1, self.lidar_data.shape[-1])  # ✅ Changed lidar to lidar_data
        
        # 4. Remove background (label == 0)
        valid_mask = labels_flat > 0
        
        # 5. Store as attributes (this is what create_dataloaders needs)
        self.coordinates = coordinates[valid_mask]
        self.labels = labels_flat[valid_mask]
        self.hsi_flat = hsi_flat[valid_mask]
        self.lidar_flat = lidar_flat[valid_mask]
        
        # Also keep the 2D version for reference
        self.gt = labels_2d
        
        print(f"  Processed dataset:")
        print(f"    Valid pixels: {len(self.labels)}")
        print(f"    Coordinates: {self.coordinates.shape}")
        print(f"    HSI flat: {self.hsi_flat.shape}")
        print(f"    LiDAR flat: {self.lidar_flat.shape}")
        print(f"    Unique classes: {np.unique(self.labels)}")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load Augsburg data"""
        # Augsburg typically uses HDF5 format
        hsi_path = self.data_root / 'augsburg_hsi.mat'
        
        if hsi_path.suffix == '.h5' or hsi_path.with_suffix('.h5').exists():
            hsi_path = hsi_path.with_suffix('.h5')
            with h5py.File(hsi_path, 'r') as f:
                hsi_data = f['augsburg_hsi'][:].astype(np.float32)
        else:
            hsi_mat = sio.loadmat(hsi_path)
            hsi_data = hsi_mat['augsburg_hsi'].astype(np.float32)
        
        # Load LiDAR
        lidar_path = self.data_root / 'augsburg_lidar.mat'
        if lidar_path.suffix == '.h5' or lidar_path.with_suffix('.h5').exists():
            lidar_path = lidar_path.with_suffix('.h5')
            with h5py.File(lidar_path, 'r') as f:
                lidar_data = f['data_DSM'][:].astype(np.float32)
        else:
            lidar_mat = sio.loadmat(lidar_path)
            lidar_data = lidar_mat['data_DSM'].astype(np.float32)
        
        if lidar_data.ndim == 2:
            lidar_data = lidar_data[:, :, np.newaxis]
        
        # Load labels
        label_path = self.data_root / 'augsburg_gt.mat'
        if label_path.suffix == '.h5' or label_path.with_suffix('.h5').exists():
            label_path = label_path.with_suffix('.h5')
            with h5py.File(label_path, 'r') as f:
                labels = f['augsburg_gt'][:].astype(np.int64)
        else:
            label_mat = sio.loadmat(label_path)
            labels = label_mat['augsburg_gt'].astype(np.int64)
        
        print(f"  HSI shape: {hsi_data.shape}")
        print(f"  LiDAR shape: {lidar_data.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Classes: {np.unique(labels)}")
        
        return hsi_data, lidar_data, labels
    
    def create_dataloaders(self, unknown_class, batch_size=32, train_ratio=0.1, val_ratio=0.1,
                           num_workers=4, random_seed=42, samples_per_class=None, val_unknown_samples=0):
        np.random.seed(random_seed)
        
        all_classes = np.unique(self.gt)
        all_classes = all_classes[all_classes > 0]
        if isinstance(unknown_class, (list, tuple, set)):
            unknown_set = set(unknown_class)
        else:
            unknown_set = {unknown_class}

        known_classes = [c for c in all_classes if c not in unknown_set]
        
        #print(f"\n  Creating datasets (unknown class: {unknown_class})")
        print(f"Creating datasets (unknown classes: {sorted(list(unknown_set))})")
        
        # Collect samples per class — use self.gt (2D) consistently
        h, w = self.gt.shape
        margin = self.patch_size // 2
        all_indices = []
        
        for i in range(margin, h - margin):
            for j in range(margin, w - margin):
                #label = self.labels[i, j]
                label = self.gt[i, j]
                if label == 0:  # Skip background
                    continue
                all_indices.append((i, j, label))

         # Separate known and unknown indices
        known_indices = [(i, j, label) for i, j, label in all_indices if label in known_classes]
        #unknown_indices = [(i, j, label) for i, j, label in all_indices if label == unknown_class]
        unknown_indices = [(i, j, label) for i, j, label in all_indices if label in unknown_set]
        
        train_indices = []
        val_indices = []
        test_known_indices = []

        VAL_PER_CLASS = samples_per_class if samples_per_class is not None else max(1, int(50 * val_ratio))

        train_indices = []
        val_indices = []
        test_known_indices = []

        if samples_per_class is not None:
            print(f"  Using FIXED {samples_per_class} train + {VAL_PER_CLASS} val samples per class")
        else:
            print(f"  Using {train_ratio*100:.0f}% training data per class")

        for cls in known_classes:
            cls_indices = [(i, j) for i, j, label in known_indices if label == cls]
            np.random.shuffle(cls_indices)

            if samples_per_class is not None:
                n_train = min(samples_per_class, len(cls_indices))
                n_val   = min(VAL_PER_CLASS, len(cls_indices) - n_train)
            else:
                n_total = len(cls_indices)
                n_train = int(n_total * train_ratio)
                n_val   = int(n_total * val_ratio)

            train_indices.extend([(i, j, cls, True) for i, j in cls_indices[:n_train]])
            val_indices.extend([(i, j, cls, True)   for i, j in cls_indices[n_train:n_train+n_val]])
            test_known_indices.extend([(i, j, cls, True) for i, j in cls_indices[n_train+n_val:]])
        
        # Split unknown samples into val AND test so AUROC can be computed on val
        # Standard protocol: unknowns are NOT seen during training,
        # but must appear in val for model selection and test for final evaluation.
        unknown_indices_list = list(unknown_indices)
        np.random.shuffle(unknown_indices_list)
        # Бүх unknown → test (val-д огт хуваарилахгүй)
        #unknown_test = [(i, j, unknown_class, False) for i, j, _ in unknown_indices_list]
        unknown_test = [(i, j, label, False) for i, j, label in unknown_indices_list]
        # val_unknown_samples>0 тохиолдолд val-д нэмж болно
        if val_unknown_samples > 0:
            n_unknown_val = min(val_unknown_samples, len(unknown_indices_list))
            #val_indices  = val_indices + [(i, j, unknown_class, False) for i, j, _ in unknown_indices_list[:n_unknown_val]]
            val_indices = val_indices + [(i, j, label, False) for i, j, label in unknown_indices_list[:n_unknown_val]]
            #unknown_test = [(i, j, unknown_class, False) for i, j, _ in unknown_indices_list[n_unknown_val:]]
            unknown_test = [(i, j, label, False) for i, j, label in unknown_indices_list[n_unknown_val:]]
            print(f"  Val unknown count: {n_unknown_val}")
        else:
            print(f"  Val after filtering: known={len([x for x in val_indices if x[3]])}, unknown=0")
        test_indices = test_known_indices + unknown_test

         # Create datasets using the PATCH-BASED MultiModalOpenSetDataset
        #self.labels,
        train_dataset = MultiModalOpenSetDataset( 
            self.hsi_data, self.lidar_data, self.gt,
            known_classes=known_classes,
            patch_size=self.patch_size,
            split='train',
            augment=True,
            normalize=True
        )
        train_dataset.samples = [(i, j, is_known) for i, j, _, is_known in train_indices]
        
        #self.labels,
        val_dataset = MultiModalOpenSetDataset(
            self.hsi_data, self.lidar_data, self.gt,
            known_classes=known_classes,
            patch_size=self.patch_size,
            split='val',
            augment=False,
            normalize=True
        )
        val_dataset.samples = [(i, j, is_known) for i, j, _, is_known in val_indices]
        
        #self.labels,
        test_dataset = MultiModalOpenSetDataset(
            self.hsi_data, self.lidar_data, self.gt,
            known_classes=known_classes,
            patch_size=self.patch_size,
            split='test',
            augment=False,
            normalize=True
        )
        test_dataset.samples = [(i, j, is_known) for i, j, _, is_known in test_indices]
        
        # ── Print correct sample counts (samples assigned above) ─────────────
        for _name, _ds in [('TRAIN', train_dataset), ('VAL', val_dataset), ('TEST', test_dataset)]:
            _nk = sum(1 for _, _, ik in _ds.samples if ik)
            _nu = sum(1 for _, _, ik in _ds.samples if not ik)
            print(f"  {_name}: {len(_ds.samples)} samples  (known={_nk}, unknown={_nu})")

        # ── Create dataloaders (guard against empty train) ───────────────────
        if len(train_dataset) > 0:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=True
            )
        else:
            # train_ratio=0.0 case — return val_loader as placeholder
            # (caller should discard it; only val_loader and test_loader matter)
            print("  INFO: train split is empty (train_ratio=0.0) — skipping train DataLoader")
            train_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True
            )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        return train_loader, val_loader, test_loader

def _load_builtin_dataset(dataset_name: str, data_root: str, **kwargs):
    """
    Factory function to load dataset by name
    
    Args:
        dataset_name: 'houston2013', 'muufl', or 'augsburg'
        data_root: Path to dataset directory
        **kwargs: Additional arguments for dataset loader
    
    Returns:
        Dataset object
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'houston2013':
        return Houston2013Dataset(data_root, **kwargs)
    elif dataset_name == 'muufl':
        return MUUFLDataset(data_root, **kwargs)
    elif dataset_name == 'augsburg':
        return AugsburgDataset(data_root, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _get_hw_from_dataset(ds):
    # Try common attribute names
    if hasattr(ds, "hsi_data"):
        h, w = ds.hsi_data.shape[:2]
        return int(h), int(w)
    if hasattr(ds, "hsi"):
        h, w = ds.hsi.shape[:2]
        return int(h), int(w)
    if hasattr(ds, "gt"):
        h, w = ds.gt.shape[:2]
        return int(h), int(w)
    if hasattr(ds, "labels") and isinstance(ds.labels, np.ndarray) and ds.labels.ndim == 2:
        h, w = ds.labels.shape[:2]
        return int(h), int(w)
    raise AttributeError("Cannot infer (H,W) from dataset object (missing hsi_data/hsi/gt/labels 2D).")


def _get_gt_2d(ds):
    """
    Return a 2D ground-truth label map if possible.
    """
    if hasattr(ds, "gt") and isinstance(ds.gt, np.ndarray) and ds.gt.ndim == 2:
        return ds.gt
    if hasattr(ds, "labels") and isinstance(ds.labels, np.ndarray) and ds.labels.ndim == 2:
        return ds.labels
    # If labels are flattened, we can't safely reshape without knowing original H,W.
    return None


class DatasetWrapper:
    """
    Wrapper that injects UNKNOWN samples into TRAIN for confidence calibration.
    Crucially:
      - Only adds unknown samples that are patch-valid (not near borders).
      - Injects into train_dataset.samples if available (your patch-based dataset uses this).
    """

    def __init__(self, dataset, patch_size=15, seed=42):
        self.dataset = dataset
        self.patch_size = int(patch_size)
        self.seed = int(seed)

    def create_dataloaders(
        self,
        unknown_class,
        batch_size,
        train_ratio=None,
        val_ratio=0.15,
        samples_per_class=None,
        unknown_train_samples=0,
        num_workers=4,
        val_unknown_samples=0,
    ):
        import inspect as _inspect
        _sig = _inspect.signature(self.dataset.create_dataloaders)
        _call_kwargs = dict(
            unknown_class=unknown_class,
            batch_size=batch_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            samples_per_class=samples_per_class,
            num_workers=num_workers,
            val_unknown_samples=val_unknown_samples,
        )
        # signature-д байхгүй параметрүүдийг хасна
        _call_kwargs = {k: v for k, v in _call_kwargs.items() if k in _sig.parameters}
        train_loader, val_loader, test_loader = self.dataset.create_dataloaders(**_call_kwargs)

        unknown_train_samples = int(unknown_train_samples or 0)
        if unknown_train_samples <= 0:
            return train_loader, val_loader, test_loader

        train_dataset = train_loader.dataset

        # We will append to train_dataset.samples (recommended for your dataset.py).
        if not hasattr(train_dataset, "samples"):
            print(
                "WARNING: train_dataset has no attribute 'samples'. "
                "Unknown injection skipped (cannot do safely)."
            )
            return train_loader, val_loader, test_loader

        gt2d = _get_gt_2d(self.dataset)
        if gt2d is None:
            print(
                "WARNING: Could not find a 2D gt/labels map on the base dataset. "
                "Unknown injection skipped (cannot filter border-safe patches)."
            )
            return train_loader, val_loader, test_loader

        h, w = _get_hw_from_dataset(self.dataset)
        margin = self.patch_size // 2

        # Candidate unknown pixels (border-safe)
        #rr, cc = np.where(gt2d == unknown_class)
        if isinstance(unknown_class, (list, tuple, set)):
            unknown_list = list(unknown_class)
        else:
            unknown_list = [unknown_class]

        rr, cc = np.where(np.isin(gt2d, unknown_list))
        if rr.size == 0:
            print("WARNING: No unknown pixels found for unknown classes; injection skipped.")
            return train_loader, val_loader, test_loader

        valid = (rr >= margin) & (rr < (h - margin)) & (cc >= margin) & (cc < (w - margin))
        rr = rr[valid]
        cc = cc[valid]

        if rr.size == 0:
            print(
                "WARNING: Unknown pixels exist but none are border-safe for the given patch_size. "
                "Injection skipped."
            )
            return train_loader, val_loader, test_loader

        # Avoid duplicates: do not re-add coordinates already present in train samples
        existing = set((int(r), int(c)) for (r, c, _) in train_dataset.samples)

        cand = [(int(r), int(c)) for r, c in zip(rr.tolist(), cc.tolist()) if (int(r), int(c)) not in existing]
        if len(cand) == 0:
            print("INFO: All border-safe unknown coords already present; injection skipped.")
            return train_loader, val_loader, test_loader

        rng = np.random.default_rng(self.seed)
        k = min(unknown_train_samples, len(cand))
        chosen_idx = rng.choice(len(cand), size=k, replace=False)
        chosen = [cand[i] for i in chosen_idx]

        # Append as (row, col, is_known=False)
        train_dataset.samples.extend([(r, c, False) for (r, c) in chosen])

        print(f"Added {len(chosen)} unknown samples to TRAIN (confidence calibration only)")
        return train_loader, val_loader, test_loader


def load_dataset(dataset_name: str, data_root: str, patch_size: int = 15, seed: int = 42):
    """Load a built-in dataset and wrap it with optional unknown-sample injection.

    This function is standalone and does not depend on any external dataset.py file.
    """
    base = _load_builtin_dataset(dataset_name, data_root, patch_size=patch_size)
    return DatasetWrapper(base, patch_size=patch_size, seed=seed)