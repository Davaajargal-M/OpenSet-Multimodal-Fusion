import argparse
import json
import math
import random
from pathlib import Path
from collections import Counter
import inspect

import numpy as np
import pandas as pd
from tqdm import tqdm
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import (roc_auc_score, roc_curve,
                              confusion_matrix, cohen_kappa_score)

import warnings
warnings.filterwarnings("ignore")

from dataset_fixed_multi import load_dataset
from openset_multimodal_fusion_change_signal_conbination import (
    NovelOpenSetMultiModalNet,
    OpenSetContrastiveLoss,
)

# ─────────────────────────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, args, seed=0):
        self.args   = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        self.model     = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "OA": [], "AA": [], "Kappa": [],
            "AUROC_ent": [], "AUROC_gate": [], "AUROC_msp": [],
            "AUROC_conv_var": [], "AUROC_conv_slope": [], "gate_u_mean": [],
        }

        self.best_val_acc       = 0.0
        self.best_monitor_auroc = 0.0
        self.best_monitor_score = -1e9
        self.best_epoch         = -1
        self.patience_counter   = 0
        self.protocol_info = {
            "dataset": args.dataset,
            "unknown_class": sorted(args.unknown_class),
            "num_known_classes": args.num_classes,
            "samples_per_class": args.samples_per_class,
            "unknown_train_samples": args.unknown_train_samples,
            "val_unknown_samples": getattr(args, "val_unknown_samples", 0),
            "patch_size": args.patch_size,
            "epochs": args.epochs,
            "seed": seed,
        }
    # ── Setup ──────────────────────────────────────────────────────

    def setup(self, train_loader):
        sample_hsi, sample_lidar, _, _ = next(iter(train_loader))

        self.model = NovelOpenSetMultiModalNet(
            hsi_channels   = sample_hsi.shape[1],
            lidar_channels = sample_lidar.shape[1],
            num_classes    = self.args.num_classes,
            feature_dim    = self.args.feature_dim,
            use_attention  = self.args.use_attention,
            num_iterations = self.args.num_iterations,
            dropout        = self.args.dropout,
            use_gate       = self.args.use_gate,
            use_feedback   = self.args.use_feedback,
            use_mch        = self.args.use_mch,
        ).to(self.device)

        # Class weights
        train_labels = []
        for _, _, labels, is_known in train_loader:
            mask = is_known.bool() if hasattr(is_known, 'bool') else is_known
            train_labels.extend(labels[mask].numpy().tolist())
        class_counts = Counter(train_labels)
        total = len(train_labels)
        class_weights = torch.FloatTensor([
            total / (self.args.num_classes * class_counts.get(c, 1))
            for c in range(self.args.num_classes)
        ]).to(self.device)

        # OpenSetContrastiveLoss
        self.criterion = OpenSetContrastiveLoss(
            num_classes        = self.args.num_classes,
            temperature        = self.args.temperature,
            class_weights      = class_weights,
            label_smoothing    = 0.1,
            ce_weight          = 1.0,
            contrastive_weight = self.args.contrastive_weight,
            separation_weight  = self.args.separation_weight,
            vicreg_weight      = self.args.vicreg_weight,
        )

        # Optimizer: check openset params
        openset_params = []
        if getattr(self.model, 'openset_head', None) is not None:
            openset_params += list(self.model.openset_head.parameters())
        #if hasattr(self.model, 'entropy_temperature'):
        #    openset_params += [self.model.entropy_temperature]
        if hasattr(self.model, 'log_logit_scale'):
            openset_params += [self.model.log_logit_scale]

        elif hasattr(self.model, 'logit_scale') and isinstance(
                self.model.logit_scale, torch.nn.Parameter):
            openset_params += [self.model.logit_scale]

        openset_ids     = {id(p) for p in openset_params}
        backbone_params = [p for p in self.model.parameters()
                           if id(p) not in openset_ids]

        self.optimizer = optim.AdamW([
            {"params": backbone_params,
             "lr": self.args.learning_rate,
             "weight_decay": self.args.weight_decay},
            {"params": openset_params,
             "lr": self.args.learning_rate * 0.3,
             "weight_decay": self.args.weight_decay},
            {"params": list(self.criterion.parameters()),
             "lr": self.args.learning_rate * 0.3,
             "weight_decay": 0.0},
        ])

        def lr_lambda(epoch):
            if epoch < 10:
                return 0.1 + 0.9 * (epoch / 10)
            progress = (epoch - 10) / max(1, self.args.epochs - 10)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        print(f"  Class weights : {class_weights.cpu().numpy().round(2)}")
        print(f"  Model params  : {sum(p.numel() for p in self.model.parameters()):,}")

    # ── Train epoch ────────────────────────────────────────────────

    def train_epoch(self, train_loader, epoch: int):
        self.model.train()
        total_loss, correct, total, valid_batches = 0.0, 0, 0, 0

        # Staged training (warmup + separation)
        warmup = getattr(self.args, 'warmup_epochs', 30)
        if epoch < warmup:
            # Stage 1: CE + contrastive only
            self.criterion.alpha_sep = 0.0
            self.criterion.stage     = 1
        else:
            # Stage 2: add separation + vicreg gradually
            progress = min(1.0, (epoch - warmup) / 20.0)
            self.criterion.alpha_sep = progress * self.args.alpha_sep
            self.criterion.stage     = 2

        pbar = tqdm(train_loader, desc=f'Train[ep{epoch+1}]', leave=False)
        for batch in pbar:
            hsi, lidar, labels, is_known = batch
            hsi      = hsi.to(self.device)
            lidar    = lidar.to(self.device)
            labels   = labels.to(self.device)
            is_known = is_known.to(self.device).bool()

            self.optimizer.zero_grad()

            logits, conf, feat, unc = self.model(
                hsi, lidar,
                return_features=True,
                return_uncertainty=True,
            )

            # Pass conf, unc to loss
            confidence_for_loss = conf if self.args.use_mch else None

            loss, loss_dict = self.criterion(
                logits       = logits,
                features     = feat,
                confidence   = confidence_for_loss,           # ← MonotonicConfidenceHead gradient
                labels       = labels,
                is_known     = is_known,
                gate_u_train = None, #unc['gate_uncertainty'], 
                fb_u_train   = unc['feedback_uncertainty'],
                ent_u_train  = unc['normalized_entropy'],
            )

            # Scale regularization: target=5.0
            scale_reg = self.model.scale_regularization_loss(
                target=5.0, weight=1e-3)
            loss = loss + scale_reg

            if valid_batches % 20 == 0:
                print(
                    f"[DBG-TRAIN] gate_mean={unc['gate_uncertainty'].mean().item():.4f}  "
                    f"fb_mean={unc['feedback_uncertainty'].mean().item():.4f}  "
                    f"ent_mean={unc['normalized_entropy'].mean().item():.4f}"
                )

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            all_params = (list(self.model.parameters())
                        + list(self.criterion.parameters()))
            grads = [p for p in all_params if p.grad is not None]
            torch.nn.utils.clip_grad_norm_(grads, 1.0)
            self.optimizer.step()

            total_loss   += float(loss.item())
            valid_batches += 1

            if is_known.sum() > 0:
                pred     = logits[is_known].argmax(dim=1)
                total   += int(is_known.sum().item())
                correct += int(pred.eq(labels[is_known]).sum().item())

            pbar.set_postfix({
                'loss':     f'{loss.item():.4f}',
                'acc':      f'{100*correct/total:.1f}%' if total > 0 else 'N/A',
                'ce':       f'{loss_dict["ce"]:.3f}',
                'sep':      f'{loss_dict["separation"]:.3f}',
                'alpha':    f'{self.criterion.alpha_sep:.2f}',
                'scale':    f'{self.model.logit_scale.item():.2f}',
            })

        return (total_loss / max(valid_batches, 1),
                100.0 * correct / total if total > 0 else 0.0)

    # ── Collect scores ─────────────────────────────────────────────

    @torch.no_grad()
    def collect_open_set_scores(self, loader):
        self.model.eval()
        rows = []
        for hsi, lidar, labels, is_known in tqdm(loader, desc='Collect', leave=False):
            hsi, lidar = hsi.to(self.device), lidar.to(self.device)
            labels     = labels.to(self.device)
            is_known   = is_known.to(self.device).bool()

            logits, conf, feat, unc = self.model(
                hsi, lidar, return_features=True, return_uncertainty=True)

            probs          = F.softmax(logits, dim=1)
            max_probs, preds = probs.max(dim=1)
            energy         = torch.logsumexp(logits, dim=1)

            gate_u = unc['gate_uncertainty']
            fb_u   = unc['feedback_uncertainty']
            ent_u  = unc['normalized_entropy']
            cv     = unc.get('conv_var',   None)
            cs     = unc.get('conv_slope', None)

            for i in range(hsi.size(0)):
                rows.append({
                    'label':      int(labels[i].item()),
                    'is_known':   int(is_known[i].item()),
                    'pred':       int(preds[i].item()),
                    'conf':       float(conf[i].item()),
                    'max_softmax':float(max_probs[i].item()),
                    'energy':     float(energy[i].item()),
                    'gate_u':     float(gate_u[i].item()),
                    'fb_u':       float(fb_u[i].item()),
                    'ent_u':      float(ent_u[i].item()),
                    'conv_var':   float(cv[i].item()) if cv is not None else 0.0,
                    'conv_slope': float(cs[i].item()) if cs is not None else 0.0,
                })
        df = pd.DataFrame(rows)
        return {k: df[k].to_numpy() for k in df.columns}

    # ── Validate ───────────────────────────────────────────────────

    @torch.no_grad()
    def validate_closed_set(self, val_loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        closed_labels, closed_preds = [], []

        for hsi, lidar, labels, is_known in tqdm(val_loader, desc='Val', leave=False):
            hsi, lidar = hsi.to(self.device), lidar.to(self.device)
            labels     = labels.to(self.device)
            is_known   = is_known.to(self.device).bool()

            logits, conf, feat, unc = self.model(
                hsi, lidar, return_features=True, return_uncertainty=True)

            # Val loss: alpha_sep=0 (CE+contrastive only)
            orig_alpha = self.criterion.alpha_sep
            if self.criterion.stage == 1:
                self.criterion.alpha_sep = 0.0
            else:
                self.criterion.alpha_sep = self.args.alpha_sep

            loss, _ = self.criterion(
                logits       = logits,
                features     = feat,
                confidence   = conf if self.args.use_mch else None,
                labels       = labels,
                is_known     = is_known,
                gate_u_train = unc['gate_uncertainty'],
                fb_u_train   = unc['feedback_uncertainty'],
                ent_u_train  = unc['normalized_entropy'],
            )
            self.criterion.alpha_sep = orig_alpha

            total_loss += float(loss.item())
            if is_known.sum() > 0:
                pred = logits[is_known].argmax(dim=1)
                total   += int(is_known.sum().item())
                correct += int(pred.eq(labels[is_known]).sum().item())
                closed_labels.extend(labels[is_known].cpu().numpy().tolist())
                closed_preds.extend(pred.cpu().numpy().tolist())

        val_loss = total_loss / len(val_loader)
        val_acc  = 100.0 * correct / total if total > 0 else 0.0

        OA = AA = Kappa = 0.0
        if closed_labels:
            y_true = np.array(closed_labels, dtype=np.int64)
            y_pred = np.array(closed_preds,  dtype=np.int64)
            cm     = confusion_matrix(y_true, y_pred,
                                      labels=list(range(self.args.num_classes)))
            OA     = 100.0 * np.trace(cm) / np.sum(cm)
            per_acc = [cm[i, i]/cm[i].sum() for i in range(self.args.num_classes)
                       if cm[i].sum() > 0]
            AA     = 100.0 * np.mean(per_acc) if per_acc else 0.0
            Kappa  = 100.0 * cohen_kappa_score(y_true, y_pred)

        # Monitor signal — no unknown in val, skip AUROC
        sp = self.collect_open_set_scores(val_loader)
        yk = sp['is_known'].astype(np.int32)
        n_unk = (yk == 0).sum()

        def safe_auc(y, sc):
            """Return 0.0 when no unknown (avoid nan)."""
            if (y == 0).sum() == 0 or (y == 1).sum() == 0:
                return 0.0
            try:
                return 100.0 * roc_auc_score(y, sc)
            except Exception:
                return 0.0

        au_ent  = safe_auc(yk, 1.0 - sp['ent_u'])
        au_gate = safe_auc(yk, sp['gate_u'])
        au_msp  = safe_auc(yk, sp['max_softmax'])
        au_fb   = safe_auc(yk, sp['fb_u'])
        au_cv   = safe_auc(yk, sp['conv_var'])
        au_cs   = safe_auc(yk, sp['conv_slope'])
        gate_mean = float(sp['gate_u'].mean())
        cv_mean = float(sp['conv_var'].mean())
        cs_mean = float(sp['conv_slope'].mean())


        return val_loss, val_acc, {
            'OA': OA, 'AA': AA, 'Kappa': Kappa,
            'AUROC_ent':  au_ent,
            'AUROC_gate': au_gate,
            'AUROC_msp':  au_msp,
            'AUROC_fb':   au_fb,
            'AUROC_conv_var':   au_cv,
            'AUROC_conv_slope': au_cs,
            'gate_u_mean':      gate_mean,
            'cv_mean':  float(sp['conv_var'].mean()),   
            'cs_mean':  float(sp['conv_slope'].mean()),
        }

    # ── Post-hoc calibrator ────────────────────────────────────────

    def fit_posthoc_calibrator(self, val_loader):
        sp_val = self.collect_open_set_scores(val_loader)

        # conv_var scale fix
        sp_val['conv_var'] = np.log(sp_val['conv_var'] + 1e-6)

        from scipy.stats import weibull_min

        sp_fit = sp_val
        known_mask_fit = sp_fit['is_known'] == 1

        print(f"  [Calibrator] EVT strict OSR: known={known_mask_fit.sum()}")

        # ─────────────────────────────────────────
        # FINAL SIGNALS
        # ─────────────────────────────────────────
        signals = ['gate_u', 'conv_var']
        feature_order = ['gate_u', 'conv_var']

        directions = {}
        wb_params = {}
        sig_stats = {}

        for sig in signals:
            k_vals = sp_fit[sig][known_mask_fit]

            if sig == 'gate_u':
                direction = -1   # high = unknown
            elif sig == 'conv_var':
                direction = +1   # high = known
            else:
                raise ValueError(f"Unknown signal: {sig}")

            sig_shift = float(k_vals.min())
            sig_scale = max(float(k_vals.max() - k_vals.min()), 0.5) + 1e-6
            sig_stats[sig] = (sig_shift, sig_scale)

            norm = np.clip((k_vals - sig_shift) / sig_scale, 0.0, 1.0)
            directions[sig] = direction
            corrected = norm if direction == +1 else 1.0 - norm

            try:
                c, loc, scale = weibull_min.fit(corrected, floc=0.0)
                if scale < 1e-4:
                    raise ValueError("Weibull scale too small")
            except Exception:
                c, loc, scale = 2.0, 0.0, 0.3

            wb_params[sig] = (float(c), float(loc), float(scale))
            print(f"    {sig}: mean={k_vals.mean():.4f} dir={direction}")

        # ─────────────────────────────────────────
        # FIXED WEIGHTS
        # ─────────────────────────────────────────
        weights = {
            'gate_u': 0.5,
            'conv_var': 0.5,
        }

        print(f"  Fixed weights: {weights}")

        # ─────────────────────────────────────────
        # EVT SCORE
        # ─────────────────────────────────────────
        def _evt_score(sp):
            n = len(sp[signals[0]])
            log_sum = np.zeros(n)

            for sig in signals:
                shift, scale = sig_stats[sig]
                vals = np.clip((sp[sig] - shift) / scale, 0.0, 1.0)

                corrected = vals if directions[sig] == +1 else 1.0 - vals
                c, loc, sc = wb_params[sig]
                p_k = weibull_min.cdf(corrected, c, loc, sc)

                log_sum += weights[sig] * np.log(np.clip(p_k, 1e-8, 1.0))

            return np.exp(log_sum / sum(weights.values()))

        known_scores = _evt_score({
            'gate_u': sp_fit['gate_u'][known_mask_fit],
            'conv_var': sp_fit['conv_var'][known_mask_fit],
        })

        best_tau = float(np.percentile(known_scores, 45))
        print(f"  tau={best_tau:.6f}")

        # ─────────────────────────────────────────
        # CALIBRATOR
        # ─────────────────────────────────────────
        class EVTCalibrator:
            def __init__(self):
                self.directions = directions
                self.wb_params = wb_params
                self.sig_stats = sig_stats
                self.weights = weights
                self.signals = signals
                self.feature_order = feature_order

            def predict_proba(self, X):
                # X: [gate_u, conv_var]
                raw_gate = X[:, 0]
                raw_cv   = X[:, 1]

                log_sum = np.zeros(X.shape[0])

                for sig in signals:
                    if sig == 'gate_u':
                        raw = raw_gate
                    elif sig == 'conv_var':
                        raw = raw_cv
                    else:
                        raise ValueError(f"Unknown signal: {sig}")

                    shift, scale = self.sig_stats[sig]
                    norm = np.clip((raw - shift) / scale, 0.0, 1.0)
                    corrected = norm if self.directions[sig] == +1 else 1.0 - norm

                    c, loc, sc = self.wb_params[sig]
                    p_k = weibull_min.cdf(corrected, c, loc, sc)

                    log_sum += self.weights[sig] * np.log(np.clip(p_k, 1e-8, 1.0))

                score = np.exp(log_sum / sum(self.weights.values()))
                return np.stack([1.0 - score, score], axis=1)

        clf = EVTCalibrator()

        # ─────────────────────────────────────────
        # SAVE CALIBRATOR  ← NEW
        # ─────────────────────────────────────────
        calibrator_ckpt = {
            "threshold": best_tau,
            "evt_directions": directions,
            "evt_wb_params": wb_params,
            "evt_signals": signals,
            "evt_sig_stats": sig_stats,
            "evt_weights": weights,
            "feature_order": ['gate_u', 'conv_var'],
            "protocol_info": self.protocol_info,
        }

        calib_path = self.output_dir / "checkpoints" / "posthoc_calibrator.pt"
        torch.save(calibrator_ckpt, calib_path)
        print(f"  Calibrator saved → {calib_path}")

        return clf, best_tau

    def evaluate_with_calibrator(self, loader, clf, tau):
        sp = self.collect_open_set_scores(loader)
        sp['conv_var'] = np.log(sp['conv_var'] + 1e-6)

        X = np.stack([
            sp['gate_u'],
            sp['conv_var']
        ], axis=1)

        y = sp['is_known'].astype(np.int32)
        p_known = clf.predict_proba(X)[:, 1]

        n_kno = int((y == 1).sum())
        n_unk = int((y == 0).sum())

        if n_unk == 0 or n_kno == 0:
            auroc = float('nan')
            fpr95 = 100.0
        else:
            auroc = 100.0 * roc_auc_score(y, p_known)
            labels_unk = (1 - y).astype(np.int32)
            fpr, tpr, _ = roc_curve(labels_unk, 1.0 - p_known)
            idx = np.where(tpr >= 0.95)[0]
            fpr95 = 100.0 * float(fpr[idx[0]]) if len(idx) > 0 else 100.0

        known_acc = 100.0 * (p_known[y == 1] >= tau).mean() if n_kno > 0 else 0.0
        unknown_acc = 100.0 * (p_known[y == 0] < tau).mean() if n_unk > 0 else float('nan')

        known_mask = y == 1
        y_true = sp['label'][known_mask]
        y_pred = sp['pred'][known_mask]

        OA = AA = Kappa = 0.0
        if y_true.size > 0:
            cm = confusion_matrix(
                y_true, y_pred,
                labels=list(range(self.args.num_classes))
            )

            OA = 100.0 * np.trace(cm) / np.sum(cm)
            per_class_acc = {}
            for i in range(self.args.num_classes):
                if cm[i].sum() > 0:
                    per_class_acc[f"class_{i}"] = float(cm[i, i] / cm[i].sum())
                else:
                    per_class_acc[f"class_{i}"] = 0.0
            per = [
                cm[i, i] / cm[i].sum()
                for i in range(self.args.num_classes)
                if cm[i].sum() > 0
            ]
            AA = 100.0 * np.mean(per) if per else 0.0
            Kappa = 100.0 * cohen_kappa_score(y_true, y_pred)

        return {
            'OA': OA,
            'AA': AA,
            'Kappa': Kappa,
            'KnownAcc': known_acc,
            'UnknownAcc': unknown_acc,
            'AUROC': auroc,
            'FPR95': fpr95,
            'ThresholdTau': float(tau),
            'per_class_acc': per_class_acc, 
        }

    def save_history(self):
        with open(self.output_dir / 'history.json', 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)

    # ── Main train loop ────────────────────────────────────────────

    def train(self, train_loader, val_loader, test_loader=None):
        self.setup(train_loader)

        print("\n" + "="*80)

        print("TRAINING START  (OpenSetContrastiveLoss + staged training)")
        print("="*80)

        # warmup threshold — defined outside loop
        warmup = getattr(self.args, 'warmup_epochs', 30)
        self.epoch_times = []

        for epoch in range(self.args.epochs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, vm = self.validate_closed_set(val_loader)
            self.scheduler.step()
        
            # History
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["OA"].append(vm["OA"])
            self.history["AA"].append(vm["AA"])
            self.history["Kappa"].append(vm["Kappa"])
            self.history["AUROC_ent"].append(vm["AUROC_ent"])
            self.history["AUROC_gate"].append(vm["AUROC_gate"])
            self.history["AUROC_msp"].append(vm["AUROC_msp"])
            self.history["AUROC_conv_var"].append(vm["AUROC_conv_var"])
            self.history["AUROC_conv_slope"].append(vm["AUROC_conv_slope"])
            self.history["gate_u_mean"].append(vm["gate_u_mean"])

            # Monitor signal
            # No unknown in val → AUROC=0, monitor by val_acc only
            val_has_unknown = any(v > 0.0 for v in [
                vm["AUROC_ent"], vm["AUROC_gate"],
                vm["AUROC_msp"], vm["AUROC_fb"],
                vm["AUROC_conv_var"], vm["AUROC_conv_slope"]
            ])

            if not val_has_unknown:
                signal_score = 0.5 * vm["AUROC_conv_var"] + 0.5 * vm["AUROC_conv_slope"]
                monitor_auroc = signal_score
                monitor_score = 0.7 * val_acc + 0.3 * signal_score
            else:
                if epoch < warmup:
                    monitor_auroc = max(vm["AUROC_msp"], vm["AUROC_conv_var"])
                else:
                    monitor_auroc = max(
                        vm["AUROC_ent"], vm["AUROC_gate"], vm["AUROC_msp"],
                        vm["AUROC_fb"], vm["AUROC_conv_var"], vm["AUROC_conv_slope"]
                    )
                monitor_score = 0.6 * monitor_auroc + 0.4 * val_acc

            print(f"\nEpoch {epoch+1}/{self.args.epochs} "
                  f"[stage={'1' if epoch < getattr(self.args,'warmup_epochs',30) else '2'}]")
            print(f"  Train  : loss={train_loss:.4f}  acc={train_acc:.2f}%")
            print(f"  Val    : loss={val_loss:.4f}   acc={val_acc:.2f}%")
            print(f"  AUROC  : ent={vm['AUROC_ent']:.2f}  gate={vm['AUROC_gate']:.2f}  "
                  f"msp={vm['AUROC_msp']:.2f}  fb={vm['AUROC_fb']:.2f}")
            scale_str = f"{self.model.logit_scale.item():.3f}" \
                        if hasattr(self.model, 'logit_scale') else "N/A"
            #T_str     = f"{self.model.entropy_temperature.item():.3f}" \
            #            if hasattr(self.model, 'entropy_temperature') else "N/A"
            print(f"  Scale  : {scale_str}    " #T={T_str}
                  f"alpha_sep={self.criterion.alpha_sep:.3f}")
            print(f"  Signals: conv_var={vm['AUROC_conv_var']:.2f}  "
                  f"conv_slope={vm['AUROC_conv_slope']:.2f}  "
                  f"gate_u_mean={vm['gate_u_mean']:.4f}")     
            
            print(f" cv_mean={vm['cv_mean']:.3f}  "
                    f"cs_mean={vm['cs_mean']:.3f}")
            # Checkpoint
            if val_acc >= 40.0:
                monitor_score = 0.6 * monitor_auroc + 0.4 * val_acc
            else:
                monitor_score = -1e9

            if monitor_score > self.best_monitor_score:
                self.best_monitor_score = monitor_score
                self.best_monitor_auroc = monitor_auroc
                self.best_val_acc       = val_acc
                self.best_epoch         = epoch
                self.patience_counter   = 0

                torch.save({
                    "epoch":               epoch,
                    "model_state_dict":    self.model.state_dict(),
                    "optimizer_state_dict":self.optimizer.state_dict(),
                    "best_monitor_score":  self.best_monitor_score,
                    "best_monitor_auroc":  self.best_monitor_auroc,
                    "best_val_acc":        self.best_val_acc,
                    "logit_scale":         self.model.logit_scale.item(),
                    "protocol_info":       self.protocol_info,
                    "unknown_class":       sorted(self.args.unknown_class),
                    "num_known_classes":   self.args.num_classes,
                }, self.output_dir / "checkpoints" / "full_model_best.pth")
                print(f"  Best checkpoint saved (score={monitor_score:.2f})")
            else:
                self.patience_counter += 1

            # Last checkpoint
            torch.save({
                "epoch":            epoch,
                "model_state_dict": self.model.state_dict(),
                "val_acc":          val_acc,
                "monitor_auroc":    monitor_auroc,
            }, self.output_dir / "checkpoints" / "full_model_last.pth")

            if self.patience_counter >= self.args.patience:
                print(f"\n[Early Stop] Epoch {epoch+1}")
                break
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            epoch_time = end_time - start_time
            self.epoch_times.append(epoch_time)

            print(f"Epoch {epoch+1}: time = {epoch_time:.2f} sec")

        self.save_history()

        print(f"\nBest epoch={self.best_epoch+1}  "
              f"val_acc={self.best_val_acc:.2f}%  "
              f"AUROC={self.best_monitor_auroc:.2f}%")

        # Load best → post-hoc calibration
        ckpt = torch.load(
            self.output_dir / "checkpoints" / "full_model_best.pth",
            map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])

        print("\n" + "="*80 + "\nPOST-HOC CALIBRATION\n" + "="*80)
        # No unknown in val → use val known only for tau
        clf, tau = self.fit_posthoc_calibrator(val_loader)

        val_final = self.evaluate_with_calibrator(val_loader, clf, tau)
        print("Val  (post-hoc):", val_final)
        with open(self.output_dir / "val_posthoc_metrics.json", "w") as f:
            json.dump(val_final, f, indent=2)

        test_final = None
        if test_loader is not None:
            test_final = self.evaluate_with_calibrator(test_loader, clf, tau)
            print("Test (post-hoc):", test_final)
            with open(self.output_dir / "test_posthoc_metrics.json", "w") as f:
                json.dump(test_final, f, indent=2)

        # ─────────────────────────────────────────
        # SEED-LEVEL FINAL SUMMARY
        # ─────────────────────────────────────────
        final_result = test_final if test_final is not None else val_final

        valid_epoch_times = self.epoch_times[5:] if len(self.epoch_times) > 5 else self.epoch_times
        final_result["epoch_time_mean"] = float(np.mean(valid_epoch_times)) if valid_epoch_times else None
        final_result["epoch_time_std"] = float(np.std(valid_epoch_times)) if valid_epoch_times else None

        seed_summary = {
            **self.protocol_info,
            "best_epoch": int(self.best_epoch + 1),
            "best_val_acc": float(self.best_val_acc),
            "best_monitor_auroc": float(self.best_monitor_auroc),
            "best_monitor_score": float(self.best_monitor_score),
            "checkpoint_path": str(self.output_dir / "checkpoints" / "full_model_best.pth"),
            "calibrator_path": str(self.output_dir / "checkpoints" / "posthoc_calibrator.pt"),
            "final_metrics": final_result,
            "per_class_acc": final_result.get("per_class_acc", {}),
        }

        with open(self.output_dir / "seed_final_summary.json", "w") as f:
            json.dump(seed_summary, f, indent=2)

        return final_result


# ─────────────────────────────────────────────────────────────────
# Seed summary
# ─────────────────────────────────────────────────────────────────

def save_seed_summary(all_results, output_dir, args=None):
    output_dir = Path(output_dir)
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "all_seed_results.csv", index=False)

    # -------------------------------------------------
    # Global metrics
    # -------------------------------------------------
    metrics = ["OA", "AA", "Kappa", "KnownAcc", "UnknownAcc", "AUROC", "FPR95", "ThresholdTau"]
    rows = []

    for m in metrics:
        if m not in df:
            continue
        mean = float(df[m].mean())
        std = float(df[m].std())
        rows.append({
            "Metric": m,
            "Mean": mean,
            "Std": std,
            "Mean±Std": f"{mean:.2f} ± {std:.2f}"
        })
    
    if "epoch_time_mean" in df.columns:
        mean_time = float(df["epoch_time_mean"].mean())
        std_time  = float(df["epoch_time_mean"].std())

        rows.append({
            "Metric": "TrainTime_per_epoch (s)",
            "Mean": mean_time,
            "Std": std_time,
            "Mean±Std": f"{mean_time:.2f} ± {std_time:.2f}"
        })
    # -------------------------------------------------
    # Per-class accuracy aggregation
    # -------------------------------------------------
    per_class_all = {}

    for r in all_results:
        if "per_class_acc" not in r:
            continue
        pc = r["per_class_acc"]
        if not isinstance(pc, dict):
            continue
        for k, v in pc.items():
            try:
                per_class_all.setdefault(k, []).append(float(v))
            except Exception:
                pass

    per_class_summary = {}
    for k, vals in per_class_all.items():
        mean = float(np.mean(vals))
        std = float(np.std(vals)) if len(vals) > 1 else 0.0
        per_class_summary[k] = {
            "mean": mean,
            "std": std,
            "mean±std": f"{mean * 100:.2f} ± {std * 100:.2f}"
        }

        # optional: also append to summary CSV
        rows.append({
            "Metric": k,
            "Mean": mean * 100,
            "Std": std * 100,
            "Mean±Std": f"{mean * 100:.2f} ± {std * 100:.2f}"
        })

    # -------------------------------------------------
    # Save summary CSV
    # -------------------------------------------------
    summary_df = pd.DataFrame(rows)
    
    summary_df.to_csv(output_dir / "seed_summary_results.csv", index=False)

    # Save per-class JSON
    with open(output_dir / "per_class_summary.json", "w", encoding="utf-8") as f:
        json.dump(per_class_summary, f, indent=2)

    # -------------------------------------------------
    # Protocol metadata
    # -------------------------------------------------
    if args is not None:
        protocol = {
            "dataset": args.dataset,
            "unknown_class": sorted(args.unknown_class),
            "num_known_classes": args.num_classes,
            "samples_per_class": args.samples_per_class,
            "unknown_train_samples": args.unknown_train_samples,
            "val_unknown_samples": getattr(args, "val_unknown_samples", 0),
            "patch_size": args.patch_size,
            "epochs": args.epochs,
            "seeds": args.seeds,
            "selection_rule": (
                "Each seed is trained independently; each seed uses its own best checkpoint "
                "and own post-hoc calibrator; final results are reported as mean±std across seeds."
            ),
        }
        with open(output_dir / "protocol_summary.json", "w", encoding="utf-8") as f:
            json.dump(protocol, f, indent=2)

    print("\n" + "=" * 40 + "\nFINAL RESULTS (mean ± std)\n" + "=" * 40)
    for r in rows:
        print(f"  {r['Metric']:<14}: {r['Mean±Std']}")

    if per_class_summary:
        print("\nPer-class accuracy (mean ± std):")
        for k, v in per_class_summary.items():
            print(f"  {k:<10}: {v['mean±std']}")
# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',      type=str, required=True,
                   choices=['houston2013','muufl','augsburg'])
    p.add_argument('--data_root',    type=str, required=True)
    p.add_argument('--unknown_class',type=int, nargs='+', required=True, 
                   help='One or more unknown classes, e.g. --unknown_class 3 10 11')
    p.add_argument('--patch_size',   type=int, default=15)
    p.add_argument('--train_ratio',  type=float, default=0.2)
    p.add_argument('--val_ratio',    type=float, default=0.15)
    p.add_argument('--unknown_train_samples', type=int, default=0)
    p.add_argument('--samples_per_class',     type=int, default=20)
    # ── Protocol switch ────────────────────────────────────────────────────
    p.add_argument('--val_unknown_samples',   type=int, default=0,
                   help="Val set-д unknown sample оруулах тоо.\n"
                        "0 = val-д unknown байхгүй (HyLiOSR protocol)\n"
                        "default: 0 (strict OSR protocol, no unknown in val)")

    p.add_argument('--feature_dim',    type=int,   default=64)
    p.add_argument('--num_iterations', type=int,   default=3)
    p.add_argument('--dropout',        type=float, default=0.1)
    p.add_argument('--use_attention',
                   type=lambda x: str(x).lower()=='true', default=True)

    p.add_argument('--contrastive_weight', type=float, default=0.30)
    p.add_argument('--temperature',        type=float, default=0.07)
    # separation loss weight
    p.add_argument('--alpha_sep',          type=float, default=0.15)
    # staged training
    p.add_argument('--warmup_epochs',      type=int,   default=30)
    p.add_argument('--separation_weight', type=float, default=0.20)
    p.add_argument('--vicreg_weight',     type=float, default=0.00)

    p.add_argument('--epochs',         type=int,   default=200)
    p.add_argument('--batch_size',     type=int,   default=32)
    p.add_argument('--learning_rate',  type=float, default=5e-4)
    p.add_argument('--weight_decay',   type=float, default=1e-2)
    p.add_argument('--patience',       type=int,   default=40)
    p.add_argument('--seeds',          type=int,   nargs='+', default=[0,1,2])
    p.add_argument('--output_dir',     type=str,   default='posthoc_experiments_fixed')

    def str2bool(v):
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("1", "true", "yes", "y", "t")

    p.add_argument('--use_gate', type=str2bool, default=True,
               help='Enable uncertainty-aware gate')
    p.add_argument('--use_feedback', type=str2bool, default=True,
                help='Enable feedback fusion')
    p.add_argument('--use_mch', type=str2bool, default=True,
                help='Enable monotonic confidence head')
    
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*80 + "\nCONFIGURATION\n" + "="*80)
    for k, v in vars(args).items():
        print(f"  {k:<22}: {v}")
    print("="*80 + "\n")

    if args.dataset == "houston2013":
        all_classes = list(range(1, 16))
    elif args.dataset == "muufl":
        all_classes = list(range(1, 12))
    else:
        all_classes = list(range(1, 8))
    #all_classes.remove(args.unknown_class)
    #args.num_classes = len(all_classes)
    unknown_classes = sorted(set(args.unknown_class))

    all_classes = [c for c in all_classes if c not in unknown_classes]
    args.num_classes = len(all_classes)

    print(f"Unknown classes: {unknown_classes}")
    print(f"Known classes: {args.num_classes}")

    val_unk = getattr(args, 'val_unknown_samples', 0)
    print(f"\n{'='*60}")
    print(f"PROTOCOL: val_unknown_samples = {val_unk}")
    if val_unk == 0:
        print("  → Val set has NO unknown samples (strict OSR protocol)")
        print("  → Calibrator: tau computed from known val only")
        print("  → Fair comparison protocol: no unknown supervision")
    else:
        print(f"  → Val set has {val_unk} unknown samples")
    print(f"{'='*60}\n")

    all_results = []
    original_out_dir = args.output_dir
    unk_tag = "class" + "_".join(map(str, sorted(args.unknown_class)))

    for seed in args.seeds:
        print(f"\n{'='*80}\nSEED {seed}  ({args.seeds.index(seed)+1}/{len(args.seeds)})\n{'='*80}")
        set_seed(seed)

        # seed-specific output dir
        args.output_dir = str(
            Path(original_out_dir) / args.dataset / unk_tag / f"seed_{seed}"
        )

        # rebuild dataset/splits for this seed
        dataset = load_dataset(args.dataset, args.data_root, patch_size=args.patch_size)

        dl_sig = inspect.signature(dataset.create_dataloaders)
        dl_kwargs = dict(
            unknown_class=args.unknown_class,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio if args.samples_per_class < 0 else None,
            val_ratio=args.val_ratio,
            samples_per_class=args.samples_per_class if args.samples_per_class > 0 else None,
            unknown_train_samples=args.unknown_train_samples,
        )

        if 'val_unknown_samples' in dl_sig.parameters:
            dl_kwargs['val_unknown_samples'] = getattr(args, "val_unknown_samples", 0)

        train_loader, val_loader, test_loader = dataset.create_dataloaders(**dl_kwargs)

        # strict known-only val filtering if needed
        n_val_unk = sum((isk == 0).sum().item() for _, _, _, isk in val_loader)
        if n_val_unk > 0 and getattr(args, "val_unknown_samples", 0) == 0:
            print(f"  → Filtering out {n_val_unk} unknown samples from val (known-only val)")
            val_ds = val_loader.dataset
            known_indices = [i for i in range(len(val_ds)) if val_ds[i][3] == 1]
            val_known_ds = Subset(val_ds, known_indices)
            val_loader = DataLoader(
                val_known_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )

            n_val_unk_after = sum((isk == 0).sum().item() for _, _, _, isk in val_loader)
            n_val_known = sum((isk == 1).sum().item() for _, _, _, isk in val_loader)
            print(f"  Val after filtering: known={n_val_known}, unknown={n_val_unk_after}")
        else:
            print(f"  Val unknown count: {n_val_unk} (protocol: {getattr(args, 'val_unknown_samples', 0)})")

        trainer = Trainer(args, seed=seed)
        result = trainer.train(train_loader, val_loader, test_loader)

        if result:
            result["seed"] = seed
            result["checkpoint_path"] = str(Path(args.output_dir) / "checkpoints" / "full_model_best.pth")
            result["calibrator_path"] = str(Path(args.output_dir) / "checkpoints" / "posthoc_calibrator.pt")
            all_results.append(result)

    # restore original output dir
    args.output_dir = original_out_dir

    if len(all_results) > 1:
        summary_dir = Path(original_out_dir) / args.dataset / unk_tag
        summary_dir.mkdir(parents=True, exist_ok=True)

        save_seed_summary(all_results, summary_dir, args=args)

        summary = {
            "dataset": args.dataset,
            "unknown_class": sorted(args.unknown_class),
            "num_known_classes": args.num_classes,
            "samples_per_class": args.samples_per_class,
            "unknown_train_samples": args.unknown_train_samples,
            "val_unknown_samples": getattr(args, "val_unknown_samples", 0),
            "seeds": args.seeds,
            "protocol": "per-seed independent training + per-seed best checkpoint + per-seed own calibrator + final mean±std",
        }
        
        if all_results and "epoch_time_mean" in all_results[0]:
            vals = [r["epoch_time_mean"] for r in all_results if "epoch_time_mean" in r]
            if vals:
                summary["TrainTime_per_epoch"] = float(np.mean(vals))
                summary["TrainTime_per_epoch_std"] = float(np.std(vals))

        for k in ["OA", "AA", "Kappa", "KnownAcc", "UnknownAcc", "AUROC", "FPR95", "ThresholdTau"]:
            vals = [r[k] for r in all_results if k in r and r[k] is not None]
            if vals:
                summary[k] = float(np.mean(vals))
                summary[f"{k}_std"] = float(np.std(vals))

        with open(summary_dir / "summary_across_seeds.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()


# Example:

# python train_no_val_unknown_calibrator_with_evalution.py --dataset houston2013 --data_root "D:/datasets/Houston, Trento, Muufl/houston2013" --unknown_class 6 13 14 15 --samples_per_class 20 --val_unknown_samples 0 --epochs 200 --seeds 0 1 2 3 4

# python train_no_val_unknown_calibrator_with_evalution_houston2013.py --dataset augsburg --data_root "D:/datasets/Houston, Trento, Muufl/augsburg" --unknown_class 4 5 --samples_per_class 20 --val_unknown_samples 0 --epochs 200 --seeds 0 1 2 3 4 

# python train_no_val_unknown_calibrator_with_evalution.py --dataset muufl --data_root "D:/datasets/Houston, Trento, Muufl/muufl" --unknown_class 7 8 9 10 11 --samples_per_class 20 --val_unknown_samples 0 --epochs 200 --seeds 0 1 2 3 4 

# python train_no_val_unknown_calibrator_with_evalution.py --dataset trento --data_root "D:/datasets/Houston, Trento, Muufl/trento" --unknown_class 3 13 --samples_per_class 20 --val_unknown_samples 0 --epochs 200 --seeds 0 1 2 3 4 