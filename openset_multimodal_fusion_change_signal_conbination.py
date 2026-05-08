"""
EVT/Weibull post-hoc calibrator (strict OSR protocol).

This implementation always fits Weibull using known validation samples only.
Unknown validation samples, if present, are ignored for fitting and thresholding.

tau = 10th percentile of known EVT scores.
═══════════════════════════════════════════════════════════════════════════════
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# 1. UncertaintyAwareGate  (өөрчлөлтгүй)
# ═══════════════════════════════════════════════════════════════════════════

class UncertaintyAwareGate(nn.Module):
    def __init__(self, dim: int, use_attention: bool = True, dropout: float = 0.3):
        super().__init__()
        self.dim = dim
        self.use_attention = use_attention

        self.gate_hsi = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.LayerNorm(dim // 2),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(dim // 2, 1), nn.Sigmoid())

        self.gate_lidar = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.LayerNorm(dim // 2),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(dim // 2, 1), nn.Sigmoid())

        if use_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=dim, num_heads=4, dropout=dropout, batch_first=True)

        self.uncertainty_net = nn.Sequential(
            nn.Linear(dim * 2 + 2, dim), nn.LayerNorm(dim),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(dim, dim // 2), nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1), nn.Sigmoid())

        self.uncertainty_weights = nn.Parameter(torch.tensor([0.33, 0.33, 0.33])) # 0.2, 0.2, 0.6
                                                                # 0.33, 0.33, 0.33   # 0.6, 0.2, 0.2
    def forward(self, feat_hsi, feat_lidar):
        gate_hsi   = self.gate_hsi(feat_hsi)
        gate_lidar = self.gate_lidar(feat_lidar)
        gate_sum   = gate_hsi + gate_lidar + 1e-8
        gate_hsi_norm   = gate_hsi   / gate_sum
        gate_lidar_norm = gate_lidar / gate_sum

        if self.use_attention:
            stacked = torch.stack([feat_hsi, feat_lidar], dim=1)
            attended, _ = self.cross_attention(stacked, stacked, stacked)
            fused = (gate_hsi_norm * attended[:, 0, :]
                + gate_lidar_norm * attended[:, 1, :])
        else:
            fused = gate_hsi_norm * feat_hsi + gate_lidar_norm * feat_lidar

        # ✅ RAW gate variance — Figure C4-аас нотлогдсон дохио
        # Known: variance өргөн (0.70~0.95) → high variance
        # Unknown: variance нарийн (0.74~0.78) → low variance
        # MW U: p=1.38e-139 → статистикийн хувьд хүчтэй ялгаа
        gate_variance = torch.abs(gate_hsi - gate_lidar)        # [0,1]
        gate_min      = torch.min(gate_hsi, gate_lidar)         # [0,1]

        unc_input = torch.cat([feat_hsi, feat_lidar,
                            gate_hsi_norm, gate_lidar_norm], dim=-1)
        base_unc  = self.uncertainty_net(unc_input)
        weights   = F.softmax(self.uncertainty_weights, dim=0)

        # gate_variance болон (1-gate_min) нь raw space-д
        # шугаман хамааралтай боловч НОРМАЛЛАГДСАН space-д тогтмол биш
        # → RAW утга ашиглах нь зөв (Figure C4 нотолсон)
        combined_unc = (weights[0] * base_unc
                    + weights[1] * gate_variance
                    + weights[2] * (1.0 - gate_min))

        stats = {
            'gate_hsi':          gate_hsi.squeeze(-1),
            'gate_lidar':        gate_lidar.squeeze(-1),
            'gate_variance':     gate_variance.squeeze(-1),
            'base_uncertainty':  base_unc.squeeze(-1),
            'final_uncertainty': combined_unc.squeeze(-1),
        }
        return fused, combined_unc.squeeze(-1), stats


# ═══════════════════════════════════════════════════════════════════════════
# 2. FeedbackOpenSetFusion  (өөрчлөлтгүй)
# ═══════════════════════════════════════════════════════════════════════════

class FeedbackOpenSetFusion(nn.Module):
    def __init__(self, dim: int, num_iterations: int = 3, dropout: float = 0.2):
        super().__init__()
        self.dim = dim
        self.num_iterations = num_iterations

        self.refine_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 3, dim * 2), nn.LayerNorm(dim * 2),
                nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.ReLU(inplace=True))
            for _ in range(num_iterations)])

        self.consistency_net = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.LayerNorm(dim),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(dim, dim // 2), nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1), nn.Sigmoid())

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, feat_hsi, feat_lidar, initial_fused=None):
        fused = initial_fused if initial_fused is not None else (feat_hsi + feat_lidar) / 2
        feature_changes    = []
        consistency_scores = []

        for refine_net in self.refine_nets:
            refinement_input = torch.cat([fused, feat_hsi, feat_lidar], dim=-1)
            refined  = refine_net(refinement_input)
            change   = torch.norm(refined - fused, p=2, dim=-1)
            feature_changes.append(change)
            consistency = self.consistency_net(
                torch.cat([refined, feat_hsi, feat_lidar], dim=-1))
            consistency_scores.append(consistency.squeeze(-1))
            fused = refined

        feature_changes    = torch.stack(feature_changes,    dim=1)
        consistency_scores = torch.stack(consistency_scores, dim=1)
        mean_change        = feature_changes.mean(dim=1)
        convergence_score  = torch.exp(-mean_change / self.temperature)
        consistency_mean   = consistency_scores.mean(dim=1)

        if consistency_scores.shape[1] > 1:
            consistency_std = consistency_scores.std(dim=1)
        else:
            consistency_std = torch.zeros(
                consistency_scores.shape[0], device=consistency_scores.device)

        consistency_stability = torch.sigmoid(1.0 / (consistency_std + 1e-6))

        # conv_slope: last - first change → negative = converging (known-like)
        # Diagnose: conv_slope AUROC=77.5%, gate_u corr=−0.051 → complementary
        if feature_changes.shape[1] > 1:
            conv_slope = feature_changes[:, -1] - feature_changes[:, 0]
        else:
            conv_slope = torch.zeros(
                feature_changes.shape[0], device=feature_changes.device)

        if self.num_iterations == 1:
            feedback_known_score = 0.7 * consistency_mean + 0.3 * convergence_score
        else:
            feedback_known_score = (0.6 * consistency_mean
                                   + 0.2 * convergence_score
                                   + 0.2 * consistency_stability)

        stats = {
            'feature_changes':      feature_changes,
            'consistency_scores':   consistency_scores,
            'convergence_score':    convergence_score,
            'consistency_mean':     consistency_mean,
            'consistency_stability':consistency_stability,
            'conv_slope':           conv_slope,  
        }
        return fused, feedback_known_score, stats


# ═══════════════════════════════════════════════════════════════════════════
# 3. FeatureExtractor  (өөрчлөлтгүй)
# ═══════════════════════════════════════════════════════════════════════════

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64,  kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64,  128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1))
        self.final_dim = feature_dim

    def forward(self, x):
        return self.projection(self.encoder(x))


# ═══════════════════════════════════════════════════════════════════════════
# 4. MonotonicConfidenceHead
# ═══════════════════════════════════════════════════════════════════════════

class MonotonicConfidenceHead(nn.Module):
    """
    3 uncertainty signal-ийг нэгтгэж conf_known [0,1] буцаана.

    Signals (all in [0,1], higher = more unknown-like):
      - gate_uncertainty      : modal disagreement
      - feedback_known_score  : high = known (NOT unknown-like!)
      - normalized_entropy    : classifier uncertainty
    """
    def __init__(self):
        super().__init__()
        self.raw_w = nn.Parameter(torch.zeros(2))       # gate_u + fb_u only
        self.bias  = nn.Parameter(torch.tensor(0.0))
        self.kappa = nn.Parameter(torch.tensor(1.0))

    def weights(self):
        w = F.softplus(self.raw_w) + 1e-6
        return w / w.sum()

    def forward(self, gate_u, fb_u, ent_u):
        """
        conf = f(gate_u, fb_u) — modal + feedback only.
        ent_u removed from MCH to avoid redundancy with calibrator signal.
        → conf: cross-modal + iterative convergence uncertainty
        → ent_u: used separately in calibrator as independent signal
        """
        s_gate = 1.0 - gate_u   # high = known
        s_fb   = 1.0 - fb_u     # high = known

        x     = torch.stack([s_gate, s_fb], dim=1)       # [B, 2]
        w     = self.weights()                             # [2]
        kappa = torch.clamp(self.kappa, 0.1, 10.0)
        z     = (x * w.unsqueeze(0)).sum(dim=1) + self.bias
        return torch.sigmoid(kappa * z)


# ═══════════════════════════════════════════════════════════════════════════
# 5. NovelOpenSetMultiModalNet  
# ═══════════════════════════════════════════════════════════════════════════

class NovelOpenSetMultiModalNet(nn.Module):
    def __init__(self, hsi_channels: int, lidar_channels: int,
                 num_classes: int, feature_dim: int = True,
                 use_attention: bool = True,
                 num_iterations: int = 3, dropout: float = True, 
                 use_gate=True,
                use_feedback=True,
                use_mch=True ):
                 
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.use_gate = use_gate
        self.use_feedback = use_feedback
        self.use_mch = use_mch

        self.hsi_encoder   = FeatureExtractor(in_channels=hsi_channels,   feature_dim=feature_dim)
        self.lidar_encoder = FeatureExtractor(in_channels=lidar_channels,  feature_dim=feature_dim)
        self.hsi_proj      = nn.Identity()
        self.lidar_proj    = nn.Identity()

        self.uncertainty_gate  = UncertaintyAwareGate(dim=feature_dim,
                                                       use_attention=use_attention,
                                                       dropout=dropout) if self.use_gate else None
        self.feedback_fusion   = FeedbackOpenSetFusion(dim=feature_dim,
                                                        num_iterations=num_iterations,
                                                        dropout=dropout) if self.use_feedback else None
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes, bias=False))

        self.openset_head = MonotonicConfidenceHead() if self.use_mch else None

        # scale_reg target=5.0 → хэт өсөхөөс хамгаална
        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(5.0)))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias,   0)
                nn.init.constant_(m.weight, 1.0)

    @property
    def logit_scale(self):
        """log-space → linear-space, [1.0, 20.0]-д хязгаарлана"""
        return torch.clamp(torch.exp(self.log_logit_scale), 1.0, 20.0)

    def forward(self, hsi, lidar, return_features=False, return_uncertainty=False):
        feat_hsi   = self.hsi_proj(self.hsi_encoder(hsi))
        feat_lidar = self.lidar_proj(self.lidar_encoder(lidar))

        # gate_u: detach feat_hsi/feat_lidar → contrastive gradient
        # cannot collapse gate_variance via feat alignment
        B = feat_hsi.size(0)
        device = feat_hsi.device

        # ----- Gate / UAG -----
        if self.use_gate and self.uncertainty_gate is not None:
            gated_features, gate_uncertainty, gate_stats = self.uncertainty_gate(
                feat_hsi.detach(), feat_lidar.detach()
            )
            gate_u = gate_uncertainty.clamp(0.0, 1.0)
        else:
            gated_features = 0.5 * (feat_hsi + feat_lidar)
            gate_u = torch.zeros(B, device=device)
            gate_stats = None

        # ----- Feedback / FBF -----
        if self.use_feedback and self.feedback_fusion is not None:
            refined_features, feedback_known_score, feedback_stats = self.feedback_fusion(
                feat_hsi, feat_lidar, initial_fused=gated_features
            )
        else:
            refined_features = gated_features
            feedback_known_score = torch.ones(B, device=device)
            feedback_stats = {
                "feature_changes": torch.zeros(B, 1, device=device),
                "conv_slope": torch.zeros(B, device=device),
            }

        # Penultimate features
        feat_penultimate = refined_features
        for layer in list(self.classifier.children())[:-1]:
            feat_penultimate = layer(feat_penultimate)

        # Cosine classifier
        feat_norm   = F.normalize(feat_penultimate, dim=1)
        weight_norm = F.normalize(self.classifier[-1].weight, dim=1)
        scale  = self.logit_scale
        logits = feat_norm @ weight_norm.T * scale

        # Entropy: T-г ХАСАВ (collapse засвар)
        # scale=5.0 → logits ∈ [-5, +5] → softmax мэдрэмтгий
        # Known: нэг ангид итгэлтэй → ent_u ≈ 0.2~0.5
        # Unknown: тархмал → ent_u ≈ 0.6~0.9 → ялгаа гарна
        probs   = F.softmax(logits.detach(), dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        ent_u   = (entropy / math.log(self.num_classes)).clamp(0.0, 1.0)

        # 3 uncertainty signals [0,1] (high = unknown)
        
        # fb_u: conv_slope (last - first feature change)
        # Known: converges → slope < 0 → after normalization fb_u low
        # Unknown: diverges → slope > 0 → fb_u high
        # Diagnose: AUROC=77.5%, gate_u corr=−0.051 → complementary ✓
        raw_slope = feedback_stats['conv_slope']
        # Normalize to [0,1]: sigmoid((slope - median) / scale)
        fb_u = torch.sigmoid(
            raw_slope / (self.feature_dim ** 0.5)
        ).clamp(0.0, 1.0)

        # openset_head: gate_u, fb_u detach → MCH gradient UAG/feedback-д буцахгүй
        #openset_confidence = self.openset_head(gate_u.detach(), fb_u.detach(), ent_u)
        gate_u_for_mch = gate_u.detach() + 0.1 * (gate_u - gate_u.detach())
        fb_u_for_mch   = fb_u.detach()   + 0.1 * (fb_u   - fb_u.detach())

        #openset_confidence = self.openset_head(gate_u_for_mch, fb_u_for_mch, ent_u)
        if self.use_mch and self.openset_head is not None:
            openset_confidence = self.openset_head(gate_u_for_mch, fb_u_for_mch, ent_u)
        else:
            openset_confidence = 1.0 - ent_u

        # Санамсаргүй print (1% магадлалтай)
        if self.training and torch.rand(1).item() < 0.005:
            print(f"  [DBG] scale={scale.item():.3f}  " #T={T.item():.3f}
                  f"gate_u={gate_u.mean():.3f} fb_u={fb_u.mean():.3f} "
                  f"ent_u={ent_u.mean():.3f} conf={openset_confidence.mean():.3f}")

        outputs = [logits, openset_confidence]
        if return_features:
            outputs.append(refined_features)
        if return_uncertainty:
            # conv_var: variance of per-iteration feature changes
            fc = feedback_stats['feature_changes']   # [B, n_iter]
            cv = fc.var(dim=1).detach() if fc.shape[1] > 1 else torch.zeros(
                fc.shape[0], device=fc.device)
            cs = feedback_stats['conv_slope'].detach()
            outputs.append({
                'gate_uncertainty':    gate_u,
                'feedback_uncertainty':fb_u,
                'normalized_entropy':  ent_u,
                'logit_scale':         scale.detach(),
                'conv_var':            cv,
                'conv_slope':          cs,
            })

        return tuple(outputs) if len(outputs) > 2 else (logits, openset_confidence)

    def predict_with_rejection(self, hsi, lidar, threshold):
        logits, openset_confidence = self.forward(hsi, lidar)
        probs  = F.softmax(logits, dim=1)
        _, class_preds = probs.max(dim=1)
        is_known    = openset_confidence >= threshold
        predictions = torch.where(
            is_known, class_preds,
            torch.tensor(-1, dtype=class_preds.dtype, device=class_preds.device))
        return predictions, openset_confidence, is_known

    def scale_regularization_loss(self, target: float = 5.0, weight: float = 1e-3):
        """
        logit_scale-г target утгад татах regularization.
        target=5.0: entropy мэдрэмтгий байхад хангалттай
        """
        return weight * (self.log_logit_scale - math.log(target)) ** 2


# ═══════════════════════════════════════════════════════════════════════════
# 6. Losses (өөрчлөлтгүй, reference болгон хадгалав)
# ═══════════════════════════════════════════════════════════════════════════

class OpenSetContrastiveLoss(nn.Module):
    """
    conf, gate_u, fb_u, ent_u-г loss-д ашигладаг бүрэн loss.
    train_posthoc_osr_fixed.py-д энийг ашиглана.
    """
    def __init__(self, num_classes: int, temperature: float = 0.07,
                 class_weights=None, label_smoothing: float = 0.1,
                 ce_weight: float = 1.0, contrastive_weight: float = 0.30,
                 separation_weight: float = 0.15, vicreg_weight: float = 0.00):
        super().__init__()
        self.num_classes        = num_classes
        self.temperature        = temperature
        self.label_smoothing    = label_smoothing
        self.ce_weight          = ce_weight
        self.contrastive_weight = contrastive_weight
        self.separation_weight  = separation_weight
        self.vicreg_weight      = vicreg_weight

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self.stage     = 1
        self.alpha_sep = 0.0
        self.prototype_initialized = torch.zeros(num_classes, dtype=torch.bool)

    def _contrastive_loss(self, features, labels):
        if features.size(0) <= 1:
            return torch.tensor(0.0, device=features.device)
        z    = F.normalize(features, dim=1)
        sim  = torch.matmul(z, z.T) / self.temperature
        same = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        eye  = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        same = same & (~eye)
        has_pos = same.sum(dim=1) > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        exp_sim  = torch.exp(sim)
        pos_sum  = (exp_sim * same.float()).sum(dim=1)
        denom    = exp_sim.sum(dim=1) - torch.exp(torch.diagonal(sim))
        loss_per = -torch.log((pos_sum / (denom + 1e-8)) + 1e-8)
        return loss_per[has_pos].mean()

    def _boundary_separation_loss(self, confidence, is_known,
                                   gate_u_train=None, fb_u_train=None, ent_u_train=None):
        device = confidence.device if confidence is not None else is_known.device
        zero   = torch.tensor(0.0, device=device)
        empty_aux = {"loss_easy":0.,"loss_hard":0.,"loss_rank":0.,
                     "loss_gate_floor":0.,"loss_fb_var":0.,"u_mean":0.}

        if confidence is None or is_known.sum() < 4:
            return zero, empty_aux

        conf_k   = confidence[is_known]
        unc_parts = []
        if gate_u_train is not None: unc_parts.append(gate_u_train[is_known])
        if fb_u_train   is not None: unc_parts.append(fb_u_train[is_known])
        if ent_u_train  is not None: unc_parts.append(ent_u_train[is_known])
        if len(unc_parts) == 0:
            return zero, empty_aux

        u = torch.stack(unc_parts, dim=0).mean(dim=0)
        u = (u - u.min().detach()) / (u.max().detach() - u.min().detach() + 1e-8)

        q_low  = torch.quantile(u.detach(), 0.30)
        q_high = torch.quantile(u.detach(), 0.70)

        easy_mask = u <= q_low
        hard_mask = u >= q_high

        loss_easy = loss_hard = loss_rank = loss_gate_floor = loss_fb_var = zero

        if easy_mask.sum() > 0:
            loss_easy = F.relu(0.90 - conf_k[easy_mask]).mean()
        if hard_mask.sum() > 0:
            loss_hard = F.relu(conf_k[hard_mask] - 0.60).mean()

        target_conf = 1.0 - u
        loss_rank   = ((conf_k - target_conf) ** 2).mean()

        if gate_u_train is not None:
            loss_gate_floor = F.relu(0.05 - gate_u_train[is_known].mean())
        if fb_u_train is not None and is_known.sum() > 1:
            loss_fb_var = F.relu(0.01 - fb_u_train[is_known].std())

        loss_sep = (0.35 * loss_easy + 0.35 * loss_hard
                   + 0.15 * loss_rank + 0.10 * loss_gate_floor
                   + 0.05 * loss_fb_var)

        aux = {"loss_easy":float(loss_easy.item()), "loss_hard":float(loss_hard.item()),
               "loss_rank":float(loss_rank.item()), "loss_gate_floor":float(loss_gate_floor.item()),
               "loss_fb_var":float(loss_fb_var.item()), "u_mean":float(u.mean().item())}
        return loss_sep, aux

    def _uncertainty_vicreg_loss(self, gate_u_train, fb_u_train, ent_u_train,
                                  is_known, var_target=0.05, var_weight=1.0, cov_weight=0.05):
        device = is_known.device
        zero   = torch.tensor(0.0, device=device)
        parts  = []
        if gate_u_train is not None: parts.append(gate_u_train[is_known])
        if fb_u_train   is not None: parts.append(fb_u_train[is_known])
        if ent_u_train  is not None: parts.append(ent_u_train[is_known])
        if len(parts) < 2:
            return zero, {"vic_var":0., "vic_cov":0.}

        U   = torch.stack(parts, dim=1)
        if U.shape[0] < 2:
            return zero, {"vic_var":0., "vic_cov":0.}

        std      = torch.sqrt(U.var(dim=0, unbiased=False) + 1e-4)
        var_loss = torch.mean(F.relu(var_target - std) ** 2)
        Uc       = U - U.mean(dim=0, keepdim=True)
        cov      = (Uc.T @ Uc) / max(U.shape[0] - 1, 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = (off_diag ** 2).sum() / U.shape[1]

        loss = var_weight * var_loss + cov_weight * cov_loss
        return loss, {"vic_var":float(var_loss.item()), "vic_cov":float(cov_loss.item())}

    def forward(self, logits, features, confidence, labels, is_known,
                gate_u_train=None, fb_u_train=None, ent_u_train=None):
        device = logits.device
        zero   = torch.tensor(0.0, device=device)

        ce_loss = contrastive_loss = separation_loss = vicreg_u_loss = zero
        sep_aux = {"loss_easy":0.,"loss_hard":0.,"loss_rank":0.,
                   "loss_gate_floor":0.,"loss_fb_var":0.,"u_mean":0.}
        vic_aux = {"vic_var":0., "vic_cov":0.}

        if is_known.sum() > 0:
            ce_loss = F.cross_entropy(
                logits[is_known], labels[is_known],
                weight=self.class_weights if self.class_weights is not None else None,
                label_smoothing=self.label_smoothing)

        if features is not None and is_known.sum() > 1:
            contrastive_loss = self._contrastive_loss(features[is_known], labels[is_known])

        if self.alpha_sep > 0.0 and confidence is not None:
            separation_loss, sep_aux = self._boundary_separation_loss(
                confidence, is_known, gate_u_train, fb_u_train, ent_u_train)

        if self.alpha_sep > 0.0 and is_known.sum() > 1:
            vicreg_u_loss, vic_aux = self._uncertainty_vicreg_loss(
                gate_u_train, fb_u_train, ent_u_train, is_known)

        total_loss = (self.ce_weight          * ce_loss
                    + self.contrastive_weight  * contrastive_loss
                    + self.alpha_sep * self.separation_weight * separation_loss
                    + self.alpha_sep * self.vicreg_weight     * vicreg_u_loss)

        return total_loss, {
            "ce": float(ce_loss.item()),
            "contrastive": float(contrastive_loss.item()),
            "separation":  float(separation_loss.item()),
            "vicreg_u":    float(vicreg_u_loss.item()),
            "alpha_sep":   float(self.alpha_sep),
            "stage":       int(self.stage),
            **sep_aux, **vic_aux,
            "total":       float(total_loss.item()),
        }