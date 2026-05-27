"""ProbeSuite for v7-unified-masked-multitask-740.

Runs every 2 epochs on a fixed 50k-row val subset (seed=42 for
selection). All probes run with the model in eval mode, no grad,
bf16 autocast.

Each probe maps a per-scenario inference path to a single scalar
metric. The metrics feed back into ScenarioSampler.update_probs for
adaptive sampling, and are checked against per-probe halt thresholds
at epoch 10.

Probes (9 total):
  pure_pregame_probe       -> val_auc on win head with pure_pregame masking
  partial_draft_probe      -> top-5 hero rec accuracy (1 hero slot masked,
                                model predicts via item_head on hero idx
                                substitute -- here we approximate via the
                                hero_embed dot-product with the encoded
                                masked slot; the top-5 set is among the
                                top-5 highest cosine scores)
  duration_cond_probe      -> val_auc on win head with TRUE duration as input
  items_cond_probe         -> val_auc on win head with TRUE items as input
  outcome_cond_probe       -> item mAP@10 with TRUE win + heroes + player_feats
  partial_items_probe      -> item BCE on partial-items scenario val rows
  kills_pair_probe_probe   -> kills MAE on kills_pair_probe scenario val rows
  gpm_probe                -> gpm MAE for full-input val
  hd_probe                 -> hd log1p MAE for full-input val

Halt thresholds are read from config; at epoch >= halt_at_epoch, if
ANY probe is below its halt threshold, train.py halts and writes the
diagnostic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from mae import PER_SLOT_GROUPS, PER_MATCH_GROUPS  # noqa: F401


def _bool_zero(B: int, T: int, device) -> torch.Tensor:
    return torch.zeros(B, T, dtype=torch.bool, device=device)


def _build_masks(B: int, device, spec: dict) -> dict:
    """spec: dict of group_name -> 'mask' or 'unmask' (default unmask)."""
    out: dict = {}
    for g in PER_SLOT_GROUPS:
        if spec.get(g) == "mask":
            out[g] = torch.ones(B, 10, dtype=torch.bool, device=device)
        else:
            out[g] = torch.zeros(B, 10, dtype=torch.bool, device=device)
    for g in PER_MATCH_GROUPS:
        if spec.get(g) == "mask":
            out[g] = torch.ones(B, dtype=torch.bool, device=device)
        else:
            out[g] = torch.zeros(B, dtype=torch.bool, device=device)
    return out


@dataclass
class ProbeResult:
    name: str
    value: float
    halt_at_value: float
    halt_below: bool   # True = halt when value <= halt_at_value
                         # False = halt when value >= halt_at_value


class ProbeSuite:
    def __init__(self, val_ds, device, autocast_dtype,
                 fixed_subset_size: int = 50_000, seed: int = 42,
                 batch_size: int = 1024,
                 halt_thresholds: dict | None = None):
        self.val_ds = val_ds
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.batch_size = int(batch_size)
        n = len(val_ds)
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=min(fixed_subset_size, n), replace=False)
        self.subset_idx = np.sort(idx)
        self.subset = Subset(val_ds, self.subset_idx.tolist())
        self.loader = DataLoader(self.subset, batch_size=self.batch_size,
                                  shuffle=False, num_workers=0, pin_memory=False)
        self.halt_thresholds = halt_thresholds or {}

    def _autocast(self):
        return torch.autocast(device_type=self.device.type,
                                dtype=self.autocast_dtype,
                                enabled=self.autocast_dtype is not None)

    def _forward_with_mask_spec(self, model, mask_spec: dict) -> dict:
        """Walks the fixed subset, builds masks per the spec, returns dict
        of np arrays for the predictions.
        """
        model.eval()
        all_win, all_y_win = [], []
        all_item_logits, all_item_targets = [], []
        all_kills, all_y_kills = [], []
        all_gpm, all_y_gpm = [], []
        all_hd, all_y_hd = [], []
        all_hero_ids = []
        all_encoded = []
        with torch.no_grad():
            for batch in self.loader:
                (hero_ids, pf, _patch_id, _acct, items,
                 kills, deaths, assists, gpm, hd,
                 dur_log, y_win) = batch
                B = hero_ids.size(0)
                device = self.device
                hero_ids = hero_ids.to(device)
                pf = pf.to(device)
                items = items.to(device)
                kills = kills.to(device); deaths = deaths.to(device); assists = assists.to(device)
                gpm = gpm.to(device); hd = hd.to(device)
                dur_log = dur_log.to(device); y_win = y_win.to(device)
                win_idx = y_win.long()
                masks = _build_masks(B, device, mask_spec)
                with self._autocast():
                    out = model(hero_ids, pf, items, kills, deaths, assists,
                                 gpm, hd, dur_log, win_idx, masks=masks)
                all_win.append(torch.sigmoid(out["win"].float()).cpu().numpy())
                all_y_win.append(y_win.cpu().numpy())
                all_item_logits.append(out["item"].float().cpu().numpy())
                all_item_targets.append(items.cpu().numpy())
                all_kills.append(out["kills"].float().cpu().numpy())
                all_y_kills.append(kills.cpu().numpy())
                all_gpm.append(out["gpm"].float().cpu().numpy())
                all_y_gpm.append(gpm.cpu().numpy())
                all_hd.append(out["hd"].float().cpu().numpy())
                all_y_hd.append(hd.cpu().numpy())
                all_hero_ids.append(hero_ids.cpu().numpy())
                all_encoded.append(out["encoded"].float().cpu().numpy())
        return {
            "win_p":     np.concatenate(all_win),
            "win_y":     np.concatenate(all_y_win),
            "item_p":    np.concatenate(all_item_logits, axis=0),
            "item_y":    np.concatenate(all_item_targets, axis=0),
            "kills_p":   np.concatenate(all_kills, axis=0),
            "kills_y":   np.concatenate(all_y_kills, axis=0),
            "gpm_p":     np.concatenate(all_gpm, axis=0),
            "gpm_y":     np.concatenate(all_y_gpm, axis=0),
            "hd_p":      np.concatenate(all_hd, axis=0),
            "hd_y":      np.concatenate(all_y_hd, axis=0),
            "hero_ids":  np.concatenate(all_hero_ids, axis=0),
            "encoded":   np.concatenate(all_encoded, axis=0),
        }

    # ----- Per-probe metric calcs -----

    @staticmethod
    def _val_auc(win_y: np.ndarray, win_p: np.ndarray) -> float:
        try:
            return float(roc_auc_score(win_y, win_p))
        except ValueError:
            return float("nan")

    @staticmethod
    def _item_map_at_k(targets: np.ndarray, logits: np.ndarray, k: int = 10) -> float:
        # targets/logits: [N, 10, V]
        T = targets.reshape(-1, targets.shape[-1])
        L = logits.reshape(-1, logits.shape[-1])
        if T.size == 0:
            return float("nan")
        V = T.shape[1]
        k = min(k, V)
        top_idx = np.argpartition(-L, kth=k - 1, axis=1)[:, :k]
        sort_order = np.argsort(-np.take_along_axis(L, top_idx, axis=1), axis=1)
        top_idx = np.take_along_axis(top_idx, sort_order, axis=1)
        hits = np.take_along_axis(T, top_idx, axis=1).astype(np.float32)
        cum = np.cumsum(hits, axis=1)
        ranks = np.arange(1, k + 1, dtype=np.float32)
        prec_at_i = cum / ranks
        n_pos = T.sum(axis=1)
        denom = np.where(n_pos > 0, np.minimum(n_pos, k), 1.0)
        ap = (prec_at_i * hits).sum(axis=1) / denom
        return float(ap.mean())

    @staticmethod
    def _item_bce(targets: np.ndarray, logits: np.ndarray) -> float:
        T = targets.reshape(-1, targets.shape[-1]).astype(np.float64)
        L = logits.reshape(-1, logits.shape[-1]).astype(np.float64)
        if T.size == 0:
            return float("nan")
        # numerically stable BCE: log(1+exp(-|L|)) + max(L,0) - L*T
        m = np.maximum(L, 0.0)
        bce = m - L * T + np.log1p(np.exp(-np.abs(L)))
        return float(bce.mean())

    @staticmethod
    def _mae(pred: np.ndarray, target: np.ndarray) -> float:
        if pred.size == 0:
            return float("nan")
        return float(np.mean(np.abs(pred - target)))

    def _hero_topk_acc(self, model, k: int = 5) -> float:
        """Probe: mask 1 hero slot per row; check if true hero is in the
        top-k cosine-similarity scores between the model's hero embedding
        table and the encoded slot.
        """
        model.eval()
        device = self.device
        hits = 0
        total = 0
        # Hero embedding -> d_model space.
        with torch.no_grad():
            W = model.hero_embed.weight                                        # [V, embed_dim]
            W_d = model.hero_proj(W) if not isinstance(model.hero_proj, torch.nn.Identity) else W
            W_norm = F.normalize(W_d.float(), dim=-1)                          # [V, d_model]
            for batch in self.loader:
                (hero_ids, pf, _patch_id, _acct, items,
                 kills, deaths, assists, gpm, hd,
                 dur_log, y_win) = batch
                B = hero_ids.size(0)
                hero_ids = hero_ids.to(device)
                pf = pf.to(device)
                items = items.to(device)
                kills = kills.to(device); deaths = deaths.to(device); assists = assists.to(device)
                gpm = gpm.to(device); hd = hd.to(device)
                dur_log = dur_log.to(device); y_win = y_win.to(device)
                win_idx = y_win.long()
                # Mask exactly one hero slot per row (deterministic for stability).
                slot_pick = torch.arange(B, device=device) % 10
                hero_mask = torch.zeros(B, 10, dtype=torch.bool, device=device)
                hero_mask[torch.arange(B, device=device), slot_pick] = True
                masks = _build_masks(B, device,
                                       {"items": "mask", "kills": "mask", "deaths": "mask",
                                        "assists": "mask", "gpm": "mask", "hd": "mask",
                                        "duration": "mask", "win": "mask"})
                masks["hero"] = hero_mask
                with self._autocast():
                    enc, _ = model.encode(hero_ids, pf, items, kills, deaths, assists,
                                            gpm, hd, dur_log, win_idx, masks)
                enc = enc.float()                                              # [B, 12, d]
                # Pick the masked slot embedding.
                picked = enc[torch.arange(B, device=device), slot_pick, :]     # [B, d]
                picked_norm = F.normalize(picked, dim=-1)
                sims = picked_norm @ W_norm.T                                  # [B, V]
                topk = sims.topk(k=k, dim=-1).indices                          # [B, k]
                true_h = hero_ids[torch.arange(B, device=device), slot_pick]   # [B]
                hits += int((topk == true_h.unsqueeze(-1)).any(dim=-1).sum().item())
                total += B
        return hits / max(total, 1)

    # ----- The full suite -----

    def run(self, model) -> dict[str, float]:
        # 1. pure_pregame: items, k, d, a, gpm, hd, dur, win masked.
        out = self._forward_with_mask_spec(model, {
            "items": "mask", "kills": "mask", "deaths": "mask", "assists": "mask",
            "gpm": "mask", "hd": "mask", "duration": "mask", "win": "mask"})
        pure_pregame = self._val_auc(out["win_y"], out["win_p"])

        # 2. partial_draft: top-5 hero rec.
        partial_draft = self._hero_topk_acc(model, k=5)

        # 3. duration_cond: items, k, d, a, gpm, hd, win masked.
        out = self._forward_with_mask_spec(model, {
            "items": "mask", "kills": "mask", "deaths": "mask", "assists": "mask",
            "gpm": "mask", "hd": "mask", "win": "mask"})
        duration_cond = self._val_auc(out["win_y"], out["win_p"])

        # 4. items_cond: k, d, a, gpm, hd, dur, win masked.
        out = self._forward_with_mask_spec(model, {
            "kills": "mask", "deaths": "mask", "assists": "mask",
            "gpm": "mask", "hd": "mask", "duration": "mask", "win": "mask"})
        items_cond = self._val_auc(out["win_y"], out["win_p"])

        # 5. outcome_cond: items, k, d, a, gpm, hd, dur masked (win visible).
        out = self._forward_with_mask_spec(model, {
            "items": "mask", "kills": "mask", "deaths": "mask", "assists": "mask",
            "gpm": "mask", "hd": "mask", "duration": "mask"})
        outcome_cond = self._item_map_at_k(out["item_y"], out["item_p"], k=10)

        # 6. partial_items: heroes + pf visible; some items masked + post-game
        # masked. We measure item BCE on the FULL targets (so the model has
        # to predict items it didn't see). Use items=mask at scenario level.
        out = self._forward_with_mask_spec(model, {
            "items": "mask", "kills": "mask", "deaths": "mask", "assists": "mask",
            "gpm": "mask", "hd": "mask", "duration": "mask", "win": "mask"})
        partial_items = self._item_bce(out["item_y"], out["item_p"])

        # 7. kills_pair_probe: most heroes masked, post-game masked.
        # NOTE: heads predict in LOG1P space (loss is on log1p targets to
        # keep multi-task scales commensurate). MAE is measured in log1p
        # space; convert to raw by exp1m at query time. Halt thresholds
        # in config.yaml are in log1p units accordingly.
        out = self._forward_with_mask_spec(model, {
            "items": "mask", "kills": "mask", "deaths": "mask", "assists": "mask",
            "gpm": "mask", "hd": "mask", "duration": "mask", "win": "mask"})
        kills_pair_probe = self._mae(out["kills_p"], np.log1p(out["kills_y"]))

        # 8. gpm_probe: full input, measure gpm MAE in log1p space.
        out_full = self._forward_with_mask_spec(model, {})
        gpm_probe = self._mae(out_full["gpm_p"], np.log1p(out_full["gpm_y"]))

        # 9. hd_probe: full input, measure hd MAE in log1p space.
        hd_probe = self._mae(out_full["hd_p"], np.log1p(out_full["hd_y"]))

        results = {
            "pure_pregame":     float(pure_pregame),
            "partial_draft":    float(partial_draft),
            "duration_cond":    float(duration_cond),
            "items_cond":       float(items_cond),
            "outcome_cond":     float(outcome_cond),
            "partial_items":    float(partial_items),
            "kills_pair_probe": float(kills_pair_probe),
            "gpm_probe":        float(gpm_probe),
            "hd_probe":         float(hd_probe),
        }
        return results

    def halt_decision(self, results: dict, epoch: int, halt_at_epoch: int = 10) -> dict:
        """Returns dict with 'halt': bool, 'reasons': list[str]."""
        if epoch < halt_at_epoch:
            return {"halt": False, "reasons": []}
        reasons = []
        for name, val in results.items():
            spec = self.halt_thresholds.get(name)
            if spec is None:
                continue
            thresh = float(spec["value"])
            direction = spec.get("direction", "below")
            if direction == "below":
                if val <= thresh:
                    reasons.append(f"{name}={val:.4f} <= halt_threshold={thresh:.4f}")
            elif direction == "above":
                if val >= thresh:
                    reasons.append(f"{name}={val:.4f} >= halt_threshold={thresh:.4f}")
        return {"halt": bool(reasons), "reasons": reasons}


__all__ = ["ProbeSuite", "ProbeResult"]
