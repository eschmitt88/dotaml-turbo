"""PMAE per-column proportional masking (Kim 2024), adapted to our token layout.

PMAE formula (Eq. 11 in the paper):

  M_j(p_obs_j) = clip(a * logit(1 - p_obs_j) + b, 0, 1)
  defaults: a = 0.05, b = 0.5
  where p_obs_j = fraction of rows where column j is observed (non-missing).

In our setting, "columns" are semantic groups:
  - "player_block" -- a whole slot's 8 player features. p_obs_j computed from
    the per-slot is_anonymous proxy (anonymous slots have most features
    artificially zeroed and so are "less observed").
  - "item_list"    -- a whole slot's item list. p_obs_j ~ fraction of rows
    where the slot has >= 1 item in the multi-hot target.
  - "hero_token"   -- a full hero token (hero_id + per-slot features).
    p_obs_j ~ 1.0 (heroes are almost always present), so its mask rate is
    determined by the b intercept after `logit(0)`.
  - "patch_token"  -- the single patch token. p_obs_j = 1.0 (always present
    on the current window); rate set by b.

For the MVP we use a SIMPLER approximation: rather than computing p_obs_j
per group from data statistics, we set per-group base rates by inverting the
PMAE formula at a desired mean rate. The result is a per-group constant
rate clamped to [min_rate, max_rate]. Per-batch we sample random masks at
those rates so 30-50% of semantic units are masked on average.

This is a practical simplification of Kim 2024's per-column proportionality
that preserves the "rare units get more attention" principle for our domain:
in particular the `player_block` mask rate is set higher for anonymous slots
because we know p_obs is empirically lower there.

The PMAE reconstruction loss is computed against the ORIGINAL token's
encoder output at the same position (a teacher-forcing-style "predict the
unmasked encoder state" objective). Concretely:
  - run encoder twice per batch: once unmasked (no_grad teacher), once masked
  - reconstruction loss = SmoothL1 between student-encoded[mask_positions]
    and teacher-encoded[mask_positions]
This is the cheap variant; the alternative (predict the raw column value
back via per-column heads) requires per-column inversion that is hairy in
our mixed-modality token setup. The encoder-state objective is what JEPA
and many recent MAE variants converge on.

`alpha_mae` is annealed externally (in train.py) from 1.0 -> 0.1 over
training so the MAE signal kickstarts representations early but yields
to the supervised heads later.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _pmae_rate(p_obs: float, a: float, b: float, min_rate: float, max_rate: float) -> float:
    """Compute per-group mask rate via the PMAE logit transform."""
    # logit(1 - p_obs); clip p_obs to avoid log(0).
    p_obs_c = max(min(p_obs, 1.0 - 1e-6), 1e-6)
    arg = (1.0 - p_obs_c) / p_obs_c   # logit(1 - p_obs_c) = log((1 - p_obs_c) / p_obs_c)
    import math
    val = a * math.log(arg) + b
    return float(max(min(val, max_rate), min_rate))


class PMAEMasker(nn.Module):
    """Group-wise PMAE masking.

    On forward(hero_ids, player_feats, patch_id, is_anonymous_per_slot), returns:
      hero_mask: [B, 10] bool -- True positions are masked (whole hero token)
      patch_mask: [B] bool   -- True means patch token is masked

    Plus a `mask_stats` dict with the per-group rates actually applied (for
    logging in train.py).
    """

    def __init__(self, a: float = 0.05, b: float = 0.5,
                 min_rate: float = 0.05, max_rate: float = 0.85,
                 groups: list[str] | None = None):
        super().__init__()
        self.a = float(a)
        self.b = float(b)
        self.min_rate = float(min_rate)
        self.max_rate = float(max_rate)
        self.groups = list(groups) if groups else ["player_block", "item_list",
                                                       "hero_token", "patch_token"]
        # Cached p_obs estimates per group (set later via set_p_obs).
        self.p_obs = {
            "player_block": 0.34,    # ~66% of slots anonymous in Turbo -> p_obs ~ 0.34
            "item_list":    0.95,    # most slots have items
            "hero_token":   0.999,
            "patch_token":  0.999,
        }
        self._refresh_rates()

    def _refresh_rates(self) -> None:
        self.rates = {g: _pmae_rate(self.p_obs[g], self.a, self.b,
                                       self.min_rate, self.max_rate)
                      for g in self.groups}

    def set_p_obs(self, **kwargs) -> None:
        """Update per-group p_obs and refresh rates."""
        for g, p in kwargs.items():
            if g in self.p_obs:
                self.p_obs[g] = float(p)
        self._refresh_rates()

    def forward(self, hero_ids: torch.Tensor,
                player_feats: torch.Tensor | None = None,
                patch_id: torch.Tensor | None = None,
                is_anonymous_per_slot: torch.Tensor | None = None) -> dict:
        B = hero_ids.size(0)
        device = hero_ids.device
        out: dict = {}

        # Hero-token mask: whole hero token (which includes per-slot feature
        # + hero_embed). Higher rate on anonymous slots if we know per-slot.
        if "hero_token" in self.groups:
            base_rate = self.rates["hero_token"]
            hero_mask = (torch.rand(B, 10, device=device) < base_rate)
            if is_anonymous_per_slot is not None:
                # Boost rate to player_block rate on anonymous slots.
                anon = is_anonymous_per_slot.bool()
                boost_rate = self.rates.get("player_block", base_rate)
                anon_mask = (torch.rand(B, 10, device=device) < boost_rate)
                hero_mask = torch.where(anon, anon_mask, hero_mask)
            out["hero_mask"] = hero_mask
        else:
            out["hero_mask"] = torch.zeros(B, 10, dtype=torch.bool, device=device)

        # Patch-token mask.
        if "patch_token" in self.groups and patch_id is not None:
            rate = self.rates.get("patch_token", 0.5)
            out["patch_mask"] = (torch.rand(B, device=device) < rate)
        else:
            out["patch_mask"] = torch.zeros(B, dtype=torch.bool, device=device)

        # Per-group rates for logging.
        out["mask_stats"] = {
            g: float(r) for g, r in self.rates.items()
        }
        out["actual_hero_mask_frac"] = float(out["hero_mask"].float().mean().item())
        out["actual_patch_mask_frac"] = float(out["patch_mask"].float().mean().item())
        return out


def pmae_reconstruction_loss(student_encoded: torch.Tensor,
                              teacher_encoded: torch.Tensor,
                              hero_mask: torch.Tensor,
                              patch_mask: torch.Tensor | None = None) -> torch.Tensor:
    """SmoothL1 between student encoder output at masked positions and the
    teacher encoder's output (computed with no_grad, no mask) at the same
    positions.

    student_encoded, teacher_encoded: [B, T, D]
    hero_mask: [B, 10] bool       -- T_hero positions (first 10).
    patch_mask: [B] bool or None  -- True means the patch token (index 10) is masked.

    Returns a scalar loss averaged over the masked positions. If no positions
    are masked, returns a zero with grad-flow preserved.
    """
    n_masked = 0
    loss_sum = student_encoded.sum() * 0.0  # detached-from-data zero, keeps grad-flow

    if hero_mask is not None and hero_mask.any():
        # student/teacher: positions 0..9
        s = student_encoded[:, :10, :]
        t = teacher_encoded[:, :10, :]
        m = hero_mask.unsqueeze(-1)
        diff = (s - t) * m
        abs_diff = diff.abs()
        elem = torch.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
        loss_sum = loss_sum + elem.sum()
        n_masked += int(hero_mask.sum().item()) * s.size(-1)

    if patch_mask is not None and patch_mask.any() and student_encoded.size(1) > 10:
        # patch token is at position 10.
        s = student_encoded[:, 10, :]
        t = teacher_encoded[:, 10, :]
        diff = s - t
        abs_diff = diff.abs()
        elem = torch.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
        # Only count rows where patch_mask is True.
        m = patch_mask.unsqueeze(-1).to(elem.dtype)
        loss_sum = loss_sum + (elem * m).sum()
        n_masked += int(patch_mask.sum().item()) * s.size(-1)

    if n_masked == 0:
        return loss_sum
    return loss_sum / float(n_masked)


__all__ = ["PMAEMasker", "pmae_reconstruction_loss"]
