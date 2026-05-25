"""PMAE per-column proportional masking (Kim 2024), adapted to our token layout.

# Bug-fix relative to foundation-mvp-740 (Bug A diagnosis)

In foundation-mvp-740, mae_loss collapsed to 0 mid-training. Root cause:
the teacher and student passes shared the SAME weights (the call was
`model(...)` for both passes). As the model trained, it learned the
trivial collapse: make the encoder output at masked positions become
invariant to whether the input was masked. Specifically, if the encoder
propagates information only from un-masked neighbors, student_encoded
[mask_pos] == teacher_encoded[mask_pos] regardless of the mask --
SmoothL1 collapses to 0 without learning anything useful. This is the
classic BYOL/JEPA representational-collapse failure mode and the
established fix is an EMA-updated teacher with stop-gradient.

# Fix applied here

`EMATeacher` wraps a deep-copy of the model and updates its weights via
exponential moving average each step (momentum=0.99 default). The teacher
is .eval()'d and the parameters have requires_grad=False, so no gradient
flows back through the teacher pass.

Additionally we add a `pmae_reconstruction_loss_logged` variant that
returns the loss AND the mask_count / mask_fraction so train.py can
log per-epoch diagnostics confirming the masking actually fires and
the loss is non-degenerate.

Original PMAE rate formula and group definitions retained below
verbatim from foundation-mvp-740.

PMAE formula (Eq. 11 in the paper):

  M_j(p_obs_j) = clip(a * logit(1 - p_obs_j) + b, 0, 1)
  defaults: a = 0.05, b = 0.5
  where p_obs_j = fraction of rows where column j is observed (non-missing).

`alpha_mae` is annealed externally (in train.py) from 1.0 -> 0.1 over
training so the MAE signal kickstarts representations early but yields
to the supervised heads later.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


def _pmae_rate(p_obs: float, a: float, b: float, min_rate: float, max_rate: float) -> float:
    """Compute per-group mask rate via the PMAE logit transform."""
    p_obs_c = max(min(p_obs, 1.0 - 1e-6), 1e-6)
    arg = (1.0 - p_obs_c) / p_obs_c
    val = a * math.log(arg) + b
    return float(max(min(val, max_rate), min_rate))


class PMAEMasker(nn.Module):
    """Group-wise PMAE masking.

    Identical to foundation-mvp-740 PMAEMasker -- the bug was not here.
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
        self.p_obs = {
            "player_block": 0.34,
            "item_list":    0.95,
            "hero_token":   0.999,
            "patch_token":  0.999,
        }
        self._refresh_rates()

    def _refresh_rates(self) -> None:
        self.rates = {g: _pmae_rate(self.p_obs[g], self.a, self.b,
                                       self.min_rate, self.max_rate)
                      for g in self.groups}

    def set_p_obs(self, **kwargs) -> None:
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

        if "hero_token" in self.groups:
            base_rate = self.rates["hero_token"]
            hero_mask = (torch.rand(B, 10, device=device) < base_rate)
            if is_anonymous_per_slot is not None:
                anon = is_anonymous_per_slot.bool()
                boost_rate = self.rates.get("player_block", base_rate)
                anon_mask = (torch.rand(B, 10, device=device) < boost_rate)
                hero_mask = torch.where(anon, anon_mask, hero_mask)
            out["hero_mask"] = hero_mask
        else:
            out["hero_mask"] = torch.zeros(B, 10, dtype=torch.bool, device=device)

        if "patch_token" in self.groups and patch_id is not None:
            rate = self.rates.get("patch_token", 0.5)
            out["patch_mask"] = (torch.rand(B, device=device) < rate)
        else:
            out["patch_mask"] = torch.zeros(B, dtype=torch.bool, device=device)

        out["mask_stats"] = {g: float(r) for g, r in self.rates.items()}
        out["actual_hero_mask_frac"] = float(out["hero_mask"].float().mean().item())
        out["actual_patch_mask_frac"] = float(out["patch_mask"].float().mean().item())
        return out


class EMATeacher(nn.Module):
    """EMA-tracked teacher copy of `model`. Stop-gradient on all parameters.

    Forward signature mirrors the wrapped model exactly so it can be called
    in place of `model(...)`. The teacher receives `hero_mask=None,
    patch_mask=None` -- it always sees the un-masked input. The student
    (the actual model) sees the masked input. Reconstruction objective:
    student_encoded[mask_pos] -> teacher_encoded[mask_pos]. Because teacher
    weights LAG student weights, the student cannot collapse to "ignore the
    mask" by drifting both encoders in tandem; the teacher's lagged-target
    forces non-trivial reconstruction.

    Update rule:
        teacher_param = m * teacher_param + (1 - m) * student_param

    Default m=0.996 (BYOL default). Should be very close to 1 so the
    teacher is meaningfully lagged; m=0.99..0.999 are all reasonable.
    """

    def __init__(self, model: nn.Module, momentum: float = 0.996):
        super().__init__()
        self.momentum = float(momentum)
        # Deep-copy on the same device.
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        m = self.momentum
        for tp, sp in zip(self.teacher.parameters(), student.parameters()):
            tp.data.mul_(m).add_(sp.data, alpha=1.0 - m)
        # Buffers (batchnorm running stats etc.) -- copy outright.
        for tb, sb in zip(self.teacher.buffers(), student.buffers()):
            tb.data.copy_(sb.data)

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> dict:
        return self.teacher(*args, **kwargs)


def pmae_reconstruction_loss(student_encoded: torch.Tensor,
                              teacher_encoded: torch.Tensor,
                              hero_mask: torch.Tensor,
                              patch_mask: torch.Tensor | None = None) -> torch.Tensor:
    """SmoothL1 between student encoder output at masked positions and the
    teacher encoder's output at the same positions.

    Returns a scalar loss averaged over the masked positions (over the elem
    count B*T*D-ish). If no positions are masked, returns a zero with
    grad-flow preserved.
    """
    n_masked = 0
    loss_sum = student_encoded.sum() * 0.0

    if hero_mask is not None and hero_mask.any():
        s = student_encoded[:, :10, :]
        t = teacher_encoded[:, :10, :]
        m = hero_mask.unsqueeze(-1)
        diff = (s - t) * m
        abs_diff = diff.abs()
        elem = torch.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
        loss_sum = loss_sum + elem.sum()
        n_masked += int(hero_mask.sum().item()) * s.size(-1)

    if patch_mask is not None and patch_mask.any() and student_encoded.size(1) > 10:
        s = student_encoded[:, 10, :]
        t = teacher_encoded[:, 10, :]
        diff = s - t
        abs_diff = diff.abs()
        elem = torch.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
        m = patch_mask.unsqueeze(-1).to(elem.dtype)
        loss_sum = loss_sum + (elem * m).sum()
        n_masked += int(patch_mask.sum().item()) * s.size(-1)

    if n_masked == 0:
        return loss_sum
    return loss_sum / float(n_masked)


def pmae_reconstruction_loss_logged(student_encoded: torch.Tensor,
                                      teacher_encoded: torch.Tensor,
                                      hero_mask: torch.Tensor,
                                      patch_mask: torch.Tensor | None = None) -> dict:
    """Same as pmae_reconstruction_loss but returns extra logging fields.

    Returns dict {loss, hero_mask_count, hero_mask_frac, patch_mask_count,
                  patch_mask_frac, target_l2_mean, target_l2_std,
                  student_l2_mean, student_l2_std}.

    The L2-mean fields let us detect representational collapse: if both
    teacher and student outputs at masked positions have L2 norm -> 0
    (or both saturate to the same constant), the loss is degenerate.
    """
    out: dict = {}
    loss = pmae_reconstruction_loss(student_encoded, teacher_encoded,
                                       hero_mask, patch_mask)
    out["loss"] = loss
    out["hero_mask_count"] = (int(hero_mask.sum().item()) if hero_mask is not None
                                 else 0)
    out["hero_mask_frac"] = (float(hero_mask.float().mean().item()) if hero_mask is not None
                                 else 0.0)
    out["patch_mask_count"] = (int(patch_mask.sum().item()) if patch_mask is not None
                                  else 0)
    out["patch_mask_frac"] = (float(patch_mask.float().mean().item()) if patch_mask is not None
                                  else 0.0)
    # Health diagnostics on the hero-position slice.
    if hero_mask is not None and hero_mask.any():
        with torch.no_grad():
            s = student_encoded[:, :10, :]
            t = teacher_encoded[:, :10, :]
            m = hero_mask.unsqueeze(-1).to(s.dtype)
            denom = max(int(hero_mask.sum().item()), 1)
            s_l2 = ((s * m).pow(2).sum() / denom).sqrt()
            t_l2 = ((t * m).pow(2).sum() / denom).sqrt()
            out["student_l2_mean_at_mask"] = float(s_l2.item())
            out["teacher_l2_mean_at_mask"] = float(t_l2.item())
    else:
        out["student_l2_mean_at_mask"] = 0.0
        out["teacher_l2_mean_at_mask"] = 0.0
    return out


__all__ = ["PMAEMasker", "EMATeacher", "pmae_reconstruction_loss",
           "pmae_reconstruction_loss_logged"]
