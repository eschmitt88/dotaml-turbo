"""PMAE-style 6-group masking + EMA teacher for v5-pretrain-finetune-740.

Forked from experiments/2026-05-23-foundation-component-isolation-740/mae.py.

Key differences vs the iso_pmae implementation:

1. **Six groups** (player_block, hero_token, item_list, kda, gpm, hd)
   are masked at the WHOLE-EXAMPLE level (per group, per row), with
   independent per-group probabilities `p_group` (default 0.4).

2. **Mask tokens, not zeros**: masking is implemented in models.py by
   REPLACING the per-group contribution with a learned mask-token
   vector (NOT zero, NOT mean-imputed). This file only generates the
   per-example per-group boolean mask dict.

3. **EMA teacher**: identical pattern to iso_pmae — deep-copy the
   student, freeze grads, EMA-update each step (momentum=0.996).
   Teacher sees the FULLY-OBSERVED (no mask) input; student sees the
   masked input. Reconstruction objective combines per-group prediction
   losses (player SmoothL1, hero CE, item BCE, kda/gpm/hd SmoothL1)
   computed at the masked positions ONLY, against the original raw
   inputs (NOT the teacher's encoded representation — the v5 recipe
   uses raw-target reconstruction, mirroring BERT MLM. The EMA teacher
   is retained for the alignment loss as an OPTIONAL stabilizer; the
   primary loss is raw-target.).

   Specifically: total_loss_pretrain = sum_g w_g * L_g(pred_g, raw_g)
   where pred_g comes from the student's per-group pretrain head and
   raw_g comes from the unmasked input. The EMA teacher is kept around
   for future BYOL-style alignment but its forward output is not
   consumed in this implementation — collapse is prevented by the
   raw-target masking objective itself (the student can't trivially
   match raw_g unless the encoder propagates information from the
   un-masked groups).

   Why keep the EMA teacher at all? For HCE-safe alignment in case
   later experiments want representational distillation, and for future
   variants. The EMA update is cheap (~1ms per step).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


PRETRAIN_GROUPS = ["player_block", "hero_token", "item_list", "kda", "gpm", "hd"]


class SixGroupMasker(nn.Module):
    """Generate per-example per-group bool masks.

    Each row independently has each group masked with probability
    `p_group`. Returns a dict: {group_name: bool tensor of shape [B]}.
    """

    def __init__(self, p_group: float = 0.4, groups: list[str] | None = None):
        super().__init__()
        self.p_group = float(p_group)
        self.groups = list(groups) if groups else PRETRAIN_GROUPS

    def forward(self, batch_size: int, device: torch.device) -> dict:
        out: dict = {}
        for g in self.groups:
            out[g] = (torch.rand(batch_size, device=device) < self.p_group)
        return out

    def mask_rates(self, mask_dict: dict) -> dict:
        return {g: float(mask_dict[g].float().mean().item()) for g in mask_dict}


class EMATeacherV5(nn.Module):
    """EMA-tracked teacher copy of `model`. Identical pattern to iso_pmae's
    EMATeacher. Forward is a no-op pass-through to the wrapped student
    forward signature (uses forward_pretrain by default; callers may
    invoke .teacher directly for other forward kinds).
    """

    def __init__(self, model: nn.Module, momentum: float = 0.996):
        super().__init__()
        self.momentum = float(momentum)
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        m = self.momentum
        for tp, sp in zip(self.teacher.parameters(), student.parameters()):
            tp.data.mul_(m).add_(sp.data, alpha=1.0 - m)
        for tb, sb in zip(self.teacher.buffers(), student.buffers()):
            tb.data.copy_(sb.data)


def _smooth_l1_at_mask(pred: torch.Tensor, target: torch.Tensor,
                         mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """SmoothL1 averaged over MASKED rows only.

    pred, target: same shape [B, ...].
    mask: [B] bool — True rows contribute.

    Returns (scalar_loss, n_mask_rows).
    """
    n_mask = int(mask.sum().item())
    if n_mask == 0:
        return pred.sum() * 0.0, 0
    # Broadcast mask to pred shape.
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(-1)
    diff = (pred - target).float()
    ad = diff.abs()
    elem = torch.where(ad < 1.0, 0.5 * diff * diff, ad - 0.5)
    elem = elem * mask.to(elem.dtype)
    # Per-row reduce; we average over masked rows AND over per-row element count.
    n_per_row = elem.numel() / pred.size(0)
    return elem.sum() / float(n_mask * n_per_row), n_mask


def _ce_at_mask(logits: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Cross-entropy averaged over MASKED rows only.

    logits: [B, T, C]; target: [B, T]; mask: [B] bool.
    Computes CE per (B, T) cell and averages over the 10 per-row cells
    of masked rows.
    """
    n_mask = int(mask.sum().item())
    if n_mask == 0:
        return logits.sum() * 0.0, 0
    B, T, C = logits.shape
    flat_logits = logits.reshape(B * T, C).float()
    flat_target = target.reshape(B * T)
    loss_per = F.cross_entropy(flat_logits, flat_target, reduction="none").reshape(B, T)
    mask_row = mask.to(loss_per.dtype).unsqueeze(-1)   # [B, 1]
    masked = loss_per * mask_row
    return masked.sum() / float(n_mask * T), n_mask


def _bce_at_mask(logits: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Per-class BCE averaged over MASKED rows only.

    logits, target: [B, T, V].
    """
    n_mask = int(mask.sum().item())
    if n_mask == 0:
        return logits.sum() * 0.0, 0
    B, T, V = logits.shape
    per_elem = F.binary_cross_entropy_with_logits(logits.float(), target.float(),
                                                    reduction="none")
    mask_row = mask.to(per_elem.dtype).view(B, 1, 1)
    per_elem = per_elem * mask_row
    return per_elem.sum() / float(n_mask * T * V), n_mask


def per_group_reconstruction_losses(pred: dict, target_inputs: dict,
                                      mask_dict: dict) -> dict:
    """Compute per-group reconstruction losses over MASKED rows only.

    pred: dict from model.forward_pretrain — keys:
      'pred_player' [B,10,F_pf], 'pred_hero' [B,10,V_hero],
      'pred_item' [B,10,V_item], 'pred_kda/gpm/hd' [B,10].
    target_inputs: dict with keys:
      'player_block' [B,10,F_pf], 'hero_token' [B,10] long,
      'item_list' [B,10,V_item], 'kda' [B,10], 'gpm' [B,10], 'hd' [B,10].
    mask_dict: dict[group_name] = [B] bool.

    Returns dict of {group: scalar_loss_tensor, '_n_mask': dict counts}.
    """
    out: dict = {}
    counts: dict = {}

    l_pb, n_pb = _smooth_l1_at_mask(pred["pred_player"],
                                       target_inputs["player_block"],
                                       mask_dict["player_block"])
    out["player_block"] = l_pb; counts["player_block"] = n_pb

    l_h, n_h = _ce_at_mask(pred["pred_hero"], target_inputs["hero_token"],
                             mask_dict["hero_token"])
    out["hero_token"] = l_h; counts["hero_token"] = n_h

    l_it, n_it = _bce_at_mask(pred["pred_item"], target_inputs["item_list"],
                                 mask_dict["item_list"])
    out["item_list"] = l_it; counts["item_list"] = n_it

    l_kda, n_kda = _smooth_l1_at_mask(pred["pred_kda"], target_inputs["kda"],
                                          mask_dict["kda"])
    out["kda"] = l_kda; counts["kda"] = n_kda

    l_gpm, n_gpm = _smooth_l1_at_mask(pred["pred_gpm"], target_inputs["gpm"],
                                          mask_dict["gpm"])
    out["gpm"] = l_gpm; counts["gpm"] = n_gpm

    l_hd, n_hd = _smooth_l1_at_mask(pred["pred_hd"], target_inputs["hd"],
                                        mask_dict["hd"])
    out["hd"] = l_hd; counts["hd"] = n_hd

    out["_n_mask"] = counts
    return out


__all__ = ["SixGroupMasker", "EMATeacherV5", "per_group_reconstruction_losses",
           "PRETRAIN_GROUPS"]
