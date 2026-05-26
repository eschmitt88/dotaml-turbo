"""JEPA-style 6-group masking + EMA teacher for v6-jepa-pretrain-finetune-740.

Forked from experiments/2026-05-26-v5-pretrain-finetune-740/mae.py.

Single conceptual change vs v5:
- v5 used BERT-style raw-target reconstruction (6 per-group losses
  against the raw input values at masked positions).
- v6 uses JEPA-style latent-space prediction: the student encodes the
  masked input, the EMA teacher encodes the UN-masked input, and the
  loss is SmoothL1 between a small predictor MLP applied to the
  student's per-slot reps and the teacher's per-slot reps at the
  positions whose tokens were affected by masking.

What is preserved verbatim from v5:
- `SixGroupMasker` — independent per-group masking with p_group=0.4,
  groups list unchanged.
- `EMATeacherV6` (renamed from V5) — deep-copy + momentum=0.996 +
  per-step EMA update + stop-gradient.

What's new:
- `slot_mask_from_groups(mask_dict)` — collapses per-group [B] booleans
  into a per-slot [B, 10] boolean mask. A slot is "masked" if ANY of
  its 6 groups is masked for that example. Since groups player_block,
  hero_token, item_list, kda, gpm, hd are masked WHOLE-EXAMPLE in v5
  (per-row [B] bool, not per-slot [B, 10]), the resulting per-slot mask
  ends up uniform across all 10 slots within an example whenever any
  group is masked. Concretely: slot_mask[b, :] = ANY(mask_dict[g][b] for g).
  This means each row contributes either ALL 10 slots or 0 slots to the
  JEPA loss. With p_group=0.4 across 6 groups, the expected fraction of
  rows with at least one group masked is 1 - (0.6)^6 = 0.953, so ~95%
  of rows contribute.

  Alternative considered: per-slot mask = mask_dict["hero_token"] OR
  any-group. Rejected because the v5 masking is fundamentally per-row,
  not per-slot, and we want all 10 slots in a masked example to
  receive gradient (the encoder's job is to reconstruct latent content
  at those positions from the OTHER groups in the same row).

- `jepa_loss(predictor, student_reps, teacher_reps, slot_mask)` — single
  scalar SmoothL1 loss, averaged over masked slots.

- `representation_diagnostics(reps)` — per-slot L2-norm and pairwise
  cosine similarity diagnostics for collapse detection. Returns dict
  with mean L2-norm, std L2-norm, mean off-diagonal cosine across slots
  in a batch, and a `collapse_warning` bool.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


PRETRAIN_GROUPS = ["player_block", "hero_token", "item_list", "kda", "gpm", "hd"]


class SixGroupMasker(nn.Module):
    """Generate per-example per-group bool masks (identical to v5).

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


class EMATeacherV6(nn.Module):
    """EMA-tracked teacher copy of `model`. Identical pattern to v5.

    Teacher params are NOT in the optimizer; updated only via .update().
    Teacher is in eval() mode, requires_grad_(False).
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


def slot_mask_from_groups(mask_dict: dict, batch_size: int,
                            device: torch.device) -> torch.Tensor:
    """Collapse per-group [B] booleans into a per-slot [B, 10] bool mask.

    Semantics: slot_mask[b, s] = True iff ANY group is masked for row b.
    Because masking is per-row (each group's mask is [B], not [B, 10]),
    the per-slot mask is uniform within a row: all 10 slots receive the
    same value. We still return [B, 10] so the JEPA loss can index
    naturally into the per-slot teacher/student reps.
    """
    if not mask_dict:
        return torch.zeros(batch_size, 10, dtype=torch.bool, device=device)
    accum = None
    for g, m in mask_dict.items():
        if m is None:
            continue
        mb = m.to(device).bool()
        accum = mb if accum is None else (accum | mb)
    if accum is None:
        return torch.zeros(batch_size, 10, dtype=torch.bool, device=device)
    return accum.view(batch_size, 1).expand(batch_size, 10).contiguous()


def jepa_loss(predictor: nn.Module, student_reps: torch.Tensor,
                teacher_reps: torch.Tensor, slot_mask: torch.Tensor
                ) -> tuple[torch.Tensor, int]:
    """SmoothL1 between predictor(student_reps) and teacher_reps at masked slots.

    student_reps, teacher_reps: [B, 10, d_model] -- per-slot latent reps.
    slot_mask: [B, 10] bool -- True where the loss applies.

    Returns (scalar_loss, n_masked_slots).
    """
    B, T, D = student_reps.shape
    n_mask = int(slot_mask.sum().item())
    if n_mask == 0:
        # Touch predictor so the autograd graph isn't dead.
        return (predictor(student_reps).sum() * 0.0), 0
    # Apply predictor to ALL student slots (cheap) and select masked.
    pred = predictor(student_reps)                # [B, 10, D]
    # Gather only the masked positions for both sides.
    mask_flat = slot_mask.view(B * T)
    pred_flat = pred.view(B * T, D)
    teach_flat = teacher_reps.detach().view(B * T, D)
    pred_sel = pred_flat[mask_flat]                # [n_mask, D]
    teach_sel = teach_flat[mask_flat]              # [n_mask, D]
    loss = F.smooth_l1_loss(pred_sel.float(), teach_sel.float(), reduction="mean")
    return loss, n_mask


@torch.no_grad()
def representation_diagnostics(reps: torch.Tensor,
                                 collapse_cos_threshold: float = 0.95,
                                 collapse_l2_threshold: float = 1e-3,
                                 ) -> dict:
    """Per-slot rep diagnostics for collapse detection.

    reps: [B, 10, d_model] -- typically the STUDENT's encoded output.

    Returns dict:
      - rep_l2_mean, rep_l2_std: per-slot L2 norms aggregated across
          batch + slots.
      - pairwise_cos_mean: mean off-diagonal cosine similarity across
          the 10 slots within each example, averaged over the batch.
      - collapse_warning: bool — True if rep_l2_mean < collapse_l2_threshold
          OR pairwise_cos_mean > collapse_cos_threshold.
    """
    reps_f = reps.float()
    B, T, D = reps_f.shape
    # L2 norm per slot.
    l2 = torch.linalg.vector_norm(reps_f, dim=-1)  # [B, T]
    rep_l2_mean = float(l2.mean().item())
    rep_l2_std = float(l2.std().item())

    # Pairwise cosine across the 10 slots per example.
    eps = 1e-9
    normed = reps_f / (l2.unsqueeze(-1) + eps)     # [B, T, D]
    # cos[b, i, j] = <normed[b,i], normed[b,j]>
    cos = torch.matmul(normed, normed.transpose(1, 2))  # [B, T, T]
    # Off-diagonal entries.
    eye = torch.eye(T, dtype=torch.bool, device=cos.device).unsqueeze(0)
    off = cos.masked_fill(eye, float("nan"))
    valid = ~torch.isnan(off)
    pairwise_cos_mean = float((off.masked_fill(~valid, 0.0).sum() / valid.sum()).item())

    collapse_warning = bool(
        (rep_l2_mean < collapse_l2_threshold)
        or (pairwise_cos_mean > collapse_cos_threshold)
    )
    return {
        "rep_l2_mean": rep_l2_mean,
        "rep_l2_std": rep_l2_std,
        "pairwise_cos_mean": pairwise_cos_mean,
        "collapse_warning": collapse_warning,
    }


class JEPAPredictor(nn.Module):
    """Small predictor MLP applied to student per-slot reps before
    matching against teacher reps.

    Architecture: Linear -> GELU -> Linear, d_model in/out, no LN.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = [
    "SixGroupMasker",
    "EMATeacherV6",
    "JEPAPredictor",
    "PRETRAIN_GROUPS",
    "slot_mask_from_groups",
    "jepa_loss",
    "representation_diagnostics",
]
