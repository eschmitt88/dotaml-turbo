"""UW-SO loss weighting (Kirchdorfer 2024).

omega_k = softmax(1 / sg[L_k] / T)_k

Where sg[] is stop-gradient so the weight does not participate in autograd
(was a load-bearing UW failure mode per the paper's inertia analysis).

Single tunable temperature T. We make T learnable but clamp to >= 0.1 to
avoid degenerate one-task-dominates collapse.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UWSO(nn.Module):
    def __init__(self, n_tasks: int, T_init: float = 1.0, learnable_T: bool = True,
                 T_min: float = 0.1):
        super().__init__()
        self.n_tasks = int(n_tasks)
        self.T_min = float(T_min)
        if learnable_T:
            self.log_T = nn.Parameter(torch.tensor(float(T_init)).log())
        else:
            self.register_buffer("log_T", torch.tensor(float(T_init)).log())

    @property
    def T(self) -> torch.Tensor:
        return torch.clamp(self.log_T.exp(), min=self.T_min)

    def weights(self, losses: torch.Tensor) -> torch.Tensor:
        """Compute omega per task. losses: [n_tasks] tensor (current batch L_k).

        Returns: [n_tasks] tensor of weights summing to 1. NO grad through
        losses (sg) -- we explicitly detach.
        """
        sg = losses.detach()
        # Defensive: replace any non-finite with the median of finite entries.
        finite_mask = torch.isfinite(sg)
        if not finite_mask.all():
            med = sg[finite_mask].median() if finite_mask.any() else torch.tensor(1.0, device=sg.device)
            sg = torch.where(finite_mask, sg, med)
        # Clamp very small losses up so 1/L_k doesn't explode.
        sg = torch.clamp(sg, min=1e-4)
        logits = (1.0 / sg) / self.T
        return torch.softmax(logits, dim=0)

    def forward(self, losses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (combined_loss_scalar, omega_per_task [n_tasks])."""
        omega = self.weights(losses)
        combined = (omega * losses).sum()
        return combined, omega


__all__ = ["UWSO"]
