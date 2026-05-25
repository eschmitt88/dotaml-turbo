"""UW-SO loss weighting (Kirchdorfer 2024) WITH per-task initial-loss normalization.

# Bug-fix relative to foundation-mvp-740 (Bug B diagnosis)

In foundation-mvp-740's `foundation_mvp` ablation, temperature T converged
to ~0.45. Combined with raw per-task loss magnitudes spanning ~30x
(items per-class BCE ~0.07; duration CE ~2.1; win BCE ~0.69), the
softmax(1 / sg[L_k] / T) formula over-weighted low-magnitude tasks by
~30x, drowning the win head's gradient. Observed symptom: train_win
loss INCREASED monotonically across epochs 1-5 (0.6947 -> 0.7020),
i.e. the model actively anti-learned the primary task. (Detail: the
softmax of 1/L over tasks with L ~ {0.69, 2.1, 0.07, ..} gives the
items task ~14x the weight of win.)

# Fix applied here

Normalize each per-task loss by a per-task initial-loss snapshot
L_k_init BEFORE feeding to the UW-SO softmax. With this, all tasks
start at L_normalized ~= 1.0 and the softmax sees comparable magnitudes
regardless of task units. The actual weighted training loss is computed
on the UN-normalized losses with the omega weights, so the final loss
magnitude remains task-faithful for backprop.

L_k_init is computed as the running mean of L_k over the first
`init_window_batches` steps of training (default 100). Before
init_window_batches is reached, omega defaults to uniform-1/n.

# Original UW-SO formula

omega_k = softmax(1 / sg[L_k] / T)_k

Where sg[] is stop-gradient (was a load-bearing UW failure mode per the
paper's inertia analysis).

Single tunable temperature T. We make T learnable but clamp to >= 0.1 to
avoid degenerate one-task-dominates collapse.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UWSO(nn.Module):
    def __init__(self, n_tasks: int, T_init: float = 1.0, learnable_T: bool = True,
                 T_min: float = 0.1, init_window_batches: int = 100):
        super().__init__()
        self.n_tasks = int(n_tasks)
        self.T_min = float(T_min)
        self.init_window_batches = int(init_window_batches)
        if learnable_T:
            self.log_T = nn.Parameter(torch.tensor(float(T_init)).log())
        else:
            self.register_buffer("log_T", torch.tensor(float(T_init)).log())
        # Per-task initial-loss tracker. We accumulate sum + count over the
        # first init_window_batches batches, then freeze L_k_init.
        self.register_buffer("init_sum", torch.zeros(n_tasks))
        self.register_buffer("init_n", torch.zeros(()))
        self.register_buffer("L_k_init", torch.ones(n_tasks))
        self.register_buffer("init_frozen", torch.zeros((), dtype=torch.bool))

    @property
    def T(self) -> torch.Tensor:
        return torch.clamp(self.log_T.exp(), min=self.T_min)

    def _update_init(self, losses: torch.Tensor) -> None:
        if bool(self.init_frozen.item()):
            return
        with torch.no_grad():
            sg = losses.detach()
            # Skip non-finite contributions in the init window.
            finite = torch.isfinite(sg)
            sg_clean = torch.where(finite, sg, torch.zeros_like(sg))
            self.init_sum += sg_clean
            # Only count batches with all-finite losses toward init_n.
            if finite.all():
                self.init_n += 1.0
            if int(self.init_n.item()) >= self.init_window_batches:
                # Freeze. Clamp to a minimum so we never divide by ~0.
                avg = self.init_sum / self.init_n
                avg = torch.clamp(avg, min=1e-4)
                self.L_k_init.copy_(avg)
                self.init_frozen.fill_(True)

    def weights(self, losses: torch.Tensor) -> torch.Tensor:
        """Compute omega per task. losses: [n_tasks] tensor (current batch L_k).

        If not yet frozen, uses uniform weights. After freeze, computes
            L_normalized = sg[L_k] / L_k_init
            omega = softmax(1 / L_normalized / T)
        """
        if not bool(self.init_frozen.item()):
            return torch.full((self.n_tasks,), 1.0 / self.n_tasks,
                                device=losses.device, dtype=losses.dtype)
        sg = losses.detach()
        finite_mask = torch.isfinite(sg)
        if not finite_mask.all():
            med = sg[finite_mask].median() if finite_mask.any() else torch.tensor(1.0, device=sg.device)
            sg = torch.where(finite_mask, sg, med)
        sg_norm = sg / self.L_k_init.to(sg.dtype)
        sg_norm = torch.clamp(sg_norm, min=1e-4)
        logits = (1.0 / sg_norm) / self.T
        return torch.softmax(logits, dim=0)

    def forward(self, losses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (combined_loss_scalar, omega_per_task [n_tasks]).

        Updates the L_k_init running stats on each call until frozen.
        Combined loss uses UN-normalized per-task losses weighted by omega
        (so the gradient scale matches the original task losses, only the
        relative balance is set by omega).
        """
        self._update_init(losses)
        omega = self.weights(losses)
        combined = (omega * losses).sum()
        return combined, omega


__all__ = ["UWSO"]
