"""ScenarioSampler for v7-unified-masked-multitask-740.

Each batch picks ONE scenario from a categorical distribution. The
scenario determines the per-group mask pattern; the same pattern
applies to every example in the batch (this keeps per-scenario probe
metrics clean and the loss accounting separable).

Initial scenario distribution + per-head loss weights are defined in
config.yaml (`scenarios:`). After every probe-suite pass, train.py
calls `update_probs` with the probe results; this implements the
adaptive sampling rule described in the proposal.

apply_mask returns a dict of bool tensors per group:
  hero, player_feat, items, kills, deaths, assists, gpm, hd  -> [B, 10]
  duration, win                                                -> [B]
True means "masked" (input replaced by learned mask embedding).
For win specifically, the `outcome_cond` scenario UNmasks win at the
true value — this is encoded as `win=False` in the mask dict (i.e.
the true win_idx flows through the model).

Compatible with the FoundationTransformerV7 mask-dict signature.
"""
from __future__ import annotations

import torch


# Canonical list of group keys (matches mask dict used by models.py).
PER_SLOT_GROUPS = ["hero", "player_feat", "items", "kills", "deaths", "assists", "gpm", "hd"]
PER_MATCH_GROUPS = ["duration", "win"]
ALL_GROUPS = PER_SLOT_GROUPS + PER_MATCH_GROUPS

# Per-head names used in loss weighting. NOTE: player_feat is not a head; the
# heads we supervise on are these 8.
HEAD_NAMES = ["win", "dur", "items", "kills", "deaths", "assists", "gpm", "hd"]


def _empty_mask_dict(B: int, device) -> dict:
    """All-False mask dict (nothing masked)."""
    out: dict = {}
    for g in PER_SLOT_GROUPS:
        out[g] = torch.zeros(B, 10, dtype=torch.bool, device=device)
    for g in PER_MATCH_GROUPS:
        out[g] = torch.zeros(B, dtype=torch.bool, device=device)
    return out


def _mask_all(B: int, device, slot_groups: list[str], match_groups: list[str]) -> dict:
    """Mask the listed groups fully; leave the rest visible."""
    out = _empty_mask_dict(B, device)
    for g in slot_groups:
        out[g] = torch.ones(B, 10, dtype=torch.bool, device=device)
    for g in match_groups:
        out[g] = torch.ones(B, dtype=torch.bool, device=device)
    return out


class ScenarioSampler:
    """Per-batch scenario sampler with adaptive sampling-probability updates.

    initial_probs: dict {scenario_name: float} -- must sum to 1.0.
    loss_weights: dict {scenario_name: {head_name: float}} -- per-head weight
        emphasis for each scenario; missing heads default to 1.0.
    initial_targets: dict {scenario_name: float} -- the per-scenario probe
        target used by update_probs. Same units as the per-scenario probe
        value passed to update_probs (typically val_auc or accuracy).
    """

    SCENARIOS = (
        "everything_visible",
        "pure_pregame",
        "partial_draft",
        "partial_items",
        "duration_cond",
        "items_cond",
        "outcome_cond",
        "kills_pair_probe",
        "random_uniform",
    )

    def __init__(self, initial_probs: dict[str, float],
                 loss_weights: dict[str, dict[str, float]],
                 initial_targets: dict[str, float] | None = None,
                 seed: int = 42):
        for s in self.SCENARIOS:
            if s not in initial_probs:
                raise ValueError(f"missing scenario in initial_probs: {s}")
        self.initial_probs = dict(initial_probs)
        self.probs = dict(initial_probs)
        self.loss_weights_per_scenario = {
            s: {h: float(w.get(h, 1.0)) for h in HEAD_NAMES}
            for s, w in loss_weights.items()
        }
        self.targets = dict(initial_targets or {})
        self.history: list = []
        # Pure-python rng (avoid torch RNG state coupling).
        import random as _rd
        self._rd = _rd.Random(seed)

    def sample_batch_scenario(self) -> str:
        items = list(self.probs.items())
        names = [it[0] for it in items]
        ps = [it[1] for it in items]
        # Normalize defensively.
        s = sum(ps)
        if s <= 0:
            return "pure_pregame"
        ps = [p / s for p in ps]
        x = self._rd.random()
        acc = 0.0
        for n, p in zip(names, ps):
            acc += p
            if x <= acc:
                return n
        return names[-1]

    def loss_weights(self, scenario: str) -> dict[str, float]:
        return self.loss_weights_per_scenario.get(
            scenario, {h: 1.0 for h in HEAD_NAMES}
        )

    def apply_mask(self, B: int, device, scenario: str,
                    win_idx: torch.Tensor | None = None) -> dict:
        """Build mask dict for the chosen scenario.

        win_idx is only consulted by `outcome_cond` (which keeps the win
        input visible). For other scenarios it is unused.
        """
        s = scenario
        if s == "everything_visible":
            return _empty_mask_dict(B, device)
        if s == "pure_pregame":
            return _mask_all(B, device,
                              slot_groups=["items", "kills", "deaths", "assists", "gpm", "hd"],
                              match_groups=["duration", "win"])
        if s == "partial_draft":
            # Pick K in [1,5] hero slots per row to mask; mask ALL post-game.
            out = _mask_all(B, device,
                             slot_groups=["items", "kills", "deaths", "assists", "gpm", "hd"],
                             match_groups=["duration", "win"])
            hero_m = torch.zeros(B, 10, dtype=torch.bool, device=device)
            for i in range(B):
                k = self._rd.randint(1, 5)
                slots = self._rd.sample(range(10), k=k)
                for sl in slots:
                    hero_m[i, sl] = True
            out["hero"] = hero_m
            return out
        if s == "partial_items":
            # Mask 1-3 items per slot's bag + ALL post-game (except items).
            # "Mask 1-3 items per bag" is implemented as masking the items
            # GROUP entirely for ~30% of slots per row -- the simplest
            # within-batch granularity given mask is per-group not per-item.
            # The probe for this scenario uses a separate per-item mask
            # in probes.py; the training signal here is still useful (the
            # model has to predict items_set with limited per-slot info).
            out = _mask_all(B, device,
                             slot_groups=["kills", "deaths", "assists", "gpm", "hd"],
                             match_groups=["duration", "win"])
            items_m = torch.zeros(B, 10, dtype=torch.bool, device=device)
            for i in range(B):
                n_mask = self._rd.randint(1, 3)
                slots = self._rd.sample(range(10), k=n_mask)
                for sl in slots:
                    items_m[i, sl] = True
            out["items"] = items_m
            return out
        if s == "duration_cond":
            # Items, k, d, a, gpm, hd, win masked; duration is INPUT (un-masked).
            return _mask_all(B, device,
                              slot_groups=["items", "kills", "deaths", "assists", "gpm", "hd"],
                              match_groups=["win"])
        if s == "items_cond":
            # K, d, a, gpm, hd, dur, win masked; items as INPUT.
            return _mask_all(B, device,
                              slot_groups=["kills", "deaths", "assists", "gpm", "hd"],
                              match_groups=["duration", "win"])
        if s == "outcome_cond":
            # Items, k, d, a, gpm, hd, dur masked; win UNmasked at true value.
            return _mask_all(B, device,
                              slot_groups=["items", "kills", "deaths", "assists", "gpm", "hd"],
                              match_groups=["duration"])
        if s == "kills_pair_probe":
            # Mask all heroes EXCEPT 1-2 ally pair; mask all post-game.
            out = _mask_all(B, device,
                             slot_groups=["items", "kills", "deaths", "assists", "gpm", "hd"],
                             match_groups=["duration", "win"])
            hero_m = torch.ones(B, 10, dtype=torch.bool, device=device)
            for i in range(B):
                # Pick an ally team uniformly; pick a 2-slot pair within that team.
                team = self._rd.randint(0, 1)
                base = 0 if team == 0 else 5
                pair = self._rd.sample(range(base, base + 5), k=2)
                for sl in pair:
                    hero_m[i, sl] = False
            out["hero"] = hero_m
            return out
        if s == "random_uniform":
            # Each per-slot group independently masked at Beta(2,4) rate per group;
            # match groups masked iid.
            from random import betavariate
            out = _empty_mask_dict(B, device)
            for g in PER_SLOT_GROUPS:
                r = betavariate(2, 4)
                out[g] = (torch.rand(B, 10, device=device) < r)
            for g in PER_MATCH_GROUPS:
                r = betavariate(2, 4)
                out[g] = (torch.rand(B, device=device) < r)
            return out
        raise ValueError(f"unknown scenario {s!r}")

    # ----- Adaptive sampling update -----

    def update_probs(self, probe_results: dict[str, float],
                      gap_pp_threshold: float = 0.02,
                      up_mul: float = 1.2, down_mul: float = 0.95,
                      cap_mul: float = 2.0, floor_mul: float = 0.5) -> dict:
        """Update sampling probs based on per-scenario probe shortfall.

        probe_results: {scenario_name: probe_value}. Compared to self.targets.
        gap = target - probe_value; positive gap means below target.

        Rule:
          gap > +threshold -> probs[s] *= up_mul
          gap < -threshold -> probs[s] *= down_mul
        Capped at [floor * initial, cap * initial], then renormalized.
        """
        new_probs = dict(self.probs)
        deltas = {}
        for s, p in new_probs.items():
            tgt = self.targets.get(s)
            val = probe_results.get(s)
            if tgt is None or val is None:
                deltas[s] = {"gap": None, "mul": 1.0}
                continue
            gap = float(tgt) - float(val)
            mul = 1.0
            if gap > gap_pp_threshold:
                mul = up_mul
            elif gap < -gap_pp_threshold:
                mul = down_mul
            init = self.initial_probs[s]
            new_p = max(init * floor_mul, min(p * mul, init * cap_mul))
            new_probs[s] = new_p
            deltas[s] = {"gap": gap, "mul": mul, "before": p, "after": new_p}
        # Renormalize.
        total = sum(new_probs.values())
        if total > 0:
            for s in new_probs:
                new_probs[s] = new_probs[s] / total
        self.probs = new_probs
        snapshot = {"probs_after": dict(new_probs), "deltas": deltas,
                     "probe_results": dict(probe_results)}
        self.history.append(snapshot)
        return snapshot


__all__ = ["ScenarioSampler", "PER_SLOT_GROUPS", "PER_MATCH_GROUPS", "ALL_GROUPS",
           "HEAD_NAMES"]
