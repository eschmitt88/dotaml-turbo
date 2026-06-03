"""Matchup probe: does v7 recommend counter-items against specific enemies?

A controlled swap experiment. Hold my hero + my team + most enemies
fixed; swap ONE enemy slot between a TARGET hero (that should demand a
counter-item) and a NEUTRAL filler hero. Measure the change in
P(counter_item | win, dur) for the counter-items. A CONTROL item
(unrelated to the matchup) gives the noise floor.

If v7 learned matchup itemization, the counter-items rise when the
target enemy is present while the control item stays flat.

Hypotheses tested (each: my hero is a natural builder of the counter):
  - Cleave (Battle Fury, Mjollnir) vs illusion heroes (Phantom Lancer,
    Naga Siren).
  - Silence (Orchid, Bloodthorn) vs mobile/escape heroes (Anti-Mage,
    Weaver).
  - Detection (Dust, Sentry, Gem) vs invisible heroes (Riki, Clinkz,
    Bounty Hunter).
  - True-strike (Monkey King Bar) vs evasion (Phantom Assassin).
  - Heal reduction (Spirit Vessel) vs healers (Dazzle, Omniknight).
  - Break (Silver Edge) vs passive-reliant heroes (Phantom Assassin,
    Bristleback).

Run directly:  python serve/test_counter_items.py
Or import:     from serve.test_counter_items import run_all
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

EXP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))

from serve.v7_inference import V7Foundation  # noqa: E402
from serve.queries import _sort_draft_track_my_slot  # noqa: E402
from serve.lookups import (  # noqa: E402
    item_name, hero_name, get_player_features_or_default)


# Item ids
BFURY, MJOLLNIR, MAELSTROM = 145, 158, 166
ORCHID, BLOODTHORN = 98, 250
DUST, SENTRY, GEM = 40, 43, 30
MKB = 135
SPIRIT_VESSEL = 267
SILVER_EDGE = 249
DESOLATOR = 168
BKB = 116
# Control items: matchup-NEUTRAL stat/durability items that a core builds
# regardless of which enemies are present (good noise-floor baselines).
POWER_TREADS, SANGE_YASHA, HEART, ASSAULT = 63, 154, 113, 96
CONTROL_BASKET = [POWER_TREADS, SANGE_YASHA]

# Hero ids
PL, NAGA = 12, 89
AM, WEAVER = 1, 63
RIKI, CLINKZ, BOUNTY = 32, 56, 62
PA, BRISTLE = 44, 99
DAZZLE, OMNI = 50, 57
# Neutral filler enemies: generic durable cores with no single counter-demand.
# Chosen to be disjoint from the test teams below.
FILLERS = [29, 49, 83]  # Tidehunter, Dragon Knight, Treant Protector
# My-hero builders (each naturally builds its hypothesis's counter)
SVEN, JUGG, SNIPER, LINA = 18, 8, 35, 25
TROLL, MEDUSA, STORM, TA, SKYWRATH = 95, 94, 17, 46, 101


@dataclass
class Hypothesis:
    label: str
    my_hero: int
    my_slot: int            # 0..4 (radiant); we always put MY team on radiant
    allies: list[int]       # 4 other radiant heroes
    base_enemies: list[int] # 5 dire heroes (the swap_slot one is replaced)
    swap_slot: int          # which dire slot (0..4) to swap
    target_enemy: int       # the counter-demanding hero
    counter_items: list[int]
    control_items: list[int]
    rationale: str


def _winbag_for_draft(f: V7Foundation, radiant: list[int], dire: list[int],
                       my_slot_radiant: int, account_id: int | None,
                       durations: list[float]) -> np.ndarray:
    """P(X in bag | win, dur=t) for MY slot, over durations. [T, vocab]."""
    heroes = radiant + dire
    accounts = [None] * 10
    accounts[my_slot_radiant] = account_id
    sorted_h, sorted_a, my_sorted = _sort_draft_track_my_slot(heroes, my_slot_radiant, accounts)
    pf = np.stack([get_player_features_or_default(a) for a in sorted_a], axis=0)
    my_team_radiant = (my_sorted < 5)

    T = len(durations)
    inputs = f.empty_inputs(batch_size=T)
    inputs["hero_ids"][:, :] = torch.tensor(sorted_h, dtype=torch.long, device=f.device).unsqueeze(0)
    inputs["player_feats"][:, :, :] = torch.tensor(pf, dtype=torch.float32, device=f.device).unsqueeze(0)
    inputs["dur_log"] = torch.tensor(np.log1p(np.array(durations) * 60.0),
                                      dtype=torch.float32, device=f.device)
    inputs["win_idx"][:] = 1 if my_team_radiant else 0
    masks = f.pure_pregame_mask(batch_size=T)
    masks["win"] = torch.zeros((T,), dtype=torch.bool, device=f.device)
    masks["duration"] = torch.zeros((T,), dtype=torch.bool, device=f.device)
    out = f.predict(inputs=inputs, masks=masks)
    return out.item_probs().cpu().numpy()[:, my_sorted, :], my_sorted


def counter_effect(f: V7Foundation, h: Hypothesis,
                    durations: list[float] | None = None,
                    account_id: int | None = None) -> dict:
    """Measure delta P(item | win) when target enemy is present vs a panel
    of neutral fillers, averaged over durations.

    Returns {item_id: {'with_target': p, 'filler_mean': p, 'delta': d}} for
    every counter + control item, plus the swapped hero names.
    """
    if durations is None:
        durations = [15.0, 25.0, 35.0]
    radiant = [h.my_hero if i == h.my_slot else h.allies[i if i < h.my_slot else i - 1]
               for i in range(5)]
    # Build radiant explicitly: my_hero at my_slot, allies fill the rest
    radiant = list(h.allies)
    radiant.insert(h.my_slot, h.my_hero)
    assert len(radiant) == 5

    all_items = h.counter_items + h.control_items
    vidx = {iid: f.item_vocab.get(str(iid)) for iid in all_items}

    def probs_for_dire(dire_team):
        wb, _ = _winbag_for_draft(f, radiant, dire_team, h.my_slot, account_id, durations)
        return {iid: float(np.mean(wb[:, vidx[iid]])) if vidx[iid] is not None else float("nan")
                for iid in all_items}

    # With target in swap_slot
    dire_target = list(h.base_enemies)
    dire_target[h.swap_slot] = h.target_enemy
    with_target = probs_for_dire(dire_target)

    # Panel of fillers in swap_slot (exclude any filler already on a team)
    used = set(radiant) | set(h.base_enemies)
    filler_panel = [fl for fl in FILLERS if fl not in used and fl != h.target_enemy][:3]
    if not filler_panel:
        filler_panel = [fl for fl in FILLERS if fl != h.target_enemy][:1]
    filler_runs = []
    for fl in filler_panel:
        dire_fl = list(h.base_enemies)
        dire_fl[h.swap_slot] = fl
        filler_runs.append(probs_for_dire(dire_fl))
    filler_mean = {iid: float(np.mean([fr[iid] for fr in filler_runs])) for iid in all_items}

    result = {}
    for iid in all_items:
        wt, fm = with_target[iid], filler_mean[iid]
        result[iid] = {
            "with_target": wt,
            "filler_mean": fm,
            "delta": wt - fm,
            "ratio": (wt / fm) if fm > 1e-6 else float("inf"),
            "is_counter": iid in h.counter_items,
        }
    return {"items": result, "filler_panel": filler_panel,
            "target": h.target_enemy, "swap_slot": h.swap_slot}


# ----- The hypothesis suite -----


def hypotheses() -> list[Hypothesis]:
    # Shared support-ish ally set (Lina, Lich, Dazzle, Pudge) to fill radiant
    # around my core. Enemies avoid the FILLERS (29/49/83).
    ALLIES = [25, 31, 50, 14]            # Lina, Lich, Dazzle, Pudge
    ENEMY_BASE = [35, 13, 26, 5, 78]     # Sniper, Puck, Lion, CM, Brewmaster
    CTRL = [POWER_TREADS, SANGE_YASHA]
    return [
        Hypothesis(
            label="Cleave vs Phantom Lancer (illusions)",
            my_hero=JUGG, my_slot=0, allies=ALLIES,  # Jugg builds Bfury/Mjollnir
            base_enemies=ENEMY_BASE, swap_slot=4, target_enemy=PL,
            counter_items=[BFURY, MJOLLNIR, MAELSTROM, RADIANCE := 137],
            control_items=CTRL,
            rationale="cleave / chain-lightning / Radiance burn clears PL illusions"),
        Hypothesis(
            label="Silence vs Anti-Mage (blink escape)",
            my_hero=TA, my_slot=0, allies=ALLIES,    # TA builds Orchid/Bloodthorn
            base_enemies=ENEMY_BASE, swap_slot=4, target_enemy=AM,
            counter_items=[ORCHID, BLOODTHORN],
            control_items=CTRL,
            rationale="silence prevents AM Blink/Manta escape"),
        Hypothesis(
            label="Silence/hex vs Weaver (Time Lapse escape)",
            my_hero=TA, my_slot=0, allies=ALLIES,
            base_enemies=ENEMY_BASE, swap_slot=4, target_enemy=WEAVER,
            counter_items=[ORCHID, BLOODTHORN],
            control_items=CTRL,
            rationale="silence stops Weaver's Time Lapse / Shukuchi escape"),
        Hypothesis(
            label="Detection vs Riki (permanent invis)",
            my_hero=LINA, my_slot=0, allies=[18, 31, 50, 14],  # Sven core ally
            base_enemies=ENEMY_BASE, swap_slot=4, target_enemy=RIKI,
            counter_items=[DUST, SENTRY, GEM],
            control_items=CTRL,
            rationale="detection reveals Riki's invisibility"),
        Hypothesis(
            label="Detection vs Clinkz (invis archer)",
            my_hero=LINA, my_slot=0, allies=[18, 31, 50, 14],
            base_enemies=ENEMY_BASE, swap_slot=4, target_enemy=CLINKZ,
            counter_items=[DUST, SENTRY, GEM],
            control_items=CTRL,
            rationale="detection reveals Clinkz's Skeleton Walk"),
        Hypothesis(
            label="MKB vs Phantom Assassin (evasion)",
            my_hero=SNIPER, my_slot=0, allies=ALLIES,
            base_enemies=ENEMY_BASE, swap_slot=4, target_enemy=PA,
            counter_items=[MKB],
            control_items=CTRL,
            rationale="MKB true-strike pierces PA's Blur evasion"),
        Hypothesis(
            label="Break (Silver Edge) vs PA (passive evasion+crit)",
            my_hero=JUGG, my_slot=0, allies=ALLIES,
            base_enemies=ENEMY_BASE, swap_slot=4, target_enemy=PA,
            counter_items=[SILVER_EDGE],
            control_items=CTRL,
            rationale="Silver Edge break disables PA's Blur + Coup de Grace"),
    ]


def run_all(f: V7Foundation | None = None, account_id: int | None = None) -> list[dict]:
    if f is None:
        f = V7Foundation()
    results = []
    print("=" * 78)
    print("MATCHUP PROBE: does v7 recommend counter-items vs specific enemies?")
    print("delta = P(item|win) with target enemy - mean over neutral fillers")
    print("=" * 78)
    n_pass = 0
    for h in hypotheses():
        res = counter_effect(f, h, account_id=account_id)
        items = res["items"]
        fillers = ", ".join(hero_name(x) for x in res["filler_panel"])
        print(f"\n### {h.label}")
        print(f"    my hero: {hero_name(h.my_hero)}  | target: {hero_name(h.target_enemy)}  "
              f"| fillers: {fillers}")
        print(f"    ({h.rationale})")
        print(f"      {'':8} {'item':<22} {'with':>8} {'filler':>8} {'delta':>8} {'ratio':>6}")
        best_ratio, best_delta = 0.0, -1.0
        for iid in h.counter_items:
            d = items[iid]
            best_ratio = max(best_ratio, d["ratio"])
            best_delta = max(best_delta, d["delta"])
            print(f"      COUNTER  {item_name(iid):<22} {d['with_target']:>8.4f} "
                  f"{d['filler_mean']:>8.4f} {d['delta']:>+8.4f} {d['ratio']:>5.2f}x")
        ctrl_deltas = []
        for iid in h.control_items:
            d = items[iid]
            ctrl_deltas.append(abs(d["delta"]))
            print(f"      control  {item_name(iid):<22} {d['with_target']:>8.4f} "
                  f"{d['filler_mean']:>8.4f} {d['delta']:>+8.4f} {d['ratio']:>5.2f}x")
        # Pass via EITHER lens (ratio is better for low-base items, scale-aware
        # delta is better for high-base items where ratios compress):
        #   - ratio route:  best counter lift >= 1.3x
        #   - delta route:  best counter delta >= 3x the control noise floor
        #                   AND a non-trivial absolute lift (>= 0.01)
        ctrl_floor = max(ctrl_deltas)
        pass_ratio = best_ratio >= 1.3
        pass_delta = (best_delta >= 3.0 * ctrl_floor) and (best_delta >= 0.01)
        passed = pass_ratio or pass_delta
        n_pass += int(passed)
        route = "ratio" if pass_ratio else ("delta" if pass_delta else "-")
        verdict = f"PASS ({route})" if passed else "no signal"
        print(f"    => best counter {best_ratio:.2f}x / {best_delta:+.4f} delta  "
              f"vs control noise +-{ctrl_floor:.4f}  [{verdict}]")
        results.append({"hypothesis": h.label, "passed": passed,
                         "best_ratio": best_ratio, "best_delta": best_delta,
                         "control_floor": ctrl_floor})
    print("\n" + "=" * 78)
    print(f"SUMMARY: {n_pass}/{len(results)} hypotheses show counter-item lift "
          f"above the control noise floor.")
    print("=" * 78)
    return results


if __name__ == "__main__":
    run_all()
