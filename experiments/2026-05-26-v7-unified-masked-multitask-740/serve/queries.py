"""Concrete downstream-query functions on top of v7.

All functions take a `V7Foundation` instance (the loaded model) and
return python-native data structures (lists, dicts, dataclasses) so the
notebook + CLI can consume them directly.

Queries implemented:
- personal_winprob: P(radiant_win | draft, player_feats)
- hero_pick_rec: top-K heroes for an open slot in a partial draft
- item_rec_for_winprob: top-K items to add (sorted by marginal win lift)
- item_rec_given_win: top-K items predicted in your final bag, given win=1
- win_vs_duration: P(radiant_win) as a function of game duration
- kills_per_minute_pair: predicted kills/min for a 1-5 hero subset
- lineup_matchup: P(radiant_win) for two given lineups (alias for
  personal_winprob with explicit teams)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .lookups import (
    ANON_ACCOUNT_IDS, get_player_features_or_default, hero_name,
    item_cost, item_id_to_info, item_name, sample_unknown_heroes,
)
from .v7_inference import V7Foundation, canonical_hero_sort


# ----- Result dataclasses -----


@dataclass
class HeroPickRec:
    hero_id: int
    hero_name: str
    mean_winprob: float
    n_samples: int


@dataclass
class ItemRec:
    vocab_idx: int
    item_id: int
    item_name: str
    score: float
    cost: int


@dataclass
class WinDurationPoint:
    duration_minutes: float
    win_prob: float


@dataclass
class KillsPerMinResult:
    hero_subset: list[int]
    kills_per_min: float
    predicted_duration_min: float
    predicted_total_kills: float
    predicted_assists: float


# ----- Helpers -----


def _build_inputs_for_draft(f: V7Foundation,
                              heroes: list[int],
                              account_ids: list[int | None] | None = None) -> torch.Tensor:
    """Build inputs dict for a single match with given heroes + per-slot
    account-id-derived player features. Applies canonical hero sort.

    Returns (inputs_dict, sorted_heroes, sorted_account_ids).
    """
    assert len(heroes) == 10
    if account_ids is None:
        account_ids = [None] * 10
    assert len(account_ids) == 10

    # Per-team canonical sort: sort each team's (hero, account_id) by hero_id
    r_pairs = sorted(zip(heroes[:5], account_ids[:5]), key=lambda p: p[0])
    d_pairs = sorted(zip(heroes[5:], account_ids[5:]), key=lambda p: p[0])
    sorted_heroes = [p[0] for p in r_pairs] + [p[0] for p in d_pairs]
    sorted_accts  = [p[1] for p in r_pairs] + [p[1] for p in d_pairs]

    # Per-slot player features
    pf = np.stack([get_player_features_or_default(a) for a in sorted_accts], axis=0)
    assert pf.shape == (10, 8)

    inputs = f.empty_inputs(batch_size=1)
    inputs["hero_ids"][0, :] = torch.tensor(sorted_heroes, dtype=torch.long, device=f.device)
    inputs["player_feats"][0, :, :] = torch.tensor(pf, dtype=torch.float32, device=f.device)
    return inputs, sorted_heroes, sorted_accts


# ----- Queries -----


def personal_winprob(f: V7Foundation,
                       heroes: list[int],
                       account_ids: list[int | None] | None = None) -> float:
    """P(radiant_win) for a 10-hero draft with optional per-slot accounts.

    heroes:       [r0, r1, r2, r3, r4, d0, d1, d2, d3, d4]
    account_ids:  same length; None for anonymous / unknown.

    Returns a single float win probability in [0, 1].
    """
    inputs, _h, _a = _build_inputs_for_draft(f, heroes, account_ids)
    masks = f.pure_pregame_mask(batch_size=1)
    out = f.predict(inputs=inputs, masks=masks)
    return float(out.win_prob()[0].cpu())


def lineup_matchup(f: V7Foundation,
                     radiant: list[int], dire: list[int],
                     radiant_accounts: list[int | None] | None = None,
                     dire_accounts: list[int | None] | None = None) -> dict:
    """Alias for personal_winprob with explicit teams. Returns a richer
    result dict including duration estimate."""
    assert len(radiant) == 5 and len(dire) == 5
    heroes = radiant + dire
    accts = (radiant_accounts or [None] * 5) + (dire_accounts or [None] * 5)
    inputs, sorted_h, _ = _build_inputs_for_draft(f, heroes, accts)
    masks = f.pure_pregame_mask(batch_size=1)
    out = f.predict(inputs=inputs, masks=masks)
    return {
        "radiant_win_prob": float(out.win_prob()[0].cpu()),
        "predicted_duration_sec": float(out.dur_seconds()[0].cpu()),
        "sorted_heroes": [hero_name(h) for h in sorted_h],
    }


def hero_pick_rec(f: V7Foundation,
                    known_radiant: list[int],
                    known_dire: list[int],
                    my_side: str,
                    account_id: int | None = None,
                    top_k: int = 10,
                    candidate_heroes: list[int] | None = None) -> list[HeroPickRec]:
    """Recommend top-K heroes for ME to pick, using v7's trained hero-mask
    token for unknown enemy/ally slots.

    known_radiant: heroes locked on radiant (excluding mine — 0 to 4 of them).
    known_dire:    heroes locked on dire (0 to 5 of them).
    my_side: 'radiant' or 'dire' (which team I'm on).
    account_id: my account ID (None = anonymous defaults).
    candidate_heroes: which hero IDs to consider; default = all 150 heroes
                     minus already-locked ones.

    Returns a list of HeroPickRec sorted descending by mean_winprob.

    Mechanism: for each candidate hero in my open slot, the unknown
    enemy/ally hero slots are filled with v7's learned `hero_mask_embed`
    token. v7 was trained on the partial_draft scenario (1-5 random hero
    slots masked + all post-game masked) — by end of training this
    scenario had the HIGHEST adaptive sampling probability (0.293),
    meaning the model saw a lot of partial drafts and the win head
    learned to predict P(win | partial draft + features). Single forward
    pass per candidate; total ~150 candidates batched ≈ 0.5s.

    Player_feats for unknown slots default to ANON_FEATS (~66% of Turbo
    slots are anonymous; this matches the training distribution).
    n_samples=1 in the returned dataclass (no longer a sampling estimator).
    """
    assert my_side in ("radiant", "dire")
    assert len(known_radiant) <= 4 if my_side == "radiant" else len(known_radiant) <= 5
    assert len(known_dire)    <= 4 if my_side == "dire"    else len(known_dire)    <= 5

    locked = set(known_radiant + known_dire)
    if candidate_heroes is None:
        candidate_heroes = [hid for hid in range(1, 151) if hid not in locked]
    else:
        candidate_heroes = [hid for hid in candidate_heroes if hid not in locked]

    n_unknown_radiant = 5 - len(known_radiant) - (1 if my_side == "radiant" else 0)
    n_unknown_dire    = 5 - len(known_dire)    - (1 if my_side == "dire"    else 0)

    from .v7_inference import ANON_FEATS
    my_feats = get_player_features_or_default(account_id)

    N = len(candidate_heroes)
    hero_ids_np = np.zeros((N, 10), dtype=np.int64)
    hero_mask_np = np.zeros((N, 10), dtype=bool)
    player_feats_np = np.tile(ANON_FEATS, (N, 10, 1)).astype(np.float32)

    for ci, cand in enumerate(candidate_heroes):
        if my_side == "radiant":
            # Pre-sort radiant_5: known heroes + candidate + masked placeholders (hero_id=0)
            radiant_5 = list(known_radiant) + [cand] + [0] * n_unknown_radiant
            radiant_mask = [False] * len(known_radiant) + [False] + [True] * n_unknown_radiant
            dire_5 = list(known_dire) + [0] * n_unknown_dire
            dire_mask = [False] * len(known_dire) + [True] * n_unknown_dire
            my_orig_slot_in_team = len(known_radiant)  # my slot index in radiant_5
        else:
            radiant_5 = list(known_radiant) + [0] * n_unknown_radiant
            radiant_mask = [False] * len(known_radiant) + [True] * n_unknown_radiant
            dire_5 = list(known_dire) + [cand] + [0] * n_unknown_dire
            dire_mask = [False] * len(known_dire) + [False] + [True] * n_unknown_dire
            my_orig_slot_in_team = len(known_dire)

        # Canonical sort by (is_masked, hero_id) — keeps masked slots together
        # at end of each team (deterministic order for masked positions)
        r_argsort = sorted(range(5), key=lambda i: (radiant_mask[i], radiant_5[i]))
        d_argsort = sorted(range(5), key=lambda i: (dire_mask[i], dire_5[i]))

        sorted_r = [radiant_5[i] for i in r_argsort]
        sorted_r_mask = [radiant_mask[i] for i in r_argsort]
        sorted_d = [dire_5[i] for i in d_argsort]
        sorted_d_mask = [dire_mask[i] for i in d_argsort]

        hero_ids_np[ci, :5] = sorted_r
        hero_ids_np[ci, 5:] = sorted_d
        hero_mask_np[ci, :5] = sorted_r_mask
        hero_mask_np[ci, 5:] = sorted_d_mask

        # Place my player features at my new slot position
        if my_side == "radiant":
            new_my_slot = r_argsort.index(my_orig_slot_in_team)
            player_feats_np[ci, new_my_slot, :] = my_feats
        else:
            new_my_slot = 5 + d_argsort.index(my_orig_slot_in_team)
            player_feats_np[ci, new_my_slot, :] = my_feats

    # ONE GPU transfer + ONE forward pass
    inputs = f.empty_inputs(batch_size=N)
    inputs["hero_ids"]    = torch.from_numpy(hero_ids_np).to(f.device, non_blocking=True)
    inputs["player_feats"] = torch.from_numpy(player_feats_np).to(f.device, non_blocking=True)
    masks = f.pure_pregame_mask(batch_size=N)
    masks["hero"] = torch.from_numpy(hero_mask_np).to(f.device, non_blocking=True)

    out = f.predict(inputs=inputs, masks=masks)
    winp = out.win_prob().cpu().numpy()
    my_winp = winp if my_side == "radiant" else (1.0 - winp)

    recs = [HeroPickRec(hero_id=hid, hero_name=hero_name(hid),
                          mean_winprob=float(p), n_samples=1)
            for hid, p in zip(candidate_heroes, my_winp)]
    recs.sort(key=lambda r: -r.mean_winprob)
    return recs[:top_k]


def item_rec_marginal_sweep(f: V7Foundation,
                              heroes: list[int],
                              my_slot: int,
                              current_bag: list[int] | None = None,
                              account_ids: list[int | None] | None = None,
                              gold_budget: int | None = None,
                              top_k: int = 10,
                              min_cost: int = 200) -> list[ItemRec]:
    """DIAGNOSTIC BASELINE: marginal-win-prob counterfactual sweep.

    For each candidate item, compute the MARGINAL win_prob increase from
    adding it to my current_bag. Return top-K by that lift.

    Caveat — strongly confounded: items that winning teams build will
    push the model's win prediction up even if the items aren't causal.
    Use item_rec_odds_ratio for a cleaner draft-contextual signal, or
    build_path for an ordered budget-aware progression.

    heroes:       [r0..r4, d0..d4] hero IDs (NOT yet canonical-sorted).
    my_slot:      0..9 index of MY slot in the input heroes array.
    current_bag:  list of item IDs already in my bag (default: empty).
    account_ids:  10-list of account IDs (None for unknown).
    gold_budget:  if set, exclude items costing more than this.
    top_k:        number of items to return.
    min_cost:     exclude items below this cost (filters out boots-1, etc.,
                  consumables, components that aren't really build choices).
    """
    assert 0 <= my_slot < 10
    if current_bag is None:
        current_bag = []
    if account_ids is None:
        account_ids = [None] * 10

    # Canonical sort tracks where my_slot lands
    heroes_full = list(heroes)
    r_pairs = sorted(enumerate(zip(heroes_full[:5], account_ids[:5])),
                      key=lambda p: p[1][0])
    d_pairs = sorted(enumerate(zip(heroes_full[5:], account_ids[5:])),
                      key=lambda p: p[1][0])
    # Map original slot -> new slot
    new_slot_of_original = {}
    for new_idx, (orig_idx, _) in enumerate(r_pairs):
        new_slot_of_original[orig_idx] = new_idx
    for new_idx, (orig_idx, _) in enumerate(d_pairs):
        new_slot_of_original[5 + orig_idx] = 5 + new_idx
    sorted_heroes  = [p[1][0] for p in r_pairs] + [p[1][0] for p in d_pairs]
    sorted_accts   = [p[1][1] for p in r_pairs] + [p[1][1] for p in d_pairs]
    my_sorted_slot = new_slot_of_original[my_slot]

    # Candidate items: from our vocab, excluding ones already in bag,
    # excluding ones outside budget, excluding ones below min_cost.
    vocab_to_iid = f.vocab_idx_to_item_id
    candidate_indices: list[int] = []
    item_info = item_id_to_info()
    in_bag = set(int(x) for x in current_bag)
    for vidx in range(2, f.item_vocab_size):  # skip PAD (0) and RARE (1)
        iid = vocab_to_iid.get(vidx)
        if iid is None or iid in in_bag:
            continue
        info = item_info.get(iid, {"cost": 0})
        c = int(info.get("cost", 0))
        if c < min_cost:
            continue
        if gold_budget is not None and c > gold_budget:
            continue
        candidate_indices.append(vidx)

    if not candidate_indices:
        return []

    # Build batch in numpy (single GPU transfer at end)
    N = len(candidate_indices)
    B = N + 1
    hero_ids_np = np.tile(np.array(sorted_heroes, dtype=np.int64), (B, 1))
    pf_one = np.stack([get_player_features_or_default(a) for a in sorted_accts], axis=0)
    player_feats_np = np.tile(pf_one, (B, 1, 1))

    # Items: row 0 = current_bag at my slot; rows 1..N add one candidate item each
    items_np = np.zeros((B, 10, f.item_vocab_size), dtype=np.float32)
    bag_vec = f.items_multihot(list(in_bag))
    items_np[0, my_sorted_slot, :] = bag_vec
    for ci, cand_vidx in enumerate(candidate_indices):
        row = ci + 1
        items_np[row, my_sorted_slot, :] = bag_vec
        items_np[row, my_sorted_slot, cand_vidx] = 1.0

    inputs = f.empty_inputs(batch_size=B)
    inputs["hero_ids"]    = torch.from_numpy(hero_ids_np).to(f.device, non_blocking=True)
    inputs["player_feats"] = torch.from_numpy(player_feats_np).to(f.device, non_blocking=True)
    inputs["items"]       = torch.from_numpy(items_np).to(f.device, non_blocking=True)

    # Mask: items UNMASKED (we want them as input); all other post-game masked.
    masks = f.pure_pregame_mask(batch_size=B)
    masks["items"] = torch.zeros((B, 10), dtype=torch.bool, device=f.device)

    out = f.predict(inputs=inputs, masks=masks)
    winp = out.win_prob().cpu().numpy()  # [B]
    baseline_wp = float(winp[0])
    my_team_baseline = baseline_wp if my_sorted_slot < 5 else (1.0 - baseline_wp)

    recs: list[ItemRec] = []
    for ci, cand_vidx in enumerate(candidate_indices):
        row = ci + 1
        cand_wp = float(winp[row])
        my_team_cand = cand_wp if my_sorted_slot < 5 else (1.0 - cand_wp)
        marginal = my_team_cand - my_team_baseline
        iid = vocab_to_iid[cand_vidx]
        info = item_info.get(iid, {})
        recs.append(ItemRec(
            vocab_idx=cand_vidx,
            item_id=iid,
            item_name=info.get("dname", f"item_{iid}"),
            score=marginal,
            cost=int(info.get("cost", 0)),
        ))

    recs.sort(key=lambda r: -r.score)
    return recs[:top_k]


def _sort_draft_track_my_slot(heroes: list[int], my_slot: int,
                                account_ids: list[int | None] | None
                                ) -> tuple[list[int], list[int | None], int]:
    """Canonical-sort heroes within each team, returning (sorted_heroes,
    sorted_accts, new_my_slot)."""
    if account_ids is None:
        account_ids = [None] * 10
    r_pairs = sorted(enumerate(zip(heroes[:5], account_ids[:5])),
                      key=lambda p: p[1][0])
    d_pairs = sorted(enumerate(zip(heroes[5:], account_ids[5:])),
                      key=lambda p: p[1][0])
    new_slot_of_original: dict[int, int] = {}
    for new_idx, (orig_idx, _) in enumerate(r_pairs):
        new_slot_of_original[orig_idx] = new_idx
    for new_idx, (orig_idx, _) in enumerate(d_pairs):
        new_slot_of_original[5 + orig_idx] = 5 + new_idx
    sorted_heroes = [p[1][0] for p in r_pairs] + [p[1][0] for p in d_pairs]
    sorted_accts  = [p[1][1] for p in r_pairs] + [p[1][1] for p in d_pairs]
    return sorted_heroes, sorted_accts, new_slot_of_original[my_slot]


def item_rec_odds_ratio(f: V7Foundation,
                         heroes: list[int],
                         my_slot: int,
                         current_bag: list[int] | None = None,
                         account_ids: list[int | None] | None = None,
                         top_k: int = 10,
                         min_cost: int = 200,
                         gold_budget: int | None = None,
                         smoothing: float = 1e-3) -> list[ItemRec]:
    """Design A: rank items by P(item | win=1) / P(item | win=0).

    For each item X, compute the odds ratio:
      P(item=X ∈ my final bag | draft, player, current_bag, my-team-wins=1)
      ÷ P(item=X ∈ my final bag | draft, player, current_bag, my-team-wins=0)

    Items with odds ratio >> 1 are DISTINCTIVELY associated with winning
    in this draft+player context. Items equally common in winning and
    losing bags get ratio ≈ 1 and rank low.

    Much cleaner than item_rec_marginal_sweep because draft/player
    context is held FIXED across both conditional distributions, so
    confounding with team strength is controlled.

    Cost: 2 forward passes total (one batch with win=1, one with win=0),
    each on a single match. ~0.3s.

    Returns ItemRec sorted descending by odds ratio (stored in `score`).
    """
    assert 0 <= my_slot < 10
    if current_bag is None:
        current_bag = []
    if account_ids is None:
        account_ids = [None] * 10

    sorted_heroes, sorted_accts, my_sorted_slot = \
        _sort_draft_track_my_slot(heroes, my_slot, account_ids)
    my_team_is_radiant = (my_sorted_slot < 5)

    pf = np.stack([get_player_features_or_default(a) for a in sorted_accts], axis=0)
    bag_vec = f.items_multihot(list(int(x) for x in current_bag))

    # Build a batch of size 2: row 0 = win=1, row 1 = win=0 (from MY-team POV)
    inputs = f.empty_inputs(batch_size=2)
    inputs["hero_ids"][:, :] = torch.tensor(
        sorted_heroes, dtype=torch.long, device=f.device).unsqueeze(0)
    inputs["player_feats"][:, :, :] = torch.tensor(
        pf, dtype=torch.float32, device=f.device).unsqueeze(0)
    inputs["items"][:, my_sorted_slot, :] = torch.tensor(bag_vec, device=f.device)
    # Win token: row 0 sets my-team-wins, row 1 sets my-team-loses
    if my_team_is_radiant:
        inputs["win_idx"][0] = 1
        inputs["win_idx"][1] = 0
    else:
        inputs["win_idx"][0] = 0  # row 0 = dire wins = my team
        inputs["win_idx"][1] = 1

    masks = f.pure_pregame_mask(batch_size=2)
    masks["items"] = torch.zeros((2, 10), dtype=torch.bool, device=f.device)
    masks["win"] = torch.zeros((2,), dtype=torch.bool, device=f.device)

    out = f.predict(inputs=inputs, masks=masks)
    item_probs = out.item_probs().cpu().numpy()   # [2, 10, 305]
    p_my_win  = item_probs[0, my_sorted_slot, :]
    p_my_lose = item_probs[1, my_sorted_slot, :]

    odds = (p_my_win + smoothing) / (p_my_lose + smoothing)

    # Rank — skip PAD/RARE + min_cost + gold_budget filter + skip items in bag
    in_bag = set(int(x) for x in current_bag)
    vocab_to_iid = f.vocab_idx_to_item_id
    info_map = item_id_to_info()
    scored: list[tuple[float, int, int, int]] = []  # (odds, vidx, iid, cost)
    for vidx in range(2, f.item_vocab_size):
        iid = vocab_to_iid.get(vidx)
        if iid is None or iid in in_bag:
            continue
        cost = int(info_map.get(iid, {}).get("cost", 0))
        if cost < min_cost:
            continue
        if gold_budget is not None and cost > gold_budget:
            continue
        scored.append((float(odds[vidx]), vidx, iid, cost))

    scored.sort(reverse=True)
    out_recs: list[ItemRec] = []
    for odds_val, vidx, iid, cost in scored[:top_k]:
        info = info_map.get(iid, {})
        out_recs.append(ItemRec(
            vocab_idx=vidx, item_id=iid,
            item_name=info.get("dname", f"item_{iid}"),
            score=odds_val, cost=cost))
    return out_recs


@dataclass
class BuildStep:
    step: int
    item_id: int
    item_name: str
    cost: int
    cumulative_cost: int
    expected_minute: float        # cumulative_cost / GPM
    odds_ratio_at_pick: float
    remaining_budget: int


def build_path(f: V7Foundation,
                 heroes: list[int],
                 my_slot: int,
                 account_ids: list[int | None] | None = None,
                 gpm: float | None = None,
                 duration_minutes: float | None = None,
                 budget_safety_factor: float = 0.85,
                 max_items: int = 6,
                 min_cost: int = 200,
                 candidate_pool_size: int = 30) -> list[BuildStep]:
    """Design B: greedy build progression with budget.

    Step-by-step build path:
    1. Predict GPM and game duration from v7's heads (if not provided)
    2. total_gold = GPM × duration_min × budget_safety_factor
       (safety factor leaves room for consumables / pings / wards)
    3. Loop until 6 items or budget exhausted:
         a. Run item_rec_odds_ratio with the current bag as input
         b. Filter by remaining budget
         c. Pick top-1 affordable item; add to bag, subtract cost
    4. Return ordered list of BuildStep records

    Each step's `expected_minute` = cumulative_cost / GPM, an estimate
    of when in the game you'd typically have enough gold to buy that
    item assuming linear gold accumulation.

    The model naturally avoids "rush Divine Rapier (6300g)" because at
    minute 8 the budget can't afford it; by the time it can, the bag
    already has earlier items and Rapier may have a different odds ratio.

    Cost: ~6 forward passes (one per build step) plus 1 to predict
    baseline GPM/duration. <1 second total.
    """
    assert 0 <= my_slot < 10
    if account_ids is None:
        account_ids = [None] * 10

    sorted_heroes, sorted_accts, my_sorted_slot = \
        _sort_draft_track_my_slot(heroes, my_slot, account_ids)
    info_map = item_id_to_info()

    # Predict baseline GPM and duration (single forward pass with everything masked)
    if gpm is None or duration_minutes is None:
        baseline_inputs = f.empty_inputs(batch_size=1)
        baseline_inputs["hero_ids"][0, :] = torch.tensor(
            sorted_heroes, dtype=torch.long, device=f.device)
        pf = np.stack([get_player_features_or_default(a) for a in sorted_accts], axis=0)
        baseline_inputs["player_feats"][0, :, :] = torch.tensor(
            pf, dtype=torch.float32, device=f.device)
        baseline_masks = f.pure_pregame_mask(batch_size=1)
        baseline_out = f.predict(inputs=baseline_inputs, masks=baseline_masks)
        if gpm is None:
            gpm = float(baseline_out.gpm()[0, my_sorted_slot].cpu())
        if duration_minutes is None:
            duration_minutes = float(baseline_out.dur_seconds()[0].cpu()) / 60.0

    total_gold = int(gpm * duration_minutes * budget_safety_factor)
    bag: list[int] = []
    remaining = total_gold
    progression: list[BuildStep] = []

    for step in range(max_items):
        # Get top affordable items by odds ratio, conditioned on current bag
        recs = item_rec_odds_ratio(
            f, heroes=heroes, my_slot=my_slot,
            current_bag=bag, account_ids=account_ids,
            top_k=candidate_pool_size,
            min_cost=min_cost, gold_budget=remaining)
        if not recs:
            break
        best = recs[0]
        bag.append(int(best.item_id))
        remaining -= int(best.cost)
        cumulative = total_gold - remaining
        progression.append(BuildStep(
            step=step + 1,
            item_id=int(best.item_id),
            item_name=best.item_name,
            cost=int(best.cost),
            cumulative_cost=cumulative,
            expected_minute=(cumulative / gpm) if gpm > 0 else 0.0,
            odds_ratio_at_pick=float(best.score),
            remaining_budget=max(0, remaining),
        ))
        if remaining < min_cost:
            break

    return progression


def item_rec_given_win(f: V7Foundation,
                         heroes: list[int],
                         my_slot: int,
                         account_ids: list[int | None] | None = None,
                         top_k: int = 10,
                         min_cost: int = 200) -> list[ItemRec]:
    """Top-K items most likely in MY final bag, given that MY TEAM wins.

    Uses the outcome_cond scenario at inference: unmask the win token at
    the value corresponding to my-team-wins; query the items head for my
    slot; return top-K probabilities.
    """
    assert 0 <= my_slot < 10
    if account_ids is None:
        account_ids = [None] * 10

    # Sort + track my slot
    heroes_full = list(heroes)
    r_pairs = sorted(enumerate(zip(heroes_full[:5], account_ids[:5])),
                      key=lambda p: p[1][0])
    d_pairs = sorted(enumerate(zip(heroes_full[5:], account_ids[5:])),
                      key=lambda p: p[1][0])
    new_slot_of_original = {}
    for new_idx, (orig_idx, _) in enumerate(r_pairs):
        new_slot_of_original[orig_idx] = new_idx
    for new_idx, (orig_idx, _) in enumerate(d_pairs):
        new_slot_of_original[5 + orig_idx] = 5 + new_idx
    sorted_heroes  = [p[1][0] for p in r_pairs] + [p[1][0] for p in d_pairs]
    sorted_accts   = [p[1][1] for p in r_pairs] + [p[1][1] for p in d_pairs]
    my_sorted_slot = new_slot_of_original[my_slot]
    my_team_is_radiant = (my_sorted_slot < 5)

    inputs = f.empty_inputs(batch_size=1)
    inputs["hero_ids"][0, :] = torch.tensor(sorted_heroes, dtype=torch.long, device=f.device)
    pf = np.stack([get_player_features_or_default(a) for a in sorted_accts], axis=0)
    inputs["player_feats"][0, :, :] = torch.tensor(pf, dtype=torch.float32, device=f.device)
    # Set win token to MY-TEAM-WINS state
    inputs["win_idx"][0] = 1 if my_team_is_radiant else 0

    # Mask: pre-game UNMASKED, win UNMASKED (conditioning), everything else MASKED
    masks = f.pure_pregame_mask(batch_size=1)
    masks["win"] = torch.zeros((1,), dtype=torch.bool, device=f.device)

    out = f.predict(inputs=inputs, masks=masks)
    probs = out.item_probs()[0, my_sorted_slot, :].cpu().numpy()  # [305]

    # Rank — skip PAD/RARE, apply min_cost filter, take top-K
    vocab_to_iid = f.vocab_idx_to_item_id
    item_info = item_id_to_info()
    scored: list[tuple[float, int, int]] = []  # (prob, vidx, iid)
    for vidx in range(2, f.item_vocab_size):
        iid = vocab_to_iid.get(vidx)
        if iid is None:
            continue
        cost = int(item_info.get(iid, {}).get("cost", 0))
        if cost < min_cost:
            continue
        scored.append((float(probs[vidx]), vidx, iid))

    scored.sort(reverse=True)
    out_recs: list[ItemRec] = []
    for prob, vidx, iid in scored[:top_k]:
        info = item_info.get(iid, {})
        out_recs.append(ItemRec(
            vocab_idx=vidx, item_id=iid,
            item_name=info.get("dname", f"item_{iid}"),
            score=prob, cost=int(info.get("cost", 0))))
    return out_recs


def win_vs_duration(f: V7Foundation,
                      heroes: list[int],
                      account_ids: list[int | None] | None = None,
                      duration_minutes: list[float] | None = None
                      ) -> list[WinDurationPoint]:
    """Sweep duration as input; return P(radiant_win) at each duration.

    Uses the duration_cond scenario: unmask duration at the queried value.
    Other post-game info masked.

    duration_minutes: list of game durations in MINUTES (default: 15-50 in 5-min steps).
    """
    if duration_minutes is None:
        duration_minutes = [15, 20, 25, 30, 35, 40, 50]
    inputs, _h, _ = _build_inputs_for_draft(f, heroes, account_ids)
    B = len(duration_minutes)
    # Replicate the single-config inputs across B duration values
    inputs2 = f.empty_inputs(batch_size=B)
    for k in ("hero_ids", "player_feats"):
        inputs2[k] = inputs[k].expand(B, *inputs[k].shape[1:]).contiguous()
    # Set duration per-row
    dur_secs = np.array(duration_minutes, dtype=np.float32) * 60.0
    inputs2["dur_log"] = torch.tensor(np.log1p(dur_secs), dtype=torch.float32, device=f.device)
    masks = f.pure_pregame_mask(batch_size=B)
    masks["duration"] = torch.zeros((B,), dtype=torch.bool, device=f.device)
    out = f.predict(inputs=inputs2, masks=masks)
    winp = out.win_prob().cpu().numpy()
    return [WinDurationPoint(duration_minutes=float(m), win_prob=float(p))
            for m, p in zip(duration_minutes, winp)]


def kills_per_minute_pair(f: V7Foundation,
                            hero_subset: list[int],
                            account_ids: list[int | None] | None = None,
                            allies_fill: list[int] | None = None,
                            enemies_fill: list[int] | None = None,
                            seed: int = 42, n_samples: int = 12) -> KillsPerMinResult:
    """Predict kills/min for a SUBSET (1-5 heroes) on the same team.

    The model needs 10 heroes to forward; we fill the unspecified ally and
    enemy slots from the empirical distribution and average predictions
    over n_samples random completions.

    hero_subset:  1-5 hero IDs assumed to be on the SAME team (radiant).
    account_ids:  per-subset account IDs (None for anonymous); same length
                  as hero_subset.
    allies_fill:  fill radiant up to 5 (if not specified, sampled).
    enemies_fill: fill dire (5 heroes; if not specified, sampled).

    Returns predicted (kills + assists) summed over the subset, divided
    by predicted duration in minutes.
    """
    assert 1 <= len(hero_subset) <= 5
    if account_ids is None:
        account_ids = [None] * len(hero_subset)
    assert len(account_ids) == len(hero_subset)
    rng = np.random.default_rng(seed)

    n_unknown_ally = 5 - len(hero_subset)
    locked = set(hero_subset)

    # Build n_samples completions
    sample_completions: list[tuple[list[int], list[int]]] = []
    for _ in range(n_samples):
        if allies_fill is None:
            unk_ally = sample_unknown_heroes(n_unknown_ally, exclude=locked, rng=rng)
        else:
            unk_ally = list(allies_fill)[:n_unknown_ally]
        excl2 = locked | set(unk_ally)
        if enemies_fill is None:
            unk_enemy = sample_unknown_heroes(5, exclude=excl2, rng=rng)
        else:
            unk_enemy = list(enemies_fill)[:5]
        sample_completions.append((unk_ally, unk_enemy))

    B = n_samples
    # Build inputs as numpy arrays first, then single GPU transfer
    hero_ids_np = np.zeros((B, 10), dtype=np.int64)
    from .v7_inference import ANON_FEATS
    player_feats_np = np.tile(ANON_FEATS, (B, 10, 1)).astype(np.float32)
    subset_slot_per_row: list[list[int]] = []

    # Cache subset's own player_feats
    subset_feats = [get_player_features_or_default(a) for a in account_ids]

    for row, (unk_ally, unk_enemy) in enumerate(sample_completions):
        radiant_5 = list(hero_subset) + list(unk_ally)
        dire_5    = list(unk_enemy)

        # Argsort radiant slots and track where each subset member lands
        r_argsort = sorted(range(5), key=lambda i: radiant_5[i])
        d_argsort = sorted(range(5), key=lambda i: dire_5[i])
        sorted_r = [radiant_5[i] for i in r_argsort]
        sorted_d = [dire_5[i] for i in d_argsort]

        # Subset members were at positions 0..len(hero_subset)-1 in radiant_5;
        # find their new positions after sorting
        new_subset_slots = []
        for orig_i in range(len(hero_subset)):
            new_subset_slots.append(r_argsort.index(orig_i))
        subset_slot_per_row.append(new_subset_slots)

        # Place subset features at their new slots
        for orig_i, new_slot in enumerate(new_subset_slots):
            player_feats_np[row, new_slot, :] = subset_feats[orig_i]

        hero_ids_np[row, :5] = sorted_r
        hero_ids_np[row, 5:] = sorted_d

    inputs = f.empty_inputs(batch_size=B)
    inputs["hero_ids"]    = torch.from_numpy(hero_ids_np).to(f.device, non_blocking=True)
    inputs["player_feats"] = torch.from_numpy(player_feats_np).to(f.device, non_blocking=True)
    masks = f.pure_pregame_mask(batch_size=B)

    out = f.predict(inputs=inputs, masks=masks)
    kills = out.kills().cpu().numpy()      # [B, 10]
    assists = out.assists().cpu().numpy()  # [B, 10]
    dur_sec = out.dur_seconds().cpu().numpy()  # [B]
    dur_min = dur_sec / 60.0

    # Sum K+A over the subset slots per row
    row_metrics = []
    for row, slots in enumerate(subset_slot_per_row):
        k_sum = float(kills[row, slots].sum())
        a_sum = float(assists[row, slots].sum())
        ka = k_sum + a_sum
        dm = max(float(dur_min[row]), 1.0)
        row_metrics.append((k_sum, a_sum, ka / dm, dm))

    k_mean = float(np.mean([r[0] for r in row_metrics]))
    a_mean = float(np.mean([r[1] for r in row_metrics]))
    kpm_mean = float(np.mean([r[2] for r in row_metrics]))
    dm_mean = float(np.mean([r[3] for r in row_metrics]))

    return KillsPerMinResult(
        hero_subset=list(hero_subset),
        kills_per_min=kpm_mean,
        predicted_duration_min=dm_mean,
        predicted_total_kills=k_mean,
        predicted_assists=a_mean,
    )


__all__ = [
    "HeroPickRec", "ItemRec", "WinDurationPoint", "KillsPerMinResult",
    "BuildStep",
    "personal_winprob", "lineup_matchup",
    "hero_pick_rec",
    "item_rec_marginal_sweep", "item_rec_odds_ratio", "item_rec_given_win",
    "build_path",
    "win_vs_duration", "kills_per_minute_pair",
]
