"""Time-integrated item build optimizer on v7.

Optimizes a full item BUILD (purchase + sell sequence over game time)
for a fixed draft, rather than recommending a single static 6-item bag.

## The math

For a fixed draft + my hero/slot, a build determines my inventory I(t)
at every minute t. The game ends at a random time tau ~ p_tau. We score
a build by how well its inventory at each moment matches the items that
WINNING players hold in a game of that length, integrated over when the
game might end, with a small efficiency regularizer:

    J(build) = sum_t  p_tau(t | draft) * value(I(t), t)
                 - lambda * (gross gold spent) / G_norm

    value(I, t) = sum_{X in I} P(X in final bag | draft, player, dur=t, win=1)

This is a "build-match score": how much my inventory at minute t overlaps
the build that winning players of this hero+draft hold in a t-length game.

## Why P(X | win, dur), NOT the win head and NOT the odds ratio

Three candidate per-item signals were evaluated:

1. WIN HEAD w(I, t): responsive (range ~0.39) but REVERSE-CAUSAL -- a Zeus
   holding 6 luxury items at minute 25 reads as 0.72 win prob because
   *having* them means the team is already snowballing. Maximizing it =
   "rush the most expensive items" = the bias we avoid.

2. ODDS RATIO P(X|win)/P(X|loss): least reverse-causal, but it only flags
   DISCRIMINATING items (the flex picks that separate winners from losers)
   and treats the CORE build as neutral (odds ~ 1, because both winners
   and losers build it). An odds-ratio objective builds 2 flex items and
   stops -- it misses the core build entirely.

3. P(X | win, dur=t) [CHOSEN]: the descriptive "what winning builds look
   like," conditioned on game length. Captures BOTH the core (high P) and
   is duration-aware. Crucially, duration-conditioning tempers the
   expensive-item bias: verified on Zeus, the ranking puts 1500g Arcane
   Boots and 505g Null Talisman above 5600g Divine Rapier at short
   durations -- it does NOT just prefer expensive items.

This is imitation ("build like winning players do, accounting for game
length"), which is what most build guides ARE -- not a causal claim, but
the most useful honest signal v7 offers. Reverse causality survives
(winners afford a bit more) but duration-conditioning + the gold budget
keep it realizable and bounded.

Tempo items have high P at short t and low P at long t, so a build holds
them early and SELLS them for luxury items as their P decays -- exactly
the slot-recycling behavior we wanted. This is why time integration and
selling matter.

- p_tau(t | draft) = game-end-time distribution. Global empirical PMF
            from val durations (low-variance backbone), LIGHTLY shifted
            by the model's duration prediction (time-axis stretch by
            s = 1 + alpha*(mu_pred/mu_global - 1), alpha=0.25, s clamped
            to +-15%). The point prediction can only nudge the center;
            we still integrate over the whole distribution.

P(X | win, dur=t) is PRECOMPUTED into a [pool x T] matrix (T forward
passes, batched), so the beam search runs on CPU with no GPU calls in
the loop.

## Gold(t) via Monte-Carlo over duration x win

The budget available by minute t is estimated by averaging GPM over all
game-lengths that SURVIVE to t (a game reaches minute t only if tau>=t),
marginalized over the win outcome:

    Gold(t) = Gold_0 + t * ( sum_{tau>=t} p_tau(tau) * gbar(tau) )
                            / ( sum_{tau>=t} p_tau(tau) )

    gbar(tau) = E_w GPM(my_slot | draft, player, dur=tau, win=w)
              = P(win)*GPM(dur=tau,win=1) + P(loss)*GPM(dur=tau,win=0)

GPM is a RATE (low variance) queried from v7's gpm head. Survival-
weighting makes the minute-20 budget reflect the GPM of games that
actually reach minute 20.

## Actions, slots, selling

At each 1-minute step, a build trajectory may BUY, SELL, or HOLD:
- BUY(X): requires marginal_cost(X | held) <= gold AND a free slot (or X
  upgrades held components, freeing slots). marginal_cost = cost(X) -
  sum(cost of held immediate-components of X that get consumed). The
  consumed components leave the inventory (Sange -> Sange & Yasha frees
  Sange's slot).
- SELL(X): recover 0.5 * cost(X), free its slot. The optimizer only
  sells when the freed slot raises J more than the half-gold loss costs.
- HOLD: accumulate gold (income added every step regardless of action).
- 6-slot cap enforced at every step.

## Solver: beam search

The exact DP over inventory states is combinatorial. We run beam search:
maintain top-K trajectories, expand each by all feasible actions, score
the resulting inventory's running contribution p_tau(t)*value(I_t,t), keep
top-K by cumulative J. value(I,t) is path-independent (depends only on
inventory + t) and PRECOMPUTED, so the beam loop runs purely on CPU.

## Caveats

- Item costs are current-patch, not 7.40-exact (serve/items.json).
- value(I, t) is imitation of winning builds, still root-correlational;
  duration conditioning + the gold budget constraint blunt the
  reverse-causality "expensive item" bias but don't fully eliminate it.
- CONSUMABLE-BUFF ITEMS (Aghanim's Scepter/Shard, Moon Shard): these are
  consumed into a PERMANENT buff and leave the inventory. Since the model
  trained on final inventories, P(item|win,dur=t) UNDERSTATES them at the
  long durations where consuming is common -- their declining late-game
  profile is partly the consume mechanic, not the item getting worse. The
  optimizer relabels dropping these as "consume" (frees slot, no gold).
  The model cannot see WHEN they were consumed (no timeline data), so the
  consume timing in the plan is approximate.
- Boots exclusivity is enforced; other item-family exclusivities (e.g.
  unique-attribute auras) are NOT modelled.
- Linear-within-game gold accrual is a first approximation.
- Beam search is not guaranteed globally optimal; widen K for more
  thorough search.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyarrow.parquet as pq
import torch

from .lookups import (
    item_cost, item_id_to_components, item_name,
    get_player_features_or_default,
)
from .queries import _sort_draft_track_my_slot
from .v7_inference import PROJECT_ROOT, V7Foundation

# Turbo starting gold (each player starts with ~600 in Turbo).
TURBO_START_GOLD = 600

# Boots family — a player may hold at most ONE pair of boots. Buying a
# second boots item is disallowed (the beam must sell the old pair first).
# Boots of Speed(29) is excluded since it's a sub-component of every other
# boots item and gets consumed on upgrade.
BOOTS_FAMILY = frozenset({63, 50, 180, 214, 231, 48, 220, 291})

# Consumable-buff items: can be CONSUMED into a permanent effect, leaving
# the inventory (Aghanim's Scepter -> Aghanim's Blessing; Aghanim's Shard;
# Moon Shard). This matters for two reasons:
#   1. The model trained on FINAL inventories, so a player who consumed the
#      item doesn't have it in their final bag. P(item|win,dur=t) therefore
#      UNDERSTATES these items at long durations (where consuming is common)
#      -- the declining late-game profile is partly the consume mechanic,
#      not "the item got worse."
#   2. When the optimizer drops one of these to free a slot, the real-world
#      action is CONSUME (keep the permanent buff, free the slot) -- not a
#      sell. Consuming returns NO gold. We model it as a 0-gold slot-free
#      and relabel the action "consume".
CONSUMABLE_BUFFS = frozenset({108, 609, 247})  # Aghs Scepter, Aghs Shard, Moon Shard


# ----- Duration distribution -----


_DUR_CACHE: dict = {}


def _global_duration_pmf(t_grid: np.ndarray) -> np.ndarray:
    """Empirical game-end-time PMF over t_grid (minutes), from val durations."""
    key = ("global_pmf", tuple(t_grid))
    if key in _DUR_CACHE:
        return _DUR_CACHE[key]
    rc_val = PROJECT_ROOT / "data" / "snapshots" / "7.40-2025-12-16" / \
             "processed" / "rich_cols_extended" / "val.parquet"
    dur_sec = pq.read_table(rc_val, columns=["duration"])["duration"].to_numpy()
    dur_min = dur_sec / 60.0
    # Histogram onto the grid (grid points are bin centers, 1-min wide)
    edges = np.concatenate([[t_grid[0] - 0.5], (t_grid[:-1] + t_grid[1:]) / 2,
                            [t_grid[-1] + 0.5]])
    counts, _ = np.histogram(dur_min, bins=edges)
    pmf = counts.astype(np.float64)
    pmf = pmf / pmf.sum()
    _DUR_CACHE[key] = pmf
    return pmf


def duration_pmf(f: V7Foundation,
                  draft: list[int],
                  my_slot: int,
                  account_ids: list[int | None] | None,
                  t_grid: np.ndarray,
                  alpha: float = 0.25,
                  clamp: float = 0.15) -> np.ndarray:
    """Lightly draft-conditioned game-end-time PMF.

    Global empirical PMF, time-axis stretched by
      s = 1 + alpha * (mu_pred / mu_global - 1),  clamped to [1-clamp, 1+clamp]
    where mu_pred is the model's duration prediction for this draft and
    mu_global is the global mean duration. alpha=0.25 keeps the shape
    mostly global; the prediction only nudges the center.

    Set alpha=0 for a pure-global (fully draft-agnostic) distribution.
    """
    global_pmf = _global_duration_pmf(t_grid)
    mu_global = float((t_grid * global_pmf).sum())
    if alpha <= 0:
        return global_pmf

    # Model duration prediction for this draft (pure pregame)
    if account_ids is None:
        account_ids = [None] * 10
    sorted_h, sorted_a, _ = _sort_draft_track_my_slot(draft, my_slot, account_ids)
    inputs = f.empty_inputs(batch_size=1)
    inputs["hero_ids"][0, :] = torch.tensor(sorted_h, dtype=torch.long, device=f.device)
    pf = np.stack([get_player_features_or_default(a) for a in sorted_a], axis=0)
    inputs["player_feats"][0, :, :] = torch.tensor(pf, dtype=torch.float32, device=f.device)
    out = f.predict(inputs=inputs, masks=f.pure_pregame_mask(batch_size=1))
    mu_pred = float(out.dur_seconds()[0].cpu()) / 60.0

    s = 1.0 + alpha * (mu_pred / max(mu_global, 1e-6) - 1.0)
    s = float(np.clip(s, 1.0 - clamp, 1.0 + clamp))

    # Stretch time axis by s: a game that the global dist places at t now
    # lands at t*s. Resample onto t_grid by interpolating the CDF.
    stretched_t = t_grid * s
    cdf = np.cumsum(global_pmf)
    # New PMF: probability mass between consecutive grid edges under the
    # stretched mapping. Interpolate CDF at grid edges scaled by 1/s.
    edges = np.concatenate([[t_grid[0] - 0.5], (t_grid[:-1] + t_grid[1:]) / 2,
                            [t_grid[-1] + 0.5]])
    cdf_at_edges = np.interp(edges / s, t_grid, cdf, left=0.0, right=1.0)
    pmf = np.diff(cdf_at_edges)
    pmf = np.clip(pmf, 0.0, None)
    if pmf.sum() <= 0:
        return global_pmf
    return pmf / pmf.sum()


# ----- Gold curve (Monte-Carlo over duration x win) -----


def _expected_gpm_by_duration(f: V7Foundation,
                                draft: list[int], my_slot: int,
                                account_ids: list[int | None],
                                durations_min: np.ndarray) -> np.ndarray:
    """gbar(tau) = E_w GPM(my_slot | draft, player, dur=tau, win=w) for each
    tau in durations_min. Marginalizes win by the model's predicted win prob.
    """
    sorted_h, sorted_a, my_sorted = _sort_draft_track_my_slot(draft, my_slot, account_ids)
    pf = np.stack([get_player_features_or_default(a) for a in sorted_a], axis=0)
    my_team_radiant = (my_sorted < 5)

    D = len(durations_min)
    # Batch: for each duration, 2 rows (win=1, win=0) = 2D rows
    B = 2 * D
    inputs = f.empty_inputs(batch_size=B)
    inputs["hero_ids"][:, :] = torch.tensor(sorted_h, dtype=torch.long, device=f.device).unsqueeze(0)
    inputs["player_feats"][:, :, :] = torch.tensor(pf, dtype=torch.float32, device=f.device).unsqueeze(0)
    dur_secs = np.repeat(durations_min, 2).astype(np.float32) * 60.0
    inputs["dur_log"] = torch.tensor(np.log1p(dur_secs), dtype=torch.float32, device=f.device)
    # win input: even rows win for my team, odd rows lose
    win_vals = np.zeros(B, dtype=np.int64)
    for d in range(D):
        win_vals[2 * d]     = 1 if my_team_radiant else 0   # my team wins
        win_vals[2 * d + 1] = 0 if my_team_radiant else 1   # my team loses
    inputs["win_idx"] = torch.tensor(win_vals, dtype=torch.long, device=f.device)

    masks = f.pure_pregame_mask(batch_size=B)
    masks["duration"] = torch.zeros((B,), dtype=torch.bool, device=f.device)
    masks["win"] = torch.zeros((B,), dtype=torch.bool, device=f.device)

    out = f.predict(inputs=inputs, masks=masks)
    gpm = out.gpm().cpu().numpy()[:, my_sorted]   # [B]

    # Predicted win prob for this draft (to weight win vs loss)
    pg_out = f.predict(
        inputs=_pure_inputs(f, sorted_h, pf),
        masks=f.pure_pregame_mask(batch_size=1))
    p_radiant = float(pg_out.win_prob()[0].cpu())
    p_my_win = p_radiant if my_team_radiant else (1.0 - p_radiant)

    gbar = np.zeros(D)
    for d in range(D):
        gbar[d] = p_my_win * gpm[2 * d] + (1.0 - p_my_win) * gpm[2 * d + 1]
    return gbar


def _pure_inputs(f: V7Foundation, sorted_h, pf):
    inp = f.empty_inputs(batch_size=1)
    inp["hero_ids"][0, :] = torch.tensor(sorted_h, dtype=torch.long, device=f.device)
    inp["player_feats"][0, :, :] = torch.tensor(pf, dtype=torch.float32, device=f.device)
    return inp


def gold_curve(f: V7Foundation,
                draft: list[int], my_slot: int,
                account_ids: list[int | None],
                t_grid: np.ndarray,
                pmf: np.ndarray,
                start_gold: int = TURBO_START_GOLD) -> np.ndarray:
    """Gold(t) for each t in t_grid, via survival-weighted MC over duration.

      Gold(t) = start_gold + t * E[gbar(tau) | tau >= t]
    """
    gbar = _expected_gpm_by_duration(f, draft, my_slot, account_ids, t_grid)
    gold = np.zeros(len(t_grid))
    for i, t in enumerate(t_grid):
        surv = pmf.copy()
        surv[t_grid < t] = 0.0
        denom = surv.sum()
        if denom <= 0:
            eff_gpm = gbar[i]
        else:
            eff_gpm = float((surv * gbar).sum() / denom)
        gold[i] = start_gold + t * eff_gpm
    return gold


# ----- Inventory bookkeeping -----


def _marginal_buy(held: frozenset[int], item_id: int
                   ) -> tuple[int, frozenset[int]] | None:
    """Return (marginal_cost, consumed_components) for buying item_id given
    `held`, or None if item_id is already held.

    consumed_components = held immediate-components of item_id (they merge
    into the finished item, freeing their slots).
    """
    if item_id in held:
        return None
    comps = set(item_id_to_components().get(item_id, []))
    consumed = frozenset(c for c in held if c in comps)
    marginal = item_cost(item_id) - sum(item_cost(c) for c in consumed)
    return max(0, int(marginal)), consumed


# ----- Beam search -----


@dataclass
class BuildAction:
    minute: int
    kind: str            # 'buy' | 'sell' | 'consume'
    item_id: int
    item_name: str
    gold_delta: int      # negative for buy, positive for sell, 0 for consume
    inventory_after: tuple[int, ...]
    note: str = ""       # e.g. "Aghanim's Blessing (permanent, frees slot)"


@dataclass
class BuildPlan:
    actions: list[BuildAction]
    final_inventory: tuple[int, ...]
    build_match_value: float          # the J value term: sum_t pmf(t)*value(I_t,t)
    objective: float                  # J including the regularizer
    predicted_gpm: float
    duration_pmf_summary: dict        # p10/p50/p90 minutes
    gold_at_end: int


@dataclass
class _BeamState:
    held: frozenset[int]
    gold: int
    score: float                       # running J (value term - regularizer)
    value_accum: float                 # running sum of pmf(t)*value(I_t,t)
    gross_spent: int
    actions: list[BuildAction]


def _candidate_pool(winbag: np.ndarray, pmf: np.ndarray, f: V7Foundation,
                     pool_size: int, min_cost: int) -> list[int]:
    """Shortlist of relevant item ids = top-N by time-integrated
    P(X | win, dur) + their immediate components. Filters cost<min_cost
    (drops consumables / neutral cost-0 items from the buildable set)."""
    comp_map = item_id_to_components()
    # Time-integrated winbag value per vocab item
    vocab_iids = [int(k) for k in f.item_vocab.keys()]
    V = {}
    for iid in vocab_iids:
        vi = f.item_vocab[str(iid)]
        V[iid] = float((pmf * winbag[:, vi]).sum())
    # Top-N priced items
    priced = [iid for iid in vocab_iids if item_cost(iid) >= min_cost]
    priced.sort(key=lambda iid: -V[iid])
    pool: set[int] = set(priced[:pool_size])
    # Add immediate components so the optimizer can buy incrementally
    for iid in list(pool):
        for c in comp_map.get(iid, []):
            if item_cost(c) > 0:
                pool.add(int(c))
    return list(pool)


def _winbag_matrix(f: V7Foundation, sorted_h, pf, my_sorted_slot: int,
                    t_grid: np.ndarray) -> np.ndarray:
    """P(X in final bag | draft, player, dur=t, win=1) for ALL vocab items
    and every t. Returns [T, item_vocab_size]. T forward passes (win=1
    only), batched."""
    T = len(t_grid)
    my_team_radiant = (my_sorted_slot < 5)
    inputs = f.empty_inputs(batch_size=T)
    inputs["hero_ids"][:, :] = torch.tensor(sorted_h, dtype=torch.long, device=f.device).unsqueeze(0)
    inputs["player_feats"][:, :, :] = torch.tensor(pf, dtype=torch.float32, device=f.device).unsqueeze(0)
    inputs["dur_log"] = torch.tensor(np.log1p(t_grid * 60.0), dtype=torch.float32, device=f.device)
    inputs["win_idx"][:] = 1 if my_team_radiant else 0   # my team wins
    masks = f.pure_pregame_mask(batch_size=T)
    masks["win"] = torch.zeros((T,), dtype=torch.bool, device=f.device)
    masks["duration"] = torch.zeros((T,), dtype=torch.bool, device=f.device)
    out = f.predict(inputs=inputs, masks=masks)
    return out.item_probs().cpu().numpy()[:, my_sorted_slot, :]   # [T, vocab]


def optimize_build(f: V7Foundation,
                    draft: list[int],
                    my_slot: int,
                    account_ids: list[int | None] | None = None,
                    t_min: int = 2, t_max: int = 45, dt: int = 1,
                    beam_width: int = 32,
                    pool_size: int = 16,
                    min_cost: int = 400,
                    lam: float = 0.05,
                    alpha_duration: float = 0.25,
                    max_actions_per_step: int = 1) -> BuildPlan:
    """Beam-search the build that maximizes time-integrated expected win.

    Returns a BuildPlan with the ordered buy/sell actions, final inventory,
    and the objective breakdown.
    """
    if account_ids is None:
        account_ids = [None] * 10
    sorted_h, sorted_a, my_sorted = _sort_draft_track_my_slot(draft, my_slot, account_ids)
    pf = np.stack([get_player_features_or_default(a) for a in sorted_a], axis=0)

    t_grid = np.arange(t_min, t_max + dt, dt).astype(float)
    pmf = duration_pmf(f, draft, my_slot, account_ids, t_grid, alpha=alpha_duration)
    gold = gold_curve(f, draft, my_slot, account_ids, t_grid, pmf)
    g_norm = max(float(gold[-1]), 1.0)

    # Precompute P(X | win, dur=t) for all vocab items, then build the pool
    winbag = _winbag_matrix(f, sorted_h, pf, my_sorted, t_grid)
    pool = _candidate_pool(winbag, pmf, f, pool_size, min_cost=min_cost)
    # Per-item value over t: P(X | win, dur=t)
    pval: dict[int, np.ndarray] = {}
    for X in pool:
        vi = f.item_vocab.get(str(X))
        pval[X] = winbag[:, vi] if vi is not None else np.zeros(len(t_grid))

    gbar = _expected_gpm_by_duration(f, draft, my_slot, account_ids, t_grid)
    pred_gpm = float((pmf * gbar).sum())

    # Tiny per-action penalty to break ties against pointless buy/sell churn
    action_eps = 1e-4

    def value_at(held: frozenset[int], ti: int) -> float:
        return float(sum(pval[X][ti] for X in held if X in pval))

    beam = [_BeamState(held=frozenset(), gold=int(gold[0]), score=0.0,
                        value_accum=0.0, gross_spent=0, actions=[])]

    for ti, t in enumerate(t_grid):
        income = int(gold[ti] - gold[ti - 1]) if ti > 0 else 0
        successors: list[_BeamState] = []
        for st in beam:
            g_now = st.gold + income
            base_actions: list[tuple[str, int, int, frozenset[int]]] = []
            base_actions.append(("hold", -1, 0, st.held))
            for X in pool:
                mb = _marginal_buy(st.held, X)
                if mb is None:
                    continue
                marginal, consumed = mb
                new_held = (st.held - consumed) | {X}
                if len(new_held) > 6 or marginal > g_now:
                    continue
                # Boots exclusivity: at most one boots-family item held
                if X in BOOTS_FAMILY and (new_held & BOOTS_FAMILY) - {X}:
                    continue
                base_actions.append(("buy", X, -marginal, new_held))
            for X in st.held:
                if X in CONSUMABLE_BUFFS:
                    # Consume for the permanent buff: frees slot, NO gold back.
                    base_actions.append(("consume", X, 0, st.held - {X}))
                else:
                    refund = int(0.5 * item_cost(X))
                    base_actions.append(("sell", X, refund, st.held - {X}))

            for kind, X, gdelta, new_held in base_actions:
                gross = st.gross_spent
                n_act = len(st.actions)
                new_actions = st.actions
                if kind in ("buy", "sell", "consume"):
                    note = ""
                    if kind == "consume":
                        note = "consumed for permanent buff (frees slot, no gold)"
                    new_actions = st.actions + [BuildAction(
                        minute=int(t), kind=kind, item_id=X,
                        item_name=item_name(X), gold_delta=gdelta,
                        inventory_after=tuple(sorted(new_held)), note=note)]
                    n_act += 1
                if kind == "buy":
                    gross = st.gross_spent + (-gdelta)
                # This timestep's value contribution under the NEW inventory
                inc_val = pmf[ti] * value_at(new_held, ti)
                new_value_accum = st.value_accum + inc_val
                score = (new_value_accum
                         - lam * (gross / g_norm)
                         - action_eps * n_act)
                successors.append(_BeamState(
                    held=new_held, gold=g_now + gdelta, score=score,
                    value_accum=new_value_accum, gross_spent=gross,
                    actions=new_actions))

        successors.sort(key=lambda s: -s.score)
        seen: set = set()
        pruned: list[_BeamState] = []
        for s in successors:
            key = (s.held, s.gold // 500)
            if key in seen:
                continue
            seen.add(key)
            pruned.append(s)
            if len(pruned) >= beam_width:
                break
        beam = pruned

    best = max(beam, key=lambda s: s.score)
    # p10/p50/p90 of the duration PMF for reporting
    cdf = np.cumsum(pmf)
    p10 = float(t_grid[np.searchsorted(cdf, 0.10)])
    p50 = float(t_grid[np.searchsorted(cdf, 0.50)])
    p90 = float(t_grid[np.searchsorted(cdf, 0.90)])

    return BuildPlan(
        actions=best.actions,
        final_inventory=tuple(sorted(best.held)),
        build_match_value=float(best.value_accum),
        objective=float(best.score),
        predicted_gpm=pred_gpm,
        duration_pmf_summary={"p10_min": p10, "p50_min": p50, "p90_min": p90},
        gold_at_end=int(gold[-1]),
    )


def format_plan(plan: BuildPlan) -> str:
    """Human-readable build plan."""
    lines = []
    s = plan.duration_pmf_summary
    lines.append(f"Build plan (duration p10/p50/p90 = "
                 f"{s['p10_min']:.0f}/{s['p50_min']:.0f}/{s['p90_min']:.0f} min, "
                 f"predicted GPM = {plan.predicted_gpm:.0f})")
    lines.append(f"Final inventory: "
                 f"{', '.join(item_name(i) for i in plan.final_inventory)}")
    lines.append(f"  {'min':>4}  {'action':<8} {'item':<24} {'gold':>7}")
    for a in plan.actions:
        note = f"   # {a.note}" if a.note else ""
        lines.append(f"  {a.minute:>4}  {a.kind:<8} {a.item_name:<24} "
                     f"{a.gold_delta:>+7}{note}")
    return "\n".join(lines)


__all__ = [
    "BuildAction", "BuildPlan",
    "duration_pmf", "gold_curve", "optimize_build", "format_plan",
    "TURBO_START_GOLD", "BOOTS_FAMILY", "CONSUMABLE_BUFFS",
]
