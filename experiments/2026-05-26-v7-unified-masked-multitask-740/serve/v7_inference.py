"""V7 inference wrapper — load model + maskable forward.

The trained v7 checkpoint at
  experiments/2026-05-26-v7-unified-masked-multitask-740/results/model_v7_unified.pt
supports ANY subset of the 10 maskable input groups (hero, player_feat,
items, kills, deaths, assists, gpm, hd, duration, win). At inference,
mask what's unknown, query what's wanted.

This wrapper provides:
- V7Foundation: loads the checkpoint once, holds the model + config
- forward(): one-shot forward pass with explicit `inputs` dict + `mask` dict
- build_inputs(): convenience to build tensors from python-native inputs
                  (lists of hero ids, dicts of player features, etc.)

All compute on GPU if available; otherwise CPU.

Output predictions are in the model's native scale:
- win: logits → sigmoid → probability
- dur: scalar log(seconds+1) — call expm1() to recover raw seconds
- items: 305-dim logits per slot → sigmoid → per-class probability
- kills/deaths/assists/gpm/hd: scalar log1p — call expm1() to recover raw
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

EXP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXP_DIR.parent.parent

# Ensure we can import the v7 models module
sys.path.insert(0, str(EXP_DIR))

import models as v7_models  # noqa: E402


# Feature names + ordering — must match v7 training
FEAT_NAMES = [
    "n_games_log1p",
    "smoothed_winrate",
    "smoothed_winrate_hero",
    "last10_winrate",
    "days_since_last_log1p",
    "n_games_hero_log1p",
    "hero_diversity_log1p",
    "is_anonymous",
]

# Anonymous-slot default features (rough population averages — used when
# we don't know a player's profile, e.g. enemy slots in a partial draft).
# These are conservative defaults; in production, you'd want per-slot
# empirical averages by hero or by skill bracket.
ANON_FEATS = np.array([
    1.6,   # n_games_log1p (~exp(1.6) ≈ 4 games)
    0.50,  # smoothed_winrate (matchmaking baseline)
    0.50,  # smoothed_winrate_hero
    0.50,  # last10_winrate
    4.0,   # days_since_last_log1p
    0.50,  # n_games_hero_log1p
    1.0,   # hero_diversity_log1p
    1.0,   # is_anonymous flag
], dtype=np.float32)

# Default scalar "unknown" values when a group is not masked but the user
# didn't provide a value — these get used for inputs that we'd MASK in
# practice; provided here as safe defaults for explicit-value paths.
DEFAULT_KILLS_DEATHS_ASSISTS = 0.0
DEFAULT_GPM = 400.0
DEFAULT_HD = 15000.0
DEFAULT_DUR_SECONDS = 1500.0  # ~25 min Turbo


@dataclass
class V7Outputs:
    """Container for one forward pass result. Tensors stay on the device.

    Use .cpu_numpy() to grab numpy arrays for downstream analysis.
    """
    win_logit: torch.Tensor       # [B]
    dur_log: torch.Tensor         # [B] log(seconds + 1)
    item_logits: torch.Tensor     # [B, 10, 305]
    kills_log1p: torch.Tensor     # [B, 10]
    deaths_log1p: torch.Tensor    # [B, 10]
    assists_log1p: torch.Tensor   # [B, 10]
    gpm_log1p: torch.Tensor       # [B, 10]
    hd_log1p: torch.Tensor        # [B, 10]
    encoded: torch.Tensor         # [B, 12, d_model]

    def win_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.win_logit.float())

    def dur_seconds(self) -> torch.Tensor:
        return torch.expm1(self.dur_log.float()).clamp_min(0.0)

    def item_probs(self) -> torch.Tensor:
        return torch.sigmoid(self.item_logits.float())

    def kills(self) -> torch.Tensor:
        return torch.expm1(self.kills_log1p.float()).clamp_min(0.0)

    def deaths(self) -> torch.Tensor:
        return torch.expm1(self.deaths_log1p.float()).clamp_min(0.0)

    def assists(self) -> torch.Tensor:
        return torch.expm1(self.assists_log1p.float()).clamp_min(0.0)

    def gpm(self) -> torch.Tensor:
        return torch.expm1(self.gpm_log1p.float()).clamp_min(0.0)

    def hd(self) -> torch.Tensor:
        return torch.expm1(self.hd_log1p.float()).clamp_min(0.0)

    def cpu_numpy(self) -> dict[str, np.ndarray]:
        return {
            "win_prob": self.win_prob().cpu().numpy(),
            "dur_seconds": self.dur_seconds().cpu().numpy(),
            "item_probs": self.item_probs().cpu().numpy(),
            "kills": self.kills().cpu().numpy(),
            "deaths": self.deaths().cpu().numpy(),
            "assists": self.assists().cpu().numpy(),
            "gpm": self.gpm().cpu().numpy(),
            "hd": self.hd().cpu().numpy(),
        }


class V7Foundation:
    """Loaded v7 model + tensor builders + maskable forward.

    Example:
        f = V7Foundation()
        out = f.predict(
            heroes=[1, 6, 22, 86, 129, 5, 11, 13, 14, 35],  # 10 hero IDs
            player_feats=None,        # fill defaults
            known={"items": False, "kills": False, ...},  # all post-game masked
        )
        print("P(radiant_win) =", out.win_prob().item())
    """

    def __init__(self,
                 ckpt_path: str | Path | None = None,
                 config_path: str | Path | None = None,
                 device: str | torch.device | None = None):
        self.exp_dir = EXP_DIR
        if ckpt_path is None:
            ckpt_path = EXP_DIR / "results" / "pretrain_encoder_v7_unified.pt"
        if config_path is None:
            config_path = EXP_DIR / "config.yaml"
        self.ckpt_path = Path(ckpt_path)
        self.config_path = Path(config_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        with open(self.config_path) as f:
            self.cfg = yaml.safe_load(f)

        # v7's config doesn't have transformer_ablations — single ablation
        # hardcoded in train.py. Read the model hyperparams directly.
        self.hp = self.cfg["transformer_model"]

        # Item vocab
        vocab_path = (EXP_DIR / self.cfg["item_vocab"]["vocab_path"]).resolve()
        with open(vocab_path) as f:
            iv = json.load(f)
        self.item_vocab = iv["vocab"]   # item_id (str) -> vocab_idx (int)
        self.item_vocab_size = int(iv["meta"]["vocab_size"])
        self.dur_bucket_edges = iv["duration_bucket_edges"]
        # Reverse map for queries that want raw item_id given vocab_idx
        self.vocab_idx_to_item_id = {int(v): int(k) for k, v in self.item_vocab.items()}

        # Build model (v7's build_model signature is simpler than v4's)
        self.model = v7_models.build_model(
            hp=self.hp,
            vocab_size=151,
            n_player_feats=8,
            item_vocab_size=self.item_vocab_size,
        )
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(ckpt, strict=True)
        self.model.eval().to(self.device)

        # Hero vocab size from the checkpoint
        self.n_heroes = int(self.model.hero_embed.weight.shape[0])  # 151 = 150 + 1 PAD

    # ----- Tensor construction -----

    def empty_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        """Build a batch of EMPTY/default inputs. All groups SHOULD be masked
        unless the caller explicitly fills them.

        Returns a dict of tensors on self.device. Slot dim = 10 always.
        """
        B = batch_size
        d = self.device
        return {
            "hero_ids":    torch.zeros((B, 10), dtype=torch.long, device=d),
            "player_feats": torch.tensor(
                np.tile(ANON_FEATS, (B, 10, 1)), dtype=torch.float32, device=d),
            "items": torch.zeros((B, 10, self.item_vocab_size), dtype=torch.float32, device=d),
            "kills":   torch.full((B, 10), DEFAULT_KILLS_DEATHS_ASSISTS, dtype=torch.float32, device=d),
            "deaths":  torch.full((B, 10), DEFAULT_KILLS_DEATHS_ASSISTS, dtype=torch.float32, device=d),
            "assists": torch.full((B, 10), DEFAULT_KILLS_DEATHS_ASSISTS, dtype=torch.float32, device=d),
            "gpm":     torch.full((B, 10), DEFAULT_GPM, dtype=torch.float32, device=d),
            "hd":      torch.full((B, 10), DEFAULT_HD, dtype=torch.float32, device=d),
            "dur_log": torch.full((B,), float(np.log1p(DEFAULT_DUR_SECONDS)),
                                   dtype=torch.float32, device=d),
            "win_idx": torch.zeros((B,), dtype=torch.long, device=d),
        }

    def full_mask(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        """All-masked mask dict — pass to model.forward to inference with
        absolutely zero conditioning info (sanity baseline)."""
        B = batch_size
        d = self.device
        return {
            "hero":        torch.ones((B, 10), dtype=torch.bool, device=d),
            "player_feat": torch.ones((B, 10), dtype=torch.bool, device=d),
            "items":       torch.ones((B, 10), dtype=torch.bool, device=d),
            "kills":       torch.ones((B, 10), dtype=torch.bool, device=d),
            "deaths":      torch.ones((B, 10), dtype=torch.bool, device=d),
            "assists":     torch.ones((B, 10), dtype=torch.bool, device=d),
            "gpm":         torch.ones((B, 10), dtype=torch.bool, device=d),
            "hd":          torch.ones((B, 10), dtype=torch.bool, device=d),
            "duration":    torch.ones((B,), dtype=torch.bool, device=d),
            "win":         torch.ones((B,), dtype=torch.bool, device=d),
        }

    def no_mask(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        """All-unmasked mask dict — pass when feeding the model everything."""
        B = batch_size
        d = self.device
        return {
            "hero":        torch.zeros((B, 10), dtype=torch.bool, device=d),
            "player_feat": torch.zeros((B, 10), dtype=torch.bool, device=d),
            "items":       torch.zeros((B, 10), dtype=torch.bool, device=d),
            "kills":       torch.zeros((B, 10), dtype=torch.bool, device=d),
            "deaths":      torch.zeros((B, 10), dtype=torch.bool, device=d),
            "assists":     torch.zeros((B, 10), dtype=torch.bool, device=d),
            "gpm":         torch.zeros((B, 10), dtype=torch.bool, device=d),
            "hd":          torch.zeros((B, 10), dtype=torch.bool, device=d),
            "duration":    torch.zeros((B,), dtype=torch.bool, device=d),
            "win":         torch.zeros((B,), dtype=torch.bool, device=d),
        }

    def pure_pregame_mask(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        """Mask all post-game info (items/k/d/a/gpm/hd/dur/win). Keep
        heroes + player_feats UNMASKED — the most-common inference path."""
        m = self.full_mask(batch_size)
        m["hero"] = torch.zeros_like(m["hero"])
        m["player_feat"] = torch.zeros_like(m["player_feat"])
        return m

    # ----- Forward pass -----

    @torch.no_grad()
    def predict(self,
                inputs: dict[str, torch.Tensor] | None = None,
                masks: dict[str, torch.Tensor] | None = None,
                heroes: list[int] | None = None,
                player_feats: np.ndarray | None = None) -> V7Outputs:
        """Run a forward pass.

        - inputs: dict of tensors as built by `empty_inputs()`. Caller fills
          in known values.
        - masks: dict of bool tensors as built by `pure_pregame_mask()` etc.
        - heroes: convenience — list of 10 hero IDs to set into inputs.
        - player_feats: convenience — [10, 8] np array to set into inputs.

        Defaults: if inputs is None, build empty 1-batch inputs. If masks
        is None, use pure_pregame_mask (everything post-game masked).
        """
        if inputs is None:
            inputs = self.empty_inputs(batch_size=1)
        if masks is None:
            masks = self.pure_pregame_mask(batch_size=inputs["hero_ids"].size(0))
        if heroes is not None:
            assert len(heroes) == 10, f"expected 10 heroes, got {len(heroes)}"
            inputs["hero_ids"][0, :] = torch.tensor(heroes, dtype=torch.long, device=self.device)
        if player_feats is not None:
            assert player_feats.shape == (10, 8), f"expected (10, 8), got {player_feats.shape}"
            inputs["player_feats"][0, :, :] = torch.tensor(
                player_feats, dtype=torch.float32, device=self.device)

        out = self.model(
            inputs["hero_ids"], inputs["player_feats"], inputs["items"],
            inputs["kills"], inputs["deaths"], inputs["assists"],
            inputs["gpm"], inputs["hd"],
            inputs["dur_log"], inputs["win_idx"],
            masks=masks,
        )

        return V7Outputs(
            win_logit=out["win"],
            dur_log=out["dur"],
            item_logits=out["item"],
            kills_log1p=out["kills"],
            deaths_log1p=out["deaths"],
            assists_log1p=out["assists"],
            gpm_log1p=out["gpm"],
            hd_log1p=out["hd"],
            encoded=out["encoded"],
        )

    # ----- Item vocab helpers -----

    def items_multihot(self, item_ids: list[int]) -> np.ndarray:
        """Build a 305-dim multi-hot vector from a list of REAL item IDs
        (e.g. 108 for power treads). Unknown IDs are ignored (no rare-bucket
        assignment at inference time)."""
        v = np.zeros(self.item_vocab_size, dtype=np.float32)
        for iid in item_ids:
            idx = self.item_vocab.get(str(iid))
            if idx is not None:
                v[idx] = 1.0
        return v


def canonical_hero_sort(heroes: list[int]) -> list[int]:
    """Apply canonical sort: sort each team's 5 heroes ascending. Must match
    v7's training-time sort (data.py:canonical_sort_by_hero)."""
    assert len(heroes) == 10
    r = sorted(heroes[:5])
    d = sorted(heroes[5:])
    return r + d
