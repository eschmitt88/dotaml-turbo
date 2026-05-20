"""MinimalTransformerWithFeatures — MinimalTransformer + per-slot player-feature injection.

Mirrors experiments/2026-05-16-transformer-hp-sweep-740/models.py:MinimalTransformer
exactly, plus a Linear(n_player_feats, d_model) projection whose output is added
to each slot's (hero_embed + team_embed) when use_features=True.

Inputs:
  hero_ids      : LongTensor[B, 10]  — Radiant 0..4, Dire 5..9, IDs in [1, 150].
  player_feats  : FloatTensor[B, 10, n_player_feats] — per-slot dense features.

When use_features=False the projection is bypassed (ignored) and player_feats
may still be supplied — the model becomes the pure MinimalTransformer and is
the architecture_only ablation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _xavier_(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MinimalTransformerWithFeatures(nn.Module):
    """MinimalTransformer with optional per-slot player-feature injection.

    Architecture (identical to MinimalTransformer when use_features=False):
      hero_embed (Embedding vocab_size × embed_dim, pad_idx=0)
      team_embed (Embedding 2 × embed_dim)
      For each token i in [0..9]:
        token[i] = hero_embed(hero_ids[i]) + team_embed(0 if i < 5 else 1)
      Optional projection if embed_dim != d_model: Linear(embed_dim → d_model)
      [if use_features] add Linear(n_player_feats → d_model)(player_feats[i])
      n_layers × TransformerEncoderLayer (d_model, n_heads,
          dim_feedforward=d_model*ff_mult, dropout, activation="gelu",
          batch_first=True, norm_first=True)
      mean-pool over 10 tokens → [B, d_model]
      Linear(d_model → 1)
    """

    def __init__(self, vocab_size: int, embed_dim: int, d_model: int,
                 n_heads: int, n_layers: int, ff_mult: int = 2,
                 dropout: float = 0.0, n_player_feats: int = 8,
                 use_features: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.use_features = bool(use_features)
        self.n_player_feats = int(n_player_feats)

        self.hero_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.team_embed = nn.Embedding(2, embed_dim)

        if embed_dim != d_model:
            self.proj = nn.Linear(embed_dim, d_model)
        else:
            self.proj = nn.Identity()

        # Always construct feat_proj (cheap, ~576 params at d_model=64) so the
        # parameter list has identical layout regardless of use_features — that
        # way a single saved-state-dict format works for both ablations and a
        # debug-load between them stays clean. When use_features=False we skip
        # the projection in forward; the unused weights just don't accrue grad.
        self.feat_proj = nn.Linear(n_player_feats, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

        team_ids = torch.zeros(10, dtype=torch.long)
        team_ids[5:] = 1
        self.register_buffer("team_ids", team_ids, persistent=False)

        self.apply(_xavier_)
        nn.init.normal_(self.hero_embed.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.hero_embed.weight[0].zero_()
        nn.init.normal_(self.team_embed.weight, mean=0.0, std=0.1)

    def forward(self, hero_ids: torch.Tensor,
                player_feats: torch.Tensor | None = None) -> torch.Tensor:
        # hero_ids:     [B, 10] long
        # player_feats: [B, 10, n_player_feats] float32 (or None when arch-only and caller knows)
        h = self.hero_embed(hero_ids)                    # [B, 10, embed_dim]
        t = self.team_embed(self.team_ids).unsqueeze(0)  # [1, 10, embed_dim]
        x = h + t                                        # [B, 10, embed_dim]
        x = self.proj(x)                                 # [B, 10, d_model]
        if self.use_features:
            if player_feats is None:
                raise ValueError("use_features=True but player_feats is None")
            x = x + self.feat_proj(player_feats)         # [B, 10, d_model]
        x = self.encoder(x)                              # [B, 10, d_model]
        pooled = x.mean(dim=1)                           # [B, d_model]
        return self.head(pooled).squeeze(-1)             # [B]


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed = 0
    for name, p in model.named_parameters():
        if "embed" in name:
            embed += p.numel()
    return {"total": int(total), "trainable": int(trainable),
            "embedding": int(embed), "non_embedding": int(total - embed)}


def build_model(hp: dict, vocab_size: int, n_player_feats: int,
                use_features: bool) -> MinimalTransformerWithFeatures:
    return MinimalTransformerWithFeatures(
        vocab_size=vocab_size,
        embed_dim=int(hp["embed_dim"]),
        d_model=int(hp["d_model"]),
        n_heads=int(hp["n_heads"]),
        n_layers=int(hp["n_layers"]),
        ff_mult=int(hp["ff_mult"]),
        dropout=float(hp["dropout"]),
        n_player_feats=int(n_player_feats),
        use_features=bool(use_features),
    )


__all__ = ["MinimalTransformerWithFeatures", "build_model", "count_params"]
