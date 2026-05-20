"""MinimalTransformerWithFeaturesAndPlayerEmbedding for player-embedding-prelim-740.

Extends `experiments/2026-05-19-upstream-data-cleanup-740/models.py`:
MinimalTransformerWithFeatures with one new optional input branch — a
learned per-account embedding lookup projected per-slot into the
hero+team(+feature) sum.

Architecture (use_player_embedding=False reduces to the cleanup-740 model):

  hero_embed    (Embedding hero_vocab_size × embed_dim, pad_idx=0)
  team_embed    (Embedding 2 × embed_dim)
  [proj]        (Linear embed_dim → d_model, or Identity if equal)
  [feat_proj]   (Linear n_player_feats → d_model)  - if use_features
  [player_embedding] (Embedding player_vocab_size × player_embed_dim)
  [player_proj]      (Linear player_embed_dim → d_model)  - if use_player_embedding

  per-slot token  =  proj(hero_embed + team_embed)
                 +  feat_proj(player_feats)              if use_features
                 +  player_proj(player_embedding(idx))   if use_player_embedding

  n_layers × TransformerEncoderLayer(d_model, n_heads,
      dim_feedforward=d_model*ff_mult, dropout, activation='gelu',
      batch_first=True, norm_first=True)
  mean-pool over 10 tokens → [B, d_model]
  Linear(d_model → 1)

Inputs:
  hero_ids:     LongTensor[B, 10]
  player_feats: FloatTensor[B, 10, n_player_feats] (or None when use_features=False)
  account_idx:  LongTensor[B, 10] (or None when use_player_embedding=False)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _xavier_(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MinimalTransformerWithFeaturesAndPlayerEmbedding(nn.Module):
    def __init__(self, hero_vocab_size: int, embed_dim: int, d_model: int,
                 n_heads: int, n_layers: int, ff_mult: int = 2,
                 dropout: float = 0.0, n_player_feats: int = 8,
                 use_features: bool = True,
                 use_player_embedding: bool = False,
                 player_vocab_size: int = 502_002,
                 player_embed_dim: int = 32,
                 player_init_std: float = 0.02):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.use_features = bool(use_features)
        self.use_player_embedding = bool(use_player_embedding)
        self.n_player_feats = int(n_player_feats)
        self.player_vocab_size = int(player_vocab_size)
        self.player_embed_dim = int(player_embed_dim)

        # Hero + team embeddings (cleanup-740 parity).
        self.hero_embed = nn.Embedding(hero_vocab_size, embed_dim, padding_idx=0)
        self.team_embed = nn.Embedding(2, embed_dim)

        if embed_dim != d_model:
            self.proj = nn.Linear(embed_dim, d_model)
        else:
            self.proj = nn.Identity()

        # Feature projection — always present (cleanup-740 parity); skipped in forward
        # if use_features=False.
        self.feat_proj = nn.Linear(n_player_feats, d_model)

        # Player embedding + projection — only allocated when use_player_embedding=True;
        # baseline ablation skips ~16M params.
        if self.use_player_embedding:
            self.player_embedding = nn.Embedding(player_vocab_size, player_embed_dim)
            self.player_proj = nn.Linear(player_embed_dim, d_model)
        else:
            self.player_embedding = None
            self.player_proj = None

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

        # Init.
        self.apply(_xavier_)
        nn.init.normal_(self.hero_embed.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.hero_embed.weight[0].zero_()
        nn.init.normal_(self.team_embed.weight, mean=0.0, std=0.1)
        if self.use_player_embedding:
            nn.init.normal_(self.player_embedding.weight, mean=0.0, std=player_init_std)

    def forward(self, hero_ids: torch.Tensor,
                player_feats: torch.Tensor | None = None,
                account_idx: torch.Tensor | None = None) -> torch.Tensor:
        h = self.hero_embed(hero_ids)                    # [B, 10, embed_dim]
        t = self.team_embed(self.team_ids).unsqueeze(0)  # [1, 10, embed_dim]
        x = h + t
        x = self.proj(x)                                  # [B, 10, d_model]
        if self.use_features:
            if player_feats is None:
                raise ValueError("use_features=True but player_feats is None")
            x = x + self.feat_proj(player_feats)         # [B, 10, d_model]
        if self.use_player_embedding:
            if account_idx is None:
                raise ValueError("use_player_embedding=True but account_idx is None")
            pe = self.player_embedding(account_idx)      # [B, 10, player_embed_dim]
            x = x + self.player_proj(pe)                 # [B, 10, d_model]
        x = self.encoder(x)                              # [B, 10, d_model]
        pooled = x.mean(dim=1)                           # [B, d_model]
        return self.head(pooled).squeeze(-1)             # [B]


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hero_embed = 0
    player_embed = 0
    for name, p in model.named_parameters():
        if "hero_embed" in name or "team_embed" in name:
            hero_embed += p.numel()
        if "player_embedding" in name:
            player_embed += p.numel()
    other = total - hero_embed - player_embed
    return {"total": int(total), "trainable": int(trainable),
            "hero_team_embedding": int(hero_embed),
            "player_embedding": int(player_embed),
            "other": int(other)}


def build_model(hp: dict, hero_vocab_size: int, n_player_feats: int,
                use_features: bool, use_player_embedding: bool,
                player_vocab_size: int, player_embed_dim: int,
                player_init_std: float = 0.02
                ) -> MinimalTransformerWithFeaturesAndPlayerEmbedding:
    return MinimalTransformerWithFeaturesAndPlayerEmbedding(
        hero_vocab_size=hero_vocab_size,
        embed_dim=int(hp["embed_dim"]),
        d_model=int(hp["d_model"]),
        n_heads=int(hp["n_heads"]),
        n_layers=int(hp["n_layers"]),
        ff_mult=int(hp["ff_mult"]),
        dropout=float(hp["dropout"]),
        n_player_feats=int(n_player_feats),
        use_features=bool(use_features),
        use_player_embedding=bool(use_player_embedding),
        player_vocab_size=int(player_vocab_size),
        player_embed_dim=int(player_embed_dim),
        player_init_std=float(player_init_std),
    )


__all__ = ["MinimalTransformerWithFeaturesAndPlayerEmbedding",
           "build_model", "count_params"]
