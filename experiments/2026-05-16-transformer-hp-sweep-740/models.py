"""MinimalTransformer architecture for transformer-hp-sweep-740.

Differences from plateau-architectures-740 DraftTransformer:
  - No side token (the 11th position).
  - No 11-position learned position embedding.
  - Binary team embedding (Radiant=0, Dire=1) added to hero embeddings.
  - Linear head (single nn.Linear) instead of LayerNorm + 2-layer MLP.

Inputs:
  hero_ids: LongTensor[B, 10] — Radiant heroes 0..4, Dire heroes 5..9, IDs in [1, 150].
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _xavier_(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MinimalTransformer(nn.Module):
    """Self-attention over 10 hero tokens with shared embed + team embed.

    Architecture:
      hero_embed (Embedding 151 × embed_dim, pad_idx=0)
      team_embed (Embedding 2 × embed_dim) — index 0 for Radiant slots, 1 for Dire
      For each token i in [0..9]:
        token[i] = hero_embed(hero_ids[i]) + team_embed(0 if i < 5 else 1)
      Optional projection if embed_dim != d_model: Linear(embed_dim → d_model)
      n_layers × TransformerEncoderLayer (d_model, n_heads, dim_feedforward=d_model*ff_mult,
                                          dropout, activation="gelu", batch_first=True,
                                          norm_first=True)
      mean-pool over 10 tokens → [B, d_model]
      Linear(d_model → 1)
    """

    def __init__(self, vocab_size: int, embed_dim: int, d_model: int, n_heads: int,
                 n_layers: int, ff_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.embed_dim = embed_dim
        self.d_model = d_model

        self.hero_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.team_embed = nn.Embedding(2, embed_dim)

        if embed_dim != d_model:
            self.proj = nn.Linear(embed_dim, d_model)
        else:
            self.proj = nn.Identity()

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

        # Pre-compute the team-id pattern for 10 slots [0,0,0,0,0,1,1,1,1,1].
        team_ids = torch.zeros(10, dtype=torch.long)
        team_ids[5:] = 1
        self.register_buffer("team_ids", team_ids, persistent=False)

        self.apply(_xavier_)
        nn.init.normal_(self.hero_embed.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.hero_embed.weight[0].zero_()
        nn.init.normal_(self.team_embed.weight, mean=0.0, std=0.1)

    def forward(self, hero_ids: torch.Tensor) -> torch.Tensor:
        # hero_ids: [B, 10]
        B = hero_ids.size(0)
        h = self.hero_embed(hero_ids)                    # [B, 10, embed_dim]
        t = self.team_embed(self.team_ids).unsqueeze(0)  # [1, 10, embed_dim]
        x = h + t                                        # broadcast → [B, 10, embed_dim]
        x = self.proj(x)                                 # [B, 10, d_model]
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


def build_minimal_transformer(hp: dict, vocab_size: int) -> MinimalTransformer:
    return MinimalTransformer(
        vocab_size=vocab_size,
        embed_dim=int(hp["embed_dim"]),
        d_model=int(hp["d_model"]),
        n_heads=int(hp["n_heads"]),
        n_layers=int(hp["n_layers"]),
        ff_mult=int(hp["ff_mult"]),
        dropout=float(hp["dropout"]),
    )


__all__ = ["MinimalTransformer", "build_minimal_transformer", "count_params"]
