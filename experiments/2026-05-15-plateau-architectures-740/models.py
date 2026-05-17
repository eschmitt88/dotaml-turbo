"""Three architectures for plateau-architectures-740.

All three consume:
  - hero_ids: LongTensor[B, 10] — Radiant heroes 0..4, Dire heroes 5..9, IDs in [1, 150]
  - side_bit: FloatTensor[B, 1] — Radiant-perspective indicator (constant 1 here, kept for
              proposal fidelity)

All produce a single sigmoid logit (binary classification: radiant_win).

Hero embedding table is shared per-model (not across models), with vocab_size=151
so id 0 stays reserved for padding (we never have padding in this task — all
matches have exactly 10 heroes — but the slot exists for safety and matches
DotaML v6 conventions).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _xavier_(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class SimpleFFN(nn.Module):
    """DotaML v4 analogue. Concat 10 hero embeddings + side bit, MLP, sigmoid."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_sizes: list[int],
                 dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        in_dim = 10 * embed_dim + 1
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)
        self.apply(_xavier_)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.embed.weight[0].zero_()

    def forward(self, hero_ids: torch.Tensor, side_bit: torch.Tensor) -> torch.Tensor:
        # hero_ids: [B, 10], side_bit: [B, 1]
        h = self.embed(hero_ids)               # [B, 10, E]
        h = h.flatten(start_dim=1)             # [B, 10*E]
        x = torch.cat([h, side_bit], dim=1)    # [B, 10*E + 1]
        return self.mlp(x).squeeze(-1)         # [B]


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.act(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return self.norm(self.act(h + residual))


class ResidualFFN(nn.Module):
    """DotaML v5 analogue. Same input as SimpleFFN, then residual MLP blocks."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 n_blocks: int, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        in_dim = 10 * embed_dim + 1
        self.proj_in = nn.Linear(in_dim, hidden_size)
        self.blocks = nn.ModuleList([_ResidualBlock(hidden_size, dropout) for _ in range(n_blocks)])
        self.proj_out = nn.Linear(hidden_size, 1)
        self.apply(_xavier_)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.embed.weight[0].zero_()

    def forward(self, hero_ids: torch.Tensor, side_bit: torch.Tensor) -> torch.Tensor:
        h = self.embed(hero_ids).flatten(start_dim=1)
        x = torch.cat([h, side_bit], dim=1)
        x = torch.relu(self.proj_in(x))
        for blk in self.blocks:
            x = blk(x)
        return self.proj_out(x).squeeze(-1)


class DraftTransformer(nn.Module):
    """DotaML v6 analogue (masking off). Self-attention over 10 hero tokens
    + 1 side token, with a learned per-position 'team' embedding. Mean-pool then
    a small head produces the logit.
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 ff_mult: int = 2, dropout: float = 0.0, pool: str = "mean"):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.pool = pool

        self.hero_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # 11 positions: 0..4 = Radiant heroes, 5..9 = Dire heroes, 10 = side token.
        self.pos_embed = nn.Embedding(11, d_model)
        # Side-token learned vector (used as the 11th token's "input" before pos_embed).
        self.side_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # Encode the scalar side bit as a small projection added to the side token.
        self.side_proj = nn.Linear(1, d_model)

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
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        self.apply(_xavier_)
        nn.init.normal_(self.hero_embed.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.hero_embed.weight[0].zero_()
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.side_token, mean=0.0, std=0.02)

    def forward(self, hero_ids: torch.Tensor, side_bit: torch.Tensor) -> torch.Tensor:
        # hero_ids: [B, 10], side_bit: [B, 1]
        B = hero_ids.size(0)
        hero_tok = self.hero_embed(hero_ids)                 # [B, 10, D]
        side_tok = self.side_token.expand(B, 1, -1) + self.side_proj(side_bit).unsqueeze(1)  # [B, 1, D]
        tokens = torch.cat([hero_tok, side_tok], dim=1)      # [B, 11, D]

        pos_ids = torch.arange(11, device=hero_ids.device)
        tokens = tokens + self.pos_embed(pos_ids).unsqueeze(0)

        encoded = self.encoder(tokens)                       # [B, 11, D]
        if self.pool == "mean":
            pooled = encoded.mean(dim=1)
        elif self.pool == "side":
            pooled = encoded[:, -1, :]
        else:
            raise ValueError(f"unknown pool {self.pool!r}")
        return self.head(pooled).squeeze(-1)


def build_model(arch: str, hero_cfg: dict, arch_cfg: dict) -> nn.Module:
    vocab_size = int(hero_cfg["vocab_size"])
    embed_dim = int(hero_cfg["embed_dim"])
    if arch == "simple_ffn":
        return SimpleFFN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_sizes=list(arch_cfg["hidden_sizes"]),
            dropout=float(arch_cfg.get("dropout", 0.0)),
        )
    if arch == "residual_ffn":
        return ResidualFFN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=int(arch_cfg["hidden_size"]),
            n_blocks=int(arch_cfg["n_blocks"]),
            dropout=float(arch_cfg.get("dropout", 0.0)),
        )
    if arch == "transformer":
        return DraftTransformer(
            vocab_size=vocab_size,
            d_model=int(arch_cfg["d_model"]),
            n_heads=int(arch_cfg["n_heads"]),
            n_layers=int(arch_cfg["n_layers"]),
            ff_mult=int(arch_cfg.get("ff_mult", 2)),
            dropout=float(arch_cfg.get("dropout", 0.0)),
            pool=str(arch_cfg.get("pool", "mean")),
        )
    raise ValueError(f"unknown arch {arch!r}")


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed = 0
    for name, p in model.named_parameters():
        if "embed" in name and "pos" not in name and "side" not in name:
            embed += p.numel()
    return {"total": int(total), "trainable": int(trainable),
            "embedding": int(embed), "non_embedding": int(total - embed)}


__all__ = ["SimpleFFN", "ResidualFFN", "DraftTransformer", "build_model", "count_params"]
