"""MultiHeadTransformer -- shared encoder + 4 task heads.

Encoder is cleanup-740's MinimalTransformerWithFeatures VERBATIM (hero_embed +
team_embed + optional Linear(n_player_feats, d_model) per-slot feature
injection + N TransformerEncoderLayers). Mean-pool over 10 tokens is preserved
as the "match representation" (the proposal called this "CLS"; cleanup-740
uses mean-pool, so we keep mean-pool to make the win_only_sanity ablation
reproduce cleanup-740 to machine precision).

Heads:
  win_head: Linear(d_model, 1)                  -- BCEWithLogits(radiant_win) on pooled
  dur_head: Linear(d_model, n_dur_buckets)      -- CrossEntropy(duration_bucket) on pooled
  item_head: Linear(d_model, item_vocab_size)   -- BCEWithLogits multi-label per slot
  aux_head: Linear(d_model, n_aux)              -- SmoothL1 per slot (standardized targets)

Notes:
  - Item and aux heads share the SAME per-slot Linear across all 10 slots.
    The slot-specific info enters through the encoder via team_embed + per-slot
    player_feats; the heads are slot-symmetric (5 radiant + 5 dire) and the
    encoder learns to differentiate. This keeps head param count linear in
    item_vocab_size rather than 10x.
  - When `multitask=False` the model still constructs the heads (for state-
    dict layout stability across ablations) but `forward()` only runs the
    win head and returns a degenerate dict.

The win_only_sanity ablation runs through forward(multitask=False) and must
return win logits computed via the EXACT cleanup-740 path so the sanity
replication tracks at ~1e-4.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _xavier_(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MultiHeadTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, d_model: int,
                 n_heads: int, n_layers: int, ff_mult: int = 2,
                 dropout: float = 0.0, n_player_feats: int = 8,
                 use_features: bool = True,
                 n_dur_buckets: int = 8,
                 item_vocab_size: int = 0,
                 n_aux: int = 0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.use_features = bool(use_features)
        self.n_player_feats = int(n_player_feats)
        self.n_dur_buckets = int(n_dur_buckets)
        self.item_vocab_size = int(item_vocab_size)
        self.n_aux = int(n_aux)

        # Encoder (cleanup-740 verbatim).
        self.hero_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.team_embed = nn.Embedding(2, embed_dim)
        if embed_dim != d_model:
            self.proj = nn.Linear(embed_dim, d_model)
        else:
            self.proj = nn.Identity()
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

        # Heads.
        # Win head (kept name 'head' so a cleanup-740 state-dict could be partial-loaded).
        self.head = nn.Linear(d_model, 1)
        # Duration head over pooled rep.
        self.dur_head = nn.Linear(d_model, max(n_dur_buckets, 1))
        # Item head: shared across slots, applied to each slot's encoded vector.
        self.item_head = nn.Linear(d_model, max(item_vocab_size, 1))
        # Aux head: shared across slots.
        self.aux_head = nn.Linear(d_model, max(n_aux, 1))

        team_ids = torch.zeros(10, dtype=torch.long)
        team_ids[5:] = 1
        self.register_buffer("team_ids", team_ids, persistent=False)

        self.apply(_xavier_)
        nn.init.normal_(self.hero_embed.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.hero_embed.weight[0].zero_()
        nn.init.normal_(self.team_embed.weight, mean=0.0, std=0.1)

    def encode(self, hero_ids: torch.Tensor,
               player_feats: torch.Tensor | None) -> torch.Tensor:
        h = self.hero_embed(hero_ids)
        t = self.team_embed(self.team_ids).unsqueeze(0)
        x = h + t
        x = self.proj(x)
        if self.use_features:
            if player_feats is None:
                raise ValueError("use_features=True but player_feats is None")
            x = x + self.feat_proj(player_feats)
        x = self.encoder(x)
        return x  # [B, 10, d_model]

    def forward(self, hero_ids: torch.Tensor,
                player_feats: torch.Tensor | None = None,
                multitask: bool = False) -> dict:
        """Returns dict with always-present 'win'; if multitask, also
        'dur', 'item', 'aux'.
        """
        x = self.encode(hero_ids, player_feats)     # [B, 10, d_model]
        pooled = x.mean(dim=1)                       # [B, d_model]
        win_logits = self.head(pooled).squeeze(-1)   # [B]
        out = {"win": win_logits}
        if multitask:
            out["dur"] = self.dur_head(pooled)              # [B, n_dur_buckets]
            out["item"] = self.item_head(x)                  # [B, 10, item_vocab_size]
            out["aux"] = self.aux_head(x)                    # [B, 10, n_aux]
        return out


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
                use_features: bool, *, n_dur_buckets: int = 8,
                item_vocab_size: int = 0, n_aux: int = 0) -> MultiHeadTransformer:
    return MultiHeadTransformer(
        vocab_size=vocab_size,
        embed_dim=int(hp["embed_dim"]),
        d_model=int(hp["d_model"]),
        n_heads=int(hp["n_heads"]),
        n_layers=int(hp["n_layers"]),
        ff_mult=int(hp["ff_mult"]),
        dropout=float(hp["dropout"]),
        n_player_feats=int(n_player_feats),
        use_features=bool(use_features),
        n_dur_buckets=int(n_dur_buckets),
        item_vocab_size=int(item_vocab_size),
        n_aux=int(n_aux),
    )


__all__ = ["MultiHeadTransformer", "build_model", "count_params"]
