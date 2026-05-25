"""FoundationTransformer for foundation-mvp-740.

Architecture (~5M params, end-to-end):

  Inputs per match:
    hero_ids:     [B, 10]           int        (canonical: sorted asc within each team)
    player_feats: [B, 10, F]        float      (F=8 in this experiment)
    patch_id:     [B]               int

  Tokenization (12-13 tokens per match before task tokens):
    10 hero tokens =
        hero_embed(hero_id) + team_embed(team_id) + feat_proj(player_feat)
        -- summed (FT-Transformer style per-feature linear projection).
        -- NO per-slot positional embedding (proposal: player_slot has no
           semantic content in Turbo; permutation-equivariance within team
           is the correct inductive bias).
    1 patch token = patch_embed(patch_id)               -- when use_patch_token.
    (optional) 1 lobby token -- not enabled in this config.

  Encoder (FT-Transformer skeleton):
    Pre-Norm Transformer with N=n_layers blocks.
    First-layer first-LayerNorm REMOVED (Gorishniy 2021 recipe).
    Per-head (team_query, team_key) 2x2 additive attention bias (Bi 2022
    Pangu-Weather, adapted to (team,team) since within-team slot order is
    arbitrary). Per-block per-head ~ 4 floats; ~ 64 extra params across
    6 layers x 8 heads. Patch token treated as a third "team" (team_id=2)
    so it has a free relation to both teams without claiming a per-slot position.

  Shared decoder (2 blocks, d_model=256, n_heads=8):
    Whisper-style: a task token is prepended; the encoded sequence is
    the memory; the decoder produces a single output per task token.
    Task vocabulary (this MVP):
      <|win|>, <|duration|>, <|items|slot=k|> for k=0..9,
      <|kda|slot=k|>, <|gpm|slot=k|>, <|hd|slot=k|>.
    Per-task projection layers:
      win  -> 1 logit                  (BCE)
      dur  -> 1 scalar                 (SmoothL1 on log(seconds+1))  -- v3 change
      item -> item_vocab_size logits   (BCE multi-label) -- shared across slots
      kda  -> 1 scalar                 (SmoothL1) -- shared across slots
      gpm  -> 1 scalar                 (SmoothL1) -- shared across slots
      hd   -> 1 scalar                 (SmoothL1) -- shared across slots

  forward returns a dict:
    {"win":  [B],
     "dur":  [B]   (scalar regression on log-seconds)  -- v3 change
     "item": [B, 10, item_vocab_size],
     "kda":  [B, 10],
     "gpm":  [B, 10],
     "hd":   [B, 10],
     "encoded": [B, T, d_model]}  -- T = 10 + (1 if patch else 0)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _xavier_(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class TeamTeamAttention(nn.Module):
    """Multi-head self-attention with an additive (team_q, team_k) 2x2 (or 3x3
    when the patch token is present) bias per head per layer.

    Bias is registered as a per-layer parameter table of shape
    [n_heads, n_team_types, n_team_types] (so the patch token, treated as
    team_id=2, gets bias slots for "attending to/from radiant or dire");
    each forward pass gathers the relevant entries based on the team_ids
    tensor passed in.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 n_team_types: int = 3, use_bias: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_bias = bool(use_bias)
        self.n_team_types = int(n_team_types)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout_p = float(dropout)
        if self.use_bias:
            # [n_heads, n_team_types, n_team_types]
            self.team_bias = nn.Parameter(torch.zeros(n_heads, n_team_types, n_team_types))

    def forward(self, x: torch.Tensor, team_ids: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]; team_ids: [B, T]
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, Hd]
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # Scores [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.use_bias:
            # Gather per-(query_team, key_team) bias per head. team_ids: [B, T]
            tq = team_ids.unsqueeze(2).expand(B, T, T)   # [B, T, T] query-team
            tk = team_ids.unsqueeze(1).expand(B, T, T)   # [B, T, T] key-team
            # team_bias: [H, n_tt, n_tt] -> index by (tq, tk)
            bias = self.team_bias[:, tq, tk]              # [H, B, T, T]
            bias = bias.permute(1, 0, 2, 3)               # [B, H, T, T]
            scores = scores + bias
        attn = F.softmax(scores, dim=-1)
        if self.dropout_p > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout_p, training=True)
        out = torch.matmul(attn, v)                       # [B, H, T, Hd]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)


class FoundationEncoderBlock(nn.Module):
    """Pre-Norm transformer block with team-team attention.

    Per FT-Transformer (Gorishniy 2021), the FIRST layer's FIRST LayerNorm
    may be removed -- controlled by `skip_first_ln` flag.
    """

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4,
                 dropout: float = 0.0, use_team_bias: bool = True,
                 skip_first_ln: bool = False):
        super().__init__()
        self.skip_first_ln = bool(skip_first_ln)
        self.norm1 = nn.LayerNorm(d_model) if not self.skip_first_ln else nn.Identity()
        self.attn = TeamTeamAttention(d_model=d_model, n_heads=n_heads,
                                       dropout=dropout, use_bias=use_team_bias)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, team_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), team_ids))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class FoundationDecoderBlock(nn.Module):
    """Standard decoder block: self-attn on task token + cross-attn to encoder + FF.

    The task token sequence is short (1 token at a time per forward), so the
    self-attention is over a single position -- we skip it and just do
    cross-attention from the task token to the encoded memory.
    """

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                  batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # q: [B, 1, D]; memory: [B, T, D]
        nq = self.norm1(q)
        attn_out, _ = self.cross_attn(nq, memory, memory, need_weights=False)
        q = q + self.dropout(attn_out)
        q = q + self.dropout(self.ff(self.norm2(q)))
        return q


# Task-token vocabulary (per-slot tokens get an offset).
# Layout (logical):
#   0: <|win|>
#   1: <|duration|>
#   2..11:  <|items|slot=0..9|>
#   12..21: <|kda|slot=0..9|>
#   22..31: <|gpm|slot=0..9|>
#   32..41: <|hd|slot=0..9|>
TASK_WIN = 0
TASK_DUR = 1
TASK_ITEMS_BASE = 2     # +slot
TASK_KDA_BASE = 12      # +slot
TASK_GPM_BASE = 22      # +slot
TASK_HD_BASE = 32       # +slot
TASK_VOCAB_SIZE = 42


class FoundationTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, d_model: int,
                 n_heads: int, n_layers: int, ff_mult: int = 4,
                 dropout: float = 0.0, n_player_feats: int = 8,
                 use_features: bool = True,
                 n_dur_buckets: int = 8,
                 item_vocab_size: int = 1,
                 patch_vocab_size: int = 8,
                 use_team_team_bias: bool = True,
                 use_patch_token: bool = True,
                 use_lobby_token: bool = False,
                 decoder_n_layers: int = 2,
                 decoder_n_heads: int = 8,
                 remove_first_layer_first_ln: bool = True,
                 dur_loss_mode: str = "regression",
                 use_player_embedding: bool = False,
                 player_vocab_size: int = 1,
                 player_embed_dim: int = 128,
                 player_init_std: float = 0.02):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.d_model = int(d_model)
        self.use_features = bool(use_features)
        self.n_player_feats = int(n_player_feats)
        self.n_dur_buckets = int(n_dur_buckets)
        self.item_vocab_size = int(item_vocab_size)
        self.patch_vocab_size = int(patch_vocab_size)
        self.use_team_team_bias = bool(use_team_team_bias)
        self.use_patch_token = bool(use_patch_token)
        self.use_lobby_token = bool(use_lobby_token)
        self.dur_loss_mode = str(dur_loss_mode)
        if self.dur_loss_mode not in ("regression", "ce"):
            raise ValueError(f"dur_loss_mode must be 'regression' or 'ce', got {dur_loss_mode}")
        self.use_player_embedding = bool(use_player_embedding)
        self.player_vocab_size = int(player_vocab_size)
        self.player_embed_dim = int(player_embed_dim)
        self.player_init_std = float(player_init_std)

        # Hero tokenizer.
        self.hero_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Three "team types" so the patch token can have its own team_id=2 if used.
        self.n_team_types = 3 if (self.use_patch_token or self.use_lobby_token) else 2
        self.team_embed = nn.Embedding(self.n_team_types, embed_dim)
        if embed_dim != d_model:
            self.proj = nn.Linear(embed_dim, d_model)
        else:
            self.proj = nn.Identity()
        self.feat_proj = nn.Linear(n_player_feats, d_model)

        # Player identity embedding (A2). When enabled, a per-account
        # embedding is added to the per-slot token (post hero+team+feat sum).
        # See concepts/embedding-vs-features-gradient-competition.md for the
        # known failure mode: dense features dominate the gradient flow and
        # the embedding stays at near-init. A2 diagnostics detect this.
        if self.use_player_embedding:
            # padding_idx=None: idx 0 is the legitimate "anonymous" embedding,
            # not a no-op pad row.
            self.player_embed = nn.Embedding(self.player_vocab_size, self.player_embed_dim)
            self.player_proj = nn.Linear(self.player_embed_dim, d_model)
        else:
            self.player_embed = None
            self.player_proj = None

        # Patch tokenizer.
        if self.use_patch_token:
            self.patch_embed = nn.Embedding(self.patch_vocab_size, d_model, padding_idx=0)
        else:
            self.patch_embed = None

        # Encoder.
        self.encoder_blocks = nn.ModuleList([
            FoundationEncoderBlock(
                d_model=d_model, n_heads=n_heads, ff_mult=ff_mult,
                dropout=dropout, use_team_bias=self.use_team_team_bias,
                skip_first_ln=(remove_first_layer_first_ln and i == 0),
            ) for i in range(n_layers)
        ])
        # Final encoder LN (standard Pre-Norm tail).
        self.encoder_norm = nn.LayerNorm(d_model)

        # Task token vocabulary (decoder side).
        self.task_token_embed = nn.Embedding(TASK_VOCAB_SIZE, d_model)

        # Decoder.
        self.decoder_blocks = nn.ModuleList([
            FoundationDecoderBlock(
                d_model=d_model, n_heads=decoder_n_heads, ff_mult=ff_mult,
                dropout=dropout,
            ) for _ in range(decoder_n_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # Per-task output projections (operate on decoder hidden states).
        self.win_head = nn.Linear(d_model, 1)
        # A1 toggle: duration head outputs n_dur_buckets logits (CE) OR 1 scalar
        # (SmoothL1 on log(seconds+1)). Controlled by dur_loss_mode.
        if self.dur_loss_mode == "ce":
            self.dur_head = nn.Linear(d_model, self.n_dur_buckets)
        else:
            self.dur_head = nn.Linear(d_model, 1)
        self.item_head = nn.Linear(d_model, max(item_vocab_size, 1))   # shared across slots
        self.kda_head = nn.Linear(d_model, 1)                          # shared across slots
        self.gpm_head = nn.Linear(d_model, 1)
        self.hd_head = nn.Linear(d_model, 1)

        # Static team ids for the 10 hero slots (canonical: 0..4 radiant, 5..9 dire).
        team_ids = torch.zeros(10, dtype=torch.long)
        team_ids[5:] = 1
        self.register_buffer("hero_team_ids", team_ids, persistent=False)

        self.apply(_xavier_)
        nn.init.normal_(self.hero_embed.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.hero_embed.weight[0].zero_()
        nn.init.normal_(self.team_embed.weight, mean=0.0, std=0.1)
        if self.patch_embed is not None:
            nn.init.normal_(self.patch_embed.weight, mean=0.0, std=0.1)
            with torch.no_grad():
                self.patch_embed.weight[0].zero_()
        nn.init.normal_(self.task_token_embed.weight, mean=0.0, std=0.1)
        if self.player_embed is not None:
            nn.init.normal_(self.player_embed.weight, mean=0.0, std=self.player_init_std)

    def _encode_tokens(self, hero_ids: torch.Tensor,
                        player_feats: torch.Tensor | None,
                        patch_id: torch.Tensor | None,
                        hero_mask: torch.Tensor | None = None,
                        patch_mask: torch.Tensor | None = None,
                        account_idx: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the per-match token sequence.

        hero_mask: [B, 10] bool -- True positions are MASKED (PMAE auxiliary).
        patch_mask: [B] bool -- True means mask the patch token.

        Returns:
          x: [B, T, d_model]
          team_ids: [B, T]
        """
        B = hero_ids.size(0)
        device = hero_ids.device
        # Hero tokens.
        h = self.hero_embed(hero_ids)                       # [B, 10, embed_dim]
        t = self.team_embed(self.hero_team_ids).unsqueeze(0).expand(B, -1, -1)   # [B, 10, embed_dim]
        x = self.proj(h + t)                                # [B, 10, d_model]
        if self.use_features:
            if player_feats is None:
                raise ValueError("use_features=True but player_feats is None")
            x = x + self.feat_proj(player_feats)
        # A2: per-slot player identity embedding added to the per-slot token.
        if self.use_player_embedding and self.player_embed is not None:
            if account_idx is None:
                raise ValueError("use_player_embedding=True but account_idx is None")
            pe = self.player_embed(account_idx)            # [B, 10, player_embed_dim]
            x = x + self.player_proj(pe)                   # [B, 10, d_model]
        if hero_mask is not None:
            # Replace masked hero tokens with a zero vector (the model must reconstruct).
            mask = hero_mask.unsqueeze(-1).to(x.dtype)      # [B, 10, 1]
            x = x * (1.0 - mask)

        team_ids = self.hero_team_ids.unsqueeze(0).expand(B, -1)   # [B, 10]

        # Append patch token.
        if self.use_patch_token and self.patch_embed is not None and patch_id is not None:
            patch_tok = self.patch_embed(patch_id).unsqueeze(1)        # [B, 1, d_model]
            # Add team_embed for the patch (team_id=2).
            patch_team_id = torch.full((B, 1), 2, dtype=torch.long, device=device)
            patch_tok = patch_tok + self.proj(self.team_embed(patch_team_id))
            if patch_mask is not None:
                pm = patch_mask.view(B, 1, 1).to(x.dtype)
                patch_tok = patch_tok * (1.0 - pm)
            x = torch.cat([x, patch_tok], dim=1)                       # [B, 11, d_model]
            team_ids = torch.cat([team_ids, patch_team_id], dim=1)
        return x, team_ids

    def encode(self, hero_ids: torch.Tensor,
               player_feats: torch.Tensor | None,
               patch_id: torch.Tensor | None,
               hero_mask: torch.Tensor | None = None,
               patch_mask: torch.Tensor | None = None,
               account_idx: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x, team_ids = self._encode_tokens(hero_ids, player_feats, patch_id,
                                            hero_mask=hero_mask, patch_mask=patch_mask,
                                            account_idx=account_idx)
        for blk in self.encoder_blocks:
            x = blk(x, team_ids)
        x = self.encoder_norm(x)
        return x, team_ids  # [B, T, d_model], [B, T]

    def _decode_task(self, memory: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Run the decoder for a batch of task tokens.

        memory: [B, T, d_model]
        task_ids: [B, n_q] long -- a per-batch sequence of task token IDs.

        Returns: [B, n_q, d_model] decoder hidden states.
        """
        q = self.task_token_embed(task_ids)        # [B, n_q, d_model]
        for blk in self.decoder_blocks:
            q = blk(q, memory)
        return self.decoder_norm(q)

    def forward(self, hero_ids: torch.Tensor,
                player_feats: torch.Tensor | None = None,
                patch_id: torch.Tensor | None = None,
                hero_mask: torch.Tensor | None = None,
                patch_mask: torch.Tensor | None = None,
                account_idx: torch.Tensor | None = None) -> dict:
        """Forward producing all task outputs in one pass.

        Per-batch, we build a task-token sequence of length:
          1 (win) + 1 (dur) + 10 (items) + 10 (kda) + 10 (gpm) + 10 (hd) = 42 tokens.
        The decoder cross-attends each task token to the same encoded memory.
        """
        memory, _team_ids = self.encode(hero_ids, player_feats, patch_id,
                                         hero_mask=hero_mask, patch_mask=patch_mask,
                                         account_idx=account_idx)
        B = hero_ids.size(0)
        device = hero_ids.device
        # Build the task-token sequence (same per batch -- expand).
        task_seq = torch.empty(TASK_VOCAB_SIZE, dtype=torch.long, device=device)
        task_seq[0] = TASK_WIN
        task_seq[1] = TASK_DUR
        for s in range(10):
            task_seq[2 + s] = TASK_ITEMS_BASE + s
            task_seq[12 + s] = TASK_KDA_BASE + s
            task_seq[22 + s] = TASK_GPM_BASE + s
            task_seq[32 + s] = TASK_HD_BASE + s
        task_ids = task_seq.unsqueeze(0).expand(B, -1).contiguous()    # [B, 42]
        h = self._decode_task(memory, task_ids)                         # [B, 42, d_model]

        win_h = h[:, 0, :]
        dur_h = h[:, 1, :]
        items_h = h[:, 2:12, :]    # [B, 10, d_model]
        kda_h = h[:, 12:22, :]
        gpm_h = h[:, 22:32, :]
        hd_h = h[:, 32:42, :]

        dur_out_raw = self.dur_head(dur_h)            # [B, 1] or [B, n_dur_buckets]
        if self.dur_loss_mode == "ce":
            dur_out = dur_out_raw                      # [B, n_dur_buckets] logits
        else:
            dur_out = dur_out_raw.squeeze(-1)          # [B] scalar regression on log-seconds
        out = {
            "win":  self.win_head(win_h).squeeze(-1),           # [B]
            "dur":  dur_out,
            "item": self.item_head(items_h),                     # [B, 10, item_vocab]
            "kda":  self.kda_head(kda_h).squeeze(-1),            # [B, 10]
            "gpm":  self.gpm_head(gpm_h).squeeze(-1),            # [B, 10]
            "hd":   self.hd_head(hd_h).squeeze(-1),              # [B, 10]
            "encoded": memory,                                   # for PMAE recon if needed
        }
        return out


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed = sum(p.numel() for n, p in model.named_parameters() if "embed" in n)
    player_embed = sum(p.numel() for n, p in model.named_parameters() if "player_embed" in n)
    return {"total": int(total), "trainable": int(trainable),
            "embedding": int(embed), "non_embedding": int(total - embed),
            "player_embedding": int(player_embed)}


def build_model(hp: dict, vocab_size: int, n_player_feats: int,
                use_features: bool, *, n_dur_buckets: int = 8,
                item_vocab_size: int = 1, patch_vocab_size: int = 8,
                use_team_team_bias: bool | None = None,
                use_patch_token: bool | None = None,
                use_lobby_token: bool | None = None,
                dur_loss_mode: str = "regression",
                use_player_embedding: bool = False,
                player_vocab_size: int = 1,
                player_embed_dim: int = 128,
                player_init_std: float = 0.02) -> FoundationTransformer:
    return FoundationTransformer(
        vocab_size=vocab_size,
        embed_dim=int(hp["embed_dim"]),
        d_model=int(hp["d_model"]),
        n_heads=int(hp["n_heads"]),
        n_layers=int(hp["n_layers"]),
        ff_mult=int(hp.get("ff_mult", 4)),
        dropout=float(hp.get("dropout", 0.0)),
        n_player_feats=int(n_player_feats),
        use_features=bool(use_features),
        n_dur_buckets=int(n_dur_buckets),
        item_vocab_size=int(item_vocab_size),
        patch_vocab_size=int(patch_vocab_size),
        use_team_team_bias=bool(use_team_team_bias if use_team_team_bias is not None
                                  else hp.get("use_team_team_bias", True)),
        use_patch_token=bool(use_patch_token if use_patch_token is not None
                               else hp.get("use_patch_token", True)),
        use_lobby_token=bool(use_lobby_token if use_lobby_token is not None
                               else hp.get("use_lobby_token", False)),
        decoder_n_layers=int(hp.get("decoder_n_layers", 2)),
        decoder_n_heads=int(hp.get("decoder_n_heads", 8)),
        remove_first_layer_first_ln=bool(hp.get("remove_first_layer_first_ln", True)),
        dur_loss_mode=str(dur_loss_mode),
        use_player_embedding=bool(use_player_embedding),
        player_vocab_size=int(player_vocab_size),
        player_embed_dim=int(player_embed_dim),
        player_init_std=float(player_init_std),
    )


__all__ = ["FoundationTransformer", "build_model", "count_params",
           "TASK_WIN", "TASK_DUR", "TASK_ITEMS_BASE", "TASK_KDA_BASE",
           "TASK_GPM_BASE", "TASK_HD_BASE", "TASK_VOCAB_SIZE"]
