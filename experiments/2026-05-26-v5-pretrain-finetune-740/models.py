"""FoundationTransformer for v5-pretrain-finetune-740.

Forked from experiments/2026-05-25-v4-iso-teambias-extended-740/models.py.

Architectural changes vs v4:

1. **Six input groups per slot** (during pre-train):
     player_block, hero_token, item_list, kda, gpm, hd.
   Each has a per-group LEARNED mask token (a d_model vector); when a
   group is masked for an example, the masked-out per-slot contribution
   is REPLACED with the mask token (broadcast to [B, 10, d_model]).

2. **New input projections**:
     - `item_proj`: nn.Linear(item_vocab_size, d_model) -- consumes the
        305-dim multi-label item vector per slot at PRE-TRAIN time.
     - `kda_proj`: nn.Linear(1, d_model)
     - `gpm_proj`: nn.Linear(1, d_model)
     - `hd_proj`:  nn.Linear(1, d_model)
   These contribute additively to the per-slot token (same FT-Transformer
   pattern as feat_proj). All four are zero-init so a model loaded with
   these heads but pre-train switched off remains identical to v4 at
   step 0.

3. **Per-group mask tokens**: 6 learnable d_model parameters
   (player_block_mask, hero_token_mask, item_mask, kda_mask, gpm_mask,
   hd_mask). Initialized N(0, 0.02). When a group is masked the
   per-slot contribution from that group is REPLACED with the mask
   token (broadcast).

4. **No player_embed, no patch_token, no UW-SO** (v5 drops all of these
   for simplicity — they're closed axes per prior experiments).

5. **Pre-train reconstruction heads** (separate from task heads):
     - `pretrain_hero_head`: Linear(d_model, hero_vocab_size) for the
        130-way hero CE reconstruction.
     - `pretrain_player_head`: Linear(d_model, n_player_feats) for the
        SmoothL1 player_block reconstruction.
     - `pretrain_item_head`: Linear(d_model, item_vocab_size) for BCE
        item reconstruction.
     - `pretrain_kda_head`, `pretrain_gpm_head`, `pretrain_hd_head`:
        Linear(d_model, 1) each, for SmoothL1 scalar reconstruction.
   These operate directly on the encoded per-slot tokens (NO decoder
   pathway) -- mirrors standard BERT MLM where the prediction head is
   a tiny linear projection from the encoder output.

6. **Forward signature** accepts a `mask_dict` with per-group boolean
   masks (shape [B] or [B, 10] depending on group). When mask_dict is
   None, behaves like the v4 forward (player_block always on; item/
   scalar inputs only contribute if scalar_inputs/items_input is
   provided).
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
    """Multi-head self-attention with an additive (team_q, team_k) 2x2 bias
    per head per layer. Same as v4.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 n_team_types: int = 2, use_bias: bool = True):
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
            self.team_bias = nn.Parameter(torch.zeros(n_heads, n_team_types, n_team_types))

    def forward(self, x: torch.Tensor, team_ids: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.use_bias:
            tq = team_ids.unsqueeze(2).expand(B, T, T)
            tk = team_ids.unsqueeze(1).expand(B, T, T)
            bias = self.team_bias[:, tq, tk]
            bias = bias.permute(1, 0, 2, 3)
            scores = scores + bias
        attn = F.softmax(scores, dim=-1)
        if self.dropout_p > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout_p, training=True)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)


class FoundationEncoderBlock(nn.Module):
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
        nq = self.norm1(q)
        attn_out, _ = self.cross_attn(nq, memory, memory, need_weights=False)
        q = q + self.dropout(attn_out)
        q = q + self.dropout(self.ff(self.norm2(q)))
        return q


TASK_WIN = 0
TASK_DUR = 1
TASK_ITEMS_BASE = 2
TASK_KDA_BASE = 12
TASK_GPM_BASE = 22
TASK_HD_BASE = 32
TASK_VOCAB_SIZE = 42


class FoundationTransformerV5(nn.Module):
    """v5 encoder: 6 input groups + per-group learned mask tokens + pre-train
    reconstruction heads + v4-style multi-task decoder heads.
    """

    def __init__(self, vocab_size: int, embed_dim: int, d_model: int,
                 n_heads: int, n_layers: int, ff_mult: int = 4,
                 dropout: float = 0.0, n_player_feats: int = 8,
                 n_dur_buckets: int = 8,
                 item_vocab_size: int = 1,
                 use_team_team_bias: bool = True,
                 decoder_n_layers: int = 2,
                 decoder_n_heads: int = 8,
                 remove_first_layer_first_ln: bool = True,
                 dur_loss_mode: str = "ce"):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.d_model = int(d_model)
        self.n_player_feats = int(n_player_feats)
        self.n_dur_buckets = int(n_dur_buckets)
        self.item_vocab_size = int(item_vocab_size)
        self.hero_vocab_size = int(vocab_size)
        self.use_team_team_bias = bool(use_team_team_bias)
        self.dur_loss_mode = str(dur_loss_mode)
        if self.dur_loss_mode not in ("regression", "ce"):
            raise ValueError(f"dur_loss_mode must be 'regression' or 'ce', got {dur_loss_mode}")

        # Encoder-side tokenizers.
        self.hero_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.n_team_types = 2
        self.team_embed = nn.Embedding(self.n_team_types, embed_dim)
        if embed_dim != d_model:
            self.proj = nn.Linear(embed_dim, d_model)
        else:
            self.proj = nn.Identity()

        # Group input projections. All four "rich" projections (item/kda/
        # gpm/hd) are zero-init so a v4-compatible no-rich-input forward
        # behaves identically.
        self.feat_proj = nn.Linear(n_player_feats, d_model)
        self.item_proj = nn.Linear(max(item_vocab_size, 1), d_model, bias=False)
        self.kda_proj = nn.Linear(1, d_model, bias=False)
        self.gpm_proj = nn.Linear(1, d_model, bias=False)
        self.hd_proj  = nn.Linear(1, d_model, bias=False)
        nn.init.zeros_(self.item_proj.weight)
        nn.init.zeros_(self.kda_proj.weight)
        nn.init.zeros_(self.gpm_proj.weight)
        nn.init.zeros_(self.hd_proj.weight)

        # Per-group learned mask tokens (one d_model vector each).
        # When a group is masked, the per-slot contribution from that
        # group is REPLACED with the mask token (broadcast to all 10 slots).
        self.player_block_mask = nn.Parameter(torch.zeros(d_model))
        self.hero_token_mask   = nn.Parameter(torch.zeros(d_model))
        self.item_mask         = nn.Parameter(torch.zeros(d_model))
        self.kda_mask          = nn.Parameter(torch.zeros(d_model))
        self.gpm_mask          = nn.Parameter(torch.zeros(d_model))
        self.hd_mask           = nn.Parameter(torch.zeros(d_model))
        for p in (self.player_block_mask, self.hero_token_mask, self.item_mask,
                  self.kda_mask, self.gpm_mask, self.hd_mask):
            nn.init.normal_(p, mean=0.0, std=0.02)

        # Encoder blocks.
        self.encoder_blocks = nn.ModuleList([
            FoundationEncoderBlock(
                d_model=d_model, n_heads=n_heads, ff_mult=ff_mult,
                dropout=dropout, use_team_bias=self.use_team_team_bias,
                skip_first_ln=(remove_first_layer_first_ln and i == 0),
            ) for i in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Pre-train reconstruction heads (run on encoded per-slot output).
        self.pretrain_player_head = nn.Linear(d_model, n_player_feats)
        self.pretrain_hero_head   = nn.Linear(d_model, vocab_size)
        self.pretrain_item_head   = nn.Linear(d_model, max(item_vocab_size, 1))
        self.pretrain_kda_head    = nn.Linear(d_model, 1)
        self.pretrain_gpm_head    = nn.Linear(d_model, 1)
        self.pretrain_hd_head     = nn.Linear(d_model, 1)

        # Decoder + multi-task heads (used at Phase 2B, frozen-discarded at Phase 2A).
        self.task_token_embed = nn.Embedding(TASK_VOCAB_SIZE, d_model)
        self.decoder_blocks = nn.ModuleList([
            FoundationDecoderBlock(
                d_model=d_model, n_heads=decoder_n_heads, ff_mult=ff_mult,
                dropout=dropout,
            ) for _ in range(decoder_n_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        self.win_head = nn.Linear(d_model, 1)
        if self.dur_loss_mode == "ce":
            self.dur_head = nn.Linear(d_model, self.n_dur_buckets)
        else:
            self.dur_head = nn.Linear(d_model, 1)
        self.item_head = nn.Linear(d_model, max(item_vocab_size, 1))
        self.kda_head = nn.Linear(d_model, 1)
        self.gpm_head = nn.Linear(d_model, 1)
        self.hd_head = nn.Linear(d_model, 1)

        # Phase 2A linear-probe head (single tiny head over pooled CLS-like
        # representation, constructed lazily — not part of pre-train state.
        # Kept as a separate module so reset between probes is trivial.
        self.linear_probe_head = None  # set externally in linear_probe.py

        team_ids = torch.zeros(10, dtype=torch.long)
        team_ids[5:] = 1
        self.register_buffer("hero_team_ids", team_ids, persistent=False)

        self.apply(_xavier_)
        # Override embedding inits with the v4 convention.
        nn.init.normal_(self.hero_embed.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.hero_embed.weight[0].zero_()
        nn.init.normal_(self.team_embed.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.task_token_embed.weight, mean=0.0, std=0.1)
        # Re-zero the four rich projections (xavier overwrote them).
        nn.init.zeros_(self.item_proj.weight)
        nn.init.zeros_(self.kda_proj.weight)
        nn.init.zeros_(self.gpm_proj.weight)
        nn.init.zeros_(self.hd_proj.weight)

    def _encode_tokens(self, hero_ids: torch.Tensor,
                        player_feats: torch.Tensor | None,
                        items_input: torch.Tensor | None,
                        scalar_inputs: torch.Tensor | None,
                        mask_dict: dict | None = None,
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the per-match [B, 10, d_model] token sequence with optional
        per-group masking.

        mask_dict (optional) maps group name -> bool tensor:
          'player_block': [B] bool  (whole-group mask per example)
          'hero_token':   [B] bool
          'item_list':    [B] bool
          'kda':          [B] bool
          'gpm':          [B] bool
          'hd':           [B] bool
        For "masked" examples, the per-slot contribution from that group
        is REPLACED by the per-group mask token (broadcast across 10 slots).
        For non-masked groups, the contribution is computed normally; if
        the corresponding input is None (e.g. items_input=None at Phase
        2A/2B), the group is treated as if FULLY masked (contributes the
        learned mask token at every position) — matching the inference
        distribution.
        """
        B = hero_ids.size(0)
        device = hero_ids.device
        md = mask_dict or {}

        # Hero token contribution.
        if md.get("hero_token") is not None:
            mh = md["hero_token"].to(device).view(B, 1, 1).float()
        else:
            mh = torch.zeros(B, 1, 1, device=device)
        hero_contrib_unmasked = self.proj(self.hero_embed(hero_ids))   # [B, 10, d_model]
        hero_mask_tok = self.hero_token_mask.view(1, 1, -1).expand(B, 10, -1)
        hero_contrib = mh * hero_mask_tok + (1.0 - mh) * hero_contrib_unmasked

        # Team token (always-on; not a maskable group).
        team_contrib = self.proj(
            self.team_embed(self.hero_team_ids).unsqueeze(0).expand(B, -1, -1)
        )

        # Player block contribution.
        if md.get("player_block") is not None:
            mp = md["player_block"].to(device).view(B, 1, 1).float()
        else:
            mp = torch.zeros(B, 1, 1, device=device)
        if player_feats is not None:
            pb_unmasked = self.feat_proj(player_feats)
        else:
            pb_unmasked = torch.zeros(B, 10, self.d_model, device=device, dtype=hero_contrib.dtype)
        pb_mask_tok = self.player_block_mask.view(1, 1, -1).expand(B, 10, -1)
        # If player_feats is None we treat it as fully masked.
        if player_feats is None:
            mp = torch.ones(B, 1, 1, device=device)
        pb_contrib = mp * pb_mask_tok + (1.0 - mp) * pb_unmasked

        # Item list contribution. Absent (items_input=None) <=> fully masked.
        mi = (md.get("item_list").to(device).view(B, 1, 1).float()
              if md.get("item_list") is not None else torch.zeros(B, 1, 1, device=device))
        if items_input is None:
            mi = torch.ones(B, 1, 1, device=device)
            it_unmasked = torch.zeros(B, 10, self.d_model, device=device,
                                       dtype=hero_contrib.dtype)
        else:
            it_unmasked = self.item_proj(items_input)
        it_mask_tok = self.item_mask.view(1, 1, -1).expand(B, 10, -1)
        it_contrib = mi * it_mask_tok + (1.0 - mi) * it_unmasked

        # Scalar inputs (kda, gpm, hd) — each is its own group.
        def _scalar_contrib(idx: int, group: str, proj: nn.Linear,
                             mask_tok: nn.Parameter) -> torch.Tensor:
            mm = (md.get(group).to(device).view(B, 1, 1).float()
                  if md.get(group) is not None else torch.zeros(B, 1, 1, device=device))
            if scalar_inputs is None:
                mm = torch.ones(B, 1, 1, device=device)
                un = torch.zeros(B, 10, self.d_model, device=device,
                                  dtype=hero_contrib.dtype)
            else:
                un = proj(scalar_inputs[:, :, idx:idx + 1])   # [B, 10, d_model]
            mt = mask_tok.view(1, 1, -1).expand(B, 10, -1)
            return mm * mt + (1.0 - mm) * un

        kda_contrib = _scalar_contrib(0, "kda", self.kda_proj, self.kda_mask)
        gpm_contrib = _scalar_contrib(1, "gpm", self.gpm_proj, self.gpm_mask)
        hd_contrib  = _scalar_contrib(2, "hd",  self.hd_proj,  self.hd_mask)

        x = (hero_contrib + team_contrib + pb_contrib + it_contrib
             + kda_contrib + gpm_contrib + hd_contrib)
        team_ids = self.hero_team_ids.unsqueeze(0).expand(B, -1)
        return x, team_ids

    def encode(self, hero_ids: torch.Tensor,
               player_feats: torch.Tensor | None = None,
               items_input: torch.Tensor | None = None,
               scalar_inputs: torch.Tensor | None = None,
               mask_dict: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x, team_ids = self._encode_tokens(hero_ids, player_feats, items_input,
                                            scalar_inputs, mask_dict=mask_dict)
        for blk in self.encoder_blocks:
            x = blk(x, team_ids)
        x = self.encoder_norm(x)
        return x, team_ids

    def _decode_task(self, memory: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        q = self.task_token_embed(task_ids)
        for blk in self.decoder_blocks:
            q = blk(q, memory)
        return self.decoder_norm(q)

    def forward_multitask(self, hero_ids: torch.Tensor,
                           player_feats: torch.Tensor | None = None,
                           items_input: torch.Tensor | None = None,
                           scalar_inputs: torch.Tensor | None = None,
                           mask_dict: dict | None = None) -> dict:
        memory, _team_ids = self.encode(hero_ids, player_feats, items_input,
                                          scalar_inputs, mask_dict=mask_dict)
        B = hero_ids.size(0)
        device = hero_ids.device
        task_seq = torch.empty(TASK_VOCAB_SIZE, dtype=torch.long, device=device)
        task_seq[0] = TASK_WIN
        task_seq[1] = TASK_DUR
        for s in range(10):
            task_seq[2 + s] = TASK_ITEMS_BASE + s
            task_seq[12 + s] = TASK_KDA_BASE + s
            task_seq[22 + s] = TASK_GPM_BASE + s
            task_seq[32 + s] = TASK_HD_BASE + s
        task_ids = task_seq.unsqueeze(0).expand(B, -1).contiguous()
        h = self._decode_task(memory, task_ids)
        win_h = h[:, 0, :]
        dur_h = h[:, 1, :]
        items_h = h[:, 2:12, :]
        kda_h = h[:, 12:22, :]
        gpm_h = h[:, 22:32, :]
        hd_h = h[:, 32:42, :]
        dur_out_raw = self.dur_head(dur_h)
        dur_out = dur_out_raw if self.dur_loss_mode == "ce" else dur_out_raw.squeeze(-1)
        return {
            "win":  self.win_head(win_h).squeeze(-1),
            "dur":  dur_out,
            "item": self.item_head(items_h),
            "kda":  self.kda_head(kda_h).squeeze(-1),
            "gpm":  self.gpm_head(gpm_h).squeeze(-1),
            "hd":   self.hd_head(hd_h).squeeze(-1),
            "encoded": memory,
        }

    def forward_pretrain(self, hero_ids: torch.Tensor,
                          player_feats: torch.Tensor,
                          items_input: torch.Tensor,
                          scalar_inputs: torch.Tensor,
                          mask_dict: dict) -> dict:
        """Pre-train forward. Returns per-group predictions PLUS the encoded
        tokens (for EMA-teacher target alignment in the train loop).

        mask_dict: required, per-group [B] bool tensors.
        """
        memory, _ = self.encode(hero_ids, player_feats, items_input,
                                  scalar_inputs, mask_dict=mask_dict)
        # Reconstruction heads operate on the [B, 10, d_model] per-slot tokens.
        return {
            "encoded": memory,
            "pred_player": self.pretrain_player_head(memory),  # [B, 10, n_pf]
            "pred_hero":   self.pretrain_hero_head(memory),    # [B, 10, hero_vocab]
            "pred_item":   self.pretrain_item_head(memory),    # [B, 10, item_vocab]
            "pred_kda":    self.pretrain_kda_head(memory).squeeze(-1),  # [B, 10]
            "pred_gpm":    self.pretrain_gpm_head(memory).squeeze(-1),
            "pred_hd":     self.pretrain_hd_head(memory).squeeze(-1),
        }

    def pooled(self, memory: torch.Tensor) -> torch.Tensor:
        """Mean-pool across the 10 hero slots — used by the linear probe."""
        return memory.mean(dim=1)


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed = sum(p.numel() for n, p in model.named_parameters() if "embed" in n)
    return {"total": int(total), "trainable": int(trainable),
            "embedding": int(embed), "non_embedding": int(total - embed)}


def encoder_param_names(model: FoundationTransformerV5) -> list[str]:
    """Names of parameters that constitute the 'encoder' for Phase 2B
    parameter-group LR splitting. Includes hero/team embed, all four rich
    input projections + per-group mask tokens, the encoder blocks, and
    the encoder LN. EXCLUDES the pre-train reconstruction heads, the
    decoder, task token embed, and the per-task output heads.
    """
    enc_prefixes = (
        "hero_embed.", "team_embed.", "proj.", "feat_proj.",
        "item_proj.", "kda_proj.", "gpm_proj.", "hd_proj.",
        "player_block_mask", "hero_token_mask", "item_mask",
        "kda_mask", "gpm_mask", "hd_mask",
        "encoder_blocks.", "encoder_norm.",
    )
    return [n for n, _ in model.named_parameters()
            if any(n.startswith(p) or n == p for p in enc_prefixes)]


def build_model_v5(hp: dict, vocab_size: int, n_player_feats: int,
                    *, n_dur_buckets: int = 8,
                    item_vocab_size: int = 1,
                    use_team_team_bias: bool | None = None,
                    dur_loss_mode: str = "ce") -> FoundationTransformerV5:
    return FoundationTransformerV5(
        vocab_size=vocab_size,
        embed_dim=int(hp["embed_dim"]),
        d_model=int(hp["d_model"]),
        n_heads=int(hp["n_heads"]),
        n_layers=int(hp["n_layers"]),
        ff_mult=int(hp.get("ff_mult", 4)),
        dropout=float(hp.get("dropout", 0.0)),
        n_player_feats=int(n_player_feats),
        n_dur_buckets=int(n_dur_buckets),
        item_vocab_size=int(item_vocab_size),
        use_team_team_bias=bool(use_team_team_bias if use_team_team_bias is not None
                                  else hp.get("use_team_team_bias", True)),
        decoder_n_layers=int(hp.get("decoder_n_layers", 2)),
        decoder_n_heads=int(hp.get("decoder_n_heads", 8)),
        remove_first_layer_first_ln=bool(hp.get("remove_first_layer_first_ln", True)),
        dur_loss_mode=str(dur_loss_mode),
    )


__all__ = ["FoundationTransformerV5", "build_model_v5", "count_params",
           "encoder_param_names",
           "TASK_WIN", "TASK_DUR", "TASK_ITEMS_BASE", "TASK_KDA_BASE",
           "TASK_GPM_BASE", "TASK_HD_BASE", "TASK_VOCAB_SIZE"]
