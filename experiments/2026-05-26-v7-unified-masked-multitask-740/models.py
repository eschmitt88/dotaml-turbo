"""FoundationTransformerV7 for v7-unified-masked-multitask-740.

Forked from experiments/2026-05-25-v4-iso-teambias-extended-740/models.py.

v7 architectural changes vs v4:

1. **All input groups MASKABLE** with a learned mask embedding per group.
   Eight per-slot input groups (hero, player_feat, items, kills,
   deaths, assists, gpm, hd) and two per-match input groups (duration,
   win). The slot token is the SUM of the 8 per-slot contributions
   (each contribution is either the projected value or the group's
   learned mask embedding when masked). Duration + win occupy two
   extra per-match positions, bringing the sequence length to 12.

2. **Separate K, D, A heads** (not composite KDA): each is its own
   Linear(d_model, 1) with its own 10 task tokens.

3. **Duration as scalar regression** (single task token, SmoothL1 on
   log(seconds+1)).

Per-slot final token (s = 0..9):
    slot[s] = (hero | hero_mask)
            + (player_feat_proj | pf_mask)
            + (items_pool      | items_mask)
            + (kills_proj      | kills_mask)
            + (deaths_proj     | deaths_mask)
            + (assists_proj    | assists_mask)
            + (gpm_proj        | gpm_mask)
            + (hd_proj         | hd_mask)
            + team_embed[team(s)]

Per-match tokens (appended to the 10 slots):
    dur_tok = (dur_proj | dur_mask) + team_embed[2]
    win_tok = (win_embed | win_mask) + team_embed[2]

Task token vocabulary (62 total):
    0:        TASK_WIN          (1 head)
    1:        TASK_DUR          (1 head)
    2..11:    TASK_ITEMS_0..9   (10)
    12..21:   TASK_KILLS_0..9   (10) NEW
    22..31:   TASK_DEATHS_0..9  (10) NEW
    32..41:   TASK_ASSISTS_0..9 (10) NEW
    42..51:   TASK_GPM_0..9     (10)
    52..61:   TASK_HD_0..9      (10)

forward returns a dict with keys:
    win  [B], dur [B], item [B,10,V],
    kills [B,10], deaths [B,10], assists [B,10],
    gpm [B,10], hd [B,10], encoded [B,T,d_model]
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
    """Same as v4: per-head (team_q, team_k) additive bias."""
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
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0):
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


# Task token layout (62 tokens).
TASK_WIN = 0
TASK_DUR = 1
TASK_ITEMS_BASE = 2     # +slot
TASK_KILLS_BASE = 12    # +slot
TASK_DEATHS_BASE = 22   # +slot
TASK_ASSISTS_BASE = 32  # +slot
TASK_GPM_BASE = 42      # +slot
TASK_HD_BASE = 52       # +slot
TASK_VOCAB_SIZE = 62

# Per-slot input group names (ordered consistently across the codebase).
SLOT_GROUPS = ["hero", "player_feat", "items", "kills", "deaths", "assists", "gpm", "hd"]
# Per-match input group names.
MATCH_GROUPS = ["duration", "win"]


class FoundationTransformerV7(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, d_model: int,
                 n_heads: int, n_layers: int, ff_mult: int = 4,
                 dropout: float = 0.0, n_player_feats: int = 8,
                 item_vocab_size: int = 1,
                 use_team_team_bias: bool = True,
                 decoder_n_layers: int = 2,
                 decoder_n_heads: int = 8,
                 remove_first_layer_first_ln: bool = True):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.d_model = int(d_model)
        self.n_player_feats = int(n_player_feats)
        self.item_vocab_size = int(item_vocab_size)
        self.use_team_team_bias = bool(use_team_team_bias)

        # Hero embedding + projection to d_model.
        self.hero_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 3 team types: radiant=0, dire=1, match-level=2 (used by dur/win tokens).
        self.n_team_types = 3
        self.team_embed = nn.Embedding(self.n_team_types, embed_dim)
        if embed_dim != d_model:
            self.hero_proj = nn.Linear(embed_dim, d_model)
        else:
            self.hero_proj = nn.Identity()

        # Per-slot continuous projections (scalar inputs -> d_model).
        self.feat_proj = nn.Linear(n_player_feats, d_model)
        self.kills_proj = nn.Linear(1, d_model)
        self.deaths_proj = nn.Linear(1, d_model)
        self.assists_proj = nn.Linear(1, d_model)
        self.gpm_proj = nn.Linear(1, d_model)
        self.hd_proj = nn.Linear(1, d_model)
        # Items: 305-dim sparse multi-hot. Use an item embedding table
        # (item_vocab_size x d_model); pool as SUM(items present) / sqrt(K).
        self.item_input_embed = nn.Embedding(item_vocab_size, d_model)
        # Per-match continuous: duration scalar (log-seconds).
        self.dur_input_proj = nn.Linear(1, d_model)
        # Per-match categorical: win in {0, 1} -> 2-row embedding.
        self.win_input_embed = nn.Embedding(2, d_model)

        # ----- Learned mask embeddings (one per input group) -----
        # Per-slot:
        self.hero_mask_embed     = nn.Parameter(torch.zeros(d_model))
        self.pf_mask_embed       = nn.Parameter(torch.zeros(d_model))
        self.items_mask_embed    = nn.Parameter(torch.zeros(d_model))
        self.kills_mask_embed    = nn.Parameter(torch.zeros(d_model))
        self.deaths_mask_embed   = nn.Parameter(torch.zeros(d_model))
        self.assists_mask_embed  = nn.Parameter(torch.zeros(d_model))
        self.gpm_mask_embed      = nn.Parameter(torch.zeros(d_model))
        self.hd_mask_embed       = nn.Parameter(torch.zeros(d_model))
        # Per-match:
        self.dur_mask_embed      = nn.Parameter(torch.zeros(d_model))
        self.win_mask_embed      = nn.Parameter(torch.zeros(d_model))

        # ----- Encoder -----
        self.encoder_blocks = nn.ModuleList([
            FoundationEncoderBlock(
                d_model=d_model, n_heads=n_heads, ff_mult=ff_mult,
                dropout=dropout, use_team_bias=self.use_team_team_bias,
                skip_first_ln=(remove_first_layer_first_ln and i == 0),
            ) for i in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # ----- Task tokens (decoder side) -----
        self.task_token_embed = nn.Embedding(TASK_VOCAB_SIZE, d_model)

        # ----- Decoder -----
        self.decoder_blocks = nn.ModuleList([
            FoundationDecoderBlock(
                d_model=d_model, n_heads=decoder_n_heads, ff_mult=ff_mult,
                dropout=dropout,
            ) for _ in range(decoder_n_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # ----- Heads -----
        self.win_head = nn.Linear(d_model, 1)
        self.dur_head = nn.Linear(d_model, 1)
        self.item_head = nn.Linear(d_model, max(item_vocab_size, 1))
        self.kills_head = nn.Linear(d_model, 1)
        self.deaths_head = nn.Linear(d_model, 1)
        self.assists_head = nn.Linear(d_model, 1)
        self.gpm_head = nn.Linear(d_model, 1)
        self.hd_head = nn.Linear(d_model, 1)

        # Static team ids: slots 0..4 radiant, 5..9 dire, plus two match
        # tokens (positions 10, 11) at team_id=2.
        team_ids_slots = torch.zeros(10, dtype=torch.long)
        team_ids_slots[5:] = 1
        self.register_buffer("hero_team_ids", team_ids_slots, persistent=False)

        self.apply(_xavier_)
        nn.init.normal_(self.hero_embed.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.hero_embed.weight[0].zero_()
        nn.init.normal_(self.team_embed.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.task_token_embed.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_input_embed.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.win_input_embed.weight, mean=0.0, std=0.1)
        # Mask embeddings: small init so they don't dominate the
        # un-masked branch at step 0.
        for p in (self.hero_mask_embed, self.pf_mask_embed, self.items_mask_embed,
                  self.kills_mask_embed, self.deaths_mask_embed, self.assists_mask_embed,
                  self.gpm_mask_embed, self.hd_mask_embed,
                  self.dur_mask_embed, self.win_mask_embed):
            nn.init.normal_(p, mean=0.0, std=0.02)

    # ----- Per-group contribution helpers -----

    def _hero_contrib(self, hero_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """hero_ids: [B, 10] long; mask: [B, 10] bool (True = masked).
        Returns [B, 10, d_model].
        """
        B = hero_ids.size(0)
        h = self.hero_embed(hero_ids)                                # [B, 10, embed_dim]
        h = self.hero_proj(h)                                        # [B, 10, d_model]
        if mask is not None and mask.any():
            m = mask.unsqueeze(-1).to(h.dtype)
            h = torch.where(mask.unsqueeze(-1),
                             self.hero_mask_embed.view(1, 1, -1).expand_as(h),
                             h)
        return h

    def _pf_contrib(self, pf: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pf: [B, 10, F]
        h = self.feat_proj(pf)
        if mask is not None and mask.any():
            h = torch.where(mask.unsqueeze(-1),
                             self.pf_mask_embed.view(1, 1, -1).expand_as(h),
                             h)
        return h

    def _scalar_contrib(self, scalar: torch.Tensor, mask: torch.Tensor,
                         proj: nn.Linear, mask_embed: torch.Tensor,
                         log1p: bool) -> torch.Tensor:
        """scalar: [B, 10] f32, mask: [B, 10] bool."""
        x = scalar
        if log1p:
            # log1p on non-negative inputs; clip for safety.
            x = torch.log1p(torch.clamp(x, min=0.0))
        h = proj(x.unsqueeze(-1))     # [B, 10, d_model]
        if mask is not None and mask.any():
            h = torch.where(mask.unsqueeze(-1),
                             mask_embed.view(1, 1, -1).expand_as(h),
                             h)
        return h

    def _items_contrib(self, items: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """items: [B, 10, V] multi-hot float. mask: [B, 10] bool.

        Pooling: SUM(item_input_embed[item_id] for item_id present) / sqrt(K),
        K = #items present (1 if K=0 to avoid div-by-zero).

        Implemented batched: items @ embedding.weight   -- [B,10,V] @ [V,d] = [B,10,d].
        K[s] = items[s].sum(-1).
        """
        B = items.size(0)
        W = self.item_input_embed.weight                              # [V, d]
        h = items @ W                                                  # [B, 10, d]
        K = items.sum(dim=-1, keepdim=True).clamp(min=1.0)             # [B, 10, 1]
        h = h / torch.sqrt(K)
        if mask is not None and mask.any():
            h = torch.where(mask.unsqueeze(-1),
                             self.items_mask_embed.view(1, 1, -1).expand_as(h),
                             h)
        return h

    def _dur_input_contrib(self, dur_log: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """dur_log: [B] f32; mask: [B] bool. Returns [B, 1, d_model]."""
        h = self.dur_input_proj(dur_log.unsqueeze(-1))                 # [B, d_model]
        if mask is not None and mask.any():
            h = torch.where(mask.unsqueeze(-1),
                             self.dur_mask_embed.view(1, -1).expand_as(h),
                             h)
        return h.unsqueeze(1)                                          # [B, 1, d_model]

    def _win_input_contrib(self, win_idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """win_idx: [B] long in {0,1}; mask: [B] bool. Returns [B, 1, d_model]."""
        h = self.win_input_embed(win_idx)                              # [B, d_model]
        if mask is not None and mask.any():
            h = torch.where(mask.unsqueeze(-1),
                             self.win_mask_embed.view(1, -1).expand_as(h),
                             h)
        return h.unsqueeze(1)

    # ----- Token assembly -----

    def encode(self, hero_ids: torch.Tensor, player_feats: torch.Tensor,
               items: torch.Tensor,
               kills: torch.Tensor, deaths: torch.Tensor, assists: torch.Tensor,
               gpm: torch.Tensor, hd: torch.Tensor,
               dur_log: torch.Tensor, win_idx: torch.Tensor,
               masks: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the 12-token sequence, run encoder, return (memory, team_ids).

        masks is a dict with bool tensors for each maskable group:
          hero, player_feat, items, kills, deaths, assists, gpm, hd  -> [B, 10]
          duration, win                                                -> [B]
        """
        B = hero_ids.size(0)
        device = hero_ids.device

        # Per-slot sum (8 groups).
        slot = self._hero_contrib(hero_ids, masks.get("hero"))
        slot = slot + self._pf_contrib(player_feats, masks.get("player_feat"))
        slot = slot + self._items_contrib(items, masks.get("items"))
        slot = slot + self._scalar_contrib(kills, masks.get("kills"),
                                            self.kills_proj, self.kills_mask_embed,
                                            log1p=True)
        slot = slot + self._scalar_contrib(deaths, masks.get("deaths"),
                                            self.deaths_proj, self.deaths_mask_embed,
                                            log1p=True)
        slot = slot + self._scalar_contrib(assists, masks.get("assists"),
                                            self.assists_proj, self.assists_mask_embed,
                                            log1p=True)
        slot = slot + self._scalar_contrib(gpm, masks.get("gpm"),
                                            self.gpm_proj, self.gpm_mask_embed,
                                            log1p=True)
        slot = slot + self._scalar_contrib(hd, masks.get("hd"),
                                            self.hd_proj, self.hd_mask_embed,
                                            log1p=True)
        # Add team embed per slot.
        team_ids_slots = self.hero_team_ids.unsqueeze(0).expand(B, -1)         # [B, 10]
        team_emb_slots = self.team_embed(team_ids_slots)                        # [B, 10, embed_dim]
        team_emb_slots = self.hero_proj(team_emb_slots)                         # [B, 10, d_model]
        slot = slot + team_emb_slots

        # Per-match tokens (dur, win), team_id=2.
        dur_tok = self._dur_input_contrib(dur_log, masks.get("duration"))       # [B, 1, d]
        win_tok = self._win_input_contrib(win_idx, masks.get("win"))            # [B, 1, d]
        match_team = torch.full((B, 1), 2, dtype=torch.long, device=device)
        match_team_emb = self.hero_proj(self.team_embed(match_team))            # [B, 1, d]
        dur_tok = dur_tok + match_team_emb
        win_tok = win_tok + match_team_emb

        x = torch.cat([slot, dur_tok, win_tok], dim=1)                          # [B, 12, d]
        team_ids = torch.cat([team_ids_slots,
                               torch.full((B, 1), 2, dtype=torch.long, device=device),
                               torch.full((B, 1), 2, dtype=torch.long, device=device)], dim=1)

        for blk in self.encoder_blocks:
            x = blk(x, team_ids)
        x = self.encoder_norm(x)
        return x, team_ids

    def _decode_task(self, memory: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        q = self.task_token_embed(task_ids)
        for blk in self.decoder_blocks:
            q = blk(q, memory)
        return self.decoder_norm(q)

    def forward(self, hero_ids: torch.Tensor, player_feats: torch.Tensor,
                items: torch.Tensor,
                kills: torch.Tensor, deaths: torch.Tensor, assists: torch.Tensor,
                gpm: torch.Tensor, hd: torch.Tensor,
                dur_log: torch.Tensor, win_idx: torch.Tensor,
                masks: dict | None = None) -> dict:
        if masks is None:
            masks = {}
        memory, _ti = self.encode(hero_ids, player_feats, items,
                                    kills, deaths, assists, gpm, hd,
                                    dur_log, win_idx, masks)
        B = hero_ids.size(0)
        device = hero_ids.device

        # Build 62-token task sequence per batch.
        task_seq = torch.empty(TASK_VOCAB_SIZE, dtype=torch.long, device=device)
        task_seq[0] = TASK_WIN
        task_seq[1] = TASK_DUR
        for s in range(10):
            task_seq[2 + s] = TASK_ITEMS_BASE + s
            task_seq[12 + s] = TASK_KILLS_BASE + s
            task_seq[22 + s] = TASK_DEATHS_BASE + s
            task_seq[32 + s] = TASK_ASSISTS_BASE + s
            task_seq[42 + s] = TASK_GPM_BASE + s
            task_seq[52 + s] = TASK_HD_BASE + s
        task_ids = task_seq.unsqueeze(0).expand(B, -1).contiguous()

        h = self._decode_task(memory, task_ids)                                 # [B, 62, d]

        win_h     = h[:, 0, :]
        dur_h     = h[:, 1, :]
        items_h   = h[:, 2:12, :]
        kills_h   = h[:, 12:22, :]
        deaths_h  = h[:, 22:32, :]
        assists_h = h[:, 32:42, :]
        gpm_h     = h[:, 42:52, :]
        hd_h      = h[:, 52:62, :]

        out = {
            "win":     self.win_head(win_h).squeeze(-1),
            "dur":     self.dur_head(dur_h).squeeze(-1),
            "item":    self.item_head(items_h),
            "kills":   self.kills_head(kills_h).squeeze(-1),
            "deaths":  self.deaths_head(deaths_h).squeeze(-1),
            "assists": self.assists_head(assists_h).squeeze(-1),
            "gpm":     self.gpm_head(gpm_h).squeeze(-1),
            "hd":      self.hd_head(hd_h).squeeze(-1),
            "encoded": memory,
        }
        return out


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed = sum(p.numel() for n, p in model.named_parameters() if "embed" in n)
    mask = sum(p.numel() for n, p in model.named_parameters() if "mask_embed" in n)
    return {"total": int(total), "trainable": int(trainable),
            "embedding": int(embed), "mask_embed": int(mask),
            "non_embedding": int(total - embed)}


def build_model(hp: dict, vocab_size: int, n_player_feats: int,
                item_vocab_size: int) -> FoundationTransformerV7:
    return FoundationTransformerV7(
        vocab_size=vocab_size,
        embed_dim=int(hp["embed_dim"]),
        d_model=int(hp["d_model"]),
        n_heads=int(hp["n_heads"]),
        n_layers=int(hp["n_layers"]),
        ff_mult=int(hp.get("ff_mult", 4)),
        dropout=float(hp.get("dropout", 0.0)),
        n_player_feats=int(n_player_feats),
        item_vocab_size=int(item_vocab_size),
        use_team_team_bias=bool(hp.get("use_team_team_bias", True)),
        decoder_n_layers=int(hp.get("decoder_n_layers", 2)),
        decoder_n_heads=int(hp.get("decoder_n_heads", 8)),
        remove_first_layer_first_ln=bool(hp.get("remove_first_layer_first_ln", True)),
    )


__all__ = ["FoundationTransformerV7", "build_model", "count_params",
           "TASK_WIN", "TASK_DUR", "TASK_ITEMS_BASE", "TASK_KILLS_BASE",
           "TASK_DEATHS_BASE", "TASK_ASSISTS_BASE", "TASK_GPM_BASE", "TASK_HD_BASE",
           "TASK_VOCAB_SIZE", "SLOT_GROUPS", "MATCH_GROUPS"]
