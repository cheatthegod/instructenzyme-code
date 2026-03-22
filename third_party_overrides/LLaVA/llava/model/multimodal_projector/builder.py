import torch
import torch.nn as nn
import re
import math


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def _build_1d_sincos_pos_embed(embed_dim, length, device, dtype):
    if embed_dim % 2 != 0:
        raise ValueError(f"1D sin/cos positional embedding requires an even embed_dim, got {embed_dim}.")

    position = torch.arange(length, device=device, dtype=torch.float32)
    omega = torch.arange(embed_dim // 2, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(1, embed_dim // 2)))
    out = position[:, None] * omega[None, :]
    pos_embed = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
    return pos_embed.to(dtype=dtype)


def _build_2d_sincos_pos_embed(embed_dim, height, width, device, dtype):
    if embed_dim % 4 != 0:
        raise ValueError(f"2D sin/cos positional embedding requires embed_dim divisible by 4, got {embed_dim}.")

    grid_h = torch.arange(height, device=device, dtype=torch.float32)
    grid_w = torch.arange(width, device=device, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')

    half_dim = embed_dim // 2
    emb_h = _build_1d_sincos_pos_embed(half_dim, height, device, torch.float32)[grid_h.long()]
    emb_w = _build_1d_sincos_pos_embed(half_dim, width, device, torch.float32)[grid_w.long()]
    pos_embed = torch.cat([emb_h, emb_w], dim=-1).reshape(height * width, embed_dim)
    return pos_embed.to(dtype=dtype)


def _infer_2d_shape(num_tokens):
    side = int(math.isqrt(num_tokens))
    if side * side == num_tokens:
        return side, side
    return None


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_mult=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * ffn_mult)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        self.ln_ffn = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv, query_pos=None, kv_pos=None, attn_mask=None, key_padding_mask=None):
        q_in = q if query_pos is None else q + query_pos
        kv_in = kv if kv_pos is None else kv + kv_pos
        attn_out, _ = self.attn(
            self.ln_q(q_in),
            self.ln_kv(kv_in),
            self.ln_kv(kv),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        q = q + attn_out
        q = q + self.ffn(self.ln_ffn(q))
        return q


class FixedQueryCrossAttentionProjector(nn.Module):
    """A Perceiver-style projector that maps variable-length vision tokens to a fixed number of queries."""

    def __init__(self, config):
        super().__init__()
        self.input_dim = config.mm_hidden_size
        self.embed_dim = config.hidden_size
        self.patch_merge_type = getattr(config, 'mm_patch_merge_type', 'flat')
        self.num_queries = int(getattr(config, 'mm_projector_num_queries', 256))
        self.num_heads = int(getattr(config, 'mm_projector_num_heads', 8))
        self.num_layers = int(getattr(config, 'mm_projector_num_layers', 1))
        self.ffn_mult = float(getattr(config, 'mm_projector_ffn_mult', 4.0))
        self.dropout = float(getattr(config, 'mm_projector_dropout', 0.0))
        self.use_query_pos = bool(getattr(config, 'mm_projector_use_query_pos', True))
        self.use_input_pos = bool(getattr(config, 'mm_projector_use_input_pos', True))
        self.pos_encoding_type = getattr(config, 'mm_projector_pos_encoding', '2d')
        self.query_grid_size = getattr(config, 'mm_projector_grid_size', None)
        self.use_post_proj = bool(getattr(config, 'mm_projector_use_post_proj', False))

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.embed_dim}) must be divisible by mm_projector_num_heads ({self.num_heads})."
            )
        if self.num_layers < 1:
            raise ValueError(f"mm_projector_num_layers must be >= 1, got {self.num_layers}.")
        if self.patch_merge_type.startswith('spatial'):
            raise ValueError(
                "The resampler mm_projector outputs a fixed-length token set and is only compatible with "
                "mm_patch_merge_type='flat'."
            )

        self.kv_proj = nn.Linear(self.input_dim, self.embed_dim, bias=False)
        self.kv_norm = nn.LayerNorm(self.embed_dim)
        self.query_norm = nn.LayerNorm(self.embed_dim)
        self.post_norm = nn.LayerNorm(self.embed_dim)
        self.query = nn.Parameter(torch.zeros(self.num_queries, self.embed_dim))
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    ffn_mult=self.ffn_mult,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False) if self.use_post_proj else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.query, std=0.02)

    def _flatten_inputs(self, x):
        if x.dim() == 4:
            bsz, channels, height, width = x.shape
            x = x.flatten(2).transpose(1, 2)
            return x, (height, width)

        if x.dim() != 3:
            raise ValueError(f"Expected vision features with shape [B, N, C] or [B, C, H, W], got {tuple(x.shape)}.")

        grid_shape = _infer_2d_shape(x.shape[1])
        return x, grid_shape

    def _build_query_pos(self, device, dtype):
        if not self.use_query_pos:
            return None

        if self.pos_encoding_type == '2d':
            if self.query_grid_size is not None:
                height = width = int(self.query_grid_size)
                if height * width != self.num_queries:
                    raise ValueError(
                        f"mm_projector_grid_size={self.query_grid_size} implies {height * width} queries, "
                        f"but mm_projector_num_queries={self.num_queries}."
                    )
                return _build_2d_sincos_pos_embed(self.embed_dim, height, width, device, dtype)

            query_shape = _infer_2d_shape(self.num_queries)
            if query_shape is not None:
                return _build_2d_sincos_pos_embed(self.embed_dim, query_shape[0], query_shape[1], device, dtype)

        return _build_1d_sincos_pos_embed(self.embed_dim, self.num_queries, device, dtype)

    def _build_input_pos(self, num_tokens, grid_shape, device, dtype):
        if not self.use_input_pos:
            return None

        if self.pos_encoding_type == '2d' and grid_shape is not None:
            return _build_2d_sincos_pos_embed(self.embed_dim, grid_shape[0], grid_shape[1], device, dtype)

        return _build_1d_sincos_pos_embed(self.embed_dim, num_tokens, device, dtype)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x, grid_shape = self._flatten_inputs(x)
        x = self.kv_proj(x)
        x = self.kv_norm(x)

        bsz, num_tokens, _ = x.shape
        query = self.query_norm(self.query).unsqueeze(0).expand(bsz, -1, -1)
        query_pos = self._build_query_pos(x.device, x.dtype)
        kv_pos = self._build_input_pos(num_tokens, grid_shape, x.device, x.dtype)

        if query_pos is not None:
            query_pos = query_pos.unsqueeze(0).expand(bsz, -1, -1)
        if kv_pos is not None:
            kv_pos = kv_pos.unsqueeze(0).expand(bsz, -1, -1)

        for layer in self.layers:
            query = layer(
                query,
                x,
                query_pos=query_pos,
                kv_pos=kv_pos,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )

        query = self.post_norm(query)
        return self.out_proj(query)

    @property
    def config(self):
        return {
            "mm_projector_type": "resampler",
            "mm_projector_num_queries": self.num_queries,
            "mm_projector_num_heads": self.num_heads,
            "mm_projector_num_layers": self.num_layers,
            "mm_projector_pos_encoding": self.pos_encoding_type,
        }


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    if projector_type in {'resampler', 'cross_attention', 'cross_attn', 'perceiver_resampler'}:
        return FixedQueryCrossAttentionProjector(config)

    raise ValueError(f'Unknown projector type: {projector_type}')
