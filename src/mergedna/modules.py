from __future__ import annotations

import torch
from torch import nn

from mergedna.merge import SourceGroups, global_merge_tokens, initial_sources


class LearnedPositionEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.embedding(positions).unsqueeze(0)


def local_attention_mask(length: int, window_size: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device)
    distance = (positions[:, None] - positions[None, :]).abs()
    return distance >= window_size


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        mlp_ratio: int = 4,
        local_window: int | None = None,
    ) -> None:
        super().__init__()
        self.local_window = local_window
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_mask = None
        if self.local_window is not None:
            attn_mask = local_attention_mask(x.size(1), self.local_window, x.device)
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class TransformerStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        local_window: int | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    dropout=dropout,
                    local_window=local_window,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


class LatentEncoder(nn.Module):
    """Full-attention Transformer with one optional ToMe-style merge step inside.

    `forward` runs all layers without merging. `forward_with_merge` inserts a
    global merge after `merge_at_layer` attention layers, so the encoder is
    called once at length L and produces (Z'_K, S') as in MergeDNA Sec. 3.4.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        merge_at_layer: int | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        if merge_at_layer is None:
            merge_at_layer = max(1, num_layers // 2)
        self.merge_at_layer = merge_at_layer

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.norm(x)

    def forward_with_merge(
        self,
        x: torch.Tensor,
        *,
        target_tokens: int,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SourceGroups, list[list[int]]]:
        latent_sources: SourceGroups | None = None
        group_map: list[list[int]] | None = None
        kpm = key_padding_mask
        for layer_idx, layer in enumerate(self.layers):
            if latent_sources is None and layer_idx == self.merge_at_layer:
                x, latent_sources, group_map = self._merge(x, target_tokens)
                kpm = None
            x = layer(x, key_padding_mask=kpm)
        if latent_sources is None:
            x, latent_sources, group_map = self._merge(x, target_tokens)
        return self.norm(x), latent_sources, group_map or []

    @staticmethod
    def _merge(
        x: torch.Tensor, target_tokens: int
    ) -> tuple[torch.Tensor, SourceGroups, list[list[int]]]:
        sources = initial_sources(x.size(0), x.size(1))
        out = global_merge_tokens(x, sources, target_tokens=target_tokens)
        return out.tokens, out.sources, out.group_map or []
