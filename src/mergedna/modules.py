from __future__ import annotations

import torch
from torch import nn


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_mask = None
        if self.local_window is not None:
            attn_mask = local_attention_mask(x.size(1), self.local_window, x.device)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
