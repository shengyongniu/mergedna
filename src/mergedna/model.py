from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mergedna.data import MASK_ID, VOCAB_SIZE
from mergedna.merge import (
    SourceGroups,
    apply_merge_pairs,
    expand_local_mask_to_bases,
    global_merge_tokens,
    initial_sources,
    sample_adaptive_local_masks,
    select_merge_pairs,
    unmerge_tokens,
)
from mergedna.modules import LearnedPositionEmbedding, TransformerBlock, TransformerStack


@dataclass
class MergeDNAConfig:
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = 256
    d_model: int = 128
    num_heads: int = 4
    local_layers: int = 2
    latent_layers: int = 2
    latent_decoder_layers: int = 1
    local_decoder_layers: int = 1
    local_window: int = 16
    merge_ratio: float = 0.25
    latent_merge_ratio: float = 0.5
    neighbor_radius: int = 1
    dropout: float = 0.0
    mask_token_id: int = MASK_ID


@dataclass
class LocalEncoding:
    tokens: torch.Tensor
    sources: SourceGroups


@dataclass
class ReconstructionOutput:
    logits: torch.Tensor
    local_tokens: torch.Tensor
    local_sources: SourceGroups
    latent_tokens: torch.Tensor
    base_representations: torch.Tensor


@dataclass
class LatentMergeInfo:
    compressed_tokens: torch.Tensor
    reconstructed_local_tokens: torch.Tensor
    group_map: list[list[int]]
    latent_lengths: list[int]


class LocalEncoder(nn.Module):
    def __init__(self, config: MergeDNAConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position = LearnedPositionEmbedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.num_heads,
                    dropout=config.dropout,
                    local_window=config.local_window,
                )
                for _ in range(config.local_layers)
            ]
        )
        self.merge_key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, input_ids: torch.Tensor) -> LocalEncoding:
        x = self.position(self.embedding(input_ids))
        sources = initial_sources(input_ids.size(0), input_ids.size(1))
        for layer in self.layers:
            x = layer(x)
            pairs = select_merge_pairs(
                self.merge_key(x),
                window_size=self.config.local_window,
                merge_ratio=self.config.merge_ratio,
                neighbor_radius=self.config.neighbor_radius,
            )
            merge_out = apply_merge_pairs(x, sources, pairs)
            x = merge_out.tokens
            sources = merge_out.sources
        return LocalEncoding(tokens=self.norm(x), sources=sources)


class MergeDNAModel(nn.Module):
    def __init__(self, config: MergeDNAConfig | None = None) -> None:
        super().__init__()
        self.config = config or MergeDNAConfig()
        self.local_encoder = LocalEncoder(self.config)
        self.latent_encoder = TransformerStack(
            self.config.latent_layers,
            self.config.d_model,
            self.config.num_heads,
            dropout=self.config.dropout,
        )
        self.latent_decoder = TransformerStack(
            self.config.latent_decoder_layers,
            self.config.d_model,
            self.config.num_heads,
            dropout=self.config.dropout,
        )
        self.local_decoder = TransformerStack(
            self.config.local_decoder_layers,
            self.config.d_model,
            self.config.num_heads,
            dropout=self.config.dropout,
            local_window=self.config.local_window,
        )
        self.base_position = LearnedPositionEmbedding(self.config.max_seq_len, self.config.d_model)
        self.output = nn.Linear(self.config.d_model, self.config.vocab_size)

    def encode_local(self, input_ids: torch.Tensor) -> LocalEncoding:
        return self.local_encoder(input_ids)

    def encode_representation(self, input_ids: torch.Tensor) -> torch.Tensor:
        local = self.encode_local(input_ids)
        return self.latent_encoder(local.tokens)

    def decode_from_local_tokens(
        self,
        local_tokens: torch.Tensor,
        sources: SourceGroups,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.latent_encoder(local_tokens)
        decoded_local = self.latent_decoder(latent)
        base_repr = unmerge_tokens(decoded_local, sources, seq_len)
        base_repr = self.base_position(base_repr)
        base_repr = self.local_decoder(base_repr)
        return self.output(base_repr), latent, base_repr

    def logits_from_decoded_local(
        self,
        decoded_local: torch.Tensor,
        sources: SourceGroups,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_repr = unmerge_tokens(decoded_local, sources, seq_len)
        base_repr = self.base_position(base_repr)
        base_repr = self.local_decoder(base_repr)
        return self.output(base_repr), base_repr

    def forward(self, input_ids: torch.Tensor) -> ReconstructionOutput:
        local = self.encode_local(input_ids)
        logits, latent, base_repr = self.decode_from_local_tokens(
            local.tokens,
            local.sources,
            input_ids.size(1),
        )
        return ReconstructionOutput(
            logits=logits,
            local_tokens=local.tokens,
            local_sources=local.sources,
            latent_tokens=latent,
            base_representations=base_repr,
        )

    def latent_merge_reconstruction(
        self,
        input_ids: torch.Tensor,
        *,
        detach_local: bool = True,
    ) -> tuple[ReconstructionOutput, LatentMergeInfo]:
        local = self.encode_local(input_ids)
        local_tokens = local.tokens.detach() if detach_local else local.tokens
        token_sources = initial_sources(input_ids.size(0), local_tokens.size(1))
        target_tokens = max(1, int(local_tokens.size(1) * (1.0 - self.config.latent_merge_ratio)))
        merged = global_merge_tokens(local_tokens, token_sources, target_tokens=target_tokens)
        latent = self.latent_encoder(merged.tokens)
        decoded_latent = self.latent_decoder(latent)
        reconstructed_local = unmerge_tokens(decoded_latent, merged.sources, local_tokens.size(1))
        logits, base_repr = self.logits_from_decoded_local(
            reconstructed_local,
            local.sources,
            input_ids.size(1),
        )
        output = ReconstructionOutput(
            logits=logits,
            local_tokens=local_tokens,
            local_sources=local.sources,
            latent_tokens=latent,
            base_representations=base_repr,
        )
        info = LatentMergeInfo(
            compressed_tokens=merged.tokens,
            reconstructed_local_tokens=reconstructed_local,
            group_map=merged.group_map or [],
            latent_lengths=[len(batch_sources) for batch_sources in merged.sources],
        )
        return output, info

    def make_adaptive_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            local = self.encode_local(input_ids)
            token_sources = initial_sources(input_ids.size(0), local.tokens.size(1))
            target_tokens = max(1, int(local.tokens.size(1) * (1.0 - self.config.latent_merge_ratio)))
            merged = global_merge_tokens(local.tokens, token_sources, target_tokens=target_tokens)
            group_map = merged.group_map or []
            latent_lengths = [len(batch_sources) for batch_sources in merged.sources]
            num_masks = max(1, target_tokens)
            local_mask = sample_adaptive_local_masks(
                group_map,
                latent_lengths,
                num_masks=num_masks,
            ).to(input_ids.device)
            return expand_local_mask_to_bases(local_mask, local.sources, input_ids.size(1))

    def forward_amtm(self, input_ids: torch.Tensor) -> tuple[ReconstructionOutput, torch.Tensor]:
        base_mask = self.make_adaptive_mask(input_ids)
        masked_ids = input_ids.masked_fill(base_mask, self.config.mask_token_id)
        return self.forward(masked_ids), base_mask
