from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mergedna.data import MASK_ID, PAD_ID, VOCAB_SIZE
from mergedna.merge import (
    SourceGroups,
    apply_merge_pairs,
    expand_local_mask_to_bases,
    initial_sources,
    sample_adaptive_local_masks,
    select_merge_pairs,
    sources_valid_mask,
    unmerge_tokens,
)
from mergedna.modules import (
    LatentEncoder,
    LearnedPositionEmbedding,
    TransformerBlock,
    TransformerStack,
)


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
    merge_ratio_jitter_std: float = 0.05
    merge_ratio_min: float = 0.10
    merge_ratio_max: float = 0.40
    latent_merge_ratio: float = 0.5
    latent_merge_at_layer: int | None = None
    neighbor_radius: int = 1
    dropout: float = 0.0
    mask_token_id: int = MASK_ID


@dataclass
class LocalEncoding:
    tokens: torch.Tensor
    sources: SourceGroups
    valid_mask: torch.Tensor | None = None


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

    def _sample_merge_ratio(self) -> float:
        cfg = self.config
        if not self.training or cfg.merge_ratio_jitter_std <= 0.0:
            return cfg.merge_ratio
        sample = torch.normal(
            mean=torch.tensor(cfg.merge_ratio),
            std=torch.tensor(cfg.merge_ratio_jitter_std),
        ).item()
        return float(min(max(sample, cfg.merge_ratio_min), cfg.merge_ratio_max))

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> LocalEncoding:
        x = self.position(self.embedding(input_ids))
        sources = initial_sources(input_ids.size(0), input_ids.size(1))
        base_valid = (
            ~key_padding_mask
            if key_padding_mask is not None
            else torch.ones_like(input_ids, dtype=torch.bool)
        )
        token_valid = base_valid
        for layer in self.layers:
            attn_pad = (~token_valid) if not bool(token_valid.all()) else None
            x = layer(x, key_padding_mask=attn_pad)
            merge_ratio = self._sample_merge_ratio()
            pairs = select_merge_pairs(
                self.merge_key(x),
                window_size=self.config.local_window,
                merge_ratio=merge_ratio,
                neighbor_radius=self.config.neighbor_radius,
                valid_mask=token_valid,
            )
            merge_out = apply_merge_pairs(x, sources, pairs)
            x = merge_out.tokens
            sources = merge_out.sources
            token_valid = sources_valid_mask(sources, base_valid).to(x.device)
        return LocalEncoding(tokens=self.norm(x), sources=sources, valid_mask=token_valid)


class MergeDNAModel(nn.Module):
    def __init__(self, config: MergeDNAConfig | None = None) -> None:
        super().__init__()
        self.config = config or MergeDNAConfig()
        self.local_encoder = LocalEncoder(self.config)
        self.latent_encoder = LatentEncoder(
            self.config.latent_layers,
            self.config.d_model,
            self.config.num_heads,
            dropout=self.config.dropout,
            merge_at_layer=self.config.latent_merge_at_layer,
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
        self.latent_position = LearnedPositionEmbedding(self.config.max_seq_len, self.config.d_model)
        self.base_position = LearnedPositionEmbedding(self.config.max_seq_len, self.config.d_model)
        self.output = nn.Linear(self.config.d_model, self.config.vocab_size)

    @staticmethod
    def _pad_mask(input_ids: torch.Tensor) -> torch.Tensor | None:
        mask = input_ids == PAD_ID
        return mask if bool(mask.any()) else None

    def encode_local(
        self, input_ids: torch.Tensor, *, key_padding_mask: torch.Tensor | None = None
    ) -> LocalEncoding:
        if key_padding_mask is None:
            key_padding_mask = self._pad_mask(input_ids)
        return self.local_encoder(input_ids, key_padding_mask=key_padding_mask)

    def encode_representation(self, input_ids: torch.Tensor) -> torch.Tensor:
        local = self.encode_local(input_ids)
        latent_pad = (~local.valid_mask) if local.valid_mask is not None else None
        if latent_pad is not None and not bool(latent_pad.any()):
            latent_pad = None
        positioned = self.latent_position(local.tokens)
        return self.latent_encoder(positioned, key_padding_mask=latent_pad)

    def decode_from_local_tokens(
        self,
        local_tokens: torch.Tensor,
        sources: SourceGroups,
        seq_len: int,
        *,
        latent_padding_mask: torch.Tensor | None = None,
        base_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positioned = self.latent_position(local_tokens)
        latent = self.latent_encoder(positioned, key_padding_mask=latent_padding_mask)
        decoded_local = self.latent_decoder(latent, key_padding_mask=latent_padding_mask)
        logits, base_repr = self.logits_from_decoded_local(
            decoded_local, sources, seq_len, base_padding_mask=base_padding_mask
        )
        return logits, latent, base_repr

    def logits_from_decoded_local(
        self,
        decoded_local: torch.Tensor,
        sources: SourceGroups,
        seq_len: int,
        *,
        base_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_repr = unmerge_tokens(decoded_local, sources, seq_len)
        base_repr = self.base_position(base_repr)
        base_repr = self.local_decoder(base_repr, key_padding_mask=base_padding_mask)
        return self.output(base_repr), base_repr

    def forward(self, input_ids: torch.Tensor) -> ReconstructionOutput:
        base_pad = self._pad_mask(input_ids)
        local = self.encode_local(input_ids, key_padding_mask=base_pad)
        latent_pad = self._latent_pad_from_local(local)
        logits, latent, base_repr = self.decode_from_local_tokens(
            local.tokens,
            local.sources,
            input_ids.size(1),
            latent_padding_mask=latent_pad,
            base_padding_mask=base_pad,
        )
        return ReconstructionOutput(
            logits=logits,
            local_tokens=local.tokens,
            local_sources=local.sources,
            latent_tokens=latent,
            base_representations=base_repr,
        )

    @staticmethod
    def _latent_pad_from_local(local: LocalEncoding) -> torch.Tensor | None:
        if local.valid_mask is None:
            return None
        pad = ~local.valid_mask
        return pad if bool(pad.any()) else None

    def latent_merge_reconstruction(
        self,
        input_ids: torch.Tensor,
        *,
        detach_local: bool = True,
    ) -> tuple[ReconstructionOutput, LatentMergeInfo]:
        base_pad = self._pad_mask(input_ids)
        local = self.encode_local(input_ids, key_padding_mask=base_pad)
        local_tokens = local.tokens.detach() if detach_local else local.tokens
        local_len = local_tokens.size(1)
        latent_pad = self._latent_pad_from_local(local)
        target_tokens = max(1, int(local_len * (1.0 - self.config.latent_merge_ratio)))
        positioned = self.latent_position(local_tokens)
        compressed, latent_sources, group_map = self.latent_encoder.forward_with_merge(
            positioned,
            target_tokens=target_tokens,
            key_padding_mask=latent_pad,
        )
        unmerged_to_L = unmerge_tokens(compressed, latent_sources, local_len)
        decoded_local = self.latent_decoder(unmerged_to_L, key_padding_mask=latent_pad)
        logits, base_repr = self.logits_from_decoded_local(
            decoded_local,
            local.sources,
            input_ids.size(1),
            base_padding_mask=base_pad,
        )
        output = ReconstructionOutput(
            logits=logits,
            local_tokens=local_tokens,
            local_sources=local.sources,
            latent_tokens=compressed,
            base_representations=base_repr,
        )
        info = LatentMergeInfo(
            compressed_tokens=compressed,
            reconstructed_local_tokens=unmerged_to_L,
            group_map=group_map,
            latent_lengths=[len(batch_sources) for batch_sources in latent_sources],
        )
        return output, info

    def make_adaptive_mask(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
        with torch.no_grad():
            base_pad = self._pad_mask(input_ids)
            local = self.encode_local(input_ids, key_padding_mask=base_pad)
            local_len = local.tokens.size(1)
            latent_pad = self._latent_pad_from_local(local)
            target_tokens = max(1, int(local_len * (1.0 - self.config.latent_merge_ratio)))
            positioned = self.latent_position(local.tokens)
            _, latent_sources, group_map = self.latent_encoder.forward_with_merge(
                positioned,
                target_tokens=target_tokens,
                key_padding_mask=latent_pad,
            )
            latent_lengths = [len(batch_sources) for batch_sources in latent_sources]
            num_masks = max(1, target_tokens)
            local_mask = sample_adaptive_local_masks(
                group_map,
                latent_lengths,
                num_masks=num_masks,
            ).to(input_ids.device)
            base_mask = expand_local_mask_to_bases(local_mask, local.sources, input_ids.size(1))
            return base_mask, num_masks

    def forward_amtm(self, input_ids: torch.Tensor) -> tuple[ReconstructionOutput, torch.Tensor, int]:
        base_mask, num_selected = self.make_adaptive_mask(input_ids)
        masked_ids = input_ids.masked_fill(base_mask, self.config.mask_token_id)
        return self.forward(masked_ids), base_mask, num_selected
