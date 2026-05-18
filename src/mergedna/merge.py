from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

SourceGroups = list[list[list[int]]]


@dataclass
class MergeOutput:
    tokens: torch.Tensor
    sources: SourceGroups
    group_map: list[list[int]] | None = None


def initial_sources(batch_size: int, seq_len: int) -> SourceGroups:
    return [[[pos] for pos in range(seq_len)] for _ in range(batch_size)]


def source_lengths(sources: SourceGroups, *, device: torch.device | None = None) -> torch.Tensor:
    lengths = [[len(group) for group in batch_sources] for batch_sources in sources]
    return torch.tensor(lengths, dtype=torch.float32, device=device)


def sources_to_mask(sources: SourceGroups, seq_len: int, *, device: torch.device | None = None) -> torch.Tensor:
    batch_size = len(sources)
    max_tokens = max(len(batch_sources) for batch_sources in sources)
    mask = torch.zeros(batch_size, max_tokens, seq_len, dtype=torch.bool, device=device)
    for batch_idx, batch_sources in enumerate(sources):
        for token_idx, group in enumerate(batch_sources):
            mask[batch_idx, token_idx, group] = True
    return mask


def sources_valid_mask(sources: SourceGroups, base_valid: torch.Tensor) -> torch.Tensor:
    """Per-merged-token validity: True if any source base is valid."""
    batch_size = base_valid.size(0)
    max_tokens = max(len(batch_sources) for batch_sources in sources)
    valid = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=base_valid.device)
    for batch_idx, batch_sources in enumerate(sources):
        for token_idx, group in enumerate(batch_sources):
            valid[batch_idx, token_idx] = bool(base_valid[batch_idx, group].any())
    return valid


def _window_pair_candidates(length: int, window_size: int, radius: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for start in range(0, length, window_size):
        end = min(start + window_size, length)
        for left in range(start, end):
            for right in range(left + 1, min(end, left + radius + 1)):
                pairs.append((left, right))
    return pairs


def select_merge_pairs(
    tokens: torch.Tensor,
    *,
    window_size: int,
    merge_ratio: float,
    neighbor_radius: int = 1,
    valid_mask: torch.Tensor | None = None,
) -> list[list[tuple[int, int]]]:
    """Select non-overlapping similar pairs per batch item."""
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [batch, length, dim]")
    batch_size, length, _ = tokens.shape
    target_pairs = max(0, int(length * merge_ratio))
    if target_pairs == 0 or length < 2:
        return [[] for _ in range(batch_size)]

    candidates = _window_pair_candidates(length, window_size, neighbor_radius)
    if not candidates:
        return [[] for _ in range(batch_size)]

    normalized = F.normalize(tokens, dim=-1)
    pairs_by_batch: list[list[tuple[int, int]]] = []
    for batch_idx in range(batch_size):
        scored: list[tuple[float, int, int]] = []
        for left, right in candidates:
            if valid_mask is not None:
                left_valid = bool(valid_mask[batch_idx, left])
                right_valid = bool(valid_mask[batch_idx, right])
                if left_valid != right_valid:
                    continue
            score = torch.dot(normalized[batch_idx, left], normalized[batch_idx, right]).item()
            scored.append((score, left, right))
        scored.sort(reverse=True)
        used: set[int] = set()
        selected: list[tuple[int, int]] = []
        for _, left, right in scored:
            if left in used or right in used:
                continue
            selected.append((left, right))
            used.add(left)
            used.add(right)
            if len(selected) >= target_pairs:
                break
        pairs_by_batch.append(sorted(selected))
    return pairs_by_batch


def apply_merge_pairs(
    tokens: torch.Tensor,
    sources: SourceGroups,
    pairs_by_batch: list[list[tuple[int, int]]],
) -> MergeOutput:
    merged_batches: list[torch.Tensor] = []
    merged_sources: SourceGroups = []
    group_maps: list[list[int]] = []

    for batch_idx, pairs in enumerate(pairs_by_batch):
        left_to_right: dict[int, int] = {left: right for left, right in pairs}
        right_to_left: dict[int, int] = {right: left for left, right in pairs}
        source_batch = sources[batch_idx]
        length = tokens.size(1)
        new_tokens: list[torch.Tensor] = []
        new_sources: list[list[int]] = []
        group_map: list[int] = [-1] * length

        for idx in range(length):
            if idx in right_to_left:
                continue
            out_idx = len(new_tokens)
            if idx in left_to_right:
                right = left_to_right[idx]
                left_weight = len(source_batch[idx])
                right_weight = len(source_batch[right])
                total = left_weight + right_weight
                token = (
                    tokens[batch_idx, idx] * left_weight
                    + tokens[batch_idx, right] * right_weight
                ) / total
                group = sorted(source_batch[idx] + source_batch[right])
                group_map[idx] = out_idx
                group_map[right] = out_idx
            else:
                token = tokens[batch_idx, idx]
                group = list(source_batch[idx])
                group_map[idx] = out_idx
            new_tokens.append(token)
            new_sources.append(group)

        merged_batches.append(torch.stack(new_tokens, dim=0))
        merged_sources.append(new_sources)
        group_maps.append(group_map)

    padded = torch.nn.utils.rnn.pad_sequence(merged_batches, batch_first=True)
    return MergeOutput(tokens=padded, sources=merged_sources, group_map=group_maps)


def merge_tokens(
    tokens: torch.Tensor,
    sources: SourceGroups,
    *,
    window_size: int,
    merge_ratio: float,
    neighbor_radius: int = 1,
    valid_mask: torch.Tensor | None = None,
) -> MergeOutput:
    pairs = select_merge_pairs(
        tokens,
        window_size=window_size,
        merge_ratio=merge_ratio,
        neighbor_radius=neighbor_radius,
        valid_mask=valid_mask,
    )
    return apply_merge_pairs(tokens, sources, pairs)


def unmerge_tokens(tokens: torch.Tensor, sources: SourceGroups, seq_len: int) -> torch.Tensor:
    batch_size, _, dim = tokens.shape
    restored = tokens.new_zeros(batch_size, seq_len, dim)
    for batch_idx, batch_sources in enumerate(sources):
        for token_idx, group in enumerate(batch_sources):
            restored[batch_idx, group] = tokens[batch_idx, token_idx]
    return restored


def global_merge_tokens(
    tokens: torch.Tensor,
    sources: SourceGroups,
    target_tokens: int,
    *,
    valid_mask: torch.Tensor | None = None,
) -> MergeOutput:
    length = tokens.size(1)
    if target_tokens >= length:
        group_map = [[idx for idx in range(length)] for _ in range(tokens.size(0))]
        return MergeOutput(tokens=tokens, sources=sources, group_map=group_map)
    merge_ratio = max(0.0, (length - target_tokens) / max(length, 1))
    return merge_tokens(
        tokens,
        sources,
        window_size=length,
        merge_ratio=merge_ratio,
        neighbor_radius=length,
        valid_mask=valid_mask,
    )


def latent_importance_weights(group_map: list[list[int]], latent_lengths: list[int]) -> list[torch.Tensor]:
    weights: list[torch.Tensor] = []
    for batch_map, latent_len in zip(group_map, latent_lengths, strict=True):
        counts = torch.zeros(latent_len, dtype=torch.float32)
        for latent_idx in batch_map:
            if latent_idx >= 0:
                counts[latent_idx] += 1
        local_weights = torch.empty(len(batch_map), dtype=torch.float32)
        for local_idx, latent_idx in enumerate(batch_map):
            size = counts[latent_idx].clamp_min(1.0)
            local_weights[local_idx] = 1.0 / (size * size)
        weights.append(local_weights / local_weights.sum().clamp_min(1e-8))
    return weights


def sample_adaptive_local_masks(
    group_map: list[list[int]],
    latent_lengths: list[int],
    *,
    num_masks: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    weights = latent_importance_weights(group_map, latent_lengths)
    max_len = max(weight.numel() for weight in weights)
    masks = torch.zeros(len(weights), max_len, dtype=torch.bool)
    for batch_idx, probs in enumerate(weights):
        count = min(num_masks, probs.numel())
        selected = torch.multinomial(probs, count, replacement=False, generator=generator)
        masks[batch_idx, selected] = True
    return masks


def expand_local_mask_to_bases(local_mask: torch.Tensor, sources: SourceGroups, seq_len: int) -> torch.Tensor:
    base_mask = torch.zeros(local_mask.size(0), seq_len, dtype=torch.bool, device=local_mask.device)
    for batch_idx, batch_sources in enumerate(sources):
        for token_idx, group in enumerate(batch_sources):
            if bool(local_mask[batch_idx, token_idx]):
                base_mask[batch_idx, group] = True
    return base_mask
