from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

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


def _base_to_token_index(
    sources: SourceGroups, seq_len: int, device: torch.device
) -> torch.Tensor:
    """For each base position, the index of the merged token that contains it.

    Used to convert per-group operations into single gather/scatter calls.
    Assumes every base in [0, seq_len) appears in exactly one group.
    """
    idx = [[0] * seq_len for _ in range(len(sources))]
    for batch_idx, batch_sources in enumerate(sources):
        row = idx[batch_idx]
        for token_idx, group in enumerate(batch_sources):
            for base in group:
                row[base] = token_idx
    return torch.tensor(idx, dtype=torch.long, device=device)


def sources_to_mask(sources: SourceGroups, seq_len: int, *, device: torch.device | None = None) -> torch.Tensor:
    batch_size = len(sources)
    max_tokens = max(len(batch_sources) for batch_sources in sources)
    base_to_token = _base_to_token_index(sources, seq_len, device or torch.device("cpu"))
    mask = torch.zeros(batch_size, max_tokens, seq_len, dtype=torch.bool, device=base_to_token.device)
    arange_b = torch.arange(batch_size, device=base_to_token.device).unsqueeze(-1).expand(batch_size, seq_len)
    arange_s = torch.arange(seq_len, device=base_to_token.device).unsqueeze(0).expand(batch_size, seq_len)
    mask[arange_b, base_to_token, arange_s] = True
    return mask


def sources_valid_mask(sources: SourceGroups, base_valid: torch.Tensor) -> torch.Tensor:
    """Per-merged-token validity: True if any source base is valid."""
    batch_size, seq_len = base_valid.shape
    max_tokens = max(len(batch_sources) for batch_sources in sources)
    base_to_token = _base_to_token_index(sources, seq_len, base_valid.device)
    counts = torch.zeros(batch_size, max_tokens, dtype=torch.long, device=base_valid.device)
    counts.scatter_add_(1, base_to_token, base_valid.long())
    return counts > 0


@lru_cache(maxsize=32)
def _window_pair_candidates(length: int, window_size: int, radius: int) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for start in range(0, length, window_size):
        end = min(start + window_size, length)
        for left in range(start, end):
            for right in range(left + 1, min(end, left + radius + 1)):
                pairs.append((left, right))
    return tuple(pairs)


@torch.no_grad()
def select_merge_pairs(
    tokens: torch.Tensor,
    *,
    window_size: int,
    merge_ratio: float,
    neighbor_radius: int = 1,
    valid_mask: torch.Tensor | None = None,
) -> list[list[tuple[int, int]]]:
    """Select non-overlapping similar pairs per batch item.

    Computes all candidate similarities in one batched matmul, then runs the
    greedy non-overlapping pick on CPU (one device sync per call instead of
    one per candidate).
    """
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [batch, length, dim]")
    batch_size, length, _ = tokens.shape
    target_pairs = max(0, int(length * merge_ratio))
    if target_pairs == 0 or length < 2:
        return [[] for _ in range(batch_size)]

    candidates = _window_pair_candidates(length, window_size, neighbor_radius)
    if not candidates:
        return [[] for _ in range(batch_size)]

    cand = torch.tensor(candidates, dtype=torch.long, device=tokens.device)
    left_idx = cand[:, 0]
    right_idx = cand[:, 1]

    normalized = F.normalize(tokens, dim=-1)
    left_tokens = normalized.index_select(1, left_idx)
    right_tokens = normalized.index_select(1, right_idx)
    sim = (left_tokens * right_tokens).sum(-1)

    if valid_mask is not None:
        lv = valid_mask.index_select(1, left_idx)
        rv = valid_mask.index_select(1, right_idx)
        sim = sim.masked_fill(lv != rv, float("-inf"))

    order = sim.argsort(dim=-1, descending=True).cpu().numpy()
    left_arr = left_idx.cpu().numpy()
    right_arr = right_idx.cpu().numpy()
    sim_cpu = sim.cpu().numpy() if valid_mask is not None else None

    pairs_by_batch: list[list[tuple[int, int]]] = []
    for batch_idx in range(batch_size):
        used = bytearray(length)
        selected: list[tuple[int, int]] = []
        for s in order[batch_idx]:
            if sim_cpu is not None and sim_cpu[batch_idx, s] == float("-inf"):
                break
            l = int(left_arr[s])
            r = int(right_arr[s])
            if used[l] or used[r]:
                continue
            selected.append((l, r))
            used[l] = 1
            used[r] = 1
            if len(selected) >= target_pairs:
                break
        selected.sort()
        pairs_by_batch.append(selected)
    return pairs_by_batch


def apply_merge_pairs(
    tokens: torch.Tensor,
    sources: SourceGroups,
    pairs_by_batch: list[list[tuple[int, int]]],
) -> MergeOutput:
    """Combine merged pairs with span-weighted averages.

    For each batch we build (primary, secondary, weights) index tensors and
    perform a single gather + weighted sum instead of one scalar op per token.
    """
    batch_size, length, dim = tokens.shape
    device = tokens.device

    merged_batches: list[torch.Tensor] = []
    merged_sources: SourceGroups = []
    group_maps: list[list[int]] = []

    for batch_idx, pairs in enumerate(pairs_by_batch):
        left_to_right: dict[int, int] = {left: right for left, right in pairs}
        right_to_left: dict[int, int] = {right: left for left, right in pairs}
        source_batch = sources[batch_idx]

        primary: list[int] = []
        secondary: list[int] = []
        primary_w: list[float] = []
        secondary_w: list[float] = []
        new_sources: list[list[int]] = []
        group_map: list[int] = [-1] * length

        for idx in range(length):
            if idx in right_to_left:
                continue
            out_idx = len(primary)
            if idx in left_to_right:
                right = left_to_right[idx]
                lw = len(source_batch[idx])
                rw = len(source_batch[right])
                total = lw + rw
                primary.append(idx)
                secondary.append(right)
                primary_w.append(lw / total)
                secondary_w.append(rw / total)
                group_map[idx] = out_idx
                group_map[right] = out_idx
                new_sources.append(sorted(source_batch[idx] + source_batch[right]))
            else:
                primary.append(idx)
                secondary.append(idx)
                primary_w.append(1.0)
                secondary_w.append(0.0)
                group_map[idx] = out_idx
                new_sources.append(list(source_batch[idx]))

        p_idx = torch.tensor(primary, dtype=torch.long, device=device)
        s_idx = torch.tensor(secondary, dtype=torch.long, device=device)
        pw = torch.tensor(primary_w, dtype=tokens.dtype, device=device).unsqueeze(-1)
        sw = torch.tensor(secondary_w, dtype=tokens.dtype, device=device).unsqueeze(-1)

        row = tokens[batch_idx]
        new_tokens = row.index_select(0, p_idx) * pw + row.index_select(0, s_idx) * sw

        merged_batches.append(new_tokens)
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
    """Broadcast each merged token back to all of its source base positions."""
    batch_size, _, dim = tokens.shape
    base_to_token = _base_to_token_index(sources, seq_len, tokens.device)
    expanded = base_to_token.unsqueeze(-1).expand(batch_size, seq_len, dim)
    return tokens.gather(1, expanded)


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
        bm = torch.tensor(batch_map, dtype=torch.long)
        # Per-merged-token count; -1 sentinels (shouldn't appear in practice) route to a dummy slot.
        valid = bm >= 0
        counts = torch.zeros(max(latent_len, 1), dtype=torch.float32)
        if valid.any():
            counts.scatter_add_(0, bm[valid], torch.ones(int(valid.sum()), dtype=torch.float32))
        safe = torch.where(valid, bm, torch.full_like(bm, latent_len - 1))
        sizes = counts[safe].clamp_min(1.0)
        local = 1.0 / (sizes * sizes)
        weights.append(local / local.sum().clamp_min(1e-8))
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
    base_to_token = _base_to_token_index(sources, seq_len, local_mask.device)
    return local_mask.gather(1, base_to_token)
