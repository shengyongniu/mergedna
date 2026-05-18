import torch

from mergedna.merge import (
    apply_merge_pairs,
    expand_local_mask_to_bases,
    initial_sources,
    merge_tokens,
    sample_adaptive_local_masks,
    unmerge_tokens,
)


def test_merge_reduces_length_and_tracks_sources() -> None:
    tokens = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    sources = initial_sources(batch_size=1, seq_len=4)
    out = apply_merge_pairs(tokens, sources, [[(0, 1), (2, 3)]])
    assert out.tokens.shape == (1, 2, 2)
    assert out.sources == [[[0, 1], [2, 3]]]


def test_unmerge_restores_base_resolution_shape() -> None:
    tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    sources = [[[0, 1], [2, 3]]]
    restored = unmerge_tokens(tokens, sources, seq_len=4)
    assert restored.shape == (1, 4, 2)
    assert torch.equal(restored[0, 0], restored[0, 1])
    assert torch.equal(restored[0, 2], restored[0, 3])


def test_similarity_merge_reduces_length() -> None:
    tokens = torch.randn(1, 8, 4)
    out = merge_tokens(tokens, initial_sources(1, 8), window_size=4, merge_ratio=0.25)
    assert out.tokens.size(1) < tokens.size(1)


def test_adaptive_mask_expands_to_source_bases() -> None:
    group_map = [[0, 0, 1]]
    local_mask = sample_adaptive_local_masks(group_map, [2], num_masks=1)
    base_mask = expand_local_mask_to_bases(local_mask, [[[0, 1], [2], [3, 4]]], seq_len=5)
    assert base_mask.shape == (1, 5)
    assert int(base_mask.sum()) in {1, 2}


def test_merge_unmerge_identity_on_equal_rows() -> None:
    tokens = torch.tensor(
        [
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [-3.0, 4.0],
                [-3.0, 4.0],
            ]
        ]
    )
    sources = initial_sources(1, 4)
    merged = apply_merge_pairs(tokens, sources, [[(0, 1), (2, 3)]])
    restored = unmerge_tokens(merged.tokens, merged.sources, seq_len=4)
    assert torch.allclose(restored, tokens)


def test_adaptive_mask_biases_singleton() -> None:
    group_map = [[0, 1, 1, 1, 1]]
    latent_lengths = [2]
    counts = torch.zeros(5, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)
    for _ in range(400):
        sample = sample_adaptive_local_masks(
            group_map, latent_lengths, num_masks=1, generator=generator
        )
        counts += sample[0].long()
    assert counts[0] > counts[1:].max()
