import torch

from mergedna.data import SyntheticDNADataset
from mergedna.losses import mergedna_loss
from mergedna.model import MergeDNAConfig, MergeDNAModel


def small_config(seq_len: int = 32) -> MergeDNAConfig:
    return MergeDNAConfig(
        max_seq_len=seq_len,
        d_model=32,
        num_heads=4,
        local_layers=1,
        latent_layers=1,
        latent_decoder_layers=1,
        local_decoder_layers=1,
        local_window=8,
        merge_ratio=0.25,
        latent_merge_ratio=0.50,
    )


def test_model_forward_shapes() -> None:
    batch = torch.stack([SyntheticDNADataset(num_sequences=1, seq_len=32)[0] for _ in range(2)])
    model = MergeDNAModel(small_config())
    output = model(batch)
    assert output.logits.shape[:2] == batch.shape
    assert output.logits.size(-1) == model.config.vocab_size
    assert len(output.local_sources[0]) < batch.size(1)


def test_amtm_mask_maps_to_bases() -> None:
    batch = torch.stack([SyntheticDNADataset(num_sequences=1, seq_len=32)[0] for _ in range(2)])
    model = MergeDNAModel(small_config())
    output, mask, num_selected = model.forward_amtm(batch)
    assert output.logits.shape[:2] == batch.shape
    assert mask.shape == batch.shape
    assert mask.any()
    assert num_selected >= 1


def test_tiny_forward_backward_has_finite_loss() -> None:
    batch = torch.stack([SyntheticDNADataset(num_sequences=1, seq_len=32)[0] for _ in range(2)])
    model = MergeDNAModel(small_config())
    losses = mergedna_loss(model, batch)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_latent_mtr_leaves_local_encoder_frozen() -> None:
    from mergedna.losses import latent_mtr_loss

    batch = torch.stack([SyntheticDNADataset(num_sequences=1, seq_len=32)[0] for _ in range(2)])
    model = MergeDNAModel(small_config())
    loss = latent_mtr_loss(model, batch)
    loss.backward()
    local_grads = [
        param.grad for param in model.local_encoder.parameters() if param.grad is not None
    ]
    assert all(float(grad.abs().sum()) == 0.0 for grad in local_grads)
    latent_grads = [
        param.grad for param in model.latent_encoder.parameters() if param.grad is not None
    ]
    assert any(float(grad.abs().sum()) > 0.0 for grad in latent_grads)


def test_latent_decoder_runs_at_length_L() -> None:
    batch = torch.stack([SyntheticDNADataset(num_sequences=1, seq_len=32)[0] for _ in range(2)])
    model = MergeDNAModel(small_config())
    model.eval()
    seen_lengths: list[int] = []
    original_forward = model.latent_decoder.forward

    def recording_forward(x, **kwargs):
        seen_lengths.append(x.size(1))
        return original_forward(x, **kwargs)

    model.latent_decoder.forward = recording_forward  # type: ignore[method-assign]
    output, info = model.latent_merge_reconstruction(batch)
    model.latent_decoder.forward = original_forward  # type: ignore[method-assign]
    expected_local_len = output.local_tokens.size(1)
    assert seen_lengths
    assert all(length == expected_local_len for length in seen_lengths)
    assert info.compressed_tokens.size(1) < expected_local_len


def test_padding_does_not_change_valid_logits() -> None:
    from mergedna.data import N_ID, PAD_ID

    torch.manual_seed(0)
    model = MergeDNAModel(small_config())
    model.eval()
    base = torch.full((1, 32), N_ID, dtype=torch.long)
    base[0, :16] = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
    base[0, 16:] = PAD_ID
    perturbed = base.clone()
    perturbed[0, 16:] = PAD_ID
    with torch.no_grad():
        out_a = model(base).logits[0, :16]
        out_b = model(perturbed).logits[0, :16]
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_compression_jitter_varies_length() -> None:
    config = small_config()
    config.merge_ratio_jitter_std = 0.15
    config.merge_ratio_min = 0.05
    config.merge_ratio_max = 0.45
    model = MergeDNAModel(config)
    model.train()
    batch = torch.stack([SyntheticDNADataset(num_sequences=1, seq_len=32)[0] for _ in range(2)])
    torch.manual_seed(0)
    seen = set()
    for _ in range(8):
        local = model.encode_local(batch)
        seen.add(local.tokens.size(1))
    assert len(seen) > 1
