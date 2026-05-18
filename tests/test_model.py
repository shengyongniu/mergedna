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
    output, mask = model.forward_amtm(batch)
    assert output.logits.shape[:2] == batch.shape
    assert mask.shape == batch.shape
    assert mask.any()


def test_tiny_forward_backward_has_finite_loss() -> None:
    batch = torch.stack([SyntheticDNADataset(num_sequences=1, seq_len=32)[0] for _ in range(2)])
    model = MergeDNAModel(small_config())
    losses = mergedna_loss(model, batch)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
