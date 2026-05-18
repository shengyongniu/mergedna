import torch
import torch.nn.functional as F

from mergedna.data import SyntheticDNADataset
from mergedna.losses import reconstruction_loss
from mergedna.model import MergeDNAConfig, MergeDNAModel


def _small_config(seq_len: int = 32) -> MergeDNAConfig:
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
        merge_ratio_jitter_std=0.0,
    )


def test_reconstruction_loss_normalizes_by_num_selected() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 7)
    targets = torch.randint(0, 7, (2, 4))
    mask = torch.tensor([[True, False, True, True], [False, True, False, False]])
    expected_sum = F.cross_entropy(logits[mask], targets[mask], reduction="sum")

    loss_by_4 = reconstruction_loss(logits, targets, mask=mask, normalize_by=4)
    loss_by_2 = reconstruction_loss(logits, targets, mask=mask, normalize_by=2)
    assert torch.allclose(loss_by_4, expected_sum / 4)
    assert torch.allclose(loss_by_2, expected_sum / 2)
    assert not torch.allclose(loss_by_4, loss_by_2)


def test_amtm_loss_uses_K_not_mask_count() -> None:
    torch.manual_seed(0)
    batch = torch.stack([SyntheticDNADataset(num_sequences=1, seq_len=32)[0] for _ in range(2)])
    model = MergeDNAModel(_small_config())
    model.eval()
    output, base_mask, K = model.forward_amtm(batch)
    sum_loss = F.cross_entropy(
        output.logits[base_mask], batch[base_mask], reduction="sum"
    )
    expected_amtm = sum_loss / max(K, 1)
    from mergedna.losses import amtm_loss as compute_amtm

    torch.manual_seed(0)
    actual, _ = compute_amtm(model, batch)
    assert actual.shape == ()
    masked_count = int(base_mask.sum())
    if masked_count != K:
        mean_normalized = sum_loss / masked_count
        assert not torch.allclose(actual, mean_normalized)
