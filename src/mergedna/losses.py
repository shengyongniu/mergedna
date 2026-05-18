from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mergedna.data import PAD_ID
from mergedna.model import MergeDNAModel


@dataclass
class MergeDNALosses:
    total: torch.Tensor
    mtr: torch.Tensor
    latent_mtr: torch.Tensor
    amtm: torch.Tensor


def reconstruction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    ignore_index: int = PAD_ID,
) -> torch.Tensor:
    if mask is not None:
        if not bool(mask.any()):
            return logits.sum() * 0.0
        logits = logits[mask]
        targets = targets[mask]
        return F.cross_entropy(logits, targets)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def mtr_loss(model: MergeDNAModel, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(input_ids)
    return reconstruction_loss(output.logits, input_ids), output.logits


def latent_mtr_loss(model: MergeDNAModel, input_ids: torch.Tensor) -> torch.Tensor:
    output, _ = model.latent_merge_reconstruction(input_ids, detach_local=True)
    return reconstruction_loss(output.logits, input_ids)


def amtm_loss(model: MergeDNAModel, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    output, base_mask, _ = model.forward_amtm(input_ids)
    return reconstruction_loss(output.logits, input_ids, mask=base_mask), base_mask


def mergedna_loss(
    model: MergeDNAModel,
    input_ids: torch.Tensor,
    *,
    latent_mtr_weight: float = 0.25,
) -> MergeDNALosses:
    output = model(input_ids)
    mtr = reconstruction_loss(output.logits, input_ids)
    latent = latent_mtr_loss(model, input_ids)
    amtm, _ = amtm_loss(model, input_ids)
    total = mtr + latent_mtr_weight * latent + amtm
    return MergeDNALosses(total=total, mtr=mtr, latent_mtr=latent, amtm=amtm)
