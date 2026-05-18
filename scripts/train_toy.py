from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mergedna.data import SequenceDataset, SyntheticDNADataset, read_sequences  # noqa: E402
from mergedna.losses import mergedna_loss  # noqa: E402
from mergedna.model import MergeDNAConfig, MergeDNAModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny MergeDNA model on DNA sequences.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--fasta",
        type=Path,
        default=None,
        help="Optional path to a FASTA or plain-text DNA file. Falls back to synthetic data when absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.fasta is not None:
        sequences = read_sequences(args.fasta, min_length=args.seq_len)
        if not sequences:
            raise SystemExit(f"No sequences with length >= {args.seq_len} in {args.fasta}")
        dataset: torch.utils.data.Dataset = SequenceDataset(sequences, seq_len=args.seq_len)
    else:
        dataset = SyntheticDNADataset(
            num_sequences=max(args.steps * args.batch_size, 64),
            seq_len=args.seq_len,
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    config = MergeDNAConfig(
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=4,
        local_layers=2,
        latent_layers=2,
        latent_decoder_layers=1,
        local_decoder_layers=1,
        local_window=16,
        merge_ratio=0.20,
        latent_merge_ratio=0.50,
    )
    model = MergeDNAModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    model.train()
    step = 0
    while step < args.steps:
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = mergedna_loss(model, batch)
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            print(
                f"step {step:04d} total={losses.total.item():.4f} "
                f"mtr={losses.mtr.item():.4f} latent={losses.latent_mtr.item():.4f} "
                f"amtm={losses.amtm.item():.4f}"
            )
            if step >= args.steps:
                break

    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": config, "model": model.state_dict()}, args.checkpoint)
        print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()
