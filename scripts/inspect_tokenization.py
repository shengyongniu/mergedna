from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mergedna.data import encode_dna, make_synthetic_dna  # noqa: E402
from mergedna.model import MergeDNAConfig, MergeDNAModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MergeDNA local token spans.")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MergeDNAConfig(max_seq_len=args.seq_len, d_model=args.d_model, merge_ratio=0.20)
    model = MergeDNAModel(config)
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location="cpu")
        if "config" in payload:
            config = payload["config"]
            model = MergeDNAModel(config)
        model.load_state_dict(payload["model"])

    sequence = args.sequence or make_synthetic_dna(config.max_seq_len)
    ids = encode_dna(sequence, pad_to=config.max_seq_len).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        local = model.encode_local(ids)

    lengths = [len(group) for group in local.sources[0]]
    counts = Counter(lengths)
    print(f"input length: {ids.size(1)}")
    print(f"local tokens: {len(lengths)}")
    print(f"compression: {ids.size(1) / max(len(lengths), 1):.2f}x")
    print("span length distribution:")
    for length, count in sorted(counts.items()):
        print(f"  {length:>2}: {count}")
    print("first spans:")
    for idx, group in enumerate(local.sources[0][:12]):
        print(f"  token {idx:>2}: bases {group[0]}..{group[-1]} ({len(group)} bases)")


if __name__ == "__main__":
    main()
