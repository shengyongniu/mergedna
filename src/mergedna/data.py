from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<pad>"
MASK_TOKEN = "<mask>"
BASES = ("A", "C", "G", "T", "N")
TOKENS = (PAD_TOKEN, *BASES, MASK_TOKEN)
TOKEN_TO_ID = {token: idx for idx, token in enumerate(TOKENS)}
ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
A_ID = TOKEN_TO_ID["A"]
C_ID = TOKEN_TO_ID["C"]
G_ID = TOKEN_TO_ID["G"]
T_ID = TOKEN_TO_ID["T"]
N_ID = TOKEN_TO_ID["N"]
MASK_ID = TOKEN_TO_ID[MASK_TOKEN]
VOCAB_SIZE = len(TOKENS)


def normalize_sequence(sequence: str) -> str:
    """Uppercase a DNA string and map unknown letters to N."""
    chars = []
    for char in sequence.upper():
        if char in BASES:
            chars.append(char)
        elif char.isspace():
            continue
        else:
            chars.append("N")
    return "".join(chars)


def encode_dna(sequence: str, *, pad_to: int | None = None) -> torch.Tensor:
    encoded = [TOKEN_TO_ID[base] for base in normalize_sequence(sequence)]
    if pad_to is not None:
        if len(encoded) > pad_to:
            encoded = encoded[:pad_to]
        encoded = encoded + [PAD_ID] * (pad_to - len(encoded))
    return torch.tensor(encoded, dtype=torch.long)


def decode_dna(ids: Iterable[int] | torch.Tensor, *, skip_special: bool = True) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    chars: list[str] = []
    for idx in ids:
        token = ID_TO_TOKEN[int(idx)]
        if skip_special and token in {PAD_TOKEN, MASK_TOKEN}:
            continue
        chars.append("N" if token in {PAD_TOKEN, MASK_TOKEN} else token)
    return "".join(chars)


def read_sequences(path: str | Path, *, min_length: int = 1) -> list[str]:
    """Read plain text or FASTA-like DNA files into normalized sequences."""
    path = Path(path)
    sequences: list[str] = []
    current: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                seq = normalize_sequence("".join(current))
                if len(seq) >= min_length:
                    sequences.append(seq)
                current = []
            continue
        current.append(line)
    if current:
        seq = normalize_sequence("".join(current))
        if len(seq) >= min_length:
            sequences.append(seq)
    return sequences


def make_synthetic_dna(seq_len: int, rng: random.Random | None = None) -> str:
    rng = rng or random
    motifs = ("TATAAA", "CGCG", "GATTACA", "ACGTACGT", "GGGCCC")
    chunks: list[str] = []
    while sum(len(chunk) for chunk in chunks) < seq_len:
        choice = rng.random()
        if choice < 0.35:
            base = rng.choice("ACGT")
            chunks.append(base * rng.randint(4, 24))
        elif choice < 0.70:
            motif = rng.choice(motifs)
            chunks.append(motif * rng.randint(1, 4))
        elif choice < 0.95:
            chunks.append("".join(rng.choice("ACGT") for _ in range(rng.randint(4, 32))))
        else:
            chunks.append("N" * rng.randint(1, 6))
    return "".join(chunks)[:seq_len]


@dataclass
class SyntheticDNADataset(Dataset):
    num_sequences: int = 1024
    seq_len: int = 256
    seed: int = 13

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        self.sequences = [make_synthetic_dna(self.seq_len, rng) for _ in range(self.num_sequences)]

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> torch.Tensor:
        return encode_dna(self.sequences[index], pad_to=self.seq_len)


class SequenceDataset(Dataset):
    def __init__(self, sequences: Iterable[str], seq_len: int) -> None:
        self.seq_len = seq_len
        self.sequences = [normalize_sequence(seq) for seq in sequences if normalize_sequence(seq)]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> torch.Tensor:
        return encode_dna(self.sequences[index], pad_to=self.seq_len)
