import torch

from mergedna.data import MASK_ID, N_ID, decode_dna, encode_dna, make_synthetic_dna


def test_encode_decode_roundtrip_with_n() -> None:
    ids = encode_dna("acgtn")
    assert ids.tolist()[-1] == N_ID
    assert decode_dna(ids) == "ACGTN"


def test_unknown_bases_map_to_n() -> None:
    ids = encode_dna("AXTG")
    assert ids.tolist()[1] == N_ID
    assert decode_dna(ids) == "ANTG"


def test_decode_specials_as_n_when_requested() -> None:
    ids = torch.tensor([MASK_ID])
    assert decode_dna(ids, skip_special=False) == "N"


def test_synthetic_dna_has_requested_length() -> None:
    assert len(make_synthetic_dna(64)) == 64
