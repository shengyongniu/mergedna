# MergeDNA

This repo is a compact PyTorch prototype of **MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging**. It is meant for a take-home implementation: faithful enough to show the model structure and training objectives, small enough to run tests locally and a toy training loop on CPU or a Colab GPU.

It is not a reproduction of the full paper run. The paper trains a roughly 380M parameter model with 4k-token contexts for 100k steps on 8 A100-80G GPUs. This implementation keeps the same moving parts at a much smaller scale.

## What Is Implemented

- Base-level DNA input over `A/C/G/T/N`, plus padding and mask tokens.
- A local encoder with local-window attention and source-tracked token merging.
- ToMe-inspired local and global merge utilities.
- A latent encoder and decoder over compressed local tokens.
- A local decoder that unmerges latent/local representations back to base resolution.
- Training helpers for MTR, latent MTR, and AMTM.
- Synthetic DNA data, toy training, tokenization inspection, and unit tests.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,notebook]"
```

On Colab, install the package from the notebook after cloning the repo.

## Run Tests

```bash
pytest
```

## Toy Training

```bash
python scripts/train_toy.py --steps 50 --device cpu
```

For a GPU run:

```bash
python scripts/train_toy.py --steps 200 --device cuda
```

The toy dataset mixes random DNA with repeats and short motifs. It is only a smoke test that the architecture trains and gradients flow.

## Inspect Tokenization

```bash
python scripts/inspect_tokenization.py
```

This prints simple token-span statistics from the local merge stack. Repetitive regions should usually produce longer merged spans than noisier regions, though this prototype is intentionally small and not pretrained at paper scale.

## How This Maps To The Paper

MergeDNA learns a dynamic tokenizer by repeatedly applying local attention and token merging. The resulting local tokens keep a source map back to original bases. A latent Transformer then models the compressed sequence globally, with decoders reconstructing base-level outputs.

This repo mirrors that flow:

1. `LocalEncoder`: embeds bases, applies local attention, and merges similar nearby tokens.
2. `LatentEncoder`: applies full self-attention to the compressed local tokens.
3. `LatentDecoder`: reconstructs local-token representations.
4. `LocalDecoder`: unmerges token representations to base resolution and predicts bases.
5. `losses.py`: exposes MTR, latent MTR, and AMTM-style losses.

The merge code favors clarity over maximum throughput. Source groups are tracked as integer base-position lists, then expanded when needed for unmerge and masking.

## Out Of Scope

- No 380M parameter training run.
- No 100k-step multi-species pretraining.
- No full Genomic Benchmark, NT, or GUE reproduction.
- No highly optimized sparse/ragged CUDA kernels.

## Scaling Up

The defaults are deliberately small. For Colab or a larger GPU, try increasing `d_model`, `seq_len`, layer counts, batch size, and training steps in `MergeDNAConfig` and `scripts/train_toy.py`. The first bottleneck will be Python-level source tracking and repeated variable-length batching, not the Transformer blocks.
