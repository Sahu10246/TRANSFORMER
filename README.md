# Hindi Abstractive Text Summarization using Transformer (From Scratch)

## Overview

This project implements a full Transformer-based encoder-decoder architecture from scratch using PyTorch for abstractive text summarization on a custom Hindi dataset.

Unlike library-based implementations, all core components are manually implemented, including:

- Multi-Head Attention
- Masked Self-Attention
- Cross-Attention
- Layer Normalization
- Feed Forward Networks
- Label Smoothing
- Padding & Causal Masking
- Transformer Learning Rate Warmup
- Gradient Clipping
- Mixed Precision Training (AMP)
- Beam Search Decoding
- ROUGE Evaluation

This project demonstrates a deep understanding of Transformer internals and sequence modeling.

---

## Architecture

- Encoder-Decoder Transformer
- Rotary Positional Encoding compatible attention
- Multi-head scaled dot-product attention
- Residual connections + LayerNorm
- Teacher forcing training strategy

---

## Training Features

- Label smoothing for better generalization
- Learning rate warmup schedule
- Mixed precision (AMP) for efficient GPU training
- Gradient clipping
- Model checkpointing
- Best model saving

---

## Inference Features

- Greedy decoding
- Beam search decoding
- Custom tokenizer
- Model reload from checkpoint

---

## Dataset Format

The dataset must be a CSV file named:

