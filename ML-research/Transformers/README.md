# Paper : Attention Is All You Need

**Authors:** Vaswani et al.  
**Year:** 2017  
**Paper:** [arXiv link](...)

---

## 1. What Problem Does It Solve?

What was difficult with the previous approaches?

- RNNs process sequences sequentially
- Long-range dependencies are difficult
- Training is difficult to parallelize

### The key idea

Explain the paper's central idea in a few paragraphs.

> The Transformer removes recurrence and relies entirely on attention...

---

## 2. How Does It Work?

A high-level view of the architecture.

![Transformer architecture](...)

### Encoder

Explain briefly.

### Decoder

Explain briefly.

### Self-Attention

Explain the important mechanism.

![Self-attention](...)

### Multi-Head Attention

Explain why multiple heads are used.

### Positional Encoding

Explain why it is necessary.

---

## 3. Key Equations

The important equations from the paper.

### Scaled Dot-Product Attention

$$
Attention(Q,K,V)
=
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Briefly explain each component.

---

## 4. Implementations

| Concept | What I implemented | Code |
|---|---|---|
| Scaled Dot-Product Attention | Basic attention mechanism | [`attention`](./attention/) |
| Multi-Head Attention | Multiple attention heads | [`multi-head-attention`](./multi-head-attention/) |
| Positional Encoding | Inject sequence position | [`positional-encoding`](./positional-encoding/) |

---

## 5. What I Learned

- ...
- ...
- ...

---


| # | Topic | Description | Difficulty |
|:-:|:------|:------------|:----------:|
| 01 | [Tokenization](https://www.tensortonic.com/research/transformer/transformers-tokenization) | Convert text to token IDs | ![Medium](https://img.shields.io/badge/Medium-eab308?style=flat-square) |
| 02 | [Embedding Layer](transformers/02-embedding-layer/) | Map tokens to vectors | ![Easy](https://img.shields.io/badge/Easy-22c55e?style=flat-square) |
| 03 | [Positional Encoding](transformers/03-positional-encoding/) | Inject position via sinusoids | ![Medium](https://img.shields.io/badge/Medium-eab308?style=flat-square) |
| 04 | [Scaled Dot-Product Attention](transformers/04-scaled-dot-product-attention/) | Core QKV mechanism | ![Medium](https://img.shields.io/badge/Medium-eab308?style=flat-square) |
| 05 | [Multi-Head Attention](transformers/05-multi-head-attention/) | Parallel attention heads | ![Hard](https://img.shields.io/badge/Hard-ef4444?style=flat-square) |
| 06 | [Feed-Forward Network](transformers/06-feed-forward-network/) | Position-wise MLP | ![Easy](https://img.shields.io/badge/Easy-22c55e?style=flat-square) |
| 07 | [Layer Normalization](transformers/07-layer-normalization/) | Normalize activations | ![Easy](https://img.shields.io/badge/Easy-22c55e?style=flat-square) |
| 08 | [Encoder Block](transformers/08-encoder-block/) | Complete encoder layer | ![Medium](https://img.shields.io/badge/Medium-eab308?style=flat-square) |
| 🏁 | **[Assemble & train a Transformer](transformers/09-end-to-end/)** | End-to-end | ![Hard](https://img.shields.io/badge/Hard-ef4444?style=flat-square) |
