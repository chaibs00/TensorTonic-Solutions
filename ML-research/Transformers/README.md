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

