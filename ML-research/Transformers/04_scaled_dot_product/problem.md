<img src="../../../assets/divider.svg" width="100%"/>

# $\color{#a3e635}{\text{04. Scaled Dot-Product Attention}}$

<div align="center">

![Medium](https://img.shields.io/badge/difficulty-medium-eab308?style=flat-square&labelColor=0d1117)&nbsp;
[![TensorTonic](https://img.shields.io/badge/solve%20it-TensorTonic-22d3ee?style=flat-square&labelColor=0d1117)](https://www.tensortonic.com/research/transformer/transformers-scaled-dot-product-attention)

</div>

<img src="../../../assets/divider.svg" width="100%"/>

<br>

## $\color{#a3e635}{\text{Problem}}$

Implement **`scaled dot-product attention`**: each token looks at all the other tokens and takes a weighted mix of their values.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

> 📄 Section 3.2.1 of the paper.

<br>

## $\color{#a3e635}{\text{Input / Output}}$

<div align="center">

| | Shape | Meaning |
|:--|:--:|:--|
| **Q** (queries) | `(batch, q_len, d_k)` | what each token is looking for |
| **K** (keys) | `(batch, k_len, d_k)` | what each token offers |
| **V** (values) | `(batch, k_len, d_v)` | the information to mix |
| **Output** | `(batch, q_len, d_v)` | one mixed vector per query |

</div>

- All tensors are `float32`. The output is **unrounded**.

<br>

## $\color{#a3e635}{\text{How It Works}}$

```mermaid
flowchart LR
    Q["Q"] --> S["Q · Kᵀ<br>scores"]
    K["K"] --> S
    S --> D["÷ √d_k"]
    D --> W["softmax<br>weights"]
    W --> O["weights · V"]
    V["V"] --> O
    O --> R["Output"]
```

<br>

### $\color{red}{\text{1. Score — how well does each query match each key?}}$

Compute `Q @ Kᵀ`. This gives a `(batch, q_len, k_len)` matrix. A bigger number means a better match.

<br>

### $\color{red}{\text{2. Scale — divide by } \sqrt{d_k}}$

Divide the scores by the square root of `d_k` (the last dimension of `K`).

> 💡 Without this, large `d_k` gives huge scores. Softmax then becomes too sharp and learning gets unstable.

<br>

### $\color{red}{\text{3. Softmax — turn scores into weights}}$

Apply softmax over the **last dimension** (the keys). Each row now sums to **1**.

<br>

### $\color{red}{\text{4. Mix — weighted sum of values}}$

Multiply the weights by `V`. Each output is an average of the value vectors, weighted by attention.

<br>

### $\color{red}{\text{Example}}$

`Q = K = [[1, 0], [0, 1]]`, `V = [[1, 2], [3, 4]]`, so `d_k = 2`

1. Scores: `Q @ Kᵀ = [[1, 0], [0, 1]]`
2. Scale by `√2`: `[[0.71, 0], [0, 0.71]]`
3. Softmax per row: `[[0.67, 0.33], [0.33, 0.67]]`
4. Mix with `V`:

<div align="center">

| Query | Weights | Output |
|:--:|:--:|:--:|
| 1st | `[0.67, 0.33]` | `[1.6605, 2.6605]` |
| 2nd | `[0.33, 0.67]` | `[2.3395, 3.3395]` |

</div>

- The 1st query matches the 1st key best, so its output leans toward `V[0]`.

<br>

## ➡️ Code: [`solution.py`](solution.py)

<br>

<img src="../../../assets/divider.svg" width="100%"/>