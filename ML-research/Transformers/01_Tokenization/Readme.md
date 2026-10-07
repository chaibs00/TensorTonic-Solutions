
<img src="/assets/divider.svg" width="100%"/>

# 01.Tokenization 

<div align="center">

![Medium](https://img.shields.io/badge/difficulty-medium-eab308?style=flat-square&labelColor=0d1117)

![Topic](https://img.shields.io/badge/topic-attention-a3e635?style=flat-square&labelColor=0d1117)

[![TensorTonic](https://img.shields.io/badge/solve%20it-TensorTonic-22d3ee?style=flat-square&labelColor=0d1117)](https://www.tensortonic.com)


</div>


<img src="/assets/divider.svg" width="100%"/>


## 📝 Problem 

Given query, key and value matrices, compute the attention output — optionally applying a mask so some positions can't be attended to.

## 💡 Intuition

Each query asks _"which keys are relevant to me?"_. Dot products score the match, softmax turns scores into weights that sum to 1, and the output is the weighted mix of values.

## 📐 Math

$$\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

| Symbol | Shape | Meaning |
|:--|:--|:--|
| $Q$ | $(n, d_k)$ | queries |
| $K$ | $(m, d_k)$ | keys |
| $V$ | $(m, d_v)$ | values |
| output | $(n, d_v)$ | attended values |

## 🛠 Solution walkthrough

1. Scores: `Q @ K.T / sqrt(d_k)`
2. Mask: set blocked positions to `-inf`
3. Softmax over the last axis (subtract the row max for stability)
4. Multiply weights by `V`

➡️ Code: [`solution.py`](solution.py)

## ⚠️ Pitfalls

- Forgetting **√dₖ** → softmax saturates for large dimensions.
- Naive softmax overflows — always subtract the max first.
- Softmax over the **wrong axis** (must be over keys).

## ⏱ Complexity

Time $O(n \cdot m \cdot d)$ · Memory $O(n \cdot m)$ for the weight matrix.

