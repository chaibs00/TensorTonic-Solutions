<img src="../../assets/divider.svg" width="100%"/>

# $\color{#b31b1b}{\text{Attention Is All You Need}}$

**Year:** 2017  
**Paper:** [![arXiv](https://img.shields.io/badge/arXiv-1706.03762-b31b1b?style=flat-square&labelColor=0d1117)](https://arxiv.org/abs/1706.03762)

<img src="../../assets/divider.svg" width="100%"/>


## Some Foundations — (RECAP)

> Before diving into  **Attention is all you need**, let first review some fundamentals concepts to undrestand the problem the paper is trying to solve.


<details>

### $\color{Cyan}{\text{Sequences and Sequence Modeling }}$  

- **A sequence** is an ordered collection of elements, where the position of each element carry a meaning. 
    - `setence` => sequence of words 
    - `time series` => sequence of measurements

- **Sequence Modeling** is the process of learning **patterns and relationships** within **sequential data**, often to understand the sequence or predict what comes next.


### $\color{Cyan}{\text{Tokenization and Embeddings}}$ 

Before a **sequence of text** can be processed by a neural network, it must be converted into a numerical representation. This involves 3 steps : 

   1. `Tokenization` :  text is divided into smaller units. 
   2. `Token IDs` : Each token is mapped to an integer ID from the tokenizer's vocabulary.
   3. `Embeddings` :  Each token ID is mapped to a dense numerical vector that the model can process and learn from.



### $\color{Cyan}{\text{RNNs, LSTMs, and GRUs}}$

There are different ways to process sequences:

- `RNNs`: Process sequences step by step, using information from previous steps.

- `LSTMs & GRUs`: Types of RNNs that use gates to help decide what information to keep or forget, making it easier to learn from longer sequences.

- `Transformers`: Use `attention` to learn relationships between sequence elements, allowing them to be processed in parallel during training.

- `Sequence-to-Sequence (Seq2Seq)`: A setup where a model takes one sequence as input and produces another sequence as output, such as translating a sentence from one language to another.


### $\color{Cyan}{\text{Attention}}$

-  An algorithm that helps neural networks **focus on the most important parts of input data** (like words in a sentence or patches in an image) rather than treating everything equally.

<br>


</details>

<br>


## $\color{#a3e635}{\text{The Motivation}}$

Before the `Transformer` was introduced, `RNNs`, particularly `LSTMs` and `GRUs`, were widely used to process sequential data. They were successful in `Sequence-to-Sequence` tasks. However, they had some limitations : 


- **Sequential computation:** Step-by-step processing limits parallelization and increases training time & cost.

- **Long-Range Dependencies:** Learning relationships between distant tokens is difficult because information passes through successive recurrent steps.

<br>

> These limitations motivated researchers to explore alternative architectures.


<br>

## $\color{#a3e635}{\text{The Researchers' Intuition}}$

- What if an `architecture` could capture relationships between tokens directly, without processing them step by step like `RNNs` or using `convolutions`?

- `Attention` allows a neural network to focus on relevant parts of the input when processing each token.

<br>


> Could `attention` alone handle sequence-to-sequence tasks, without recurrence or convolutions?

<br>

## $\color{#a3e635}{\text{The Transformer Architecture}}$

The authors introduced the `Transformer`, an architecture that replaced slow, word-by-word sequential processing with `self-attention`, allowing entire sequences of text to be processed in parallel during training.


<div align="center">

<img src="../../assets/transformer.png" width="30%"/>

<p><em>Figure from the original paper, <a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</a>.</em></p>

</div>

### The Big Picture

The Transformer keeps the **encoder–decoder** setup used for translation:

1. The **encoder** reads the whole input sentence and builds a rich representation of each word.
2. The **decoder** generates the output sentence one word at a time, looking at the encoder's representations.
3. A final **Linear + Softmax** layer turns the decoder's output into probabilities over the vocabulary to pick the next word.

> Each side is a stack of **6 identical layers** (`N×` in the figure).

<br>

### 1. Input Embedding + Positional Encoding

- Words are turned into **token IDs**, then into **vectors** (embeddings).
- Attention looks at all words at once, so it has **no sense of order** by itself.
- **Positional encoding** adds a position signal (built from sine/cosine waves) to each embedding, so *"dog bites man"* ≠ *"man bites dog"*.

### 2. Self-Attention, the core idea

Each word looks at **every other word** in the sentence and decides how much each one matters to it.

> In *"The animal didn't cross the street because **it** was too tired"*, attention helps **it** focus on **animal**.

Each word plays three roles:
- **Query**: what am I looking for?
- **Key**: what do I contain?
- **Value**: what information do I pass on?

**Multi-Head Attention** runs several attentions in parallel, so each head can learn a different type of relationship (grammar, meaning, references…).

### 3. The Encoder

Each encoder layer has 2 parts:
1. **Self-attention**: every input word gathers context from the other input words.
2. **Feed-forward network**: refines each word's representation.

**Output:** one context-aware vector per input word.

### 4. The Decoder

Each decoder layer has 3 parts:
1. **Masked self-attention**: a word can only look at **previous** words, not future ones (no cheating while generating).
2. **Encoder–decoder attention**: the decoder looks at the encoder's output to find the relevant input words (e.g. which English word to translate next).
3. **Feed-forward network**: same as in the encoder.

### 5. Feed-Forward Network

A small 2-layer neural network applied to **each word separately**.
Attention **mixes** information between words; the feed-forward network **processes** it within each word.

### 6. Add & Norm (Residual Connections + Layer Normalization)

Around every block, you see **Add & Norm** in the figure:
- **Add (residual)**: the block's input is added to its output → information and gradients flow easily through deep stacks.
- **Norm (layer normalization)**: keeps values in a stable range → faster, more stable training.






<br>

<img src="../../assets/divider.svg" width="100%"/>

## Roadmap  

| # | Topic | Description | Difficulty |
|:-:|:------|:------------|:----------:|
| 01 | [Tokenization](./01_Tokenization/) | Convert text to token IDs | ![Medium](https://img.shields.io/badge/Medium-eab308?style=flat-square) |
| 02 | Embedding Layer | Map tokens to vectors | ![Easy](https://img.shields.io/badge/Easy-22c55e?style=flat-square) |
| 03 | Positional Encoding | Inject position via sinusoids | ![Medium](https://img.shields.io/badge/Medium-eab308?style=flat-square) |
| 04 | Scaled Dot-Product Attention | Core QKV mechanism | ![Medium](https://img.shields.io/badge/Medium-eab308?style=flat-square) |
| 05 | Multi-Head Attention | Parallel attention heads | ![Hard](https://img.shields.io/badge/Hard-ef4444?style=flat-square) |
| 06 | Feed-Forward Network | Position-wise MLP | ![Easy](https://img.shields.io/badge/Easy-22c55e?style=flat-square) |
| 07 | Layer Normalization | Normalize activations | ![Easy](https://img.shields.io/badge/Easy-22c55e?style=flat-square) |
| 08 | Encoder Block | Complete encoder layer | ![Medium](https://img.shields.io/badge/Medium-eab308?style=flat-square) |
| 🏁 | **Assemble & train a Transformer** | End-to-end | ![Hard](https://img.shields.io/badge/Hard-ef4444?style=flat-square) |

<img src="../../assets/divider.svg" width="100%"/>