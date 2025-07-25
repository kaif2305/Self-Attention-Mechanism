# Self-Implemented Attention Mechanism (Dot-Product Attention)

This project provides a basic, self-contained Python implementation of the core mechanism behind **dot-product attention**, often seen in transformer networks and other advanced neural architectures. It demonstrates how queries, keys, and values interact to compute attention weights and subsequently a context vector.

## What is Attention?

In neural networks, the "attention mechanism" allows a model to weigh the importance of different parts of an input sequence when processing another part of the sequence. Instead of processing an entire sequence as a single, fixed-size vector, attention allows the model to selectively focus on relevant segments of the input.

The core idea revolves around three main components:

* **Queries (Q)**: Represent what you are currently looking for.
* **Keys (K)**: Represent what's available to be matched against the queries.
* **Values (V)**: The actual information associated with the keys that will be weighted and aggregated.

The process typically involves:

1.  **Calculating Scores**: A measure of similarity or relevance between each Query and all Keys. Common methods include dot product (as in this example), scaled dot product, or additive (Bahdanau) attention.
2.  **Normalizing Scores**: Applying a softmax function to convert these raw scores into attention weights, which sum up to 1 for each query. This represents the "probability" or "importance" of each key's corresponding value with respect to the query.
3.  **Computing Context Vector**: Taking a weighted sum of the Values, where the weights are the attention weights. This aggregated vector is the "context" that the model will use, having selectively focused on the most relevant parts of the input.

## Project Overview

The Python script `attention_example.py` (assuming you save the code as such) demonstrates a simplified dot-product attention mechanism using NumPy:

1.  **Define Inputs**: Initializes sample `queries`, `keys`, and `values` as NumPy arrays.
2.  **Compute Attention Scores**: Calculates the dot product between `queries` and the transpose of `keys` (`queries @ keys.T`).
3.  **Apply Softmax**: Implements a `softmax` function to normalize the raw attention scores into `attention_weights`. This step is crucial as it ensures the weights sum to 1 for each query, representing a probability distribution.
4.  **Compute Context Vector**: Calculates the weighted sum of `values` using the `attention_weights`.
5.  **Print Results**: Displays the computed `attention_weights` and the final `context` vector.

## Mathematical Formulation (Dot-Product Attention)

Given:
* Queries $Q$ (shape: $N_Q \times D_K$)
* Keys $K$ (shape: $N_K \times D_K$)
* Values $V$ (shape: $N_K \times D_V$)

Where:
* $N_Q$ is the number of queries.
* $N_K$ is the number of keys (and values).
* $D_K$ is the dimensionality of keys and queries.
* $D_V$ is the dimensionality of values.

1.  **Attention Scores**:
    $Scores = Q K^T$
    (shape: $N_Q \times N_K$)

2.  **Attention Weights**:
    $AttentionWeights = \text{softmax}(Scores)$
    (shape: $N_Q \times N_K$)

3.  **Context Vector**:
    $Context = AttentionWeights V$
    (shape: $N_Q \times D_V$)

## Code Walkthrough

```python
import numpy as np

# Define queries, keys and values
# Queries: What we are looking for (e.g., words in a target sentence)
# Keys: What is available to be matched (e.g., words in a source sentence)
# Values: The information associated with the keys
queries = np.array([[1, 0, 1], [0, 1, 1]])
keys = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
values = np.array([[10, 0], [0, 10], [5, 5]])

# Compute attention scores
# This measures how relevant each key is to each query.
# The result will be a (num_queries x num_keys) matrix.
# For query 1 ([1,0,1]):
#   vs key 1 ([1,0,1]): 1*1 + 0*0 + 1*1 = 2
#   vs key 2 ([1,1,0]): 1*1 + 0*1 + 1*0 = 1
#   vs key 3 ([0,1,1]): 1*0 + 0*1 + 1*1 = 1
# Similarly for query 2 ([0,1,1])
scores = np.dot(queries, keys.T)
# scores will be:
# [[2, 1, 1],
#  [1, 1, 2]]

# Apply softmax to normalize scores
# Softmax converts raw scores into probability-like weights (summing to 1 for each row).
def softmax(x):
    # Subtract max for numerical stability (prevents exp(large_number) overflow)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(scores)
# Example attention_weights (approximate after softmax):
# [[0.707, 0.146, 0.146],  <- For query 1, mostly attends to key 1
#  [0.146, 0.146, 0.707]]  <- For query 2, mostly attends to key 3

# Compute weighted sum of values
# This combines the values based on their calculated importance.
# For each query, a new "context" vector is created by weighting the original values.
# context[0] = attention_weights[0,0]*values[0] + attention_weights[0,1]*values[1] + attention_weights[0,2]*values[2]
# and so on.
context = np.dot(attention_weights, values)

print("Attention Weights: \n", attention_weights)
print("Context Vector:\n", context)