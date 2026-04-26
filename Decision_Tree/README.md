# Decision Trees from Scratch

## Overview
This notebook demonstrates the implementation of a **Decision Tree Classifier** from scratch using only **NumPy**.

Decision Trees are supervised learning algorithms used for:
- Classification
- Regression

---

## Core Idea
Decision Trees work on the principle:

> "Split the data using the best possible question at each step."

Steps:
1. Choose the best feature and threshold to split the data
2. Divide the dataset into subsets
3. Repeat the process recursively
4. Stop when the data is sufficiently pure

---

## Tree Structure

A Decision Tree consists of:

- Root Node → first split
- Internal Nodes → decision rules (feature + threshold)
- Leaf Nodes → final prediction

Each path from root to leaf represents a decision rule.

---

## Splitting Criteria

To decide the best split, we measure how "pure" the resulting groups are.

### Gini Impurity

G = 1 - Σ (p_i)^2

- Measures how mixed the classes are
- Lower value → better split

---

### Entropy

H = - Σ p_i log2(p_i)

- Measures disorder in the data
- Lower value → better split

---

## Algorithm Steps

1. For each feature:
   - Try different split thresholds
2. Compute impurity (Gini/Entropy)
3. Select the split with lowest impurity
4. Recursively build left and right subtrees
5. Stop when:
   - Node is pure
   - Maximum depth is reached
   - Minimum samples condition is met

---

## Features of This Implementation

- No external ML libraries (no sklearn)
- Fully implemented using NumPy
- Recursive tree construction
- Gini-based splitting
- Visualization using matplotlib
- Decision boundary plotting
- Depth control to study model behavior

---

## Depth vs Overfitting

- Small depth → Underfitting (model too simple)
- Large depth → Overfitting (model memorizes data)

Controlling tree depth helps balance bias and variance.

---

## Limitations

- Prone to overfitting
- Sensitive to small changes in data
- Axis-aligned splits only
- Can be biased with imbalanced data

---

## Dataset Design (Important)

The dataset used in this notebook is synthetically generated but designed to mimic real-world conditions:
- Overlapping classes
- Noise in labels
- Non-linear patterns

This ensures the model learns meaningful decision boundaries rather than trivial splits.

---