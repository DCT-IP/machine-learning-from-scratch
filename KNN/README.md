# K-Nearest Neighbors (KNN) from Scratch

##  Overview
This notebook demonstrates the implementation of the **K-Nearest Neighbors (KNN)** algorithm from scratch using only **NumPy**.

KNN is a simple, non-parametric, supervised learning algorithm used for:
- Classification
- Regression

---

## Core Idea
KNN works on the principle:

> "Data points that are close to each other are likely to have the same label."

Steps:
1. Compute distance from test point to all training points
2. Select K nearest neighbors
3. Perform majority voting (classification)

---

## Distance Metric

We use **Euclidean Distance**:

d(x, y) = sqrt(Σ (xi - yi)^2)

---

## Algorithm Steps

1. Store training data
2. For each test point:
   - Compute distances to all training points
   - Sort distances
   - Select top K neighbors
   - Perform majority vote

---

## Features of This Implementation

- No external ML libraries (no sklearn)
- Fully implemented using NumPy
- Synthetic dataset generated programmatically
- Visualization using matplotlib
- Decision boundary plotting
- Accuracy evaluation using leave-one-out validation

---

## Choosing K

- Small K → Overfitting (high variance)
- Large K → Underfitting (high bias)

---

## Limitations

- Slow for large datasets (O(n) per prediction)
- Sensitive to feature scaling
- Performance degrades in high dimensions

---

