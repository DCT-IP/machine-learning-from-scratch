# Neural Network from Scratch using NumPy

## Overview
This notebook demonstrates the implementation of a simple **Neural Network** from scratch using only **NumPy**.

The goal of this project is to understand:
- Forward propagation
- Backpropagation
- Activation functions
- Gradient descent
- Parameter updates

without using external deep learning frameworks such as TensorFlow or PyTorch.

---

## Core Idea

A neural network learns patterns by:
1. Computing predictions
2. Measuring error
3. Updating parameters to reduce error

The learning process is repeated over many iterations until the network improves its predictions.

---

## Neural Network Structure

This implementation uses:

- Input Layer
- Hidden Layer
- Output Layer

Each neuron performs:
1. Weighted sum
2. Bias addition
3. Activation function

---

## Mathematical Flow

Input  
→ Weighted Sum  
→ Activation Function  
→ Prediction  
→ Loss Calculation  
→ Backpropagation  
→ Parameter Updates  
→ Learning

---

## Forward Propagation

Forward propagation computes predictions using:

```text
Z = XW + b
A = activation(Z)
```
---

## Activation Functions
### ReLU
Used in hidden layers.
f(x) = max(0, x)

### Sigmoid
Used in output layer for binary classification.
σ(x) = 1 / (1 + e^(-x))

### Soft Point Activation
f(x) = ln(1+e^x)

---

## Backpropagation

Backpropagation computes gradients of the loss with respect to:
  >weights
  >biases
  using the chain rule.

These gradients tell the network:
how parameters should change to reduce error.

---

## Project

