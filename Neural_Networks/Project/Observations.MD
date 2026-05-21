# Comprehensive Comparison: NumPy From-Scratch NN vs. PyTorch Implementation

This document provides a technical post-mortem and structural comparison between the custom, from-scratch neural network (`terrain_classifier_NN.ipynb`) and the PyTorch equivalent (`terrain_classifier_pytorch.ipynb`) designed to classify terrain features using the Covertype dataset (~581,012 samples, 55 features).

---

## 1. Architectural Design & Representation

| Feature | NumPy (From-Scratch) Implementation | PyTorch Implementation |
| :--- | :--- | :--- |
| **Layer Definitions** | Hand-coded matrix math ($X \cdot W + b$) inside variable blocks. | Encapsulated inside an `nn.Module` object using `nn.Linear` layers. |
| **Activation Layers** | Custom NumPy threshold masks (`np.maximum(0, x)` and manual exponent scaling). | Highly optimized, vector-parallelized native functional variants (`nn.ReLU`). |
| **Output Representation** | Raw probabilities generated manually via a normalized `softmax` function. | Raw unactivated linear projections (**Logits**). |
| **State Management** | Local execution `cache` dictionaries mapping batch states to matrix dimensions. | Automated internal computational graph tracking via dynamic Autograd engine. |

---

## 2. Mathematical Pipeline Breakdown

### Custom NumPy Pipeline (The Core Mechanics)
In the manual implementation, every phase of backpropagation must be explicitly mapped via matrix calculus. 

1. **Forward Equations**:
   $$Z_1 = X \cdot W_1 + b_1 \quad \rightarrow \quad A_1 = \max(0, Z_1)$$
   $$Z_2 = A_1 \cdot W_2 + b_2 \quad \rightarrow \quad A_2 = \max(0, Z_2)$$
   $$Z_3 = A_2 \cdot W_3 + b_3 \quad \rightarrow \quad A_3 = \frac{e^{Z_3 - \max(Z_3)}}{\sum e^{Z_3 - \max(Z_3)}}$$

2. **Backward Scaling**: Gradients must be strictly normalized relative to the current mini-batch slice dimension ($m$):
   $$\delta_3 = \frac{\partial L}{\partial Z_3} = \frac{A_3 - Y_{\text{one-hot}}}{m}$$
   $$\frac{\partial L}{\partial W_3} = A_2^T \cdot \delta_3, \quad \frac{\partial L}{\partial b_3} = \sum \delta_3$$

### PyTorch Pipeline (The Automated Engine)
PyTorch replaces explicit calculus mapping with an underlying **Directed Acyclic Graph (DAG)** tracking structural tensors.
* Linear transformations automatically assign random weight matrices using refined Kaiming (He) Uniform initialization.
* Executing `.backward()` calculates a chain-rule backward pass on parameters flagged with `requires_grad=True`, while `optimizer.step()` optimizes weights instantly via algorithms like Adam or SGD.

---

## 3. Performance & Optimization Metrics

When executing full-batch gradient descent on massive datasets (~464,809 training rows), execution runtimes diverge massively based on vector memory strategies.

### The Full-Batch vs. Mini-Batch Dilemma
* **NumPy Full-Batch (Initial Approach):** Took **~5 minutes for 500 epochs**. This occurred because processing a $464,809 \times 55$ matrix directly in a single pass flushes out CPU L1/L2 hardware caches, forcing severe RAM-to-CPU transit latency. Additionally, it computed only 1 optimization weight update per epoch.
* **NumPy Mini-Batch (Optimized Approach - Batch Size 2048):** Dropped the runtime down to **a few seconds**. Keeping matrix sizes bounded within 2048 rows localizes execution to fast hardware cache lines, while performing **~227 weight updates per epoch**, leading to rapid convergence.
* **PyTorch Vector Acceleration:** Native performance leverages low-level C++ wrappers (THNN/ATen libraries) and basic linear algebra subprograms (BLAS), running even fast batches smoothly on CPUs, or accelerating exponentially when targeting a GPU (`cuda` / `mps`).

---

## 4. Key Differences & Bug Manifestations

Debugging a self-built neural network vs. an abstracted framework presents entirely distinct classes of software anomalies.

### Common Anomalies Found in NumPy From-Scratch
1. **Global Variable Mutation:** Using `global Z1, A1` causes validation routines to accidentally overwrite training shapes mid-epoch, mutating network layers and leading to unexpected matrix size mismatches.
2. **Gradient Explosion/Vanishing:** Failing to balance the output error calculation ($A_3 - Y$) with the mini-batch divisor ($m$) scales gradients too high, causing learning curves to drop accuracy metrics to zero.
3. **Numerical Overflows:** Calculating raw mathematical exponents like `np.exp(x)` on large values returns `inf` or `NaN` outputs. This requires manual safety offsets (`x - np.max(x)`).

### Common Anomalies Found in PyTorch
1. **Double Activation Conflicts:** Adding a manual `nn.Softmax` layer before an `nn.CrossEntropyLoss` function results in calculating logs on small fractions ($0$ to $1$). This distorts loss landscapes and breaks convergence.
2. **Data-Type System Mismatches:** NumPy processes datasets as 64-bit precision floats (`float64`/Double), while PyTorch layers default to 32-bit floats (`float32`). Passing data directly throws a explicit type runtime error.
3. **Target Vector Shape Errors:** PyTorch expects classifications to be 1D target arrays containing class numbers (`[0, 1, 4...]` with type `.long()`), whereas NumPy networks generally require explicit 2D one-hot target matrices.

---

## 5. Engineering Takeaways

1. **Building from scratch teaches the foundational mathematical mechanics.** Writing a custom NumPy model provides an intuitive understanding of the underlying calculus of backpropagation, structural learning rates, scale factors, and parameter updates.
2. **Frameworks provide production-grade scaling and optimization.** Transitioning to PyTorch abstracts mathematical overhead, automates gradient steps, and simplifies data streams with high-performance primitives (`DataLoader`, `Dataset`), making it much cleaner to build, maintain, and scale complex model architectures.