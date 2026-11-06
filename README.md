# Machine Learning from Scratch
Classical machine learning algorithms implemented from scratch using NumPy, with an emphasis on mathematical foundations, optimization, and failure analysis.
This repository documents my journey of learning machine learning by understanding how algorithms work internally rather than relying entirely on existing libraries.
The focus is on mathematical intuition, manual implementation, optimization techniques, and understanding model behaviour.

---

## Scope and Rules
- Classical ML algorithms are implemented using NumPy and standard Python libraries.
- Core mathematical operations and optimization procedures are implemented manually.
- External ML libraries are used as references for comparison and validation, not as substitutes for the implementations.
- Experiments focus on understanding model performance, limitations, and failure cases.
- Implementations prioritize clarity and mathematical correctness over unnecessary abstraction.

---

## Topics Covered
### 1. Linear Regression
- Normal Equation
- Gradient Descent
- Cost Functions
- Convergence Analysis
- Prediction and Evaluation

### 2. Logistic Regression
- Binary Classification
- Sigmoid Function
- Cross-Entropy Loss
- Gradient Descent
- Precision and Recall
- Accuracy Analysis

### 3. Data Pipelines
- Data Cleaning
- Feature Encoding
- Data Preprocessing
- Train/Test Splitting

### 4. K-Nearest Neighbors (KNN)
- Distance Metrics
- Nearest Neighbor Selection
- Choosing Appropriate K
- Classification and Prediction
- Accuracy Measurements

### 5. Decision Trees
- Tree Structure
- Recursive Splitting
- Gini Impurity
- Entropy and Information Gain
- Tree Depth and Overfitting

### 6. Neural Networks
- Network Architecture
- Forward Propagation
- Backpropagation
- Activation Functions
- Loss Computation
- Weight Updates

---

## Planned Topics
The repository will gradually expand beyond classical ML into model evaluation, deep learning, and modern ML frameworks.

### 1. Additional Classical ML
- Naive Bayes
- K-Means Clustering
- Principal Component Analysis (PCA)
- Random Forests
- Gradient Boosting Fundamentals

### 2. Model Evaluation and Experimental Design
- Train, Validation, and Test Splits
- Cross-Validation
- Data Leakage
- Bias-Variance Trade-off
- Regularization
- Confusion Matrices
- Precision, Recall, F1-Score
- ROC-AUC and PR-AUC
- Probability Calibration
- Threshold Selection
- Feature Importance
- Error Analysis
- Reproducible Experiments

### 3. PyTorch
- Tensors and Tensor Operations
- Computational Graphs
- Automatic Differentiation (Autograd)
- Datasets and DataLoaders
- Neural Network Modules
- Loss Functions and Optimizers
- Training and Validation Loops
- Checkpointing and GPU Basics

### 4. Deep Learning
- Backpropagation in Greater Depth
- Gradient Descent and Optimization
- Weight Initialization
- Batch Normalization and Layer Normalization
- Dropout and Regularization
- Multilayer Perceptrons (MLPs)
- Convolutional Neural Networks (CNNs)
- Sequence Models
- Attention Mechanisms
- Transformers
- Transfer Learning

### 5. Advanced Applied ML
- Unsupervised Learning
- Representation Learning
- Anomaly Detection
- Time-Series Forecasting
- Uncertainty Estimation
- Advanced Feature Engineering
- Hyperparameter Optimization
- Experiment Tracking

### 6. ML Systems
- Model Serialization and Versioning
- Inference Pipelines
- Batch and Online Inference
- Model Serving
- Latency and Memory Profiling
- Model and Data Drift
- Quantization Fundamentals
- Testing and Reproducibility

---

## Learning Methodology
For each algorithm or concept:
1. Understand the underlying mathematics.
2. Derive the important equations.
3. Implement the algorithm where appropriate.
4. Test it on meaningful datasets.
5. Compare against established implementations.
6. Investigate failure cases and limitations.
7. Document observations and experimental results.

---

## Guiding Principle
The objective is not to implement every existing ML algorithm from scratch.
It is to understand the mathematical foundations, develop an intuition for optimization and model behaviour, and gradually transition toward building, evaluating, and deploying reliable machine learning systems.

**Understand the mathematics. Implement the fundamentals. Evaluate rigorously. Build reliable models.**