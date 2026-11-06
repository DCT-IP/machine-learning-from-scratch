# Naive Bayes

Implementation of Naive Bayes classifiers from scratch, focusing on probabilistic reasoning, conditional independence, likelihood estimation, and classification.

## Objectives

* Understand Bayes' theorem and conditional probability.
* Derive the Naive Bayes classification rule.
* Understand the conditional independence assumption.
* Implement prior and likelihood estimation.
* Handle zero-frequency problems using Laplace smoothing.
* Understand log probabilities and numerical stability.
* Explore different Naive Bayes variants.
* Evaluate classification performance and analyze failure cases.

## Mathematical Foundations

The classifier is based on Bayes' theorem:

$$P(C\mid X)=\frac{P(X\mid C)P(C)}{P(X)}$$

Under the conditional independence assumption:

$$P(C\mid X_1,\ldots,X_n) \propto P(C)\prod_{i=1}^{n}P(X_i\mid C)$$

The predicted class is:

$$\hat C=\arg\max_C \left[ P(C)\prod_{i=1}^{n}P(X_i\mid C) \right]$$

Log probabilities are used to avoid numerical underflow.

## Implementations

The module explores:

* Categorical Naive Bayes
* Multinomial Naive Bayes
* Bernoulli Naive Bayes
* Gaussian Naive Bayes

Core implementations use Python, NumPy, and dictionaries where appropriate, without relying on pre-built machine learning classifiers.

## Concepts Covered

* Prior probability
* Likelihood estimation
* Posterior probability
* Conditional independence
* Maximum a posteriori (MAP) classification
* Laplace smoothing
* Log likelihood
* Feature distributions

## Experiments

Classification experiments using categorical and text-based datasets.
Evaluation includes:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Prediction confidence and probability behavior

## Failure Analysis

Investigate:

* Zero-frequency problems
* Correlated features
* Conditional independence violations
* Unseen feature values
* Poor probability calibration
* Sensitivity to feature representation

## Learning Outcome

Develop an understanding of probabilistic classification and the assumptions behind Naive Bayes, while implementing and evaluating classifiers without depending on pre-built machine learning algorithms.