# Spam Classifier (Logistic Regression from Scratch)

This project demonstrates the application of logistic regression to classify messages into **spam** and **not spam (ham)** using a real-world dataset.

---

## Concepts Used

### Logistic Regression (From Scratch)

* Linear model: ( Z = Xw + b )
* Sigmoid function for probability estimation
* Binary classification using thresholding

### Optimization

* Likelihood and log-likelihood
* Cross-entropy loss (negative log-likelihood)
* Gradient descent
* Parameter updates using gradients

### Model Evaluation

* Confusion matrix (TP, TN, FP, FN)
* Accuracy
* Precision
* Recall
* Precision–Recall tradeoff

### Text Processing (NLP Basics)

* Text cleaning
* Tokenization
* Bag-of-Words representation
* Feature normalization

---

## Dataset

* SMS Spam Collection Dataset
* Contains labeled messages categorized as:
* Link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download

  * `spam`
  * `ham` (not spam)

---

## Pipeline

```
Raw Text
→ Cleaning
→ Tokenization
→ Vocabulary Creation
→ Vectorization (Bag of Words)
→ Normalization
→ Logistic Regression
→ Evaluation
```

---

## Features

* Vocabulary limited to top 1000 words to reduce overfitting
* Manual Bag-of-Words vectorization
* Logistic regression implemented from scratch
* Training using cross-entropy loss
* Evaluation using precision, recall, and accuracy

---

## Sample Output

* Accuracy: ~85–95%
* Precision: High (low false positives)
* Recall: Moderate to high

Results may vary depending on preprocessing and randomness.

---

## Key Learnings

* Logistic regression is a probabilistic model, not just a classifier
* Cross-entropy loss is directly linked to likelihood maximization
* Accuracy alone is insufficient for evaluating classification models
* Real-world datasets contain noise and are not perfectly separable
* Feature engineering significantly impacts model performance

---

## Notes

* Implemented using NumPy, Pandas, and Matplotlib
* No use of machine learning libraries (e.g., scikit-learn)
* Focus is on understanding core concepts and building from first principles
