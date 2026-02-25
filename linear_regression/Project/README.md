#  Housing Price Prediction Using Linear Regression

This project implements **Linear Regression from scratch** to predict housing prices using a real-world dataset.  
The focus is not only on building a predictive model, but also on understanding the **convergence behavior of Gradient Descent**, feature scaling, and evaluation without relying on machine learning libraries like `scikit-learn`.

---

##  Dataset

The dataset used in this project was downloaded from:

 **House Price Dataset (Kaggle)**  
https://www.kaggle.com/datasets/juhibhojani/house-price?resource=download

> The dataset is *not included* in this repository due to GitHub’s file size limitations.  
> After downloading, please place the file in the same directory as the python file OR make apt changes for the file location

##  Project Overview

In this project, you will find:

###  Data Cleaning and Preprocessing
- Handling missing and inconsistent values
- Extracting numeric features from mixed text formats
- Removing irrelevant or sparse columns
- Feature scaling (standardization)

###  Linear Regression from Scratch
- Implementation of the loss function (MSE)
- Calculation of gradients
- Gradient Descent optimizer
- Normal Equation as a closed-form baseline

###  Convergence Analysis
- Experiments with different learning rates
- Loss vs. iterations plots
- Weight trajectory visualization
- Understanding how scaling impacts convergence

###  Evaluation Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² score

---

