# Neural Network Terrain Classification 

## Overview 
>This project implements a Terrain / Forest Cover Classification Neural Network using:
  - Neural Networks From Scratch (NumPy)
  - PyTorch Neural Networks
  - A custom-built Data Pipeline
  - A real-world environmental dataset
>The objective is to classify terrain/forest cover types using cartographic and geographical features.

---

## Dataset 
https://www.kaggle.com/datasets/aashish31476/covertype-dataset

### Dataset Information
>The dataset contains cartographic variables such as:
  - Elevation
  - Aspect
  - Slope
  - Hillshade
  - Wilderness Area
  - Soil Type
>Target:
  - Forest Cover Type (7 classes)

---

## Pipeline used 
>This project uses a custom preprocessing pipeline built separately in the repository.

>Pipeline components used:
  - SimpleImputer
  - StandardScaler
  - train_test_split

---

## Project 1: self made Neural Network 
### Features

>Implemented using only:
  - NumPy
  - Matrix operations
  - Manual gradients
  - Manual backpropagation

---

## Project 2: PyTorch usage
### Features

>Implemented using:
  - PyTorch
  - Autograd
  - Adam Optimizer
  - CrossEntropyLoss

---
