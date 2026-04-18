import sys
import os
sys.path.append(os.path.abspath(".."))
import numpy as np

from Data_Pipeline import (
    Pipeline,
    StandardScaler,
    SimpleImputer,
    train_test_split
)

# Data
X = np.array([[1, 2], [3, np.nan], [5, 6]])
y = np.array([1, 2, 3])

# Pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler())
])

# Transform
X = pipeline.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Output
print("X_train:", X_train)
print("y_train:", y_train)
print("X_test:", X_test)
print("y_test:", y_test)