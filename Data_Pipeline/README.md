# Data Pipeline Module

Implements preprocessing utilities from scratch inspired by sklearn.

---

### Features
- Data Cleaning (SimpleImputer)
- Encoding (OneHotEncoder)
- Scaling (StandardScaler)
- Train/Test Split
- Pipeline system

---

### Design
- All transformers follow fit → transform interface
- Compatible with Pipeline chaining
- Reusable across all ML models

---

### How to use 
'python -m Example_DataPipe.somemodel' to run from root 
from Data_Pipeline.scaling import StandardScaler
from Data_Pipeline.cleaning import SimpleImputer
from Data_Pipeline.pipeline import Pipeline
from Data_Pipeline.split import train_test_split

---

### Example

from Data_Pipeline import Pipeline, StandardScaler, SimpleImputer

pipeline = Pipeline([
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler())
])

X = pipeline.fit_transform(X)
