# Observations

## 1. Convergence Behavior 

  - as seen in the file "convrg_analysis.ipynb" and the graph plotted in our housing price project, the loss decreases steadily over iterations
  - initialization of weights and bias has no effects on convergence due to convexity of the MSE loss 

---

## Reason for feature scaling 

  - Gradient descent converged very slowly in the absence of scaly
  - After scaling, the descent was steady and stable 

---

## Model Assumptions 

  - A linear relation between features and target 
  - Homoscedasticity 
  - no extreme multicollinearity
These assumptions are why the train data has comparetively higher error than training, classification problems will require a different hypothesis function. 