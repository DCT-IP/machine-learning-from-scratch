# linear regression 

this repository implements linear regression from first principles, starting from the mathematical formulation and translating each step into code incrementally

--- 

## learning objectives
 - Derice linear regression model from assumption
 - express the losst function in vector form
 - compute gradients analytically 
 - implement gradient descent step-by step
 - debu training using loss function

---

## repository structure
- `theory.md` – Mathematical derivation and explanations
- `linear_regression.py` – Core implementation (updated as learning progresses)
- `experiments.ipynb` – Final experiments and visualizations

names may change for the notebook
---
## log updates
_________________________________________________________________________________________________
|    dates   |                                 updates                                          |
|   9-01-26  | made the repo strcture and readme and theory files will moveonto proper work asap|

|  12-01-26  | made the initial step of regression, utilizing theory from gilbert strang and    |
|            | thus utilising least squares method and residuals to predict the equation of line|

|  16-01-26  | Added a inv(X@X.T) @ X.T @ Y, for understanding might remove them or make them   |
|            | comments when needed to as they r for personal understanding                     |

|  24-01-26  | Changed code for gradient descent, allows for looping uses epochs and learning ra|
|            | te to update values accordingly                                                  |