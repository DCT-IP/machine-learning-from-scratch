import numpy as np
X_train = np.array([1,2,3,4,5])
Y_train = np.array([4,7,10,13,16]) #assume the function to be y = 3x + 1

X_train = np.column_stack((np.stack(np.ones(len(X_train))),X_train))
w = np.linalg.pinv(X_train) @ Y_train #least squares soln
print(w) #should be two

Y_test = X_train @ w
print("\nPredictions from training model",Y_test) #seeing error margin
residual = Y_train - Y_train  #E = ||y - Ax||
#to check 
print("\nOrthogonality check:- ", X_train.T @ residual)
