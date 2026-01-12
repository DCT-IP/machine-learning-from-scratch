import numpy as np
X_train = np.array([1,1.5,2,3,3.8,4,5])
Y_train = np.array([3,2,5,7,8.9,9,11]) #assuming the equation to be y = 2x+1
#above is our target

X = np.array(np.column_stack((np.stack(np.ones(len(X_train))),X_train)))  #basis
#this allows to write y = b + mx in the form of b = Ax
#the matrix of X is [1 xi]        system observed while solving a linear sys

w = np.linalg.pinv(X)@Y_train          #coordinates 
#solving for least squares solution     - here x will store our [bias and weight]

print("bias, weight:",w) # to check the values

#to check orthognality as X^T.e = 0
residual = Y_train - X@w
print("orthogonality check", X.T@residual)
print("residual norm: ",np.linalg.norm(residual))
#to see predictability
X_test = np.array([6,7,8]);
X_test= np.array(np.column_stack((np.stack(np.ones(len(X_test))),X_test)))
Y_test = X_test@w
print("test values: ",Y_test)
# added noise to understand its effects on training the model
