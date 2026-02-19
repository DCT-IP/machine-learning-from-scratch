#modify grad descent to track convergence 
#add closed form solution
#compute learning rate bound 
#observation for loss curve's decrease, gradeitn norm tending to 0, LR's size effects
#plot 
import numpy as np
def Design_mat(X):
    # for creation of design matrix 
    #changes matrix to suit the form of linear equations ' Ax = b ' better
    X = np.array(np.column_stack((np.stack(np.ones(len(X))),X)))
    return X

def loss(Y, X, w):
    n = len(Y)
    err = X@w - Y
    return 1/n * np.dot(err,err)

def gradient(X, Y, w):
    n = len(Y)
    return 2/n * X.T @ (X@w - Y)

def Grad_Descent(w, X, Y, epoch = 100, lr = 0.01):
    for _ in range(0,epoch):
        w = w - lr * gradient(X, Y, w)

    return w

def predict(x,w):
    #to test out the data 
    return x@w

def main():
    X_train = np.array([1,1.5,2,3,3.8,4,5])
    Y_train = np.array([3,2,5,7,8.9,9,11])
    X_test = np.array([6,7,8])
    X = Design_mat(X_train)
    w = np.array([0.0, 1.0]) 
    print("Loss before the descent:-", loss(Y_train,X,w))
    w = Grad_Descent(w, X, Y_train)
    print("Loss of new w:-", loss(Y_train,X, w))
    print("new Bias:- ", w[0],"new weight", w[1])
    print(X_test, " this will be used to test")
    X_test = Design_mat(X_test)
    print(predict(X_test, w))


if __name__ == "__main__":
    main()


# def Normal_Eqn(X,Y):
#     #using Least Squares to solve
#     w = np.linalg.pinv(X)@Y
#     return w
# w2 = Normal_Eqn_wrtTranspose(X,Y_train)
# print("by pinv method:- ", w1)
# print("by X^TX method ",w2)
# print("Norm: ", np.linalg.norm(w1-w2))
# print("Rank of X:-", np.linalg.matrix_rank(X))
# print("Rank of X.X^T:-",np.linalg.matrix_rank(X.T@X))
# def Normal_Eqn_wrtTranspose(X,Y):
#     w = np.linalg.inv(X.T@X) @ X.T @ Y
#     return w

# def design_matrix(x):
#     return np.column_stack((np.ones(len(x)), x))

# def loss(X, y, w):
#     n = len(y)
#     err = X @ w - y
#     return (1/n) * np.dot(err, err)

# def gradient(X, y, w):
#     n = len(y)
#     return (2/n) * X.T @ (X @ w - y)

# def gradient_descent(X, y, lr=0.01, epochs=1000):
#     w = np.zeros(X.shape[1])
#     for _ in range(epochs):
#         w -= lr * gradient(X, y, w)
#     return w

# # Data
# X_train = np.array([1,1.5,2,3,3.8,4,5])
# Y_train = np.array([3,2,5,7,8.9,9,11])

# X = design_matrix(X_train)

# w_gd = gradient_descent(X, Y_train, lr=0.01, epochs=5000)

# print("Bias:", w_gd[0])
# print("Weight:", w_gd[1])
# print("Final loss:", loss(X, Y_train, w_gd))
# def OrthoCheck(x,res):
#     #should be 0 or something equivalent
#     return x.T@res
    # X_test = Design_mat(X_test)
    # X = Design_mat(X_train)
    # Err = residual(X,Y_train,w)
    # print("Norm for the residual: ", np.linalg.norm(Err))
    # print("On Orthogonality Check:-", OrthoCheck(X,Err))
    # print("On trying out the equation with ", w, " as bias and weight and data as \n",X_test,"\n: ",predict(X_test,w))
    # print("Comparison of using pinv(X)Y and using X^TX:-\n")
