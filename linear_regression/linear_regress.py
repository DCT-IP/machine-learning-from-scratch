
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


