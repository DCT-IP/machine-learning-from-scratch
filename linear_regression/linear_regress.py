import numpy as np
def Design_mat(X):
    # for creation of design matrix 
    #changes matrix to suit the form of linear equations ' Ax = b ' better
    X = np.array(np.column_stack((np.stack(np.ones(len(X))),X)))
    return X

def Normal_Eqn(X,Y):
    #using Least Squares to solve
    w = np.linalg.pinv(X)@Y
    return w

def residual(x,y,w):
    #calculates residuals
    return y-x@w

def OrthoCheck(x,res):
    #should be 0 or something equivalent
    return x.T@res

def predict(x,w):
    #to test out the data 
    return x@w

def main():
    X_train = np.array([1,1.5,2,3,3.8,4,5])
    Y_train = np.array([3,2,5,7,8.9,9,11])
    X_test = np.array([6,7,8])
    X_test = Design_mat(X_test)
    X = Design_mat(X_train)
    w = Normal_Eqn(X,Y_train)
    Err = residual(X,Y_train,w)
    print("Norm for the residual: ", np.linalg.norm(Err))
    print("On Orthogonality Check:-", OrthoCheck(X,Err))
    print("On trying out the equation with ", w, " as bias and weight and data as \n",X_test,"\n: ",predict(X_test,w))

if __name__ == "__main__":
    main()

