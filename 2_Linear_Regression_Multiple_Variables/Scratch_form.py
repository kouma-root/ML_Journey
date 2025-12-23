import numpy as np 
import os
from Feature_scaling import feature_scaling

def hypothesis(x,theta):
    h=0.0
    for j in range(len(theta)):
        h+= theta[j] * x[j]
    return h


def CostFunction(x, y, theta):
    m= len(y)
    sum = 0.0
    
    for i in range(m):
        h = hypothesis(x[i], theta)
        error = h - y[i]
        sum += error**2
         
    J = sum/(2*m)
    return J

def GradientDescent(x, y, theta, alpha, iterations):
    m = len(y)
    n= len(theta)
    
    for _ in range(iterations):
        grad = [0] * n #Size of the gradient vector dynamicaly determined
        
        for i in range(m):
            h = hypothesis(x[i], theta)
            error = h - y[i]
            
            #Feature loop
            for j in range(n):
                grad[j] += error * x[i][j]
                
        #update theta
        for k in range(n):
            theta[k] -= (alpha/m) * grad[k]
    
    return theta


if __name__ == "__main__":
    # Example usage
    data = np.loadtxt(os.path.join('../Data/', 'ex1data2.txt'), delimiter = ',')
    X = data[:,:-1]
    y = data[:,-1]
    m,n = data.shape
    X_scal, _, _= feature_scaling(X) 
    X_intercept = np.ones((m,n))
    X_intercept[:,1:] = X_scal
    theta = np.zeros(X_intercept.shape[1])
    

    Cost = CostFunction(X_intercept,y,theta)

    print("Cost value:",Cost)
    
    
    alpha = 0.00000001
    iterations = 1000
    Gradient = GradientDescent(X_intercept, y, theta, alpha, iterations)
    print("Theta Value: ", Gradient)