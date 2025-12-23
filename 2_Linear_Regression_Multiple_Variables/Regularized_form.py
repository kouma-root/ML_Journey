import numpy as np
import os
from Polynomial_feature import polynomial_fearures
from Feature_scaling import feature_scaling



def hypothesis(x,theta):
    return x.dot(theta)


def GradientDescent(x, y, theta, alpha, lambda_, iterations):
    m = len(y)
    cost_history =[]
    theta_history = []
    for i in range(iterations):
        h = hypothesis(x, theta)
        error = h -y
        grad = (1/m) * (x.T @ error)
        theta -= alpha * grad
        
        theta_reg = theta.copy()
        theta_reg[0] = 0   # do not regularize bias
        grad += (lambda_/m) * theta_reg
        cost = (1/(2*m)) * np.sum(error **2)
        cost_history.append(cost)
        theta_history.append(theta.copy())   # VERY IMPORTANT: use .copy()
    return theta, cost_history, np.array(theta_history)



if __name__ == "__main__":
    # Example usage
    data = np.loadtxt(os.path.join('../Data/', 'ex1data2.txt'), delimiter = ',')
    X = data[:,:-1]
    y = data[:,-1]
    m,n = data.shape
    x_poly = polynomial_fearures(X, 2)
    X_scal, _, _= feature_scaling(x_poly) 
    X_intercept = np.ones((m,n+2))
    X_intercept[:,1:] = X_scal
    theta = np.zeros(X_intercept.shape[1])
    lambda_ = 0.1

    alpha = 0.01
    iterations = 1000
    
    gradient, cost_his, theta_his = GradientDescent(X_intercept, y, theta, alpha, lambda_, iterations)
    print(gradient)
   # print(cost_his)
   # print(theta_his)