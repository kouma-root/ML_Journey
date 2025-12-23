import numpy as np
import os
from matplotlib import pyplot as plt
from Feature_scaling import feature_scaling
from Normal_equation import normal_equation
from Polynomial_feature import polynomial_fearures

def hypothesis(x,theta):
    return x.dot(theta)

def CostFunction(x,y,theta):
    m= len(y)
    h = hypothesis(x,theta)
    error = h - y
    J = error.T @ error /(2*m)
    return J

def GradientDescent(x, y, theta, alpha, iterations):
    m = len(y)
    cost_history =[]
    theta_history = []
    for i in range(iterations):
        h = hypothesis(x, theta)
        error = h -y
        grad = (1/m) * (x.T @ error)
        theta -= alpha * grad
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

    Cost = CostFunction(X_intercept,y,theta)

    print(Cost)
    
    alpha = 0.01
    iterations = 1000
    
    gradient, cost_his, theta_his = GradientDescent(X_intercept, y, theta, alpha, iterations)
    print(gradient)
    #print(cost_his)
    
    plt.plot(range(iterations), cost_his)
    plt.ylabel('theta')
    plt.xlabel('Cost History')
   # plt.show()
    
    for j in range(theta_his.shape[1]):
        plt.plot(theta_his[:, j], label=f'theta_{j}')

    plt.xlabel('Iterations')
    plt.ylabel('Theta Value')
    plt.title('Theta Convergence During Gradient Descent')
    plt.legend()
   # plt.show()
    
    #Normal_theta = normal_equation(X_scal, y)
    #print('Normal equation theta: ', Normal_theta)