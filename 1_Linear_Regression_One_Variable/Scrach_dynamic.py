import numpy as np
import os


def hypothesis(x, Theta):
    h =0.0
    for i in range(len(Theta)):
        h+= Theta[i]*x**i
    return h


def Cost_function (x, y, theta):
    
    m = y.size
    left_sum = 0.0
    
    for i in range(m):
        h = hypothesis(x[i], theta)
        cost = (h - y[i])**2
        left_sum = left_sum + cost
    J = left_sum / (2 * m)
    return J

def Gradient_Descent(x, y, theta, alpha, number_of_iterations):
    m = y.size
    for i in range(number_of_iterations):
        
        grad = np.zeros(len(theta))
        
        for j in range(m):
            
            h = hypothesis(x[j], theta)
            error = h - y[j]
            
            for k in range(len(theta)):
                grad[k] = grad[k] + error * (x[j][k])
        
        for j in range(len(theta)):
            theta[j] = theta[j] - (alpha / m) * grad[j]
    return theta

if __name__ == "__main__":
    
    Theta = np.array([0.0, 0.0])  # Initial parameters
    data = np.loadtxt(os.path.join('../Data/', 'ex1data1.txt'), delimiter = ',')
    X = data[:,0]
    y = data[:,1]
    m,n = data.shape
    X_intercept = np.ones((m,n))
    X_intercept[:,1] = X
    
    Cost = Cost_function(X_intercept, y, Theta)
    print("Cost:", Cost)
    
    alpha = 0.01
    number_of_iterations = 10000
    
    Theta = Gradient_Descent(X_intercept, y, Theta, alpha, number_of_iterations)
    print("Theta after Gradient Descent:", Theta)