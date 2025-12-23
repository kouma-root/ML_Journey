import numpy as np
import os



def hypothesis(x, Theta):
    return Theta[0] + Theta[1]*x

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
        sum_0 = 0.0
        sum_1 = 0.0
        for j in range(m):
            h = hypothesis(x[j], theta)
            sum_0 += (h - y[j])
            sum_1 += (h - y[j]) * x[j]
        theta[0] = theta[0] - (alpha/ m)*sum_0
        theta[1] = theta[1] - (alpha/ m)*sum_1
    return theta

if __name__ == "__main__":
    # Example usage
    Theta = np.array([0.0, 0.0])  # Initial parameters
    #X = np.array([1,2,3,4])
    #y = np.array([40,50,65,75])
    data = np.loadtxt(os.path.join('../Data/', 'ex1data1.txt'), delimiter = ',')
    X = data[:,0]
    y = data[:,1]
    
    Cost = Cost_function(X, y, Theta)
    print("Cost:", Cost)
    
    #Defining the Learning rate and the number of iterations
    alpha = 0.01
    number_of_iterations = 10000
    
    Theta = Gradient_Descent(X, y, Theta, alpha, number_of_iterations)
    print("Theta after Gradient Descent:", Theta)
    
    predict = hypothesis(X,Theta)
    print("Predictions:", predict)