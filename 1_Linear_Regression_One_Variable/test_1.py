from Scratch_example_1 import *

X = np.array([2,4,6,8])
y = np.array([5,9,13,17])

Theta = np.array([0.0, 0.0])  # Initial parameters
Cost = Cost_function(X, y, Theta)
print("Cost:", Cost)
#Defining the Learning rate and the number of iterations
alpha = 0.02
number_of_iterations = 10000
Theta = Gradient_Descent(X, y, Theta, alpha, number_of_iterations)
print("Theta after Gradient Descent:", Theta)