from sklearn.linear_model import LinearRegression
import numpy as np
import os

if __name__ == "__main__":
    # Example usage
    data = np.loadtxt(os.path.join('../Data/', 'ex1data1.txt'), delimiter = ',')
    X = data[:,0].reshape(-1,1)
    y = data[:,1]

    model = LinearRegression() # Create a Linear Regression model
    model.fit(X, y) # Fit the model to the data

    print("Coefficient (Theta1):", model.coef_[0]) # Slope
    print("Intercept (Theta0):", model.intercept_) # Intercept