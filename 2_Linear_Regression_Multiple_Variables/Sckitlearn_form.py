from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures

if __name__ == "__main__":
    # Example usage
    data = np.loadtxt(os.path.join('../Data/', 'ex1data2.txt'), delimiter = ',')
    X = data[:,:-1]
    y = data[:,-1]
    
    #poly = PolynomialFeatures(degree=3)
   # X_poly = poly.fit_transform(X)

    #model = LinearRegression() # Create a Linear Regression model
    #model = Ridge(alpha=1.0)
    model = Lasso(alpha=1.0)
    model.fit(X, y) # Fit the model to the data

    print("Coefficient (Theta1):", model.coef_) # Slope
    print("Intercept (Theta0):", model.intercept_) # Intercept 