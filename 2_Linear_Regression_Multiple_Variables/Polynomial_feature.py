import numpy as np
def polynomial_fearures(x, degree):
    x_poly = x.copy() # Working with the copy of the data to keep the original one
    for d in range(2, degree+1):
        x_poly = np.hstack([x_poly, x**d])
        
    return x_poly