import numpy as np


def normal_equation(x,y):
    theta = np.linalg.inv(x.T @ x) @ x.T @ y
    return theta


# sometime matrix is not inversible so we will use:
#theta = np.linalg.pinv(x) @ y


if __name__ == "__main__":

    X = np.array([[1], [2], [3]], float)
    y = np.array([[2], [4], [6]], float)
    print(normal_equation(X, y))