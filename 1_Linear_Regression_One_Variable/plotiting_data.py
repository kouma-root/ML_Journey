
import numpy as np
import os
import matplotlib.pyplot as plt


data = np.loadtxt(os.path.join('../Data/', 'ex1data1.txt'), delimiter = ',')
X = data[:,0]
y = data[:,1]

plt.scatter(X, y, color='red', marker='x')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Scatter plot of training data')
plt.grid(True)
plt.show()