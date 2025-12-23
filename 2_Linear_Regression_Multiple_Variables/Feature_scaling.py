import numpy as np

def feature_scaling(x):
    mu = np.mean(x, axis=0)
    std_dev = np.std(x, axis=0)
    x_scaled = (x-mu)/ std_dev
    
    return x_scaled, mu, std_dev