
import numpy as np

def gaussian_kernel(x, y, sigma = 1):
    return np.exp(-0.5 * ((x-y)/sigma)**2)/ sigma * np.sqrt(2 * np.pi)
