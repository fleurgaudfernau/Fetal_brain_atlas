import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np


def normalize_image_intensities(intensities):
    dtype = str(intensities.dtype)
    if dtype == 'uint8':
        return np.array(intensities / 255.0, dtype="float32"), dtype
    else:
        return np.array(intensities, dtype="float32"), dtype


def rescale_image_intensities(intensities, dtype):
    tol = 1e-10
    if dtype == 'uint8':
        return (np.clip(intensities, tol, 1 - tol) * 255).astype('uint8')
    else:
        return intensities.astype('float32')
