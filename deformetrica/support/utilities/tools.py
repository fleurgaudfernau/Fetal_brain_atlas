import torch
import numpy as np
import nibabel as nib
from ...core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from ...core import default, GpuMode
from ...support import utilities

gpu_mode = GpuMode.KERNEL

def gaussian_kernel(x, y, sigma = 1):
    return np.exp(-0.5 * ((x-y)/sigma)**2) / sigma * np.sqrt(2 * np.pi)

def residuals_change(liste):
    return round((liste[-2] - liste[-1]) / liste[-2], 2)

def total_residuals_change(liste):
    try:
        return round((liste[0] - liste[-1]) / liste[0], 2)
    except:
        return 0

def ratio(current_value, initial_value):
    if initial_value != 0:
        return round(100 * (initial_value - current_value) / initial_value, 2)
    else:
        return 0

def change(liste, n_subjects = 1):
    """
    SSD between two consecutive iterations
    """
    if len(liste) == 1: return 0

    a = list(liste[-2])[0]
    b = list(liste[-1])[0]
    
    return np.sum((a -b)**2) / n_subjects

def scalar_product_(kernel, cp, mom1, mom2):
    return torch.sum(mom1 * kernel.convolve(cp, cp, mom2))

def get_norm_squared(cp, momenta):
    return scalar_product_(cp, momenta, momenta) 

def orthogonal_projection(cp, momenta_to_project, momenta):
    sp = scalar_product_(cp, momenta_to_project, momenta) / get_norm_squared(cp, momenta)
    orthogonal_momenta = momenta_to_project - sp * momenta

    return orthogonal_momenta

def compute_RKHS_matrix(global_cp_nb, dimension, kernel_width, global_initial_cp):
    K = torch.zeros((global_cp_nb * dimension, global_cp_nb * dimension))
    global_initial_cp = torch.from_numpy(global_initial_cp)
    #K = np.zeros((global_cp_nb * dimension, global_cp_nb * dimension))
    for i in range(global_cp_nb):
        for j in range(global_cp_nb):
            cp_i = global_initial_cp[i, :]
            cp_j = global_initial_cp[j, :]
            kernel_distance = torch.exp(- torch.sum((cp_j - cp_i) ** 2) / (kernel_width ** 2))
            for d in range(dimension):
                K[dimension * i + d, dimension * j + d] = kernel_distance
                K[dimension * j + d, dimension * i + d] = kernel_distance
    return K