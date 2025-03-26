import torch

def scalar_product(kernel, cp, mom1, mom2):
    return torch.sum(mom1 * kernel.convolve(cp, cp, mom2))

def get_norm_squared(cp, momenta):
    return scalar_product(cp, momenta, momenta) 

def orthogonal_projection(cp, momenta_to_project, momenta):
    sp = scalar_product(cp, momenta_to_project, momenta) / get_norm_squared(cp, momenta)
    orthogonal_momenta = momenta_to_project - sp * momenta

    return orthogonal_momenta

def compute_RKHS_matrix(global_cp_nb, dimension, kernel_width, global_initial_cp):
    K = np.zeros((global_cp_nb * dimension, global_cp_nb * dimension))
    for i in range(global_cp_nb):
        for j in range(global_cp_nb):
            cp_i = global_initial_cp[i, :]
            cp_j = global_initial_cp[j, :]
            kernel_distance = math.exp(- np.sum((cp_j - cp_i) ** 2) / (kernel_width ** 2))
            for d in range(dimension):
                K[dimension * i + d, dimension * j + d] = kernel_distance
                K[dimension * j + d, dimension * i + d] = kernel_distance
    return K

