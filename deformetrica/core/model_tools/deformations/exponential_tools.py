import torch
from numpy import sqrt
from ....core import default
from ....support.utilities import move_data, get_best_device, detach
from ....core.model_tools.deformations.exponential import Exponential
from ....support import kernels as kernel_factory

def scalar_product(cp, momenta, momenta2, deformation_kernel_width):
    device, _ = get_best_device()
    cp = move_data(cp, device=device)
    momenta = move_data(momenta, device=device)
    momenta_ = move_data(momenta2, device=device)
    
    deformation_kernel = kernel_factory.factory(kernel_width=deformation_kernel_width)
    
    exponential = Exponential(kernel=deformation_kernel,
                            n_time_points=default.number_of_time_points,
                            use_rk2_for_shoot=default.use_rk2_for_shoot, 
                            use_rk2_for_flow=default.use_rk2_for_flow,
                            transport_cp = False) 
    
    sp = exponential.scalar_product(cp, momenta, momenta_)

    return sp.cpu().numpy()

def norm_squared(cp, momenta, deformation_kernel_width):
    return scalar_product(cp, momenta, momenta, deformation_kernel_width)

def norm(cp, momenta, deformation_kernel_width):
    
    squared = norm_squared(cp, momenta, deformation_kernel_width)

    return sqrt(squared)


def current_distance(cp, momenta, momenta2, deformation_kernel_width):

    norm = norm_squared(cp, momenta, deformation_kernel_width)
    norm_ = norm_squared(cp, momenta2, deformation_kernel_width)
    sp = scalar_product(cp, momenta, momenta2, deformation_kernel_width)

    return norm + norm_ - 2 * sp

