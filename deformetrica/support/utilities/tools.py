import torch
import numpy as np
from ...core.model_tools.deformations.exponential import Exponential
from ...support import kernels as kernel_factory
from ...core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from ...core import default, GpuMode
from ...support import utilities

gpu_mode = GpuMode.KERNEL

def residuals_change(liste):
    return (liste[-2] - liste[-1])/liste[-2]

def ratio(current_value, initial_value):
        if initial_value !=0:
            return 100 * (1 - current_value/initial_value)
        else:
             return 0

def change(list, n_subjects = 1):
    """
    SSD between two consecutive iterations
    """
    if len(list) == 1: return 0
    
    return np.sum((list[-2]-list[-1])**2) / n_subjects

def norm_squared(cp, momenta, deformation_kernel_width):
    device, _ = utilities.get_best_device(gpu_mode)
    cp = utilities.move_data(cp, dtype=default.tensor_scalar_type, device=device)
    momenta = utilities.move_data(momenta, dtype=default.tensor_scalar_type, device=device)
    
    deformation_kernel = kernel_factory.factory(default.deformation_kernel_type, gpu_mode=gpu_mode, 
                                                kernel_width=deformation_kernel_width)
    
    exponential = Exponential(dense_mode=default.dense_mode,
                            kernel=deformation_kernel, shoot_kernel_type=default.shoot_kernel_type,
                            number_of_time_points=default.number_of_time_points,
                            use_rk2_for_shoot=default.use_rk2_for_shoot, 
                            use_rk2_for_flow=default.use_rk2_for_flow,
                            transport_cp = False) 
    
    norm_squared = exponential.scalar_product(cp, momenta, momenta)

    return norm_squared.cpu().numpy()

def norm(cp, momenta, deformation_kernel_width):
    
    squared = norm_squared(cp, momenta, deformation_kernel_width)

    return np.sqrt(squared)

def scalar_product(cp, momenta, momenta2, deformation_kernel_width):
    device, _ = utilities.get_best_device(gpu_mode)
    cp = utilities.move_data(cp, dtype=default.tensor_scalar_type, device=device)
    momenta = utilities.move_data(momenta, dtype=default.tensor_scalar_type, device=device)
    momenta_ = utilities.move_data(momenta2, dtype=default.tensor_scalar_type, device=device)
    
    deformation_kernel = kernel_factory.factory(default.deformation_kernel_type, gpu_mode=gpu_mode, 
                                                kernel_width=deformation_kernel_width)
    
    exponential = Exponential(dense_mode=default.dense_mode,
                            kernel=deformation_kernel, shoot_kernel_type=default.shoot_kernel_type,
                            number_of_time_points=default.number_of_time_points,
                            use_rk2_for_shoot=default.use_rk2_for_shoot, 
                            use_rk2_for_flow=default.use_rk2_for_flow,
                            transport_cp = False) 
    
    sp = exponential.scalar_product(cp, momenta, momenta_)

    return sp.cpu().numpy()

def current_distance(cp, momenta, momenta2, deformation_kernel_width):

    norm = norm_squared(cp, momenta, deformation_kernel_width)
    norm_ = norm_squared(cp, momenta2, deformation_kernel_width)
    sp = scalar_product(cp, momenta, momenta2, deformation_kernel_width)

    return norm + norm_ - 2 * sp