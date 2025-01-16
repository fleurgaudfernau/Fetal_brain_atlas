import torch 
import os
import os.path as op
import numpy as np
from sklearn.decomposition import FastICA
from ..core import default, GpuMode
from ..in_out.array_readers_and_writers import read_3D_array, write_3D_array, concatenate_for_paraview
from ..support import kernels as kernel_factory
from ..support import utilities
from .compute_parallel_transport import PiecewiseParallelTransport
from itertools import combinations
from ..support.utilities.plot_tools import plot_sources


import logging
logger = logging.getLogger(__name__)
gpu_mode = GpuMode.KERNEL


def perform_ICA(output_dir, cp = None, global_kernel_width = None, momenta = None, 
                subjects_momenta = None, number_of_sources = None, target_time = None, 
                tR = None, overwrite = True):
    
    path_to_sources = os.path.join(output_dir, "Estimated_Sources.txt")
    path_to_mm = os.path.join(output_dir, "Estimated_Modulation_matrix.txt")

    if not overwrite and op.exists(path_to_mm):
        print("\n ICA Already performed - Stopping")
        return path_to_sources, path_to_mm

    cp = read_3D_array(cp)
    momenta = read_3D_array(momenta)

    tensor_scalar_type = utilities.get_torch_scalar_type('float32')
    device, _ = utilities.get_best_device(gpu_mode=gpu_mode)

    cp = utilities.move_data(cp, dtype=tensor_scalar_type, requires_grad=False, device=device)
    momenta = utilities.move_data(momenta, dtype=tensor_scalar_type, requires_grad=False, device=device)
    
    # TODO for adapting to different t0
    # Get component at t0
    for l, t in enumerate(tR):
        print(t)
        print(target_time)
        if np.abs(int(target_time) - int(t)) <= 1:
            momenta_t0 = momenta[l+1]
            print("component", l+1)
            break
    
    # Orthogonalisation
    kernel = kernel_factory.factory('keops', gpu_mode=gpu_mode, kernel_width=global_kernel_width)
    norm_squared = torch.sum(momenta_t0 * kernel.convolve(cp, cp, momenta_t0))
    w = []
    for i in range(len(subjects_momenta)):
        subject_momenta = read_3D_array(subjects_momenta[i])
        subject_momenta = utilities.move_data(subject_momenta, dtype=tensor_scalar_type, 
                                              requires_grad=False, device=device)
        scalar_product = torch.sum(subject_momenta * kernel.convolve(cp, cp, momenta_t0))
        orthogonal = subject_momenta.flatten() - scalar_product * momenta_t0.flatten() / norm_squared
        w.append(orthogonal.cpu().numpy())

    w = np.array(w)

    # ICA
    ica = FastICA(n_components = number_of_sources, max_iter=50000)
    sources = ica.fit_transform(w)
    modulation_matrix = ica.mixing_

    # Rescale.
    for s in range(number_of_sources):
        std = np.std(sources[:, s])
        sources[:, s] /= std
        modulation_matrix[:, s] *= std

    # Write    
    write_3D_array(sources, output_dir, "Estimated_Sources.txt")
    write_3D_array(modulation_matrix, output_dir, "Estimated_Modulation_matrix.txt")

    return path_to_sources, path_to_mm


def plot_ica(path_to_sources, path_to_mm, template_specifications, dataset_specifications,
            dimension=default.dimension, tensor_scalar_type=default.tensor_scalar_type,
            deformation_kernel_type=default.deformation_kernel_type,
            deformation_kernel_width=default.deformation_kernel_width,
            dense_mode=default.dense_mode, shoot_kernel_type=None, 
            initial_control_points=default.initial_control_points,
            initial_momenta_tR=default.initial_momenta, tmin=default.tmin, tmax=default.tmax,
            concentration_of_time_points=default.concentration_of_time_points,
            t0=default.t0, nb_components = 4, tR = [], number_of_time_points=default.number_of_time_points,
            use_rk2_for_shoot=default.use_rk2_for_shoot, use_rk2_for_flow=default.use_rk2_for_flow,
            gpu_mode=default.gpu_mode, output_dir=default.output_dir, 
            overwrite = True, flow_path = None, target_time = None, **kwargs):

    # Visualize
    modulation_matrix = read_3D_array(path_to_mm)
    sources = read_3D_array(path_to_sources)
    cp = read_3D_array(initial_control_points)

    ages = [a[0] for a in dataset_specifications["visit_ages"]]
    
    plot_sources(output_dir, sources, ages, overwrite)
    plot_sources(output_dir, sources, ages, overwrite, xlim = [-3, 3], ylim = [-3, 3])
    
    for s in range(sources.shape[1]):

        # Write space shift
        space_shift = modulation_matrix[:, s]
        space_shift = space_shift.reshape((int(modulation_matrix.shape[0] / dimension), dimension))
        
        write_3D_array(space_shift, output_dir, "Space_shift_{}.txt".format(s))
        concatenate_for_paraview(space_shift, cp, output_dir, "For_paraview_Space_shift_{}.vtk".format(s))

        space_shift_folder = op.join(output_dir, "Space_shift_{}".format(s))

        if not op.exists(space_shift_folder): 
            os.mkdir(space_shift_folder)

        print(">>> Parallel transport along main geodesic of space shift {}".format(s))
        
        pt = PiecewiseParallelTransport(template_specifications, dimension, tensor_scalar_type, 
                                        tmin, tmax, concentration_of_time_points, t0, 
                                        start_time = target_time, target_time = None, tR = tR, 
                                        gpu_mode = gpu_mode, output_dir = space_shift_folder, 
                                        flow_path = flow_path, initial_control_points = initial_control_points)

        #Initialize for later shooting
        initial_momenta_to_transport = op.join(output_dir, "Space_shift_{}.txt".format(s))

        pt.initialize_(deformation_kernel_type, deformation_kernel_width, dense_mode, 
                        shoot_kernel_type, initial_control_points, initial_momenta_tR,
                        initial_momenta_to_transport, number_of_time_points,
                        use_rk2_for_shoot, use_rk2_for_flow, nb_components)   
        pt.set_geodesic()     
        
        if not overwrite and pt.check_pt_for_ica():
            logger.info("\n Parallel Transport already performed")
        else:
            pt.transport(is_orthogonal=True) #already ortho in ICA
            pt.write_for_ica()
        
        # Shoot space shift
        for sigma in [-3, -1, 1, 3]:
            print("\n >>> Shooting space shift {} - with sigma {}".format(s, sigma))
            pt.shoot_for_ica(sigma)
            #pt.screenshot_ica()
    
