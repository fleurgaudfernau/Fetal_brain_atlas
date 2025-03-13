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

    device, _ = utilities.get_best_device(gpu_mode=gpu_mode)
    cp = utilities.move_data(cp, requires_grad=False, device=device)
    momenta = utilities.move_data(momenta, requires_grad=False, device=device)
    
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
        subject_momenta = utilities.move_data(subject_momenta, device=device)
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
            deformation_kernel_width=default.deformation_kernel_width,
            initial_cp=default.initial_cp,initial_momenta_tR=default.initial_momenta, 
            tmin=default.tmin, tmax=default.tmax, time_concentration=default.time_concentration,
            t0=default.t0, nb_components = 4, tR = [], n_time_points=default.n_time_points,
            gpu_mode=default.gpu_mode, output_dir=default.output_dir, 
            overwrite = True, flow_path = None, target_time = None, **kwargs):

    # Visualize
    modulation_matrix = read_3D_array(path_to_mm)
    sources = read_3D_array(path_to_sources)
    cp = read_3D_array(initial_cp)

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
        
        pt = PiecewiseParallelTransport(template_specifications, tmin, tmax, time_concentration, t0, 
                                        start_time = target_time, target_time = None, tR = tR, 
                                        gpu_mode = gpu_mode, output_dir = space_shift_folder, 
                                        flow_path = flow_path, initial_cp = initial_cp)

        #Initialize for later shooting
        initial_momenta_to_transport = op.join(output_dir, "Space_shift_{}.txt".format(s))

        pt.initialize_(deformation_kernel_width, initial_cp, initial_momenta_tR,
                        initial_momenta_to_transport, n_time_points, nb_components)   
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
    
