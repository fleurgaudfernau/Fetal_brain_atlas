import torch

from ..core import default
from ..core.model_tools.deformations.geodesic import Geodesic
from ..core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ..in_out.array_readers_and_writers import *
from ..in_out.dataset_functions import create_template_metadata
from ..support import kernels as kernel_factory
from ..core.models.model_functions import create_regular_grid_of_points

import logging
logger = logging.getLogger(__name__)

def vector_field_to_new_support(control_points, momenta_torch, new_template):   
    old_spacing = control_points[1][0] - control_points[0][0]     
    old_image_size = control_points[-1][0] + control_points[0][0] + 1
    new_image_size = new_template.bounding_box[0, 1] + 1

    new_spacing = old_spacing

    print("old control points", control_points.shape)

    if old_image_size != new_image_size:
        print("Transport vector field to new support!")
        print("old_image_size", old_image_size, " >> new_image_size", new_image_size)

        #(int(new_image_size / new_spacing) + 1)**2 > control_points.shape[0]
        new_spacing = int(new_image_size/ (np.sqrt(control_points.shape[0]) - 1))
        ratio = new_spacing/ old_spacing
        print("new_spacing", new_spacing)
        print("ratio", ratio)

        # ratio = int(new_image_size/old_image_size)
        # ratio = new_image_size/old_image_size
        # new_spacing = ratio * old_spacing
        # print("old_spacing", old_spacing)
        # print("new_spacing", new_spacing)
        # print(new_template.bounding_box)
        # print(new_template.dimension)


        # Create the control points for the new image
        control_points = create_regular_grid_of_points(new_template.bounding_box, new_spacing, new_template.dimension)
        
        momenta_torch = momenta_torch * ratio / 1.5
        print("new_control_points", control_points.shape)

    return control_points, momenta_torch, new_spacing

def vector_field_to_new_support_(control_points, momenta_torch, new_template):   
    old_spacing = control_points[1][0] - control_points[0][0]     
    old_image_size = control_points[-1][0] + control_points[0][0] + 1
    new_image_size = new_template.bounding_box[0, 1] + 1

    new_spacing = old_spacing

    print("old control points", control_points.shape)

    if old_image_size != new_image_size:
        print("Transport vector field to new support!")
        print("old_image_size", old_image_size, " >> new_image_size", new_image_size)

        new_spacing = int(new_image_size/ (np.sqrt(control_points.shape[0]) - 1))
        ratio = new_spacing/ old_spacing
        print("old_spacing", old_spacing, ">> new_spacing", new_spacing)
        print("ratio", ratio)

        if old_image_size < new_image_size:
            new_spacing = new_spacing / 2

        # Create the control points for the new image
        kernel = kernel_factory.factory(gpu_mode=default.gpu_mode, kernel_width = old_spacing)
        new_control_points_1 = create_regular_grid_of_points(new_template.bounding_box, new_spacing * 2, new_template.dimension)
        new_control_points = create_regular_grid_of_points(new_template.bounding_box, new_spacing, new_template.dimension)

        new_control_points = torch.from_numpy(new_control_points)
        new_control_points_1 = torch.from_numpy(new_control_points_1)
        ones = torch.ones(momenta_torch.size())
        print(ones)
        print("new_control_points", new_control_points)
        print("new_control_points_1", new_control_points_1)
        norm = kernel.convolve(new_control_points, new_control_points_1, ones)
        print("old momenta_torch", momenta_torch)
        print(norm)
        momenta_torch = kernel.convolve(new_control_points, new_control_points_1, momenta_torch) 
        print("new momenta_torch", momenta_torch)
        print("new momenta_torch", momenta_torch / norm)
        
    return new_control_points.numpy(), momenta_torch, new_spacing



def compute_shooting(template_specifications,
                     dimension=default.dimension,
                     deformation_kernel_width=default.deformation_kernel_width,
                     deformation_kernel_device=default.deformation_kernel_device,

                     initial_control_points=default.initial_control_points,
                     initial_momenta=default.initial_momenta,
                     concentration_of_time_points=default.concentration_of_time_points,
                     t0=None, tmin=default.tmin, tmax=default.tmax,
                     number_of_time_points=default.number_of_time_points,
                     gpu_mode=default.gpu_mode,
                     output_dir=default.output_dir, 
                     write_adjoint_parameters = True, 
                     **kwargs
                     ):
    """
    Create the template object
    """
    (object_list, t_name, t_name_extension,
     t_noise_variance, multi_object_attachment) = create_template_metadata(
        template_specifications, dimension, gpu_mode=gpu_mode)

    template = DeformableMultiObject(object_list)

    """
    Reading Control points and momenta
    """

    if initial_control_points is not None:
        control_points = read_2D_array(initial_control_points)
    else:
        raise RuntimeError('Please specify a path to control points to perform a shooting')

    if initial_momenta is not None:
        momenta = read_3D_array(initial_momenta)
    else:
        raise RuntimeError('Please specify a path to momenta to perform a shooting')
    
    if dimension == 2 and len(momenta.shape) == 3:
        momenta = momenta[:,:,0]
    if dimension == 2 and len(control_points.shape) == 3:
        control_points = control_points[:,:,0]

    deformation_kernel = kernel_factory.factory(gpu_mode=gpu_mode, kernel_width=deformation_kernel_width)

    momenta_torch = torch.from_numpy(momenta).type(default.tensor_scalar_type)
    template_points = {key: torch.from_numpy(value).type(default.tensor_scalar_type)
                       for key, value in template.get_points().items()}
    template_data = {key: torch.from_numpy(value).type(default.tensor_scalar_type)
                     for key, value in template.get_data().items()}
    
    control_points_torch = torch.from_numpy(control_points).type(default.tensor_scalar_type)
    geodesic = Geodesic(concentration_of_time_points=concentration_of_time_points, t0=t0,
                        kernel=deformation_kernel)

    print("Old Norm", geodesic.forward_exponential.scalar_product(control_points_torch, momenta_torch, momenta_torch))
    print("deformation_kernel_width", deformation_kernel_width)

    control_points, momenta_torch, deformation_kernel_width = vector_field_to_new_support(control_points, momenta_torch, template)
    write_2D_array(control_points, output_dir, "NewControlPoints.txt")
    
    deformation_kernel = kernel_factory.factory(gpu_mode=gpu_mode, kernel_width=deformation_kernel_width)

    control_points_torch = torch.from_numpy(control_points).type(default.tensor_scalar_type)
    geodesic.set_kernel(deformation_kernel)

    if t0 is None:
        logger.warning('Defaulting geodesic t0 to 0.')
        geodesic.t0 = 0.
    else:
        geodesic.t0 = t0

    if tmax == -float('inf'):
        logger.warning('Defaulting geodesic tmax to 1.')
        geodesic.tmax = 1.
    else:
        geodesic.tmax = tmax

    if tmin == float('inf'):
        logger.warning('Defaulting geodesic tmin to 0.')
        geodesic.tmin = 0.
    else:
        geodesic.tmin = tmin

    print("tmax: {}, tmin: {}, t0:{}".format(geodesic.tmax, geodesic.tmin, geodesic.t0))

    assert geodesic.tmax >= geodesic.t0, 'The max time {} for the shooting should be larger than t0 {}' \
        .format(geodesic.tmax, geodesic.t0)
    assert geodesic.tmin <= geodesic.t0, 'The min time for the shooting should be lower than t0.' \
        .format(geodesic.tmin, geodesic.t0)

    geodesic.set_control_points_t0(control_points_torch)
    geodesic.set_kernel(deformation_kernel)
    geodesic.set_template_points_t0(template_points)

    # Single momenta: single shooting
    if len(momenta.shape) == 2:
        geodesic.set_momenta_t0(momenta_torch)
        geodesic.update()

        print("geodesic.backward_exponential.number_of_time_points", geodesic.backward_exponential.number_of_time_points)
        print("geodesic.forward_exponential.number_of_time_points", geodesic.forward_exponential.number_of_time_points)
        names = [elt for elt in t_name]
        geodesic.write('Shooting', names, t_name_extension, template, template_data, output_dir,
                       write_adjoint_parameters=write_adjoint_parameters)

    # Several shootings to compute
    else:
        for i in range(len(momenta_torch)):
            geodesic.set_momenta_t0(momenta_torch[i])
            geodesic.update()
            names = [elt for elt in t_name]
            geodesic.write('Shooting' + "_" + str(i), names, t_name_extension, template, template_data, output_dir,
                           write_adjoint_parameters=write_adjoint_parameters)
