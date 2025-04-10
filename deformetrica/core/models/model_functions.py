import math
import torch

from ...in_out.array_readers_and_writers import *

import logging
logger = logging.getLogger(__name__)

def gaussian_kernel(x, y, sigma = 1):
    return np.exp(-0.5 * ((x-y)/sigma)**2)/ sigma * np.sqrt(2 * np.pi)

def residuals_change(avg_residuals):
    return (avg_residuals[-2] - avg_residuals[-1])/avg_residuals[-2]


def initialize_cp(initial_cp, template, deformation_kernel_width, bounding_box = None):
    
    if initial_cp is not None:
        control_points = read_2D_array(initial_cp)
        logger.info('\n>> Reading %d initial control points from file %s.' % (len(control_points), initial_cp))

    else:
        if bounding_box is not None:
            logger.info("Creating larger bounding box to fit all subjects")
        bounding_box = bounding_box if bounding_box is not None else template.bounding_box
        control_points = create_regular_grid_of_points(bounding_box, deformation_kernel_width, template.dimension)
        logger.info('\n>> Set of %d control points defined.' % len(control_points))

    return control_points


def initialize_momenta(initial_momenta, number_of_control_points, dimension, 
                       n_subjects=0, random=False):
    if initial_momenta is not None:
        momenta = read_3D_array(initial_momenta)
        logger.info('>> Reading initial momenta from file: %s.' % initial_momenta)

    else:
        if n_subjects == 0:
            if random:
                momenta = np.random.randn(number_of_control_points, dimension) / math.sqrt(number_of_control_points * dimension)
                logger.info('>> Momenta randomly initialized.')
            else:
                momenta = np.zeros((number_of_control_points, dimension))
                logger.info('>> Momenta initialized to zero.')
        else:
            momenta = np.zeros((n_subjects, number_of_control_points, dimension))
            logger.info('>> Momenta initialized to zero for %d subjects (or components).' % n_subjects)

    return momenta

def initialize_covariance_momenta_inverse(control_points, kernel, dimension):
    return np.kron(kernel.get_kernel_matrix(torch.from_numpy(control_points)).detach().numpy(), np.eye(dimension))

def initialize_rupture_time(tR, t0, t1, nb_components):
    if not tR: # set tR at regular intervals
        tR = []
        segment = int((math.ceil(t1) - math.trunc(t0))/nb_components)
        for i in range(nb_components - 1):
            tR.append(math.trunc(t0) + segment * (i+1), i)
    
    return tR 

def initialize_modulation_matrix(initial_modulation_matrix, number_of_control_points, number_of_sources, dimension):
    if initial_modulation_matrix is not None:
        modulation_matrix = read_3D_array(initial_modulation_matrix)
        if len(modulation_matrix.shape) == 1:
            modulation_matrix = modulation_matrix.reshape(-1, 1)
        logger.info('>> Reading ' + str(
            modulation_matrix.shape[1]) + '-source initial modulation matrix from file: ' + initial_modulation_matrix)

    else:
        if number_of_sources is None:
            raise RuntimeError('The number of sources must be set')
        modulation_matrix = np.zeros((number_of_control_points * dimension, number_of_sources))

    return modulation_matrix


def initialize_sources(initial_sources, n_subjects, number_of_sources):
    if initial_sources is not None:
        sources = read_3D_array(initial_sources).reshape((-1, number_of_sources))
        logger.info('>> Reading initial sources from file: ' + initial_sources)
    else:
        sources = np.zeros((n_subjects, number_of_sources))
        logger.info('>> Initializing all sources to zero')
    return sources

def create_regular_grid_of_points(box, spacing, dimension):
    """
    Creates a regular grid of 2D or 3D points, as a numpy array of size nb_of_points x dimension.
    box: (dimension, 2)
    """
    axis = []
    for d in range(dimension):
        min = box[d, 0]
        max = box[d, 1]
        length = max - min
        assert (length >= 0.0)

        offset = 0.5 * (length - spacing * math.floor(length / spacing))
        axis.append(np.arange(min + offset, max + 1e-10, spacing))

    if dimension == 1:
        control_points = np.zeros((len(axis[0]), dimension))
        control_points[:, 0] = axis[0].flatten()

    elif dimension == 2:
        x_axis, y_axis = np.meshgrid(axis[0], axis[1])

        assert (x_axis.shape == y_axis.shape)
        number_of_control_points = x_axis.flatten().shape[0]
        control_points = np.zeros((number_of_control_points, dimension))

        control_points[:, 0] = x_axis.flatten()
        control_points[:, 1] = y_axis.flatten()

    elif dimension == 3:
        x_axis, y_axis, z_axis = np.meshgrid(axis[0], axis[1], axis[2])

        assert (x_axis.shape == y_axis.shape)
        assert (x_axis.shape == z_axis.shape)
        number_of_control_points = x_axis.flatten().shape[0]
        control_points = np.zeros((number_of_control_points, dimension))

        control_points[:, 0] = x_axis.flatten()
        control_points[:, 1] = y_axis.flatten()
        control_points[:, 2] = z_axis.flatten()

    elif dimension == 4:
        x_axis, y_axis, z_axis, t_axis = np.meshgrid(axis[0], axis[1], axis[2], axis[3])

        assert (x_axis.shape == y_axis.shape)
        assert (x_axis.shape == z_axis.shape)
        number_of_control_points = x_axis.flatten().shape[0]
        control_points = np.zeros((number_of_control_points, dimension))

        control_points[:, 0] = x_axis.flatten()
        control_points[:, 1] = y_axis.flatten()
        control_points[:, 2] = z_axis.flatten()
        control_points[:, 3] = t_axis.flatten()

    else:
        raise RuntimeError('Invalid ambient space dimension.')

    return control_points


def remove_useless_control_points(control_voxels, image, kernel_width):

    intensities = image.get_intensities()
    image_shape = intensities.shape

    threshold = 1e-5
    region_size = 2 * kernel_width

    final_control_points = []
    for control_point, control_voxel in zip(control_points, control_voxels):

        axes = []
        for d in range(image.dimension):
            axe = np.arange(max(int(control_voxel[d] - region_size), 0),
                            min(int(control_voxel[d] + region_size), image_shape[d] - 1))
            axes.append(axe)

        neighbouring_voxels = np.array(np.meshgrid(*axes))
        for d in range(image.dimension):
            neighbouring_voxels = np.swapaxes(neighbouring_voxels, d, d + 1)
        neighbouring_voxels = neighbouring_voxels.reshape(-1, image.dimension)

        if (image.dimension == 2 and np.any(intensities[neighbouring_voxels[:, 0],
                                                        neighbouring_voxels[:, 1]] > threshold)) \
                or (image.dimension == 3 and np.any(intensities[neighbouring_voxels[:, 0],
                                                                neighbouring_voxels[:, 1],
                                                                neighbouring_voxels[:, 2]] > threshold)):
            final_control_points.append(control_point)

    return np.array(final_control_points)
