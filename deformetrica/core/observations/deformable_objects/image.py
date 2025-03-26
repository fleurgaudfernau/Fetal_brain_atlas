import os.path as op
import PIL.Image as pimg
import nibabel as nib
import numpy as np
import torch
from copy import deepcopy

from ....in_out.image_functions import rescale_image_intensities
from ....support.utilities import detach, move_data

from ....core import default
from cv2 import blur
from scipy.ndimage import gaussian_filter, median_filter

import logging
logger = logging.getLogger(__name__)

def image_average_filter(intensities):
        return blur(intensities, ksize=(2,2))
    
def image_median_filter(intensities, image_scale):
    #width of the filter = rayon (nb de pixels inclus dans calcul médianes)
    size = int(image_scale)+1 if int(image_scale) >= 1 else 1

    return median_filter(intensities, size = size)

def image_gaussian_filter(intensities, image_scale):
    return gaussian_filter(intensities, sigma = image_scale, mode = "constant")

class Image:
    """
    Landmarks (i.e. labelled point sets).
    The Landmark class represents a set of labelled points. This class assumes that the source and the target
    have the same number of points with a point-to-point correspondence.

    """
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, intensities, intensities_dtype, affine, interpolation=default.interpolation, 
                object_filename = None):
        self.object_filename = object_filename

        for extension in ['.nii', '.nii.gz', '.png']:
            if self.object_filename.endswith(extension):
                self.extension = extension
        
        self.interpolation = interpolation

        self.dimension = len(intensities.shape)
        assert self.dimension in [2, 3], 'Ambient-space dimension must be either 2 or 3.'

        self.type = 'Image'
        self.is_modified = False

        self.intensities = intensities
        self.original_intensities = deepcopy(intensities)
        self.intensities_dtype = intensities_dtype
        self.affine = affine

        self.downsampling_factor = 1

        self.image_filter = image_gaussian_filter

        self._update_corner_point_positions()
        self.update_bounding_box()

        self.distance = {"ssim" : 0, "mse":0}

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def get_number_of_points(self):
        raise RuntimeError("Not implemented for Image yet.")

    def set_affine(self, affine_matrix):
        """
        The affine matrix A is a 4x4 matrix that gives the correspondence between the voxel coordinates and their
        spatial positions in the 3D space: (x, y, z, 1) = A (u, v, w, 1).
        See the nibabel documentation for further details (the same attribute name is used here).
        """
        self.affine = affine_matrix

    def set_intensities(self, intensities):
        self.is_modified = True
        self.intensities = intensities

    def get_intensities(self):
        return self.intensities

    def get_points(self):

        image_shape = self.intensities.shape

        axes = []
        for d in range(self.dimension):
            axe = np.linspace(self.corner_points[0, d], self.corner_points[2 ** d, d],
                              image_shape[d] // self.downsampling_factor)
            axes.append(axe)

        points = np.array(np.meshgrid(*axes, indexing='ij')[:])
        for d in range(self.dimension):
            points = np.swapaxes(points, d, d + 1)

        return points

    def filter(self, scale):
        #intensities = self.get_intensities()
        intensities = self.image_filter(deepcopy(self.original_intensities), scale)
        self.set_intensities(intensities)

        return intensities

    def get_deformed_intensities(self, deformed_voxels, intensities):
        """
        Torch input / output.
        Interpolation function with zero-padding.
        """
        
        # for intensities: mode = "bi/trilinear"
        # for segmentations: mode = "nearest"

        options = {}        
        options["align_corners"] = True
        options["mode"] = "bilinear" if self.dimension == 2 else "trilinear"

        assert isinstance(deformed_voxels, torch.Tensor)
        assert isinstance(intensities, torch.Tensor)

        intensities = move_data(intensities, dtype=deformed_voxels.type(), device=deformed_voxels.device)
        assert deformed_voxels.device == intensities.device

        tensor_integer_type = {'cpu': 'torch.LongTensor','cuda': 'torch.cuda.LongTensor'}[deformed_voxels.device.type]

        image_shape = self.intensities.shape

        if self.dimension == 2:
            if not self.downsampling_factor == 1:
                shape = deformed_voxels.shape
                deformed_voxels = torch.nn.functional.interpolate(deformed_voxels.permute(2, 0, 1).contiguous().view(1, shape[2], shape[0], shape[1]),
                                                                  size=image_shape, mode=options["mode"], align_corners=options["align_corners"])[0].permute(1, 2, 0).contiguous()

            u, v = deformed_voxels.view(-1, 2)[:, 0], deformed_voxels.view(-1, 2)[:, 1]

            u1 = torch.floor(u.detach())
            v1 = torch.floor(v.detach())

            u1 = torch.clamp(u1, 0, image_shape[0] - 1)
            v1 = torch.clamp(v1, 0, image_shape[1] - 1)
            u2 = torch.clamp(u1 + 1, 0, image_shape[0] - 1)
            v2 = torch.clamp(v1 + 1, 0, image_shape[1] - 1)

            fu = u - u1
            fv = v - v1
            gu = (u1 + 1) - u
            gv = (v1 + 1) - v

            if self.interpolation == "nearest":
                gu = gu > 0.5
                fu = ~gu
                gv = gv > 0.5
                fv = ~gv

            deformed_intensities = (intensities[u1.type(tensor_integer_type), v1.type(tensor_integer_type)] * gu * gv +
                                    intensities[u1.type(tensor_integer_type), v2.type(tensor_integer_type)] * gu * fv +
                                    intensities[u2.type(tensor_integer_type), v1.type(tensor_integer_type)] * fu * gv +
                                    intensities[u2.type(tensor_integer_type), v2.type(tensor_integer_type)] * fu * fv).view(image_shape)

        elif self.dimension == 3:
            if not self.downsampling_factor == 1:
                shape = deformed_voxels.shape
                deformed_voxels = torch.nn.functional.interpolate(deformed_voxels.permute(3, 0, 1, 2).contiguous().view(1, shape[3], shape[0], shape[1], shape[2]),
                                                                  size=image_shape, mode=options["mode"], align_corners=options["align_corners"])[0].permute(1, 2, 3, 0).contiguous()

            u, v, w = deformed_voxels.view(-1, 3)[:, 0], \
                      deformed_voxels.view(-1, 3)[:, 1], \
                      deformed_voxels.view(-1, 3)[:, 2]

            u1 = torch.floor(u.detach()) #a new tensor from u
            v1 = torch.floor(v.detach())
            w1 = torch.floor(w.detach())

            u1 = torch.clamp(u1, 0, image_shape[0] - 1)
            v1 = torch.clamp(v1, 0, image_shape[1] - 1)
            w1 = torch.clamp(w1, 0, image_shape[2] - 1)
            u2 = torch.clamp(u1 + 1, 0, image_shape[0] - 1)
            v2 = torch.clamp(v1 + 1, 0, image_shape[1] - 1)
            w2 = torch.clamp(w1 + 1, 0, image_shape[2] - 1)

            fu = u - u1
            fv = v - v1
            fw = w - w1
            gu = (u1 + 1) - u
            gv = (v1 + 1) - v
            gw = (w1 + 1) - w

            if self.interpolation == "nearest":
                gu = gu > 0.5
                fu = ~ gu
                gv = gv > 0.5
                fv = ~gv
                gw = gw > 0.5
                fw = ~gw
            
            deformed_intensities = (intensities[u1.type(tensor_integer_type), v1.type(tensor_integer_type), w1.type(tensor_integer_type)] * gu * gv * gw +
                                    intensities[u1.type(tensor_integer_type), v1.type(tensor_integer_type), w2.type(tensor_integer_type)] * gu * gv * fw +
                                    intensities[u1.type(tensor_integer_type), v2.type(tensor_integer_type), w1.type(tensor_integer_type)] * gu * fv * gw +
                                    intensities[u1.type(tensor_integer_type), v2.type(tensor_integer_type), w2.type(tensor_integer_type)] * gu * fv * fw +
                                    intensities[u2.type(tensor_integer_type), v1.type(tensor_integer_type), w1.type(tensor_integer_type)] * fu * gv * gw +
                                    intensities[u2.type(tensor_integer_type), v1.type(tensor_integer_type), w2.type(tensor_integer_type)] * fu * gv * fw +
                                    intensities[u2.type(tensor_integer_type), v2.type(tensor_integer_type), w1.type(tensor_integer_type)] * fu * fv * gw +
                                    intensities[u2.type(tensor_integer_type), v2.type(tensor_integer_type), w2.type(tensor_integer_type)] * fu * fv * fw).view(image_shape)

        else:
            raise RuntimeError('Incorrect dimension of the ambient space: %d' % self.dimension)
        
        return deformed_intensities

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update_bounding_box(self):
        """
        Compute a tight bounding box that contains all the 2/3D-embedded image data.
        """
        self.bounding_box = np.zeros((self.dimension, 2))
        for d in range(self.dimension):
            self.bounding_box[d, 0] = np.min(self.corner_points[:, d])
            self.bounding_box[d, 1] = np.max(self.corner_points[:, d])

    def write(self, output_dir, name, intensities = None):
        if intensities is None:
            intensities = self.get_intensities()

        intensities = detach(intensities)
        
        intensities_rescaled = rescale_image_intensities(intensities, self.intensities_dtype)

        if self.extension == ".png":
            pimg.fromarray(intensities_rescaled).save(op.join(output_dir, name) + self.extension)
        elif self.extension == ".nii":
            if "/" in name:
                name = name.split("/")[-1] #ajout fg
            img = nib.Nifti1Image(intensities_rescaled, self.affine)
            nib.save(img, op.join(output_dir, name) + self.extension)
        elif self.extension == ".npy":
            np.save(op.join(output_dir, name) + self.extension, intensities_rescaled)
        else:
            raise ValueError('Writing images with the given extension "%s" is not coded yet.' % name)

    ####################################################################################################################
    ### Utility methods:
    ####################################################################################################################

    def _update_corner_point_positions(self):
        if self.dimension == 2:
            corner_points = np.zeros((4, 2))
            umax, vmax = np.subtract(self.intensities.shape, (1, 1))
            corner_points[1] = [umax, 0]
            corner_points[2] = [0, vmax]
            corner_points[3] = [umax, vmax]

        elif self.dimension == 3:
            corner_points = np.zeros((8, 3))
            umax, vmax, wmax = np.subtract(self.intensities.shape, (1, 1, 1))
            corner_points[1] = [umax, 0, 0]
            corner_points[2] = [0, vmax, 0]
            corner_points[3] = [umax, vmax, 0]
            corner_points[4] = [0, 0, wmax]
            corner_points[5] = [umax, 0, wmax]
            corner_points[6] = [0, vmax, wmax]
            corner_points[7] = [umax, vmax, wmax]

        else:
            raise RuntimeError('Invalid dimension: %d' % self.dimension)

        self.corner_points = corner_points
