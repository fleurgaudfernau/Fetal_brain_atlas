import numpy as np
import torch
from torch.autograd import Variable

from core.observations.deformable_objects.landmarks.landmark import Landmark


class SurfaceMesh(Landmark):
    """
    3D Triangular mesh.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, dimension, tensor_scalar_type, tensor_integer_type):
        Landmark.__init__(self, dimension, tensor_scalar_type, tensor_integer_type)
        self.type = 'SurfaceMesh'

        # All of these are torch tensor attributes.
        self.connectivity = None
        self.centers = None
        self.normals = None

    # Clone.
    def clone(self):
        clone = SurfaceMesh(self.dimension, (self.tensor_scalar_type, self.tensor_integer_type))
        clone.points = np.copy(self.points)
        clone.is_modified = self.is_modified
        clone.bounding_box = self.bounding_box
        clone.norm = self.norm
        clone.connectivity = self.connectivity.clone()
        clone.centers = self.centers.clone()
        clone.normals = self.normals.clone()
        return clone

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        self.get_centers_and_normals()
        Landmark.update(self)

    def get_centers_and_normals(self, points=None):
        """
        Given a new set of points, use the corresponding connectivity available in the polydata
        to compute the new normals, all in torch
        """
        if points is None:
            if self.is_modified:
                torch_points_coordinates = Variable(torch.from_numpy(self.points).type(self.tensor_scalar_type))
                a = torch_points_coordinates[self.connectivity[:, 0]]
                b = torch_points_coordinates[self.connectivity[:, 1]]
                c = torch_points_coordinates[self.connectivity[:, 2]]
                centers = (a+b+c)/3.
                self.centers = centers
                self.normals = torch.cross(b-a, c-a)/2
        else:
            a = points[self.connectivity[:, 0]]
            b = points[self.connectivity[:, 1]]
            c = points[self.connectivity[:, 2]]
            centers = (a+b+c)/3.
            self.centers = centers
            self.normals = torch.cross(b-a, c-a)/2
        return self.centers, self.normals
