import os.path as op
import numpy as np
import pyvista as pv
import torch
from ....support.utilities import detach
from ....support.utilities.plot_tools import plot_vtk_png


import logging
logger = logging.getLogger(__name__)


class Landmark:
    """
    Landmarks (i.e. labelled point sets).
    The Landmark class represents a set of labelled points. This class assumes that the source and the target
    have the same number of points with a point-to-point correspondence.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    # Constructor.
    def __init__(self, points, object_filename = None):
        self.dimension = points.shape[1]
        assert self.dimension in [2, 3], 'Ambient-space dimension must be either 2 or 3.'

        self.type = 'Landmark'
        self.is_modified = False
        self.norm = None

        self.points = points
        self.connectivity = None
        self.object_filename = object_filename
        
        if self.object_filename is not None:
            for extension in ['.pny', '.vtk', '.stl']:
                if self.object_filename.endswith(extension):
                    self.extension = extension

        self.update_bounding_box()

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def n_points(self):
        return len(self.points)

    def set_points(self, points):
        """
        Sets the list of points of the poly data, to save at the end.
        """
        self.is_modified = True
        self.points = points

    def set_connectivity(self, connectivity):
        self.connectivity = connectivity
        self.is_modified = True

    # Gets the geometrical data that defines the landmark object, as a matrix list.
    def get_points(self):
        return self.points

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    # Compute a tight bounding box that contains all the landmark data.
    def update_bounding_box(self):
        self.bounding_box = np.zeros((self.dimension, 2))
        for d in range(self.dimension):
            self.bounding_box[d, 0] = np.min(self.points[:, d])
            self.bounding_box[d, 1] = np.max(self.points[:, d])

    def write(self, output_dir, name, points=None, momenta = None, cp = None, kernel = None):
        
        name = name + self.extension

        # Write the VTK polydata
        connec_names = {2: 'LINES', 3: 'POLYGONS'}
        points = detach(self.points) if points is None else detach(points)
        
        with open(op.join(output_dir, name), 'w', encoding='utf-8') as f:
            s = '# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS {} float\n'.format(len(self.points))
            f.write(s)

            # Formats points as strings, ensuring each point has 3 coordinates
            for p in points:
                f.write(' '.join(map(str, list(p) + ([0.0] if len(p) == 2 else []))) + '\n')
            
            if self.connectivity is not None:
                connec = detach(self.connectivity)
                a, degree = connec.shape
                s = connec_names[degree] + ' {} {}\n'.format(a, a * (degree + 1))
                f.write(s)
                for face in connec:
                    f.write(f"{degree} {' '.join(map(str, face))}\n")

        # Save as png image 
        plot_vtk_png(op.join(output_dir, name))

        # Write the VTK polydata with norm of the vector fields convolved at polydata points
        if momenta is not None:
            points = torch.tensor(points, dtype=torch.float32, device='cuda:0')
            if not isinstance(cp, torch.Tensor):
                cp = torch.tensor(cp, dtype=torch.float32, device='cuda:0')
                momenta = torch.tensor(momenta, dtype=torch.float32, device='cuda:0')
            momenta_to_points = kernel.convolve(points, cp, momenta) 

            polydata.point_data["momenta_to_mesh"] = detach(momenta_to_points)
            polydata.save(op.join(output_dir, name))

