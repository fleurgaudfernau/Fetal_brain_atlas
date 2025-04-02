import os.path as op
import numpy as np
import pyvista as pv
import torch
import vtk
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from mpl_toolkits.mplot3d import Axes3D
from vtk.util.numpy_support import vtk_to_numpy
from ....support.utilities import detach, move_data, get_best_device

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
        """
            Write the VTK polydata
        """        
        points = detach(self.points) if points is None else detach(points)
        
        if self.connectivity is not None:
            connec = detach(self.connectivity)
            a, degree = connec.shape
            faces = np.hstack((np.full((connec.shape[0], 1), degree), connec)).flatten()
            mesh = pv.PolyData(points, faces = faces)
        else:
            mesh = pv.PolyData(points)
        
        # Norm of the vector fields convolved at polydata points
        if momenta is not None:
            points = move_data(points, device = get_best_device() )
            cp = move_data(cp, device = get_best_device() )
            momenta = move_data(momenta, device = get_best_device() )

            momenta_to_points = kernel.convolve(points, cp, momenta) 
            mesh.point_data["momenta_to_mesh"] = detach(momenta_to_points)
        
        self.last_vtk_saved = op.join(output_dir, name + self.extension)
        mesh.save(self.last_vtk_saved)

        # Taubin smoothing
        smooth = mesh.smooth_taubin(n_iter = 1000)
        name = op.join(output_dir, name + "_taubin" + self.extension)
        smooth.save(name)

    def write_png(self, output_dir, name, *args):
        """Plots a .vtk mesh using matplotlib and saves it as a PNG."""   

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.last_vtk_saved)
        reader.Update()
        polydata = reader.GetOutput()

        points = vtk_to_numpy(polydata.GetPoints().GetData())
        cells = vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1, 4)[:, 1:]

        fig, axes = plt.subplots(2, 3, figsize=(12, 6), subplot_kw = {'projection': '3d'}) 
        plt.subplots_adjust(wspace = 0.1, hspace = 0.1)# reduce frames
        views = [((0, 0), "Sagittal"), ((0, 100), "Coronal"), ((90, 90), "Axial")]

        beige_color = (0.93, 0.89, 0.83)

        for i, (view, title) in enumerate(views):
            ax = axes[0, i] #top row for 3d plots.
            ax.set_axis_off()
            ax.grid(False)
            ax.view_init(elev=view[0], azim=view[1])
            ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], 
                            triangles=cells, color = "bisque", shade = True, 
                            lightsource = cm.LightSource(270, 45))
            ax.set_title(title)

            # Zoom on mesh
            ranges = np.array([points[:, i].max() - points[:, i].min() for i in range(3)]) / 2.0
            midpoints = np.array([(points[:, i].max() + points[:, i].min()) / 2.0 for i in range(3)])

            ax.set_xlim(midpoints[0] - ranges[0], midpoints[0] + ranges[0])
            ax.set_ylim(midpoints[1] - ranges[1], midpoints[1] + ranges[1])
            ax.set_zlim(midpoints[2] - ranges[2], midpoints[2] + ranges[2])

            # Zoom factor text
            # zoom_factor = ranges.max() / np.array([points[:, i].max() - points[:, i].min() for i in range(3)]).max()
            # ax.text(midpoints[0] + ranges[0] * 0.9, midpoints[1] + ranges[1] * 0.9,
            #         midpoints[2] + ranges[2] * 0.9, f'Zoom: {zoom_factor:.2f}x', ha='right', va='top')
        
        scale_cm = ranges.max() / 5  # Scale bar represents 1/5th of max range
        scale_bar_x = [midpoints[0] - ranges[0] * 0.9, midpoints[0] - ranges[0] * 0.9 + scale_cm]
        scale_bar_y = [midpoints[1] - ranges[1] * 0.9, midpoints[1] - ranges[1] * 0.9]
        scale_bar_z = [midpoints[2] - ranges[2] * 0.9, midpoints[2] - ranges[2] * 0.9]

        # Create a single scale bar below the figures
        scale_ax = axes[1, :] #bottom row, spanning all columns.
        scale_ax[0].plot(scale_bar_x, [0,0], [0,0], color='black', linewidth=2)
        scale_ax[0].text(scale_bar_x[-1] / 2 + scale_bar_x[0]/2, 0, 0, f'{scale_cm:.1f} cm', ha='center', va='bottom')
        scale_ax[0].set_axis_off()
        scale_ax[1].set_axis_off()
        scale_ax[2].set_axis_off()

        plt.savefig(op.join(output_dir, name + ".png"))
        plt.close(fig)
