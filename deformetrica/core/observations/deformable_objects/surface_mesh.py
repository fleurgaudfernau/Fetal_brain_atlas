import numpy as np
import torch
import vtk
from typing import Union

from copy import deepcopy
import os.path as op
import pyvista as pv
from itertools import product
from pyvista import _vtk
from ....core import default
from ....core.observations.deformable_objects.landmark import Landmark
from ....support.utilities import move_data, detach
from vtk.util import numpy_support as nps
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull
from ....support import kernels as kernel_factory

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import logging
logger = logging.getLogger(__name__)

def vect_norm(array, order = 2):
    return np.linalg.norm(array, ord = order, axis = 1)

class SurfaceMesh(Landmark):
    """
    3D Triangular mesh.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, points, triangles, object_filename, polydata= None):
        super().__init__(points, object_filename) # initialize all attributes in Landmark
        self.type = 'SurfaceMesh'
        
        self.connectivity = triangles

        self.original_points = deepcopy(points)

        self.curv = {"gaussian": dict(), "maximum": dict(), "mean": dict(), "absolute": dict(),
                    "surface_area": dict(), "GI": dict()}
        self.distance = {"current" : 0, "signed":0, "hausdorff": 0, "average_min_dist" : 0}

        ##################################################################################
        # Create polydata (ajout fg)
        self.update_polydata(self.original_points, triangles)
        self.original_polydata = deepcopy(self.polydata)

        # # All of these are torch tensor attributes.
        self.centers, self.normals = SurfaceMesh._get_centers_and_normals(
                                    torch.from_numpy(points), torch.from_numpy(triangles))

        self.change = True
                
        # ajout fg to avoid norm recomputation (useless)
        self.filtered = False
                            
    ####################################################################################################################
    ### Polydata tools (ajout fg):
    ####################################################################################################################
    def set_points(self, points):
        """
        Override Landmark.set_points to update mesh polydata automatically.
        """
        super().set_points(points)  # still stores points and bounding box
        
        self.update_polydata(points, self.connectivity)

    def update_polydata(self, points, triangles):
        """
            Update the object polydata if position of vertices have changed
        """
        vertices = deepcopy(points).astype('float32')
        connectivity = deepcopy(triangles)

        # Edges first column: nb of points in each line
        new_column = np.asarray([self.dimension] * connectivity.shape[0]).reshape((connectivity.shape[0], 1))
        edges = np.hstack(np.append(new_column, connectivity, axis=1))

        # For Laplacian filter (multiscale objects)
        self.polydata = pv.PolyData(vertices, edges, n_faces = edges.shape[0])
    
    def _update_from_polydata(self):
        # Get new points and edges for object
        points = nps.vtk_to_numpy(self.polydata.GetPoints().GetData()).astype('float64')[:, :self.dimension]
        
        connectivity = self.get_connectivity()
        points = points[:, :self.dimension]

        self.set_points(points)
        self.set_connectivity(connectivity)

        return points
    
    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################
    def get_connectivity(self):
        lines = nps.vtk_to_numpy(self.polydata.GetLines().GetData()).reshape((-1, 3))[:, 1:]
        polygons = nps.vtk_to_numpy(self.polydata.GetPolys().GetData()).reshape((-1, 4))[:, 1:]

        if len(lines) == 0 and len(polygons) == 0:
            return None
        elif (len(lines) > 0 and len(polygons) == 0) or self.dimension == 2:
            return lines
        elif len(lines) == 0 and len(polygons) > 0:
            return polygons
        else:
            return polygons
    
    def remove_null_normals(self):
        _, normals = self.get_centers_and_normals()
        triangles_to_keep = torch.nonzero(torch.norm(normals, 2, 1) != 0)

        if len(triangles_to_keep) < len(normals):
            logger.info('Detected {} null area triangles, removing them'.format(len(normals) - len(triangles_to_keep)))
            new_connectivity = self.connectivity[triangles_to_keep.view(-1)]
            new_connectivity = np.copy(new_connectivity)
            self.connectivity = new_connectivity

            # Updating the centers and normals consequently.
            self.centers, self.normals = SurfaceMesh._get_centers_and_normals(
                torch.from_numpy(self.points), torch.from_numpy(self.connectivity))

    def filter(self, n_iter):
        """
        Laplacian smoothing of the mesh
        """
        self.filtered = True
        self.polydata = self.original_polydata.smooth(n_iter, relaxation_factor=0.01) # relaxation=displacement factor of points
        
        points = self._update_from_polydata()
        
        return points
    
    def filter_taubin(self, n_iter = 1000):
        self.update_polydata(self.points, self.connectivity)

        self.polydata.smooth_taubin(n_iter)
        self._update_from_polydata()

    def curvature_metrics(self, type):
        type = type.replace("Curvature_", "")
        if type == "normals": # Compute normals
            self.current()
        elif type == "varifold":  # Compute normals normalized
            self.varifold()
        elif type == "surface_area":
            self.surface_area()
        elif type == "GI":
            self.gyrification_index()
        else:
            self.curvature(type) # Compute curvature metrics

    def surface_area(self):
        """
        Computes the mesh total surface
        """
        self.curv["surface_area"]["mean"] = self.polydata.area
    
    def gyrification_index(self):
        convexhull = ConvexHull(self.polydata.points)
        self.curv["GI"]["mean"] = self.polydata.area / convexhull.area

    def curvature(self, type = "gaussian"):
        """
        Compute point-wise curvature measures 
        4 possible types: "mean", "gaussian", "maximum", "minimum", "absolute"
        # mean curvature: change in normal direction along surface in degrees (<<0 sulci >> 0 gyri)
        # M(v)=avg over neigbors edges e of M(e); M(e)=length(e)*dihedral_angle(e)
        # absolute (mean curvature): local amounnt of gyrification
        # gaussian(v)=2*PI-sum_(facets neigbors of v) angle f at v"""

        # pv polydata: source points need to be replaced by new ones!
        curv = self.polydata.curvature(type)
        self.curv[type]["values"] = curv
        self.curv[type]["mean"] = np.mean(np.abs(self.curv[type]["values"]))
        self.curv[type]["norm"] = np.linalg.norm(self.curv[type]["values"])

        self.polydata.point_data["Curv_"+type] = self.curv[type]["values"]

    def current(self):
        """
            Save normal vectors in polydata
            pv/vtk: The algorithm works by determining normals for each polygon and then averaging them 
            at shared points. When sharp edges are present, the edges are split and new points generated to prevent blurry edges (due to Gouraud shading).
        """
        points = torch.from_numpy(self.polydata.points)
        connectivity = torch.from_numpy(self.connectivity)
        centers, normals = self._get_centers_and_normals(points, connectivity, device=points.device)
        self.polydata.cell_data["Normals"] = normals
        self.polydata.cell_data["Centers"] = centers
    
    def varifold(self):
        """
            Varifold representation of mesh (ie with its normals) - same output as pyvista compute normals
        """
        points = torch.from_numpy(self.polydata.points)
        connectivity = torch.from_numpy(self.connectivity)
        _, normals = self._get_centers_and_normals(points, connectivity, device=points.device)

        normals_normalized = normals / torch.norm(normals, 2, 1).unsqueeze(1)
        self.polydata.cell_data["Normals_normalized"] = detach(normals_normalized)    
    
    @staticmethod
    def _get_centers_and_normals(points, triangles, device='cpu'):
        """
         Computed by hand so that centers and normals keep the autograd of point
        """
        points = move_data(points, device=device)
        triangles = move_data(triangles, integer = True, device=device)

        a = points[triangles[:, 0]]
        b = points[triangles[:, 1]]
        c = points[triangles[:, 2]]

        centers = (a + b + c) / 3.
        normals = torch.cross(b - a, c - a) / 2

        assert torch.device(device) == centers.device == normals.device

        return centers, normals

    def get_centers_and_normals(self, points=None, device='cpu', residuals = False):
        """
        Given a new set of points, use the corresponding connectivity available in the polydata
        to compute the new normals, all in torch
        """
        if points is None:
            if not self.is_modified:
                return (move_data(self.centers, device=device), move_data(self.normals, device=device))

            else:
                points = torch.from_numpy(self.points)

        centers, normals = SurfaceMesh._get_centers_and_normals(points, torch.from_numpy(self.connectivity), 
                             device=device)

        return (centers, normals)

    def get_centers_and_normals_pyvista(self, points = None, device='cpu'):
        
        if points is not None:
            self.polydata.points = points
        
        self.polydata = self.polydata.compute_normals(cell_normals=True, point_normals=False, 
                                                    consistent_normals = False)
        polydata_centers = self.polydata.cell_centers()

        normals = torch.from_numpy(self.polydata.cell_data["Normals"])
        centers = torch.from_numpy(polydata_centers.points)

        return centers, normals

