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
from ....support import utilities
from vtk.util import numpy_support as nps
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull
from ....support import kernels as kernel_factory


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

    def __init__(self, points, triangles, object_filename = None, polydata= None,
                kernel = None, kernel_width = None):
        Landmark.__init__(self, points) #self.points = points
        self.type = 'SurfaceMesh'
        
        self.connectivity = triangles
        self.object_filename = object_filename

        self.original_points = deepcopy(points)

        self.curv = {"gaussian": dict(), "maximum": dict(), "mean": dict(), "absolute": dict(),
                    "surface_area": dict(), "GI": dict()}
        self.distance = {"current" : 0, "signed":0, "hausdorff": 0, "average_min_dist" : 0}

        # Create polydata (ajout fg)
        vertices = deepcopy(self.original_points).astype('float32')
        connectivity = deepcopy(triangles)

        # Edges first column: nb of points in each line
        new_column = np.asarray([self.dimension] * connectivity.shape[0]).reshape((connectivity.shape[0], 1))
        edges = np.hstack(np.append(new_column, connectivity, axis=1))

        # For Laplacian filter (multiscale "img")
        self.original_polydata = pv.PolyData(vertices, edges, n_faces = edges.shape[0])
        self.polydata = deepcopy(self.original_polydata)

        self.centers_polydata = None

        # Multiscale mesh v1
        self.multiscale_mesh = False
        self.current_scale = None
        self.faces = dict()
        self.change = True

        # Multiscale mesh v2
        self.smoothing_kernel = None
        if kernel is not None:
            self.smoothing_kernel = kernel_factory.factory(kernel, kernel_width=kernel_width)

        # All of these are torch tensor attributes.
        self.centers, self.normals = SurfaceMesh._get_centers_and_normals(
                                    torch.from_numpy(points), torch.from_numpy(triangles))
        self.original_normals = self.normals.clone()
        self.update_polydata_(self.centers.detach().clone().cpu().numpy(), 
                              self.normals.detach().clone().cpu().numpy())
        
        # ajout fg to avoid norm recomputation (useless)
        self.filtered = False
                            
    ####################################################################################################################
    ### Polydata tools (ajout fg):
    ####################################################################################################################
    def save(self, output_dir, name):
        self.centers_polydata.save(op.join(output_dir, name), binary=False)
    
    def update_polydata_(self, centers = None, normals = None):
        """
            Update the object polydata if position of vertices have changed
        """
        # store the initial centers positions (to check how points move)
        if self.change and centers is not None:
            #if self.centers_polydata is None:
            self.centers_polydata = pv.PolyData(centers)
            # else:
            #     self.centers_polydata.points = centers
            self.centers_polydata.point_data["Normals"] = normals#.detach().clone().cpu().numpy()

    def set_centers_point_data(self, key, value, indices = None):
        if indices is None:
            self.centers_polydata.point_data[str(key)] = value
        else:
            self.centers_polydata.point_data[str(key)][indices] = value
    
    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################
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

    def set_scale(self, new_scale):
        self.current_scale = new_scale
        self.multiscale_mesh = True

    def filter(self, n_iter):
        """
        Laplacian smoothing of the mesh
        """
        self.filtered = True
        self.polydata = self.original_polydata.smooth(n_iter, relaxation_factor=0.01) # relaxation=displacement factor of points

        # Get new points and edges for object
        points = nps.vtk_to_numpy(self.polydata.GetPoints().GetData()).astype('float64')
        points = points[:, :self.dimension]
        
        lines = nps.vtk_to_numpy(self.polydata.GetLines().GetData()).reshape((-1, 3))[:, 1:]
        polygons = nps.vtk_to_numpy(self.polydata.GetPolys().GetData()).reshape((-1, 4))[:, 1:]

        if len(lines) == 0 and len(polygons) == 0:
            connectivity = None
        elif len(lines) > 0 and len(polygons) == 0:
            connectivity = lines
        elif len(lines) == 0 and len(polygons) > 0:
            connectivity = polygons
        elif self.dimension == 2:
            connectivity = lines
        else:
            connectivity = polygons

        # Save points and edges
        self.set_points(points)
        self.set_connectivity(connectivity)

        return points
    
    def curvature_metrics(self, type):
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
        self.polydata.cell_data["Normals_normalized"] = normals_normalized.cpu().numpy()    
    
    @staticmethod
    def _get_centers_and_normals(points, triangles,
                                 device='cpu'):
        """
         Computed by hand so that centers and normals keep the autograd of point
        """
        points = utilities.move_data(points, device=device)
        triangles = utilities.move_data(triangles, integer = True, device=device)

        a = points[triangles[:, 0]]
        b = points[triangles[:, 1]]
        c = points[triangles[:, 2]]

        centers = (a + b + c) / 3.
        normals = torch.cross(b - a, c - a) / 2

        assert torch.device(device) == centers.device == normals.device

        return centers, normals

    def get_centers_and_normals(self, points=None,
                                device='cpu', residuals = False):
        """
        Given a new set of points, use the corresponding connectivity available in the polydata
        to compute the new normals, all in torch
        """
        if points is None:
            if not self.is_modified:

                return (utilities.move_data(self.centers, device=device),
                        utilities.move_data(self.normals, device=device))

            else:
                logger.debug('Call of SurfaceMesh.get_centers_and_normals with is_modified=True flag.')
                points = torch.from_numpy(self.points)

        centers, normals = SurfaceMesh._get_centers_and_normals(points, torch.from_numpy(self.connectivity), 
                             device=device)

        # Haar transform normals
        if self.multiscale_mesh and self.current_scale is not None: # iter 0, 1st residuals computation
            if not residuals:
                self.filter_new_normals(centers.clone(), normals.clone())
                normals = utilities.move_data(self.normals, 
                                            requires_grad = (normals.grad_fn is not None), 
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

    ####################################################################################################################
    ### Multiscale meshes - Filtering of normals
    ####################################################################################################################    
    
    def set_kernel_scale(self, kernel_width):
        self.smoothing_kernel.update_width(kernel_width)
        self.multiscale_mesh = True
        self.current_scale = kernel_width

    def filter_vectors(self, centers, normals):
        filtered_normals = self.smoothing_kernel.convolve(centers, centers, normals, mode = "gaussian_weighted")

        old_norm = vect_norm(normals.detach().cpu().numpy())
        new_norm = vect_norm(filtered_normals.detach().cpu().numpy())
        ratio = np.expand_dims(old_norm/new_norm, axis = 1)
        ratio = utilities.move_data(torch.from_numpy(ratio), dtype=filtered_normals.dtype, 
                                    device=filtered_normals.device)
        filtered_normals = filtered_normals * ratio

        # Works well but too long!
        # weights = torch.sum((centers[:, None, :] - centers[None, :, :]) ** 2, 2)
        # #weights -= torch.max(weights, dim=1)[0][:, None]  # subtract the max for robustness
        # gamma = -1/(self.smoothing_kernel.kernel_width**2)
        # weights = torch.exp(weights*gamma) / torch.sum(torch.exp(weights*gamma), dim=1)[:, None]
        # filtered_normals = weights @ normals

        return filtered_normals


    def filter_normals(self):
        """
         Called by multiscale meshes.
         Modify self.normals as set by the filter width & polydata (for vizualisation)
        """
        self.filtered = True
                
        centers, normals = self.centers, self.original_normals.clone()

        if self.current_scale > 0:
            normals = self.filter_vectors(centers, normals)
            #self.set_kernel_scale(10)
            #filtered_centers = self.filter_vectors(centers, centers)
                    
        self.normals = normals
        self.set_centers_point_data("New_normals", self.normals.detach().cpu().numpy())
        
    def filter_new_normals(self, centers = None, normals = None):
        centers_ = centers.detach().clone().cpu().numpy()
        normals_ = normals.detach().clone().cpu().numpy()
        self.check_change_(centers_)
        self.update_polydata_(centers_, normals_)

        if self.current_scale > 0:
            normals = self.filter_vectors(centers, normals)
        
        self.normals = normals
        self.set_centers_point_data("New_normals", self.normals.detach().cpu().numpy())

        
