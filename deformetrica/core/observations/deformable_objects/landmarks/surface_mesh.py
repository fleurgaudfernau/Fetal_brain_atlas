import numpy as np
import torch
from copy import deepcopy
import os.path as op
import pyvista as pv
from itertools import product
from pyvista import _vtk
from .....core import default
from .....core.observations.deformable_objects.landmarks.landmark import Landmark
from .....support import utilities
from vtk.util import numpy_support as nps
import vtk
from scipy.spatial import KDTree
from typing import Union
from scipy.spatial import ConvexHull

from .....support.utilities.wavelets import haar_backward_transpose, get_max_scale, haar_filter_HF
from .....support import kernels as kernel_factory


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
                kernel = None, gpu_mode = None, kernel_width = None):
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
        self.coarser_mesh_scales = []
        self.closest_points = None
        self.faces = dict()
        self.change = True
        self.cube_grid = None

        # Multiscale mesh v2
        self.smoothing_kernel = None
        if kernel is not None:
            self.smoothing_kernel = kernel_factory.factory(kernel, gpu_mode=gpu_mode,
                                                       kernel_width=kernel_width)

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


    # def write_polydata(self, output_dir, name, points=None):
    #     name = name.replace(".vtk", "_polydata.vtk")
    #     if points is not None:
    #         centers, normals = SurfaceMesh._get_centers_and_normals(torch.from_numpy(points), torch.from_numpy(self.connectivity))
    #         self.check_change_(centers.cpu().numpy())
    #         self.update_polydata_(centers.cpu().numpy(), normals.cpu().numpy())

    #     self.save(output_dir, name)
    
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

                #print("self.normals[", self.normals[:1])
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
        ##print("normals[:1]", normals[:1])
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
        ratio = utilities.move_data(torch.from_numpy(ratio), dtype=filtered_normals.dtype, device=filtered_normals.device)
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
        
            #print("Filtering normals")
            
        self.normals = normals
        self.set_centers_point_data("New_normals", self.normals.detach().cpu().numpy())
        #self.set_centers_point_data("Old_centers", centers.detach().cpu().numpy())
        #self.centers_polydata.points = filtered_centers.detach().cpu().numpy()
        
    def filter_new_normals(self, centers = None, normals = None):
        centers_ = centers.detach().clone().cpu().numpy()
        normals_ = normals.detach().clone().cpu().numpy()
        self.check_change_(centers_)
        self.update_polydata_(centers_, normals_)

        if self.current_scale > 0:
            normals = self.filter_vectors(centers, normals)
        
        self.normals = normals
        self.set_centers_point_data("New_normals", self.normals.detach().cpu().numpy())

        

    ####################################################################################################################
    ### Multiscale meshes - Wavelet Transform - USELESS
    ####################################################################################################################    
    
    def compute_volume(self):
        volume = self.polydata.compute_cell_sizes()
        total_area = volume.cell_data["Area"]
        return np.sum(total_area)
    
    def find_nearest_neigbors(source, target):
        """
         Find the nearest neighbords of source that are on target (and distances to the NNs)
        """
        tree = KDTree(target.points)
        distances, neighbors_id = tree.query(source.points)
        closest_points = target.points[neighbors_id]
        
        return closest_points

    def find_closest_cell(self, grid, point, return_closest_point): 
        singular = False
        locator = _vtk.vtkCellLocator()
        locator.SetDataSet(grid)
        locator.BuildLocator()
        cell = _vtk.vtkGenericCell()

        closest_cells: list[int] = []
        closest_points: list[list[float]] = []

        for node in point:
            closest_point = [0.0, 0.0, 0.0]
            cell_id = _vtk.mutable(0)
            sub_id = _vtk.mutable(0)
            dist2 = _vtk.mutable(0.0)

            locator.FindClosestPoint(node, closest_point, cell, cell_id, sub_id, dist2)
            closest_cells.append(int(cell_id))
            closest_points.append(closest_point)

        out_cells: Union[int, np.ndarray] = (closest_cells[0] if singular else np.array(closest_cells))
        out_points = np.array(closest_points[0]) if singular else np.array(closest_points)

        if return_closest_point:
            return out_cells, out_points
        return out_cells
    
    
    def store_faces_information(self):
        """
            We extract 2D grids from the cube (=6 faces) to perform Haar transformation
        """
        if not self.faces or self.change:
            for f, (d, max_min) in enumerate(zip(list(range(self.dimension))* 2, [np.min]*self.dimension + [np.max]*self.dimension)):
                
                # select face points and normals
                m = max_min(self.closest_points, axis = 0)[d] #axis 0: max over axis 0 (points axis) 
                face = np.where(self.closest_points[:, d] == m)[0]

                self.faces[f] = dict()
                self.faces[f] = {"constant" : m, "axis" : d, "indices": face, 
                                "points": self.closest_points[face], 
                                "normals": self.centers_polydata.point_data["Normals"][face]}
    
    def new_coord(self, pos, min):
        return int(np.ceil(pos-min)*2)

    def haar_filter(self, haar_coeff, grid_normals):
        """
        Filter grid normal vectors (only if a scale is provided)
        """
        for d, haar in enumerate(haar_coeff):
            haar = haar_filter_HF(haar, self.current_scale)
            
            # Backward transform   
            grid_normals[:,:, d] = haar.haar_backward()            
        
        return grid_normals
    
    def replace(self, c, face_points, face_normals, grid_normals, min, order = 1):
        """
        Before haar transformation, assign face normals value to the grid
        After haar transformation, grid transformed normals are sent back to the face
        """
        for i, point in enumerate(face_points):
            new_coords = [self.new_coord(p, min[n]) for n, p in enumerate(point) if n!=c]
            if order == 1:
                grid_normals[new_coords[0], new_coords[1], :] = face_normals[i]
            else:
                face_normals[i] = grid_normals[new_coords[0], new_coords[1], :] 

        return face_normals, grid_normals

    def find_normals_max_scale(self):
        self.project_normals_to_grid_()
        self.store_faces_information()

        for f in self.faces.keys():
            d = self.faces[f]["axis"]
            face_points = self.faces[f]["points"]

            # if we consider intervals between points to be 1 ! in reality points are closer!
            # here we upsample the grid
            max, min = list(np.max(face_points, axis = 0)), list(np.min(face_points, axis = 0))
            grid_normals_shape = np.zeros([self.new_coord(max[n], min[n]) + 1 \
                                            for n in range(self.dimension) if n!=d])

            self.coarser_mesh_scales.append(get_max_scale(grid_normals_shape))


    def check_change_(self, centers = None):        
        if centers is not None:
            self.change = not (np.all(np.round(centers, 3) == np.round(self.centers_polydata.points, 3)))
        
            ##print(np.round(centers, 3)[10:15])
            ##print(np.round(self.centers_polydata.points, 3)[10:15])
        ##print("check change!", self.change)

    

    # self.centers_polydata.point_data["Original_centers"] = np.copy(centers)                
    # self.centers_polydata.point_data["Original_color"] = np.zeros((centers.shape[0]))

    # #combinations = list(product(range(2), repeat=3))
    # middle = np.median(centers, axis = 0)
    # c=0
    # for k in range(int(np.min(centers[:,1])), int(np.max(centers[:,1])), 4):
    #     c+=1
    #     print(k-2<centers[:,1]<k+2)
    #     self.centers_polydata.point_data["Original_color"][np.all(k-2<centers[:,1]<k+2, axis=1)] = c

    # self.centers_polydata.point_data["Original_color"][np.any(centers < middle, axis = 1)] = 1
    # self.centers_polydata.point_data["Original_color"][np.all(centers < middle, axis = 1)] = 2
    # self.centers_polydata.point_data["Original_color"][np.all(centers < 0.2*middle, axis = 1)] = 3
    # self.centers_polydata.point_data["Original_color"][np.all(centers > middle, axis = 1)] = 4
    # self.centers_polydata.point_data["Original_color"][np.all(centers > 1.5*middle, axis = 1)] = 5
    # self.centers_polydata.point_data["Original_color"][np.all(centers > 1.5*middle, axis = 1)] = 5
    # self.centers_polydata.point_data["Original_color"][np.where(np.all((middle[0]*0.8< centers[0] < middle[0]*1.2) &(centers[1] > middle[1]) & (centers[2] < middle[2]), axis =1))] = 6

    #     print(np.shape(centers[:,c]))
    #     print(np.shape(self.centers_polydata.point_data["Original_centers_colors"][:,c]\
    #     [centers[:,c] < middle]))
    #     self.centers_polydata.point_data["Original_centers_colors"][:,c]\
    #     [centers[:,c] < middle] = 0
    #     self.centers_polydata.point_data["Original_centers_colors"][:,c]\
    #     [centers[:,c] >= middle] = 1
    # for i, c in enumerate(combinations):
    #     self.centers_polydata.point_data["Original_centers_colors"]\
    #     [self.centers_polydata.point_data["Original_centers_colors"] == c] = i    

    def project_normals_to_grid_(self, sp = 1):
        """
        Create a cube that has the shape of the brain
        We project brain points (ie cell centers) to their nearest neigbords on the grid cube
        """
        # Projection only at iteration 0, or if new points were provided
        if self.closest_points is None or self.change:
            
            if self.cube_grid is None:
                #logger.info('Project normals onto a grid cube')
                unique_positions = [list(set(list(self.centers_polydata.points[:, k]))) for k in range(self.dimension)]
                limits = [[np.min(p), np.max(p)] for p in unique_positions]
                range_ = [int(l[1]-l[0]) for l in limits]

                # Create the spatial reference: a cube grid with only outer surface
                cube_grid = pv.UniformGrid(dims = range_, origin = [l[0] for l in limits],
                                            spacing = [sp] * self.dimension)      
                cube_grid = cube_grid.extract_geometry()

                # a middle brain grid: X cst
                middle = pv.UniformGrid(dims = [1]+range_[1:], origin = [l[0] for l in limits],
                                            spacing = [sp] * self.dimension)     
                self.cube_grid = cube_grid.merge(middle)
                self.cube_grid = middle

            # Extract correspondence between brain centers and closest points in grid (neirest neighbords)
            _, self.closest_points = self.find_closest_cell(self.cube_grid, self.centers_polydata.points, return_closest_point=True)
            self.centers_polydata.point_data["Closest_points"] = self.closest_points
    
    def haar_transform_normals(self, centers = None, normals = None):        
        # Update points position in polydata and grid projection
        self.check_change_(centers)
        self.update_polydata_(centers, normals)

        # Extract 2D grids from the grid to perform Haar transform
        self.project_normals_to_grid_()
        self.store_faces_information()

        self.set_centers_point_data("New_normals", np.zeros((self.centers_polydata.points.shape)))
        
        for f in self.faces.keys():
            face, d = self.faces[f]["indices"], self.faces[f]["axis"]
            face_points, face_normals = self.faces[f]["points"], self.faces[f]["normals"]
            
            # if we consider intervals between points to be 1 ! in reality points are closer!
            # here we upsample the grid
            max, min = list(np.max(face_points, axis = 0)), list(np.min(face_points, axis = 0))
            grid_normals = np.zeros([self.new_coord(max[n], min[n]) + 1 \
                                    for n in range(self.dimension) if n!=d] + [self.dimension])
            
            _, grid_normals = self.replace(d, face_points, face_normals, grid_normals, min, 1)

            haar_coeff = [haar_backward_transpose(grid_normals[:,:, d]) for d in range(self.dimension)]
            grid_normals = self.haar_filter(haar_coeff, grid_normals)

            # Save on grid    
            old_norm = vect_norm(face_normals)
            face_normals, _ = self.replace(d, face_points, face_normals, grid_normals, min, 2) 
            
            # Back to brain space - Make sure the vectors norms are preserved.
            new_norm = vect_norm(face_normals)
            face_normals *= np.expand_dims(old_norm/new_norm, axis = 1)
            self.set_centers_point_data("New_normals", face_normals, indices = face)
        
        # Compute 
        ratio = self.centers_polydata.point_data["New_normals"]/self.centers_polydata.point_data["Normals"]
        self.set_centers_point_data("Ratio", ratio)
        self.set_centers_point_data("New_normals", self.centers_polydata.point_data["Normals"] * ratio)

        return self.centers_polydata, self.cube_grid


#############################################################################

    # def compute_centers_polydata(self):
    #     """
    #         Creates a polydata whose points=cell centers, with data=normal vectors
    #     """
    #     # centers_normals = self.polydata.compute_normals(cell_normals=True, point_normals=False)
    #     # self.centers_polydata = self.polydata.cell_centers()
    #     # self.centers_polydata.point_data["Normals"] = centers_normals.cell_data["Normals"]
    #     # self.centers_polydata.point_data["Normals_2"] = self.normals.cpu().numpy()
    #     points = torch.from_numpy(self.polydata.points)
    #     triangles = torch.from_numpy(self.connectivity)
    #     centers, normals = SurfaceMesh._get_centers_and_normals(points, triangles)

    #     self.centers_polydata = pv.PolyData(centers.cpu().numpy())
    #     self.centers_polydata.point_data["Normals"] = normals.cpu().numpy()


    # def check_change(self, new_points = None):
    #     """
    #         We check that the object "new points" are actually different from old ones
    #     """
    #     self.change = False
        
    #     if new_points is not None:
    #         self.change = not (np.all(np.round(new_points, 3) == np.round(self.polydata.points, 3)))
        
    #     #print("check change!", self.change)
    
    # def update_polydata(self, new_points = None):
    #     """
    #         Update the object polydata if position of vertices have changed
    #     """
    #     if self.change and new_points is not None:
    #         # Edges first column: nb of points in each line
    #         new_column = np.asarray([self.dimension] * self.connectivity.shape[0]).reshape((self.connectivity.shape[0], 1))
    #         edges = np.hstack(np.append(new_column, self.connectivity, axis=1))
    #         self.polydata = pv.PolyData(new_points, edges, n_faces = edges.shape[0])
    
    # def project_normals_to_grid(self, sp = 1):
    #     """
    #     Create a cube that has the shape of the brain
    #     We project brain points (ie cell centers) to their nearest neigbords on the grid cube
    #     """
    #     # Projection only at iteration 0, or if new points were provided
    #     if self.closest_points is None or self.change:
    #         #logger.info('Project normals onto a grid cube')
    #         self.compute_centers_polydata()

    #         unique_positions = [list(set(list(self.centers_polydata.points[:, k]))) for k in range(self.dimension)]
    #         limits = [[np.min(p), np.max(p)] for p in unique_positions]
    #         range_ = [int(l[1]-l[0]) for l in limits]

    #         # Create the spatial reference: a cube grid with only outer surface
    #         cube_grid = pv.UniformGrid(dims = range_, origin = [l[0] for l in limits],
    #                                     spacing = [sp] * self.dimension)      
    #         cube_grid = cube_grid.extract_geometry()

    #         # Extract correspondence between brain centers and closest points in grid (neirest neighbords)
    #         _, self.closest_points = self.find_closest_cell(cube_grid, self.centers_polydata.points, return_closest_point=True)
    #         self.centers_polydata.point_data["Closest_points"] = self.closest_points

    # def haar_transform_normals(self, new_points = None):
    #     # Update points position in polydata and grid projection
    #     self.check_change(new_points)
    #     self.update_polydata(new_points)

    #     # Extract 2D grids from the grid to perform Haar transform
    #     self.project_normals_to_grid()
    #     self.store_faces_information()

    #     self.set_centers_point_data("New_normals", np.zeros((self.centers_polydata.points.shape)))
        
    #     for f in self.faces.keys():
    #         face, d = self.faces[f]["indices"], self.faces[f]["axis"]
    #         face_points, face_normals = self.faces[f]["points"], self.faces[f]["normals"]
            
    #         # if we consider intervals between points to be 1 ! in reality points are closer!
    #         # here we upsample the grid
    #         max, min = list(np.max(face_points, axis = 0)), list(np.min(face_points, axis = 0))
    #         grid_normals = np.zeros([self.new_coord(max[n], min[n]) + 1 \
    #                                 for n in range(self.dimension) if n!=d] + [self.dimension])
            
    #         _, grid_normals = self.replace(d, face_points, face_normals, grid_normals, min, 1)

    #         haar_coeff = [haar_backward_transpose(grid_normals[:,:, d]) for d in range(self.dimension)]
    #         grid_normals = self.haar_filter(haar_coeff, grid_normals)

    #         # Save on grid    
    #         old_norm = vect_norm(face_normals)
    #         face_normals, _ = self.replace(d, face_points, face_normals, grid_normals, min, 2) 
            
    #         # Back to brain space - Make sure the vectors norms are preserved.
    #         new_norm = vect_norm(face_normals)
    #         face_normals *= np.expand_dims(old_norm/new_norm, axis = 1)
    #         self.set_centers_point_data("New_normals", face_normals, indices = face)
        

    #     self.normals = torch.from_numpy(self.centers_polydata.point_data["New_normals"])
        
    #     return self.centers_polydata