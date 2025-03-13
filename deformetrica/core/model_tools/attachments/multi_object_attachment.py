import torch
import logging
import pyvista as pv
from copy import deepcopy
from scipy.spatial import KDTree
import numpy as np
import vtk
from ....support.utilities import move_data, get_best_device, get_torch_scalar_type, get_torch_integer_type
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

logger = logging.getLogger(__name__)

def get_points(data, pos, obj):
    return data['landmark_points'][pos:pos + obj.n_points()]


class MultiObjectAttachment:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, attachment_types, kernels):
        # List of strings, e.g. 'varifold' or 'current'.
        self.types = attachment_types

        #self.attachments = {"curvature" : self.curvature}
        # List of kernel objects.
        self.kernels = kernels

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    
    def compute_weighted_curvature_distance(self, data, multi_obj1, multi_obj2, inverse_weights,
                                            save = False):
        """
            If we want to add curvature as a new attachement term (for meshes)
        """
        # Unused so far
        device = next(iter(data.values())).device  # deduce device from template_data
        dtype = next(iter(data.values())).dtype  # deduce dtype from template_data
        distances = torch.zeros((len(multi_obj1.object_list),), device=device, dtype=dtype)
        pos = 0
        for i, (obj1, obj2) in enumerate(zip(multi_obj1.object_list, multi_obj2.object_list)):
            distances[i] = self.mean_curvature_distance(get_points(data, pos, obj1), obj1, obj2)            
        pos += obj1.n_points()
        inverse_weights_torch = move_data(inverse_weights, device=device, dtype=dtype)
        res = torch.sum(distances / inverse_weights_torch) #shape [1]
        
        return res

    def compute_distances(self, data, multi_obj1, multi_obj2, residuals = False):
        """
        Takes two multiobjects and their new point positions to compute the distances.
        If residuals is True, we ignore the smoothing of the normals in multiscale meshes
        """
        assert len(multi_obj1.object_list) == len(multi_obj2.object_list), \
            "Cannot compute distance between multi-objects with different number of objects"

        device = next(iter(data.values())).device  # deduce device from template_data
        dtype  = next(iter(data.values())).dtype   # deduce dtype from template_data
        distances = torch.zeros((len(multi_obj1.object_list),), device=device, dtype=dtype)
        pos = 0

        for i, (obj1, obj2, attachment, kernel) in enumerate(zip(multi_obj1.object_list, 
                                multi_obj2.object_list, self.types, self.kernels)):
            if attachment == 'L2':
                assert obj1.type.lower() == obj2.type.lower() == 'image'
                distances[i] = self.L2_distance(data['image_intensities'], obj2)
            
            else:
                if attachment.lower() == 'current':
                    distances[i] = self.current_distance(get_points(data, pos, obj1), obj1, obj2, kernel, residuals)
                    
                elif attachment.lower() == 'varifold':
                    distances[i] = self.varifold_distance(get_points(data, pos, obj1), obj1, obj2, kernel, residuals)

                elif attachment.lower() == 'landmark':
                    distances[i] = self.landmark_distance(get_points(data, pos, obj1), obj2, residuals)
                else:
                    assert False, "The distance {} does exist".format(attachment)
                pos += obj1.n_points()

        return distances

    def compute_weighted_distance(self, data, multi_obj1, multi_obj2, inverse_weights):
        """
        Takes two multiobjects and their new point positions to compute the distances
        data: deformed data
        multi_obj1: template
        multi_obj2: target
        """
        distances = self.compute_distances(data, multi_obj1, multi_obj2)
        
        assert distances.size()[0] == len(inverse_weights)
        device = next(iter(data.values())).device  # deduce device from template_data
        dtype = next(iter(data.values())).dtype  # deduce dtype from template_data

        inverse_weights_torch = move_data(inverse_weights, device=device, dtype=dtype)

        return torch.sum(distances / inverse_weights_torch)    

    def compute_vtk_distance(self, data, multi_obj1, multi_obj2, type):
        pos = 0
        for i, (obj1, obj2, attachment) in enumerate(zip(multi_obj1.object_list, 
                                    multi_obj2.object_list, self.types)):
            if attachment in ['Current', "Varifold"]:
                obj1.polydata.points = get_points(data, pos, obj).cpu().numpy()

                type2 = "hausdorff" if type == "average_min_dist" else type

                obj1.polydata = self.vtk_distance(obj1, obj2, type2)
                obj2.polydata = self.vtk_distance(obj2, obj1, type2)
                
                if type != "hausdorff":
                    obj1.distance[type] = np.mean(obj1.polydata.point_data[type2])
                    obj2.distance[type] = np.mean(obj2.polydata.point_data[type2])
                else:
                    obj1.distance["hausdorff"] = obj1.polydata["HausdorffDistance"]
                    obj2.distance["hausdorff"] = obj2.polydata["HausdorffDistance"]

                    pos += obj1.n_points()
                    
    ####################################################################################################################
    ### Auxiliary methods:
    ####################################################################################################################
    @staticmethod
    def mean_curvature_distance(points, source, target):
        # A curvature value per vertex
        # Cannot compare deformed template and target curvatures: different nb of vertices
        source.curvature(type="absolute", points = points) #shape (29691,)
        target.curvature(type="absolute") #shape (29691,)
        difference = target.global_mean_curv_absolute - source.global_mean_curv_absolute

        return difference

    @staticmethod
    def current_distance(points, source, target, kernel):
        """
        Compute the current distance between source and target, assuming points are the new points of the source
        We assume here that the target never moves.
        Dimension of the source
        """
        device = get_best_device()
        c1, n1, c2, n2 = MultiObjectAttachment.__get_source_and_target_centers_and_normals(points, source, target, device=device)
        
        def current_scalar_product(points_1, points_2, normals_1, normals_2):
            #return a single value
            assert points_1.device == points_2.device == normals_1.device == normals_2.device, 'tensors must be on the same device'
            return torch.dot(normals_1.view(-1), kernel.convolve(points_1, points_2, normals_2).view(-1))

        if target.norm is None or target.filtered: #only computed once
            target.norm = current_scalar_product(c2, c2, n2, n2)
        
        return current_scalar_product(c1, c1, n1, n1) + target.norm.to(c1.device) \
                - 2 * current_scalar_product(c1, c2, n1, n2)
    
    @staticmethod
    def point_current_distance(points, source, target, kernel):
        """
        Compute the current distance between source and target, assuming points are the new points of the source
        We assume here that the target never moves.

        Dimension of centers and normals != number of points -> number of cells!
        A current representation is center-wise
        To be point wise the current distance 
        """
        device = get_best_device()
        c1, n1, c2, n2 = MultiObjectAttachment.__get_source_and_target_centers_and_normals(points, source, target, device=device)

        def point_current_scalar_product(points_1, points_2, normals_1, normals_2):
            assert points_1.device == points_2.device == normals_1.device == normals_2.device, 'tensors must be on the same device'
            
            product = torch.zeros((normals_1.size()), device = points_1.device, dtype = normals_1.dtype)
            # normals_1 = values of normals at points_1 = n_centers_1 x 3

            # ~ We get the values of normals 2 convolved at points_1
            # = n_centers_1 x 3
            normals_2_convolve = kernel.convolve(points_1, points_2, normals_2)#.view(-1)

            for i in range(normals_1.size()[0]): # from 0 to n points
                for j in range(normals_1.size()[1]): # from 0 to 3
                    product[i][j] = normals_1[i][j] * normals_2_convolve[i][j]

            return product
        
        def norm_convolved_to_source(points_1, points_2, normals_2):
            # this is wrong to convolve twice!
            # We send the values of normals_2 to positions of points_1
            product = torch.zeros((points_1.size()), device = points_1.device, dtype = normals_2.dtype)
            normals_2_convolve = kernel.convolve(points_1, points_2, normals_2) #size of product
            for i in range(normals_2_convolve.size()[0]):
                for j in range(normals_2_convolve.size()[1]):
                    product[i][j] = normals_2_convolve[i][j] * normals_2_convolve[i][j]

            return product

        # return point_current_scalar_product(c1, c1, n1, n1) + norm_convolved_to_source(c1, c2, n2) \
        #     - 2 * point_current_scalar_product(c1, c2, n1, n2)
        
        norm_1 = point_current_scalar_product(c1, c1, n1, n1)
        norm_2_to_1 = norm_convolved_to_source(c1, c2, n2)
        scalar_product_1 = point_current_scalar_product(c1, c2, n1, n2)
        distance_1 = norm_1 + norm_2_to_1 - scalar_product_1

        norm_1_to_2 = norm_convolved_to_source(c2, c1, n1)
        norm_2 = point_current_scalar_product(c2, c2, n2, n2)
        scalar_product_2 = point_current_scalar_product(c2, c1, n2, n1)
        distance_2 = norm_1_to_2 + norm_2 - scalar_product_2

        return distance_1

    
    @staticmethod
    def vtk_distance(source, target, type = "hausdorff"):
        # Modify the source polydata (already done)
        #points = vtk.vtkPointSet(points.cpu().numpy()) #does not work: needs path
        #source_polydata.SetPoints(points) # no SetPoints attribute

        # largest distance between a point in S to points in T
        # point data = minimal distance for each point to another point
        if type.lower() in ["hausdorff"]:
            dist = vtk.vtkHausdorffDistancePointSetFilter()
            dist.SetTargetDistanceMethodToPointToPoint()
            dist.SetTargetDistanceMethodToPointToCell() # the two methods seem to get same output
        elif type.lower() == "signed":
            dist = vtk.vtkDistancePolyDataFilter()
            dist.SignedDistanceOn() #no difference signed unsigned
            dist.GetSecondDistanceOutput()
        elif type.lower() == "unsigned":
            dist = vtk.vtkDistancePolyDataFilter()
            dist.SignedDistanceOff()
            dist.GetSecondDistanceOutput()

        dist.SetInputData(0, source.polydata)
        dist.SetInputData(1, target.polydata)
        dist.Update()
        pd = dist.GetOutput() # a polydata - distance in the cells

        # convert to pyvista
        polydata = pv.wrap(pd)
        polydata.point_data[type] = polydata.point_data['Distance']
        polydata.point_data.remove('Distance')

        return polydata
    
    @staticmethod
    def KDTreeDistance(source, target):
        # ajout fg
        tree = KDTree(source.points) #nxd: n data points of dim d
        d_kdtree, _ = tree.query(target.points)
        
        return np.mean(d_kdtree)
    
    @staticmethod
    def varifold_distance(points, source, target, kernel, residuals):
        """
        Returns the varifold distance between the 3D meshes
        """
        def varifold_scalar_product(x, y, n1_norm, n2_norm, nalpha, nbeta):
            
            return torch.dot(n1_norm.view(-1), kernel.convolve((x, nalpha), (y, nbeta), 
                            n2_norm.view(-1, 1), mode='varifold').view(-1))
        
        device = get_best_device()
        c1, n1, c2, n2 = MultiObjectAttachment.__get_source_and_target_centers_and_normals(points, source, target, residuals, device=device)

        # alpha = normales non unitaires
        n1_norm = torch.norm(n1, 2, 1) # 2 for L2 norm. 1 to compute norm across axis 1 -> a norm per row
        n2_norm = torch.norm(n2, 2, 1) #size = n1_rows x 1 = n_centers x 1 -> unsqueeze: n_centers

        nalpha = n1 / n1_norm.unsqueeze(1) # each normal vector / its norm -> norm = 1
        nbeta = n2 / n2_norm.unsqueeze(1)
        
        # We always recompute target norm (in case of multiscale meshes...)
        if target.norm is None or target.filtered:
            target.norm = varifold_scalar_product(c2, c2, n2_norm, n2_norm, nbeta, nbeta)

        return varifold_scalar_product(c1, c1, n1_norm, n1_norm, nalpha, nalpha) + target.norm \
               - 2 * varifold_scalar_product(c1, c2, n1_norm, n2_norm, nalpha, nbeta)
            
    @staticmethod
    def landmark_distance(points, target, residuals):
        """
        Point correspondance distance
        """
        target_points = move_data(target.get_points(), dtype=str(points.type()), device=points.device)
        assert points.device == target_points.device, 'tensors must be on the same device'
        return torch.sum((points.contiguous().view(-1) - target_points.contiguous().view(-1)) ** 2)

    @staticmethod
    def compute_ssim_distance(data, multi_obj1, multi_obj2, type):
        for i, obj1 in enumerate(multi_obj1.object_list):
            obj2 = multi_obj2.object_list[i]

            source = data['image_intensities'].cpu().numpy()
            target = obj2.get_intensities().cpu().numpy()

            if type == "mse":
                dist = mean_squared_error(source, target)   
            else:
                dist = ssim(source, target)

            obj1.distance[type] = dist
            obj2.distance[type] = dist

    @staticmethod
    def L2_distance(intensities, target): #intensit
        """
        L2 image distance.
        intensities: template deformed intensities
        target: object intensities
        """
        assert isinstance(intensities, torch.Tensor)
        target_intensities = move_data(target.get_intensities(), dtype=intensities.type(), device=intensities.device)

        assert intensities.device == target_intensities.device, 'tensors must be on the same device'

        result = torch.sum((intensities.contiguous().view(-1) - target_intensities.contiguous().view(-1)) ** 2)
        return result

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    @staticmethod
    def __get_source_and_target_centers_and_normals(points, source, target, residuals = False, device=None):
        if device is None:
            device = points.device

        dtype = str(points.dtype)

        # ajout fg: check wether points belong to source or target
        if source.points.shape[0] == points.shape[0]:
            c1, n1 = source.get_centers_and_normals(points, device=device, residuals = residuals)
            c2, n2 = target.get_centers_and_normals(device=device, residuals = residuals)
        else:
            c1, n1 = source.get_centers_and_normals(device=device, residuals = residuals)
            c2, n2 = target.get_centers_and_normals(points, device=device, residuals = residuals)

        assert c1.device == n1.device == c2.device == n2.device, 'all tensors must be on the same device, c1.device=' + str(c1.device) \
                                                                 + ', n1.device=' + str(n1.device)\
                                                                 + ', c2.device=' + str(c2.device)\
                                                                 + ', n2.device=' + str(n2.device)
        return c1, n1, c2, n2
