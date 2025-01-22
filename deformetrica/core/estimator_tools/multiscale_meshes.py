import copy
import logging
import os.path as op
import numpy as np
from ...support.utilities.tools import residuals_change

logger = logging.getLogger(__name__)

def compute_mesh_scale(model_name, momenta_scale, current_mesh_scale = None):
    if momenta_scale > 1:
        mesh_scale = min(30000, 7**(momenta_scale)) #before 27/01: 6
    else:
        mesh_scale = 0
    ## smaller steps
    # if not current_mesh_scale:
    #     if momenta_scale > 1:
    #         mesh_scale = 8**(momenta_scale)
    # else: 
    #     mesh_scale = current_mesh_scale/2
    
    # if mesh_scale < 100:
    #     mesh_scale = 0
        
    return int(mesh_scale)

class MultiscaleMeshes():
    def __init__(self, multiscale, multiscale_momenta, model, dataset, output_dir, 
                ctf_interval, ctf_max_interval, points):
        self.model = model
        self.model_name = model.name
        self.original_dataset = copy.deepcopy(dataset)
        self.ext = self.model.objects_name_extension[0]

        self.output_dir = output_dir

        self.initial_convergence_threshold = 0.001 
        self.convergence_threshold = 0.001  
        self.ctf_interval = ctf_interval
        self.ctf_max_interval = ctf_max_interval

        self.multiscale = multiscale
        self.multiscale_momenta = multiscale_momenta
        self.iter = [0] if multiscale else []

        self.points = points 
        self.compute_scale = compute_mesh_scale

    ####################################################################################################################
    ### Coarse to fine on meshes - smoothing of the normal vectors
    ####################################################################################################################    

    def initialize(self, momenta_coarser_scale = None):
        """
            Compute coarser scale of each mesh
            - Modifies original dataset: grid projection
            - but doesnt change normals as we need them for recovering original data
            !! in case of RG, template can be= one of observations
        """
        if self.multiscale:
            if self.multiscale_momenta:
                self.scale = momenta_coarser_scale-1
                self.step = 1    

            else: 
                self.scale = 6
                self.step = 1    

    def coarse_to_fine_condition(self, iteration, avg_residuals, end = False):
        """
        Authorizes coarse to fine if more than 2 iterations after previous CTF (or beginning of algo)
        and if residuals diminution is low
        """
        #if self.model.k == 0: return False
        #self.scale = [[e if e>1 else 0 for e in f] for f in self.scale]

        #if all(e==0 for e in sum(self.scale,[])): return False
        if not self.multiscale: return False

        if self.multiscale_momenta: return False

        if self.scale == 0: return False

        if end: return True

        # if CTF just happened: we check that residuals change is enough
        if self.after_ctf(iteration):
            print("after ctf", np.abs(residuals_change(avg_residuals)))
            if np.abs(residuals_change(avg_residuals)) < 0.01:
                return True

        return (self.enough(iteration) and (residuals_change(avg_residuals) < self.convergence_threshold))\
                or (self.too_much(iteration) and residuals_change(avg_residuals) < 0.01)  

    def filter(self, parameters, iteration, new_dataset = None):
        """
        !! Distance computed from normals. Normals computed from new landmark points positions
        (in case of deformed template!)
        Here only the targets are filtered -> need to filter also deformed source
        """
        if new_dataset is None:
            new_dataset = copy.deepcopy(self.original_dataset)

        if self.multiscale:
            if iteration == 0: logger.info("Initialization - filter meshes at scale {}".format(self.scale))

            for i in range(new_dataset.subject_ids):
                for o, observation in enumerate(new_dataset.deformable_objects[i]):
                    for object in observation.object_list: # Surface mesh
                        print("\n Filter subject", i, "object", o, "scale", self.scale)
                        object.set_kernel_scale(self.scale)
                        object.filter_normals()
            object.save(self.output_dir, "Filtered_subject_{}_obs_{}_scale_{}.vtk".format(i, o, object.current_scale))
                
            for object in self.model.template.object_list:
                object.set_kernel_scale(self.scale)
                object.filter_normals()
                if not self.model.freeze_template:
                    object.save(self.output_dir, "Filtered_template_scale_{}.vtk".format(object.current_scale))
    
        return new_dataset, parameters  

    def coarse_to_fine_step(self, parameters, iteration):
        #self.scale = [[max(0,e-1) for e in self.scale[i]] for i in range(len(self.scale))]
        self.scale = max(0, self.scale - self.step)

        self.iter.append(iteration)

        logger.info("\nCoarse to fine on meshes")
        #logger.info("Mesh scales:" + " ".join(str(s) for s in sum(self.scale,[])))
        logger.info("Mesh scales:{}".format(self.scale))

        return self.filter(parameters, iteration)

    def coarse_to_fine(self, parameters, current_dataset, iteration, avg_residuals, end):

        if self.coarse_to_fine_condition(iteration, avg_residuals, end):
            return self.coarse_to_fine_step(parameters, iteration)
        
        print("\n No Coarse to Fine allowed")  

        return current_dataset, parameters

    ####################################################################################################################
    ### Optimization tools
    ####################################################################################################################     
    def enough(self, iteration):
        return iteration - self.iter[-1] > self.ctf_interval
    
    def too_much(self, iteration):
        return iteration - self.iter[-1] > self.ctf_max_interval

    def convergence(self, iteration, cond):
        """
        Check that convergence conditions fulfilled
        """
        if not self.multiscale: return cond

        return cond and (self.scale == 0 and self.enough(iteration))
    
    def after_ctf(self, iteration):
        """
        Check that CTF just happened (to prevent convergence)
        """

        return self.multiscale and (self.iter[-1] == iteration)
        
    def folder_name(self, name):
        if not self.multiscale: return name

        return name + "Mesh_scale_{}".format(self.scale)
    

    ####################################################################################################################
    ### Coarse to fine v2 - grid projection
    ####################################################################################################################        
    # def initialize(self):
    #     """
    #     Coarse to fines on data attachment term (on kernel width)
    #     """
    #     # Kernel width in data attachment function
    #     self.mesh_kernel_width = self.model.multi_object_attachment.kernels[0].kernel_width
    #     # Index of the coarsest kernel width
    #     self.model.k = len(self.model.multi_object_attachments_k) - 1
    #     # Interval between two kernel width
    #     self.step_k = self.model.multi_object_attachments_k[-1].kernels[0].kernel_width \
    #                 - self.model.multi_object_attachments_k[-2].kernels[0].kernel_width
    def initialize_(self):
        """
            Compute coarser scale of each mesh
            - Modifies original dataset: grid projection
            - but doesnt change normals as we need them for recovering original data
            !! in case of RG, template can be= one of observations
        """
        logger.info("\nInitialisation - coarse to fine on meshes")

        self.scale = []

        for i, _ in enumerate(self.original_dataset.subject_ids): 
            max_scales = []
            for o, observation in enumerate(self.original_dataset.deformable_objects[i]):
                for object in observation.object_list: # Surface mesh
                    object.find_normals_max_scale()
                    max_scales.append(min(object.coarser_mesh_scales)-1)
            self.scale.append(max_scales)
        
        max_scales = []
        for object in self.model.template.object_list:
            object.find_normals_max_scale()
            max_scales.append(min(object.coarser_mesh_scales)-1)
        self.scale.append(max_scales)
        
        print("self.scale", self.scale)

    def filter_template_mesh_(self, parameters, iteration):
        """
            Function that filters the template image when it is frozen (ie during registration)
            /!\ this only modifies the template in the model, not the estimator -> works if template not optimized!
        """
        if (iteration == 0 and  self.model_name in ["DeterministicAtlas"])\
        or (self.model_name in ["Registration", "GeodesicRegression"] and self.freeze_template):

            for o, object in enumerate(self.model.template.object_list):
                object.set_scale(self.scale[-1][0])
                haar_filtered_normals = object.haar_transform_normals()[0]
                haar_filtered_normals.save(op.join(self.output_dir, "Template_iter_{}_scale_{}{}".format(iteration, self.scale[-1][0], self.ext)), binary=False) 
            
            # the points=landmark_points= position of vertices
            # we did not modify these.
            # self.model.set_template_data({self.points : points})
            # if self.points in parameters:
            #     parameters[self.points] = points

        return parameters

    def filter_(self, parameters, iteration):
        """
        !! Distance computed from normals. Normals computed from new landmark points positions
        (in case of deformed template!)
        Here only the targets are filtered -> need to filter also deformed source
        """
        new_dataset = copy.deepcopy(self.original_dataset)

        if self.multiscale:
            
            if iteration == 0: logger.info('\nInitial filtering of meshes...\n')

            for i, _ in enumerate(new_dataset.subject_ids): #for each subject
                for o, observation in enumerate(new_dataset.deformable_objects[i]):
                    for object in observation.object_list:
                        print("Subject {} osb {}".format(i, o))
                        object.set_scale(self.scale[i][o])
                        haar_filtered_normals = object.haar_transform_normals()
                        haar_filtered_normals[0].save(op.join(self.output_dir, "Filtered_subject_{}_obs_{}_scale_{}.vtk".format(i, o, object.current_scale)), binary=False) 
                        haar_filtered_normals[1].save(op.join(self.output_dir, "Grid_subject_{}_obs_{}_scale_{}.vtk".format(i, o, object.current_scale)), binary=False) 
        
            parameters = self.filter_template_mesh(parameters, iteration)

        return new_dataset, parameters