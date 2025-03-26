import copy
import logging
import os.path as op
from ...support import utilities
import numpy as np
from ...support.utilities.tools import residuals_change
logger = logging.getLogger(__name__)


class MultiscaleObjects():
    def __init__(self, multiscale, multiscale_momenta, model, dataset, output_dir, 
                ctf_interval, ctf_max_interval, points, points_per_axis):
                
        self.model = model
        self.model_name = model.name
        self.original_dataset = copy.deepcopy(dataset)
        self.type = self.model.template.object_list[0].type

        self.output_dir = output_dir

        # Model options
        self.freeze_template = model.freeze_template
        self.deformation_kernel_width = model.deformation_kernel_width

        self.threshold = 0.001
        self.ctf_interval = ctf_interval
        self.ctf_max_interval = ctf_max_interval

        self.multiscale = multiscale
        self.multiscale_momenta = multiscale_momenta
        self.iter = [0] if multiscale else []

        self.points = points
        self.step = 0.5 if not self.multiscale_momenta else 1
        self.step = 1
        #self.step = 0.5

        self.points_per_axis = points_per_axis
        if model.dimension == 3:
            self.points_per_axis = [self.points_per_axis[1], self.points_per_axis[0], self.points_per_axis[2]]

    def initialize(self, momenta_coarser_scale = None):
        """
            Initialization to perform the multiscale optimization of the template object
            The current object scale is proportional to the current momenta scale
            The template object is filtered according to the current object scale
        """
        if self.multiscale:
            logger.info("\n** Initialisation - coarse to fine on objects with step {} **".format(self.step))
            
            self.momenta_scale = momenta_coarser_scale
            self.momenta_coarser_scale = momenta_coarser_scale
            
            # Compute the momenta corresponding scale
            self.scale = self.compute_scale()

            logger.info("Object filtering scale={} (corresp. momenta scale: {})".format(self.scale, self.momenta_scale))
    
    def compute_scale(self):
        d = 1 #d = 3 # modif FITNG in self.order to start at a lower scale
        
        if self.momenta_scale > d:
            wavelet_support = 2**(self.momenta_scale - d) * self.deformation_kernel_width

            if self.type == "Image":
                object_scale = round(wavelet_support/12, 2)  # for brains
            else:
                object_scale = round(wavelet_support/6, 2)  # for brains
        else:
            object_scale = 0
        
        if object_scale == 0:
            return 0
        
        if self.type == "SurfaceMesh": #NB: if filtering too big we can have gradient error: <= 1000
            object_scale = min(5000, int((np.exp(object_scale/1.5)-1) * 500))

        return object_scale
    

    def coarse_to_fine_condition(self, iteration, avg_residuals, end):
        """
        Authorizes coarse to fine if more than 2 iterations after previous CTF (or beginning of algo)
        and if residuals diminution is low
        """
        if self.multiscale_momenta: return False # dual multiscale handled later

        if not self.multiscale: return False
        
        if self.scale == 0: return False

        if end: return True
        
        if (self.enough(iteration) and (residuals_change(avg_residuals) < self.threshold)):
            logger.info("\n Residual change {} < Multiscale threshold ({})"\
                        .format(residuals_change(avg_residuals), self.threshold))
        
        return (self.enough(iteration) and (residuals_change(avg_residuals) < self.threshold))\
                or (self.too_much(iteration))
    
    def filter_template(self, parameters, iteration):
        """
            Function that filters the template object when it is frozen (ie during registration)
            /!\ this only modifies the template in the model, not the estimator -> works if template not optimized!
        """
        # only initialize filter for models that optimize template
        # filter template when template is frozen
        # 1st row: to avoid refiltering when template already filtered 
        if (self.type == "Image" and self.model_name in ["DeterministicAtlas"])\
            and (iteration == 0 and (self.model_name in ["DeterministicAtlas"] or\
            "Regression" in self.model_name)) or self.freeze_template:
            logger.info("Filtering of template...")

            for e in self.model.template.object_list:
                points = e.filter(self.scale)

            self.model.set_template_data({self.points : points})
            e.write(self.output_dir, "Template_iter_{}_width_{}".format(iteration, self.scale))
            
            if self.points in parameters:
                parameters[self.points] = points

        return parameters
    
    def filter(self, parameters, iteration, new_dataset = None):
        if new_dataset is None:
            new_dataset = copy.deepcopy(self.original_dataset)

        if self.multiscale:
            if iteration == 0: logger.info("Initial filtering of objects...")
            
            for i in range(new_dataset.n_subjects):
                print("\t Object {}/{} done".format(i + 1, new_dataset.n_subjects))
                for observation in new_dataset.objects[i]:
                    for object in observation.object_list:
                        object.filter(self.scale)
            
            object.write(self.output_dir, "Subject_{}_iter_{}_width_{}".format(i, iteration, self.scale))
            
            parameters = self.filter_template(parameters, iteration)

        return new_dataset, parameters

    def coarse_to_fine(self, parameters, current_dataset, iteration, avg_residuals, end):
        
        if self.coarse_to_fine_condition(iteration, avg_residuals, end):

            old_scale = self.scale
            while self.scale == old_scale:
                self.momenta_scale = max(0, self.momenta_scale - self.step) 
                self.scale = self.compute_scale()
                if self.multiscale_momenta: break
            self.iter.append(iteration)

            logger.info("*** Coarse to fine on objects")
            logger.info("\t Momenta_scale {} - object_scale {}".format(self.momenta_scale, self.scale))            
                        
            return self.filter(parameters, iteration)
        
        return current_dataset, parameters
    
    def dual_coarse_to_fine_step(self, parameters, iteration, step = 1):
        if step > 0:
            logger.info("*** Coarse to fine on objects")
        else:
            logger.info("*** Reverse coarse to fine on objects")

        self.momenta_scale = max(0, self.momenta_scale - step) 
        self.scale = self.compute_scale()
        self.iter.append(iteration)

        logger.info("Momenta_scale {} - object_scale {}".format(self.momenta_scale, self.scale))            
                        
        current_dataset, parameters = self.filter(parameters, iteration)

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

        return name + "Object_scale_{}".format(self.scale)