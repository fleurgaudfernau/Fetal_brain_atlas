import copy
import logging
import os.path as op
from ...support import utilities
import numpy as np
from ...support.utilities.tools import residuals_change
logger = logging.getLogger(__name__)


class MultiscaleImages():
    def __init__(self, multiscale, multiscale_momenta, model, dataset, output_dir, 
                ctf_interval, ctf_max_interval, points, points_per_axis):
                
        self.model = model
        self.model_name = model.name
        self.original_dataset = copy.deepcopy(dataset)
        self.ext = self.model.objects_extension[0]

        self.output_dir = output_dir

        # Model options
        self.freeze_template = model.freeze_template
        self.deformation_kernel_width = model.deformation_kernel_width

        self.initial_convergence_threshold = 0.001
        self.convergence_threshold = 0.001
        self.ctf_interval = ctf_interval
        self.ctf_max_interval = ctf_max_interval

        self.multiscale = multiscale
        self.multiscale_momenta = multiscale_momenta
        self.iter = [0] if multiscale else []

        self.points = points
        self.step = 0.5 if not self.multiscale_momenta else 1
        self.step = 1
        self.step = 0.5

        self.points_per_axis = points_per_axis
        if model.dimension == 3:
            self.points_per_axis = [self.points_per_axis[1], self.points_per_axis[0], self.points_per_axis[2]]

    def initialize(self, momenta_coarser_scale = None):
        """
            Initialization to perform the multiscale optimization of the template image
            The current image scale is proportional to the current momenta scale
            The template image is filtered according to the current image scale
        """
        if self.multiscale:
            logger.info("\nInitialisation - coarse to fine on images with step {}".format(self.step))
            
            self.momenta_scale = momenta_coarser_scale
            self.momenta_coarser_scale = momenta_coarser_scale
            
            # Compute the momenta corresponding scale
            self.scale = self.compute_scale()

            logger.info("\n Image scale {}".format(self.scale))
    
    def compute_scale(self):
        d = 1 #d = 3 # modif FITNG in self.order to start at a lower scale
        print("self.momenta_scale", self.momenta_scale)
        if self.momenta_scale > d:
            wavelet_support = 2**(self.momenta_scale - d) * self.deformation_kernel_width

            if self.ext == ".nii":
                image_scale = round(wavelet_support/12, 2)  # for brains
            else:
                #image_scale = round(wavelet_support/6, 2) #gaussian filter and bilateral filter
                image_scale = round(wavelet_support/6, 2)  # for brains
        else:
            image_scale = 0
        
        if image_scale == 0:
            return 0
        
        if self.ext == ".vtk": #NB: if filtering too big we can have gradient error: <= 1000
            #image_scale = min(600, int(np.exp(image_scale/2)*60))
            image_scale = min(5000, int((np.exp(image_scale/1.5)-1)*500))

        return image_scale
    

    def coarse_to_fine_condition(self, iteration, avg_residuals, end):
        """
        Authorizes coarse to fine if more than 2 iterations after previous CTF (or beginning of algo)
        and if residuals diminution is low
        """
        if self.multiscale_momenta: return False # dual multiscale handled later

        if not self.multiscale: return False
        
        if self.scale == 0: return False

        if end: return True
        
        return (self.enough(iteration)\
                and (residuals_change(avg_residuals) < self.convergence_threshold))\
                or (self.too_much(iteration)) #and residuals_change(avg_residuals) < 0.01)  

    def save_image(self, intensities, names = "Subject_"):
        device, _ = utilities.get_best_device(gpu_mode=self.model.gpu_mode)
        image_data = self.model._fixed_effects_to_torch_tensors(False, device = device)[0]

        image_data[self.points] = intensities

        names = names + "width_" + str(self.scale) 
        # if self.multiscale_momenta:#!!!!!
        #     names = names + "_momenta_scale_" + str(self.momenta_scale)
        names += self.ext

        self.model.template.write(self.output_dir, [names], {key: value for key, value in image_data.items()})

        del image_data
    

    def filter_template_img(self, parameters, iteration):
        """
            Function that filters the template image when it is frozen (ie during registration)
            /!\ this only modifies the template in the model, not the estimator -> works if template not optimized!
        """
        # only initialize filter for models that optimize template
        # filter template when template is frozen
        # 1st row: to avoid refiltering when template already filtered 
        if (not self.ext == ".vtk" and self.model_name in ["DeterministicAtlas"])\
            and (iteration == 0 and (self.model_name in ["DeterministicAtlas"] or\
            "Regression" in self.model_name)) or self.freeze_template:
            logger.info("Filtering of template...")

            # self.model.template = DeformableMultiObject -> self.model.template.object_list
            for e in self.model.template.object_list:
                points = e.filter(self.scale)

            self.model.set_template_data({self.points : points})
            e.write(self.output_dir, "Template_iter_{}_width_{}{}".format(iteration, self.scale, self.ext))
            
            if self.points in parameters:
                parameters[self.points] = points

        return parameters
    
    def filter(self, parameters, iteration, new_dataset = None):
        if new_dataset is None:
            new_dataset = copy.deepcopy(self.original_dataset)

        if self.multiscale:
            if iteration == 0: logger.info("Initial filtering of objects...")
            
            for i in range(len(new_dataset.subject_ids)):
                #new_dataset.deformable_objects: a list of lists. 1 list/subject
                #new_dataset.deformable_objects[i]: a list of observations for subject i
                for o, observation in enumerate(new_dataset.deformable_objects[i]):
                    for object in observation.object_list:
                        object.filter(self.scale)
            
            object.write(self.output_dir, "Subject_{}_obs_{}_iter_{}_width_{}{}".format(i, o, iteration, self.scale, self.ext))
            
            parameters = self.filter_template_img(parameters, iteration)

        return new_dataset, parameters

    def coarse_to_fine(self, parameters, current_dataset, iteration, avg_residuals, end):
        
        if self.coarse_to_fine_condition(iteration, avg_residuals, end):

            old_scale = self.scale
            while self.scale == old_scale:
                self.momenta_scale = max(0, self.momenta_scale - self.step) 
                self.scale = self.compute_scale()
                if self.multiscale_momenta: break
            self.iter.append(iteration)

            logger.info("*** Coarse to fine on images")
            logger.info("Momenta_scale {} - Image_scale {}".format(self.momenta_scale, self.scale))            
                        
            return self.filter(parameters, iteration)
        
        return current_dataset, parameters
    
    def fetch_current_momenta_scale(self):
        """
            We fetch the momenta scale equivalent to the current image scale
        """
        # self.scale: the current image scale (filter width)
        # self.compute_scale: img scale corresponding to self.momenta_scale
        # 

        print("self.momenta_coarser_scale", self.momenta_coarser_scale)
        print("self.scale", self.scale)
        print("self.momenta_scale", self.momenta_scale)
        scale = self.momenta_coarser_scale

        print("self.compute_scale() ", self.compute_scale())
        while self.momenta_scale > 0 and self.compute_scale() > self.scale:
            print(self.compute_scale())
            self.momenta_scale = max(0, self.momenta_scale-1)
        #return scale

    def dual_coarse_to_fine_step(self, parameters, iteration, step = 1):
        if step > 0:
            logger.info("*** Coarse to fine on images")
        else:
            logger.info("*** Reverse coarse to fine on images")

        # self.fetch_current_momenta_scale() 
        # self.momenta_scale = self.momenta_scale - step # CTF step
        # self.scale = self.compute_scale()
        # self.iter.append(iteration)
        # current_dataset, parameters = self.filter(parameters, iteration)

        self.momenta_scale = max(0, self.momenta_scale - step) 
        self.scale = self.compute_scale()
        self.iter.append(iteration)

        logger.info("Momenta_scale {} - Image_scale {}".format(self.momenta_scale, self.scale))            
                        
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

        return name + "Image_scale_{}".format(self.scale)