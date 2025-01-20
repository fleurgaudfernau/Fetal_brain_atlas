import copy
import math
import os
import logging
from decimal import Decimal
import numpy as np
from ...core.estimator_tools.multiscale_meshes import MultiscaleMeshes
from ...core.estimator_tools.multiscale_images import MultiscaleImages
from ...core.estimator_tools.multiscale_momenta import MultiscaleMomenta
from ...core.estimator_tools.multiscale_momenta_piecewise import MultiscaleMomentaPiecewise
from ...support.utilities.tools import residuals_change

logger = logging.getLogger(__name__)

class Multiscale():
    def __init__(self, multiscale_momenta, multiscale_images, multiscale_meshes, 
                multiscale_strategy, gamma, naive, model, initial_step_size, 
                scale_initial_step_size, output_dir, dataset, start_scale):
        
        # Data information
        self.model = model
        self.name = model.name
        self.n_subjects = len(dataset.subject_ids)
        
        self.initial_cp_spacing = model.initial_cp_spacing

        self.initial_step_size = initial_step_size
        self.scale_initial_step_size = scale_initial_step_size
        self.output_dir = output_dir

        self.points_per_axis = [len(set(list(model.fixed_effects['control_points'][:, k]))) for k in range(model.dimension)]

        ctf_interval = 15
        ctf_max_interval = 80 if "Regression" in self.name else 50
        self.convergence_threshold = 1e-3

        # Adaptation to meshes or images
        self.points = 'image_intensities'

        if 'landmark_points' in self.model.fixed_effects['template_data'].keys():
            self.points = 'landmark_points'

        if multiscale_momenta and "Piecewise" in self.name:
            multiscale_momenta_piecewise = True
        multiscale_momenta_piecewise = False

        # Multiscale options
        self.momenta = MultiscaleMomenta(multiscale_momenta, multiscale_images, multiscale_meshes, model, ctf_interval, 
                                        ctf_max_interval, self.points_per_axis, start_scale)
        self.momenta_piecewise = MultiscaleMomentaPiecewise(multiscale_momenta_piecewise, multiscale_images, multiscale_meshes, model, ctf_interval, 
                                                            ctf_max_interval, self.points_per_axis)
        self.images = MultiscaleImages(multiscale_images, multiscale_momenta, model, dataset, output_dir, ctf_interval, 
                                        ctf_max_interval, self.points, self.points_per_axis)
        self.meshes = MultiscaleMeshes(multiscale_meshes, multiscale_momenta, model, dataset, output_dir, ctf_interval, 
                                        ctf_max_interval, self.points)
        self.dual = DualMultiscale(model, self.momenta, self.images, self.meshes, multiscale_strategy, 
                                    ctf_interval, ctf_max_interval)
        
        self.model.curvature = False# consider curvature in cost fct (not CTF)

        # Store gradient norm
        self.gradient_norms = {}

        # Indepent from CTF
        self.contain_A0 = False
    
    ####################################################################################################################
    ### INITIALIZE
    ####################################################################################################################
    def filter(self, parameters, iteration):
        """
        Function for the initial filter of parameters (eg template mesh/image)
        """
        new_dataset, parameters = self.images.filter(parameters, iteration)
        new_dataset, parameters = self.meshes.filter(parameters, iteration, new_dataset )

        return new_dataset, parameters

    def initialize(self):
        self.momenta.initialize()
        #self.momenta_piecewise.initialize()
        self.images.initialize(self.momenta.coarser_scale)
        self.meshes.initialize(self.momenta.coarser_scale)
        self.dual.initialize()

    ####################################################################################################################
    ### Coarse to fine
    ####################################################################################################################
    def coarse_to_fine(self, new_parameters, dataset, iteration, avg_residuals, 
                       component_residuals = None, end = False): #TODO
        
        dataset, new_parameters = self.momenta.coarse_to_fine(new_parameters, dataset, iteration, avg_residuals, end)
        #dataset, new_parameters = self.momenta_piecewise.coarse_to_fine(new_parameters, dataset, iteration, component_residuals, end)
        dataset, new_parameters = self.images.coarse_to_fine(new_parameters, dataset, iteration, avg_residuals, end)
        dataset, new_parameters = self.dual.coarse_to_fine(new_parameters, dataset, iteration, avg_residuals, end)
        dataset, new_parameters = self.meshes.coarse_to_fine(new_parameters, dataset, iteration, avg_residuals, end)

        # Ajout fg: MAJ A0 et sources
        if self.contain_A0 and iteration > 10 and residuals_change(avg_residuals) < self.convergence_threshold:
            self.contain_A0 = False
            logger.info("Update sources and modulation matrix")

        return dataset, new_parameters

    ####################################################################################################################
    ### OPTIMIZATION TOOLS
    ####################################################################################################################
    def no_convergence_after_ctf(self, iteration):
        if self.images.after_ctf(iteration - 1) or self.meshes.after_ctf(iteration - 1) \
            or self.momenta.after_ctf(iteration - 1): #or self.momenta_piecewise.after_ctf(iteration - 1):
            print("Do not allow convergence after CTF")
            return True
                        
        return False
    
    def save_model_after_ctf(self, iteration):
        """
        Create a folder to save the model state after each CTF step
        """
        return 
        if self.name == "GeodesicRegression" \
            and (self.images.after_ctf(iteration) or self.meshes.after_ctf(iteration) \
            or self.momenta.after_ctf(iteration))\
            and iteration > 1: 
            name = "Iter_{}_".format(iteration)
            name = self.momenta.folder_name(name)
            name = self.images.folder_name(name)
            name = self.meshes.folder_name(name)

            output_dir = os.path.join(self.output_dir, name)
            if not os.path.exists(output_dir): os.mkdir(output_dir)
            
            return output_dir
        
        return
    
    def check_convergence_condition(self, iteration):
        """
            Check if the optimization is allowed to end
        """
        cond = True        
        cond = self.momenta.convergence(iteration, cond)
        cond = self.images.convergence(iteration, cond)
        cond = self.meshes.convergence(iteration, cond)

        print("check_convergence_condition", cond)
        print(self.momenta.convergence(iteration, cond))
            
        return cond
    
    ####################################################################################################################
    ### GRADIENT TOOLS
    ####################################################################################################################
    def momenta_keys(self, gradient, steps):
        if "haar_coef_momenta" in gradient.keys() and "haar_coef_momenta" in steps.keys():
            return ["haar_coef_momenta"]
        elif "momenta" in gradient.keys() and "momenta" in steps.keys():
            return ["momenta"]
        
        return []
    
    def template_keys(self, gradient, steps):
        if "image_intensities" in gradient.keys():
            return ["image_intensities"]
        elif "landmark_points" in gradient.keys():
            return ["landmark_points"]
        return []

    def space_shift_keys(self, gradient):
        if "sources" in gradient.keys() and "modulation_matrix" in gradient.keys():
            return ["sources", "modulation_matrix"]
        
        return []

    def all_keys(self, gradient, steps):

        return self.template_keys(gradient, steps) + self.momenta_keys(gradient, steps)
    
    def compute_gradients(self, gradient):
        # Compute an additional gradient: the WT of the momenta gradient
        # And silence some coefficients (actual coarse to fine)
        gradient = self.momenta.compute_haar_gradient(gradient)
        #gradient = self.momenta_piecewise.compute_haar_gradient(gradient)
        
        # Store gradient norms
        if not self.gradient_norms:
            self.gradient_norms = {k : [] for k in gradient.keys()}
        for k in gradient.keys():
            self.gradient_norms[k].append(self.compute_gradient_norm(gradient, k))

        return gradient

    def regular_gradient_ascent(self, new_parameters, gradient, step, exclude = []):
        for key in [g for g in gradient.keys() if g not in exclude]:
            new_parameters[key] += gradient[key] * step[key]


            # Avoir negative intensities for images
            if key == "image_intensities":
                new_parameters[key][new_parameters[key] < 0] = 0
            
            # Normalize columns (space shifts of) Modulation Matrix
            # if key == "modulation_matrix":
            #     for s in range(new_parameters[key].shape[1]):
            #         space_shift = new_parameters[key][:, s]
            #         norm = np.linalg.norm(space_shift)
            #         if norm != 0: 
            #             new_parameters[key][:, s] = new_parameters[key][:, s] / norm 

            # Normalize sources
            if key == "sources":
                sources = new_parameters[key]
                for s in range(sources.shape[1]):
                    mean = np.mean(sources[:, s])
                    std = np.std(sources[:, s])
                    new_parameters[key][:,s] = (new_parameters[key][:, s] - mean) / std

        return new_parameters

    def gradient_ascent(self, parameters, gradient, step):
        new_parameters = copy.deepcopy(parameters)

        # Multiscale gradient ascent
        if self.momenta.ctf_is_happening():
            # Update other parameters
            new_parameters = self.regular_gradient_ascent(new_parameters, gradient, step, \
                                                        exclude = ["momenta", "haar_coef_momenta"])
            new_parameters = self.momenta.gradient_ascent(parameters, new_parameters, gradient, step) 
            
            return new_parameters

        # elif self.momenta_piecewise.ctf_is_happening():
        #     # Update other parameters
        #     new_parameters = self.regular_gradient_ascent(new_parameters, gradient, step, \
        #                                                 exclude = ["momenta", "haar_coef_momenta"])
        #     new_parameters = self.momenta_piecewise.gradient_ascent(parameters, new_parameters, gradient, step) 
            
        #     return new_parameters
        
        # Regular gradient ascent
        return self.regular_gradient_ascent(new_parameters, gradient, step)
    
    ####################################################################################################################
    ### STEPS
    ####################################################################################################################    
    def reinitialize_step_size(self, gradient, key, scale = False):
        """
            Reinitialize step size if too small or too large
        """
        logger.info("(Re)-initialize_step_size of {}".format(key))

        if self.scale_initial_step_size:
            value = gradient[key]
            gradient_norm = math.sqrt(np.sum(value ** 2))

            if math.isinf(gradient_norm):
                return 1e-10

            if scale:
                return self.initial_step_size / gradient_norm
            
            return 1 / gradient_norm

    def initialize_momenta_step(self, steps, gradient, optimizer, iteration):
        if self.momenta.ctf_is_happening(): #gradient[haar_coef_momenta] = [[haar_d1, haar_d2, haar_d3] for each subj]
            
            gradient_norm = self.compute_gradient_norm(gradient, "haar_coef_momenta")
                        
            if iteration > 1 and steps["haar_coef_momenta"] < 0.1 / gradient_norm\
                and steps["haar_coef_momenta"] > 0.01 / gradient_norm:
                logger.info("No need to reinitialize ")
                return steps

            logger.info("(Re)-initialize haar_coef_momenta step size")
            steps["haar_coef_momenta"] = 0.1 / gradient_norm if gradient_norm > 1e-8 else 1e-5
            #steps["haar_coef_momenta"] = self.initial_step_size / gradient_norm if gradient_norm > 1e-8 else 1e-5
            
        # else:
        #     steps["momenta"] = self.reinitialize_step_size(gradient, "momenta")

        return steps  

    def reduce_step(self, steps, key, factor =0.1):
        logger.info("Contain step size of {}".format(key))
        steps[key] = factor * steps[key] 

        return steps
    
    def compute_gradient_norm(self, gradient, key):
        value = gradient[key]
        
        if key == "haar_coef_momenta":
            value = np.concatenate([np.concatenate([array.wc for array in gradient["haar_coef_momenta"][s]])\
                                for s in range(len(gradient["haar_coef_momenta"]))])
        
        value = np.array(math.sqrt(np.sum(value ** 2)))
        value[value == float("inf")] = 0.1

        return float(value)

    

    def handle_other_steps(self, steps, gradient):
        ### Check that steps are big enough (indpt from CTF)
        # for key in steps.keys():
        #     if key in gradient.keys():
        #         if steps[key] < 1e-6 / self.gradient_norms[key][-1]:
        #             logger.info("Reinitializing {} step size".format(key))
        #             steps[key] = self.reinitialize_step_size(gradient, key)

        # Ajout fg: contain gradient of sources and A0
        # if self.name == "BayesianGeodesicRegression":
        #     if self.contain_A0:
        #         for key in self.space_shift_keys(gradient):
        #             while steps[key] > 1e-3 / self.gradient_norms[key][-1]: #modif before 1
        #                 steps = self.reduce_step(steps, key, factor=0.01)
        #     else:
        #         for key in self.space_shift_keys(gradient):
        #             if steps[key] < 1e-3 / self.gradient_norms[key][-1]: #modif before 1
        #                 steps[key] = 10*self.reinitialize_step_size(gradient, key)
        #                 logger.info("Reinitialize step of {}".format(key))

        return steps

    def contain_step_size(self, steps, gradient):
        """
        Prevents the step size to increase too much
        """
        if self.meshes.multiscale:
            for key in self.momenta_keys(gradient, steps):
                while steps[key] > 10 / self.gradient_norms[key][-1]: #modif before 1
                    steps = self.reduce_step(steps, key, factor=0.5)
                if steps[key] > 1 / self.gradient_norms[key][-1]: #modif before 1
                    steps = self.reduce_step(steps, key, factor=0.5)

        elif self.images.multiscale:
            for key in self.momenta_keys(gradient, steps):#self.momenta_keys(gradient, steps):
                while steps[key] > 10 / self.gradient_norms[key][-1]: #modif before 1
                    steps = self.reduce_step(steps, key, factor=0.5)
                # if steps[key] > 1 / self.gradient_norms[key][-1]: #modif before 1
                #     steps = self.reduce_step(steps, key, factor=0.5)

                # severe threshold for brains in atlas and regression
                # Registration and Regression -- MAYBE too severe?? -> NO
                if self.images.scale > 1 and len(self.template_keys(gradient, steps)) > 0: # IMPORTANT in Deterministic atlas for .nii
                     while steps[key] > 0.1 / self.gradient_norms[key][-1]: #modif before 1
                        steps = self.reduce_step(steps, key)

                                
        elif self.momenta.multiscale:
            for key in self.momenta_keys(gradient, steps):
                while steps[key] > 10 / self.gradient_norms[key][-1]: #modif before 1
                    steps = self.reduce_step(steps, key, factor=0.5)
                # Registration and Regression
                # good for the brains
                # if self.n_subjects == 1 and steps[key] > 0.01 / self.gradient_norms[key][-1]: # before 1
                #     steps = self.reduce_step(steps, key)
                if self.n_subjects == 1 and len(self.template_keys(gradient, steps)) > 0: 
                    while steps[key] > 1 / self.gradient_norms[key][-1]: # before 1
                        steps = self.reduce_step(steps, key)

        return steps 

    def reinitialize_step(self, optimizer, gradient, iteration, steps):
        """
        Coarse to fine steps -> often big increase of the gradients (of momenta)
        After multiscale on template: reinitialize momenta steps
        Ater multiscale momenta: reinitialize momenta and template steps
        """
        #TODO test
        # if self.images.multiscale and iteration == 0:
        #     for key in self.template_keys(gradient, steps):
        #             optimizer.step[key] *= 10

        ### Reinitialize some steps after CTF
        if self.images.after_ctf(iteration) or self.meshes.after_ctf(iteration):
            if self.name != "Registration": #do not reinitialize in registration
                 optimizer.step = self.initialize_momenta_step(steps, gradient, optimizer, iteration)
            
            for key in self.template_keys(gradient, steps):
                if steps[key] > 1e2 / self.gradient_norms[key][-1] \
                    or steps[key] < 1e-3 / self.gradient_norms[key][-1]: #modif before 1
                        optimizer.step[key] = self.reinitialize_step_size(gradient, key)

        # useful in RG with brains
        elif self.momenta.after_ctf(iteration):
            if "image_intensities" in gradient.keys(): #do not reinitialize in registration
                optimizer.step = self.initialize_momenta_step(steps, gradient, optimizer, iteration)
                if 'Atlas' not in self.name:
                    optimizer.step["image_intensities"] = self.reinitialize_step_size(gradient, key = "image_intensities")
            
            elif "Regression" in self.name: #when template frozen in regression
                optimizer.step = self.initialize_momenta_step(steps, gradient, optimizer, iteration)
            else:
                optimizer.step = self.initialize_momenta_step(steps, gradient, optimizer, iteration)
        
        # elif self.momenta_piecewise.after_ctf(iteration):
        #     optimizer.step = self.initialize_momenta_step(steps, gradient, optimizer, iteration)

        ### Contain some steps during CTF
        optimizer.step = self.contain_step_size(steps, gradient)

        optimizer.step = self.handle_other_steps(steps, gradient)


    ####################################################################################################################
    ### PRINT AND WRITE
    ####################################################################################################################    
    def info(self, steps, gradient):
        logger.info(">> Step size and gradient norm: ")
        for key in [s for s in steps.keys() if s in gradient.keys()]:
            logger.info("\t\t%.3E   and   %.3E \t[ %s ]" % (Decimal(str(steps[key])),
                Decimal(str(self.compute_gradient_norm(gradient, key))), key))

    def dump_state_file(self, d):
        if self.images.multiscale:
            d["image_scale"]= self.images.scale
            d["iter_multiscale_images"] = self.images.iter
        else:
            d["image_scale"] = 0
            d["iter_multiscale_images"] = []
        if self.momenta.multiscale:
            d["momenta_scale"] = self.momenta.scale
            d["iter_multiscale_momenta"] = self.momenta.iter
        else:
            d["momenta_scale"] = 0
            d["iter_multiscale_momenta"] = []
        if self.images.multiscale and self.momenta.multiscale:
            d["order"] = self.dual.order
        else:
            d["order"] = []

        return d

####################################################################################################################
### DUAL MULTISCALE
####################################################################################################################

class DualMultiscale():
    def __init__(self, model, momenta, images, meshes, multiscale_strategy, ctf_interval, 
                ctf_max_interval):
        self.model = model
        self.name = model.name
        self.ext = self.model.objects_name_extension[0]

        self.momenta = momenta
        self.images = images
        self.meshes = meshes
        self.multiscale_momenta_images = momenta.multiscale and images.multiscale
        self.multiscale_momenta_meshes = momenta.multiscale and meshes.multiscale
        self.multiscale_strategy = multiscale_strategy

        self.order = []

        self.initial_convergence_threshold = 1e-3
        self.convergence_threshold = 1e-3
        self.ctf_interval = ctf_interval
        self.ctf_max_interval = ctf_max_interval

    def initialize(self):
        """
            Initialization of the dual coarse-to-fine (momenta + images)
            The multiscale strategy is set (i.e. self.order in which to perform the coarse-to-fine steps)
        """
        if self.multiscale_momenta_images or self.multiscale_momenta_meshes:
            self.first = "Image"
            self.first = "Momenta"
            logger.info("\nDual Multiscale, {} first".format(self.first))
            self.second = ["Image" if self.first != "Image" else "Momenta"][0]
            
            if self.multiscale_strategy == "simultaneous":
                for k in range(self.momenta.coarser_scale - 1):
                    self.step = 1
                    self.order.append({self.first: 1, self.second: 1})

            elif self.multiscale_strategy == "mountains":
                for k in range(self.momenta.coarser_scale - 1):
                    self.step = 1
                    self.order = self.order + [{self.first: 1, self.second: None}] * (self.momenta.coarser_scale - 1 - k)
                    self.order.append({self.first: k - self.momenta.coarser_scale + 2, self.second: 1})
            
            elif self.multiscale_strategy == "separate":
                for k in range(self.momenta.coarser_scale - 1):
                    self.order.append({"Image": 1, "Momenta": None})
                for k in range(self.momenta.coarser_scale - 1):
                    self.order.append({"Image": None, "Momenta": 1})

                self.ctf_max_interval = 50 # to avoid rapid transitions
                self.ctf_interval = 30

            elif self.multiscale_strategy == "unbalanced_stairs": #unbalanced_stairs
                self.order.append({self.first: 1, self.second: None}) 
                self.order.append({self.first: -1, self.second: 1}) 

                for k in range(self.momenta.coarser_scale - 2):
                    #go down
                    self.order.append({self.first: 1, self.second: None}) 
                    self.order.append({self.first: 1, self.second: None}) 
                    #go up
                    self.order.append({self.first: -1, self.second: 1}) 
                
                self.order.append({self.first: 1, self.second: None}) 
                self.order.append({self.first: None, self.second: 1}) 

            else: #stair like strategy
                for k in range(self.momenta.coarser_scale - 1):
                    self.step = 1
                    self.order.append({self.first: 1, self.second: None})
                    self.order.append({self.first: None, self.second: 1})
        
            logger.info("Chosen multiscale strategy: {}".format(self.multiscale_strategy))
            
            
            
    def coarse_to_fine_condition(self, iteration, avg_residuals, end, objects):
        """
        Authorizes coarse to fine if more than 2 iterations after previous CTF (or beginning of algo)
        and if residuals diminution is low
        """
        if not (self.order): return False
            
        if end: return True

        # if end and residuals_change(avg_residuals) < self.convergence_threshold: 
        #     return True

        enough_iterations = (self.momenta.enough(iteration) and objects.enough(iteration))
        too_much = (self.momenta.too_much(iteration) and objects.too_much(iteration))

        return (enough_iterations and residuals_change(avg_residuals) < self.convergence_threshold)\
                or too_much

    def coarse_to_fine(self, parameters, current_dataset, iteration, avg_residuals, end):
        if self.multiscale_momenta_images:
            return self.coarse_to_fine_momenta_images(parameters, current_dataset, iteration, avg_residuals, end)
        if self.multiscale_momenta_meshes:
            return self.coarse_to_fine_momenta_meshes(parameters, current_dataset, iteration, avg_residuals, end)

        return current_dataset, parameters

    def coarse_to_fine_momenta_images(self, parameters, current_dataset, iteration, avg_residuals, end):
        """
            Dual coarse to fine between momenta and images
        """

        if self.coarse_to_fine_condition(iteration, avg_residuals, end, objects = self.images):
            step_momenta, step_images = self.order[0]["Momenta"], self.order[0]["Image"]

            if step_momenta:
                self.momenta.coarse_to_fine_step(iteration, step_momenta)
                
            if step_images:
                current_dataset, parameters = self.images.dual_coarse_to_fine_step(parameters, iteration, step_images)                         
                            
            if step_momenta or step_images:
                self.order = self.order[1:]
        
        print("self.momenta.iter", self.momenta.iter)
        print("self.images.iter", self.images.iter)
                        
        return current_dataset, parameters
    
    def coarse_to_fine_momenta_meshes(self, parameters, current_dataset, iteration, avg_residuals, end):
        """
            Dual coarse to fine between momenta and meshes
        """
        print("self.momenta.iter", self.momenta.iter)
        print("self.meshes.iter", self.meshes.iter)

        if self.coarse_to_fine_condition(iteration, avg_residuals, end, objects = self.meshes):
            step_momenta, step_images = self.order[0]["Momenta"], self.order[0]["Image"]

            if step_momenta:
                self.momenta.coarse_to_fine_step(iteration, step_momenta)
                
            if step_images:
                current_dataset, parameters = self.meshes.coarse_to_fine_step(parameters, iteration)                         
                            
            if step_momenta or step_images:
                self.order = self.order[1:]
                        
        return current_dataset, parameters
    