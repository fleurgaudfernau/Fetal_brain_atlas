import copy
import math
import os
import logging
from decimal import Decimal
import numpy as np
from ...core.estimator_tools.multiscale_objects import MultiscaleObjects
from ...core.estimator_tools.multiscale_momenta import MultiscaleMomenta
from ...support.utilities.tools import residuals_change

logger = logging.getLogger(__name__)

class Multiscale():
    def __init__(self, multiscale_momenta, multiscale_objects, multiscale_strategy, 
                model, initial_step_size, output_dir, dataset):
        
        # Data information
        self.model = model
        self.name = model.name
        self.n_subjects = dataset.n_subjects
        
        self.initial_step_size = initial_step_size
        self.output_dir = output_dir

        self.points_per_axis = [len(set(list(model.cp[:, k]))) for k in range(model.dimension)]

        ctf_interval = 15
        ctf_max_interval = 80 if "Regression" in self.name else 50
        self.threshold = 1e-3

        self.points = 'image_intensities'

        if 'landmark_points' in self.model.fixed_effects['template_data'].keys():
            self.points = 'landmark_points'

        # Multiscale options
        self.momenta = MultiscaleMomenta(multiscale_momenta, multiscale_objects, model, ctf_interval, 
                                        ctf_max_interval, self.points_per_axis)
        self.objects = MultiscaleObjects(multiscale_objects, multiscale_momenta, model, dataset, output_dir, ctf_interval, 
                                        ctf_max_interval, self.points, self.points_per_axis)
        self.dual = DualMultiscale(model, self.momenta, self.objects, multiscale_strategy, 
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
        new_dataset, parameters = self.objects.filter(parameters, iteration)

        return new_dataset, parameters

    def initialize(self):
        self.momenta.initialize()
        self.objects.initialize(self.momenta.coarser_scale)
        self.dual.initialize()

    ####################################################################################################################
    ### Coarse to fine
    ####################################################################################################################
    def coarse_to_fine(self, new_parameters, dataset, iteration, avg_residuals, 
                       component_residuals = None, end = False): #TODO
        
        dataset, new_parameters = self.momenta.coarse_to_fine(new_parameters, dataset, iteration, avg_residuals, end)
        dataset, new_parameters = self.objects.coarse_to_fine(new_parameters, dataset, iteration, avg_residuals, end)
        dataset, new_parameters = self.dual.coarse_to_fine(new_parameters, dataset, iteration, avg_residuals, end)

        # Ajout fg: MAJ A0 et sources
        if self.contain_A0 and iteration > 10 and residuals_change(avg_residuals) < self.threshold:
            self.contain_A0 = False
            logger.info("Update sources and modulation matrix")

        return dataset, new_parameters

    ####################################################################################################################
    ### OPTIMIZATION TOOLS
    ####################################################################################################################
    def no_convergence_after_ctf(self, iteration):
        if self.objects.after_ctf(iteration - 1) or self.momenta.after_ctf(iteration - 1): 
            print("Do not allow convergence after CTF")
            return True
                        
        return False
        
    def check_convergence_condition(self, iteration):
        """
            Check if the optimization is allowed to end
        """
        cond = True        
        cond = self.momenta.convergence(iteration, cond)
        cond = self.objects.convergence(iteration, cond)
            
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
        gradient = self.momenta.compute_haar_gradient(gradient)
        
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

            elif key == "sources": # Normalize sources (each column of sources)
                sources = new_parameters[key]
                new_parameters[key] = (sources - np.mean(sources, axis=0)) / np.std(sources, axis=0)
            
            # Normalize columns (space shifts of) Modulation Matrix
            # if key == "modulation_matrix":
            #     for s in range(new_parameters[key].shape[1]):
            #         space_shift = new_parameters[key][:, s]
            #         norm = np.linalg.norm(space_shift)
            #         if norm != 0: 
            #             new_parameters[key][:, s] = new_parameters[key][:, s] / norm 

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
        if self.objects.multiscale:
            for key in self.momenta_keys(gradient, steps):#self.momenta_keys(gradient, steps):
                print("Step", steps[key])
                print("1/gradient", 1 / self.gradient_norms[key][-1])

                while steps[key] > 10 / self.gradient_norms[key][-1]: #modif before 1
                    steps = self.reduce_step(steps, key, factor=0.5)

                if self.n_subjects == 1:
                    while steps[key] > 1 / self.gradient_norms[key][-1]: # before 0.01
                        steps = self.reduce_step(steps, key)

                # severe threshold for brains in atlas and regression (important)
                if self.objects.scale > 1 and len(self.template_keys(gradient, steps)) > 0: # IMPORTANT in Deterministic atlas for .nii
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
        ### Reinitialize some steps after CTF
        if self.objects.after_ctf(iteration):
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
        if self.objects.multiscale:
            d["object_scale"]= self.objects.scale
            d["iter_multiscale_objects"] = self.objects.iter
        else:
            d["object_scale"] = 0
            d["iter_multiscale_objects"] = []
        if self.momenta.multiscale:
            d["momenta_scale"] = self.momenta.scale
            d["iter_multiscale_momenta"] = self.momenta.iter
        else:
            d["momenta_scale"] = 0
            d["iter_multiscale_momenta"] = []
        if self.objects.multiscale and self.momenta.multiscale:
            d["order"] = self.dual.order
        else:
            d["order"] = []

        return d

####################################################################################################################
### DUAL MULTISCALE
####################################################################################################################

class DualMultiscale():
    def __init__(self, model, momenta, objects, multiscale_strategy, ctf_interval, 
                ctf_max_interval):
        self.model = model
        self.name = model.name

        self.momenta = momenta
        self.objects = objects
        self.multiscale_momenta_objects = momenta.multiscale and objects.multiscale
        self.multiscale_strategy = multiscale_strategy

        self.order = []

        self.initial_threshold = 1e-3
        self.threshold = 1e-3
        self.ctf_interval = ctf_interval
        self.ctf_max_interval = ctf_max_interval

    def initialize(self):
        """
            Initialization of the dual coarse-to-fine (momenta + objects)
            The multiscale strategy is set (i.e. self.order in which to perform the coarse-to-fine steps)
        """
        if self.multiscale_momenta_objects:
            self.first = "Momenta"

            logger.info("\nDual Multiscale, {} first".format(self.first))
            self.second = ["Object" if self.first != "Object" else "Momenta"][0]
            
            if self.multiscale_strategy == "simultaneous":
                for k in range(self.momenta.coarser_scale - 1):
                    self.step = 1
                    self.order.append({self.first: 1, self.second: 1})
            
            elif self.multiscale_strategy == "separate":
                for k in range(self.momenta.coarser_scale - 1):
                    self.order.append({"Object": 1, "Momenta": None})
                for k in range(self.momenta.coarser_scale - 1):
                    self.order.append({"Object": None, "Momenta": 1})

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

        enough_iterations = (self.momenta.enough(iteration) and objects.enough(iteration))
        too_much = (self.momenta.too_much(iteration) and objects.too_much(iteration))

        return (enough_iterations and residuals_change(avg_residuals) < self.threshold) or too_much

    def coarse_to_fine(self, parameters, current_dataset, iteration, avg_residuals, end):
        if self.multiscale_momenta_objects:
            return self.coarse_to_fine_momenta_objects(parameters, current_dataset, iteration, avg_residuals, end)

        return current_dataset, parameters

    def coarse_to_fine_momenta_objects(self, parameters, current_dataset, iteration, avg_residuals, end):
        """
            Dual coarse to fine between momenta and objects
        """

        if self.coarse_to_fine_condition(iteration, avg_residuals, end, objects = self.objects):
            step_momenta, step_objects = self.order[0]["Momenta"], self.order[0]["Object"]

            if step_momenta:
                self.momenta.coarse_to_fine_step(iteration, step_momenta)
                
            if step_objects:
                current_dataset, parameters = self.objects.dual_coarse_to_fine_step(parameters, iteration, step_objects)                         
                            
            if step_momenta or step_objects:
                self.order = self.order[1:]
        
        print("self.momenta.iter", self.momenta.iter)
        print("self.objects.iter", self.objects.iter)
                        
        return current_dataset, parameters
    