import copy
import math
import os
import logging
from decimal import Decimal
import numpy as np
from ...core.estimator_tools.multiscale_meshes import MultiscaleMeshes
from ...core.estimator_tools.multiscale_images import MultiscaleImages
from ...core.estimator_tools.multiscale_momenta import MultiscaleMomenta
logger = logging.getLogger(__name__)

class Gradient():
    def __init__(self):
        pass

    def compute_gradient_norm(self, gradient, key):
        value = gradient[key]
        
        if key == "haar_coef_momenta":
            value = np.concatenate([np.concatenate([array.wc for array in gradient["haar_coef_momenta"][s]])\
                                for s in range(len(gradient["haar_coef_momenta"]))])
        
        value = np.array(math.sqrt(np.sum(value ** 2)))
        value[value == float("inf")] = 0.1

        return float(value)
        
    
    def compute_gradients(self, gradient):
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

            logger.info("Re)-initialize haar_coef_momenta step size")
            steps["haar_coef_momenta"] = 0.1 / gradient_norm if gradient_norm > 1e-8 else 1e-5
            
        else:
            steps["momenta"] = self.reinitialize_step_size(gradient, "momenta")

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

    def momenta_keys(self, gradient, steps):
        if "haar_coef_momenta" in gradient.keys() and "haar_coef_momenta" in steps.keys():
            return ["haar_coef_momenta"]
        elif "momenta" in gradient.keys() and "momenta" in steps.keys():
            return ["momenta"]
    
    def template_keys(self, gradient, steps):
        if "image_intensities" in gradient.keys():
            return ["image_intensities"]
        return []

    def space_shift_keys(self, gradient):
        if "sources" in gradient.keys() and "modulation_matrix" in gradient.keys():
            return ["sources", "modulation_matrix"]
        
        return []


    def all_keys(self, gradient, steps):

        return self.template_keys(gradient, steps) + self.momenta_keys(gradient, steps)


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

                # severe threshold for brains in atlas and regression
                # Registration and Regression
                # MAYBE too severe?? -> NO
                if self.n_subjects == 1:
                    while steps[key] > 1 / self.gradient_norms[key][-1]: # before 0.01
                        steps = self.reduce_step(steps, key)
                # test 12/01: (22h): before was less severe
                elif self.images.scale > 1: # IMPORTANT in Deterministic atlas
                     while steps[key] > 0.1 / self.gradient_norms[key][-1]: #modif before 1
                        steps = self.reduce_step(steps, key)
                elif self.images.scale > 0: # Deterministic atlas
                     while steps[key] > 1 / self.gradient_norms[key][-1]: #modif before 1
                        steps = self.reduce_step(steps, key)
            
                                
        elif self.momenta.multiscale:
            for key in self.momenta_keys(gradient, steps):

                # Registration and Regression
                # good for the brains
                # if self.n_subjects == 1 and steps[key] > 0.01 / self.gradient_norms[key][-1]: # before 1
                #     steps = self.reduce_step(steps, key)
                if self.n_subjects == 1: 
                    while steps[key] > 5 / self.gradient_norms[key][-1]: # before 1
                        steps = self.reduce_step(steps, key)

        return steps 

    def reinitialize_step(self, optimizer, gradient, iteration, steps):
        """
        Coarse to fine steps -> often big increase of the gradients (of momenta)
        After multiscale on template: reinitialize momenta steps
        Ater multiscale momenta: reinitialize momenta and template steps
        """
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
        
        ### Contain some steps during CTF
        optimizer.step = self.contain_step_size(steps, gradient)

        ### Check that steps are big enough (indpt from CTF)
        for key in steps.keys():
            if key in gradient.keys():
                if steps[key] < 1e-6 / self.gradient_norms[key][-1]:
                    logger.info("Reinitializing {} step size".format(key))
                    optimizer.step[key] = self.reinitialize_step_size(gradient, key, scale = True)

        # Ajout fg
        if self.name == "BayesianPiecewiseRegression":
            for key in self.space_shift_keys(gradient):
                while steps[key] > 1e-5 / self.gradient_norms[key][-1]: #modif before 1
                    steps = self.reduce_step(steps, key, factor=1)