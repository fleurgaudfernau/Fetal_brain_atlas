import copy
import logging
import torch
import itertools
import os.path as op
from math import floor
from ...support import utilities
import numpy as np
from ...core import GpuMode
from ...support.utilities import move_data, detach
from ...support.utilities.tools import residuals_change
from ...support.utilities.wavelets import haar_backward_transpose
from ...support import kernels as kernel_factory

logger = logging.getLogger(__name__)

class MultiscaleMomenta():
    def __init__(self, multiscale, multiscale_objects, model, ctf_interval, 
                ctf_max_interval, points_per_axis):
                
        self.model = model
        self.model_name = model.name
        self.n_momenta = model.get_momenta().shape[0]
        self.n_cp = model.get_momenta().shape[1]
        self.dimension = model.dimension

        # Model options
        self.freeze_template = model.freeze_template
        
        self.threshold = 1e-3
        self.ctf_interval = ctf_interval
        self.ctf_max_interval = ctf_max_interval

        self.multiscale = multiscale
        self.multiscale_wavelet = False
        self.smooth = multiscale

        self.multiscale_objects = multiscale_objects
        self.iter = [0] if multiscale else []

        self.coarser_scale = None 

        self.points_per_axis = points_per_axis
        
        if self.dimension == 3:
            self.points_per_axis = [self.points_per_axis[1], self.points_per_axis[0], self.points_per_axis[2]]

    def initialize(self):
        """
            Initialization to perform the multiscale optimization of the momentum vectors
            Initializes the current scale as the maximum scale 
        """
        # This is used to initialize the other multiscales
        haar_coef_momenta = self.compute_haar_transform(self.model.fixed_effects['momenta'])
        self.coarser_scale = haar_coef_momenta[0][0].J - 1
        self.scale = 0

        if self.multiscale_wavelet:
            self.scale = haar_coef_momenta[0][0].J

            logger.info("\n** Initialisation - multiscale momenta at scale {}".format(self.scale))  
        
        if self.smooth:
            self.scale = self.model.deformation_kernel_width * 3
            self.smoothing_kernel = kernel_factory.factory('keops', kernel_width = self.scale)  
            logger.info("\n** Initialisation - smoothing momenta at scale {} with gaussian kernel".format(self.scale)) 
                
    def compute_haar_transform(self, momenta):
        if len(momenta.shape) == 2:
            momenta = momenta[np.newaxis, :, :]
        
        haar_coef_momenta = [[haar_backward_transpose(momenta[s, :, d].reshape(tuple(self.points_per_axis))) \
                            for d in range(self.dimension)] for s in range(len(momenta))]
        
        return haar_coef_momenta
    
    ####################################################################################################################
    ### Coarse to fine on momenta
    ####################################################################################################################
    def coarse_to_fine_condition(self, iteration, avg_residuals, end):
        """
        Authorizes coarse to fine if more than 2 iterations after previous CTF (or beginning of algo)
        and if residuals diminution is low
        """
        if self.multiscale_objects: return False # dual multiscale handled later
        
        if self.scale <= 1: return False

        if not self.enough(iteration): return False
        
        if end: return True

        print("residuals_change()", residuals_change(avg_residuals))
        print("self.threshold", self.threshold)
                
        return self.too_much(iteration) or residuals_change(avg_residuals) < self.threshold
            
    def coarse_to_fine_step(self, iteration, step = 1):
        if self.multiscale_wavelet:
            self.scale = max(1, self.scale - step)

        if self.smooth:
            self.scale = max(1, floor(self.scale * 0.5))
            self.smoothing_kernel = kernel_factory.factory('keops', kernel_width = self.scale)  
        
        self.iter.append(iteration)

        logger.info(">> Coarse to fine on momenta: new scale={}".format(self.scale))    
        
    def coarse_to_fine(self, new_parameters, current_dataset, iteration, avg_residuals, end):

        if self.coarse_to_fine_condition(iteration, avg_residuals, end):
            self.coarse_to_fine_step(iteration)
        
        return current_dataset, new_parameters
    
    ####################################################################################################################
    ### Gradient tools
    ####################################################################################################################         
    def compute_haar_momenta_gradient(self, gradient):
        """
            Wavelet transform gradient of momenta, Set to 0 fine-scale coefficients
        """
        if len(gradient['momenta'].shape) == 2:
            gradient['momenta'] = gradient['momenta'][np.newaxis, :, :]

        gradient["haar_coef_momenta"] = []
        for s, momenta_gradient in enumerate(gradient['momenta']): #for each subject
            subject_gradient = []

            for d in range(self.dimension):
                gradient_of_momenta = momenta_gradient[:, d].reshape(tuple(self.points_per_axis))
                                
                subject_gradient.append(haar_backward_transpose(gradient_of_momenta))

            gradient["haar_coef_momenta"].append(subject_gradient)

        return gradient 

    
    def compute_haar_gradient(self, gradient):
        """
            Compute haar transform of the gradient of the momenta
        """
        if self.multiscale_wavelet:
            gradient = self.compute_haar_momenta_gradient(gradient)
        
        return gradient
    
    def gradient_ascent(self, parameters, new_parameters, gradient, step):
        """
            Wavelet transform the momenta, update the momenta with wavelet-transformed gradient
            Backward transform the momenta
        """
        if self.multiscale_wavelet:
            haar_coef_momenta = self.compute_haar_transform(parameters['momenta'])

            for c in range(self.n_momenta): 
                for d in range(self.dimension):
                    # Haar transform of momenta update
                    haar_coef_momenta[c][d].wc += gradient["haar_coef_momenta"][c][d].wc * step["haar_coef_momenta"]
                    
                    # Momenta update
                    new_parameters['momenta'][c][d] = haar_coef_momenta[c][d].haar_backward().flatten()
                
        return new_parameters
    
    
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

        return self.scale <= 1 and self.enough(iteration)
        
    def after_ctf(self, iteration):
        """
        Check that CTF just happened (to prevent convergence)
        """
        return self.multiscale and (self.iter[-1] == iteration)
        
    ####################################################################################################################
    ### Smooth gradient
    ####################################################################################################################     

    def smooth_gradient(self, gradient):
        if self.smooth:
            cp = move_data(self.model.cp, device = self.model.device)
            ones = torch.ones_like(cp)
            sum_kernel_along_rows = self.smoothing_kernel.convolve(cp, cp, ones)

            if len(gradient['momenta'].shape) == 2:
                gradient['momenta'] = gradient['momenta'][np.newaxis, :, :]

            for s in range(gradient['momenta'].shape[0]):
                momenta = move_data(gradient["momenta"][s], device=self.model.device)
                smoothed = self.smoothing_kernel.convolve(cp, cp, momenta)       
                gradient["momenta"][s] = detach(smoothed / sum_kernel_along_rows)
            
            # Symetry properties
            # LR symetry is around X = 5,7... 
    

        if "regression" in self.model_name.lower() and "kernel" not in self.model_name.lower():
            cp = move_data(self.model.cp, device = self.model.device)
            try:
                n_comp = len(self.model.fixed_effects['rupture_time']) + 1
            except:
                return gradient
                
            for comp, t in enumerate(self.model.fixed_effects['rupture_time']):
                if t < 30 and comp < n_comp - 1:
                    momenta = move_data(gradient["momenta"][comp], device=self.model.device)
                    smoothed = self.smoothing_kernel.convolve(cp, cp, momenta)  
                    
                    # normalize 
                    ones = torch.from_numpy(np.ones(momenta.shape)).float()
                    sum_kernel_along_rows = self.smoothing_kernel.convolve(cp, cp, ones)
                    gradient["momenta"][comp] = detach(smoothed) / sum_kernel_along_rows.numpy()

        return gradient  
