import copy
import logging
import itertools
import os.path as op
from ...support import utilities
import numpy as np
from ...core import GpuMode
import torch
from ...support.utilities.tools import residuals_change
from ...support.utilities.wavelets import haar_backward_transpose
from ...support import kernels as kernel_factory

logger = logging.getLogger(__name__)

class MultiscaleMomenta():
    def __init__(self, multiscale, multiscale_images, multiscale_meshes, model, 
                ctf_interval, ctf_max_interval,
                points_per_axis):
                
        self.model = model
        self.model_name = model.name
        self.n_momenta = model.get_momenta().shape[0]
        self.n_cp = model.get_momenta().shape[1]
        self.dimension = model.dimension

        # Model options
        self.freeze_template = model.freeze_template
        
        self.initial_convergence_threshold = 0.001
        self.convergence_threshold = 0.001
        self.ctf_interval = ctf_interval
        self.ctf_max_interval = ctf_max_interval

        self.multiscale = multiscale
        self.multiscale_images = multiscale_images
        self.multiscale_meshes = multiscale_meshes
        self.iter = [0] if multiscale else []
        self.zones = dict()
        self.silent_haar_coef_momenta_subjects = dict()

        self.smoothing_kernel = kernel_factory.factory('keops', kernel_width=3.5)
        self.smoothing_kernel_2 = kernel_factory.factory('keops', kernel_width=2.5)

        # if we want to stop before scale 0
        self.end_scale = 1

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

        if self.multiscale:
            self.scale = haar_coef_momenta[0][0].J

            logger.info("\nInitialisation - coarse to fine on momenta at scale {}".format(self.scale))    
            logger.info("Momenta end scale {}".format(self.end_scale))

            for s in range(self.n_momenta): #!! for local adaptation
                self.silent_haar_coef_momenta_subjects[s] = dict()
                self.silent_haar_coef_momenta_subjects[s][0] = []
                self.silent_haar_coef_momenta_subjects[s][self.scale] = []
            
            self.zones[self.scale] = dict() 
    
    def compute_haar_transform(self, momenta):
        if len(momenta.shape) == 2:
            #momenta = list(momenta)
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
        if self.multiscale_images: return False # dual multiscale handled later
        
        if self.multiscale_meshes: return False

        if not self.multiscale: return False

        if self.scale == 1: self.scale = 0

        if self.scale in [self.end_scale, 0]: return False
        
        if end: return True
                
        if self.too_much(iteration): return True
            
        if self.enough(iteration): return residuals_change(avg_residuals) < self.convergence_threshold
        
        return False

    def coarse_to_fine_step(self, iteration, step = 1):
        #go to smaller scale #beginning : self.scale = J
        self.scale = max(1, self.scale - step) #we only go to scale 0 so that the local adaptation is not called again ... 
        
        if step > 0:
            logger.info("Coarse to fine on momenta")
        else:
            logger.info("\nReverse coarse to fine on momenta")
            
        logger.info("Scale {}".format(self.scale))
        self.iter.append(iteration)
        
        # if self.scale in range(1, self.coarser_scale):
        #     #self.local_adaptation(iteration, new_parameters, output_dir, dataset, residuals)
        #     self.local_adaptation_subject(iteration, new_parameters, output_dir, current_dataset, residuals)

    def coarse_to_fine(self, new_parameters, current_dataset, iteration, 
                        avg_residuals, end):

        if self.coarse_to_fine_condition(iteration, avg_residuals, end):
            self.coarse_to_fine_step(iteration)
        
        return current_dataset, new_parameters
    
    ####################################################################################################################
    ### Gradient tools
    ####################################################################################################################         
    def compute_haar_momenta_gradient(self, gradient):
        """
            Wavelet transform gradient of momenta
            Set to 0 fine-scale coefficients
        """
        if len(gradient['momenta'].shape) == 2:
            gradient['momenta'] = gradient['momenta'][np.newaxis, :, :]

        haar_gradient = []
        for s, momenta_gradient in enumerate(gradient['momenta']): #for each subject
            subject_gradient = []

            for d in range(self.dimension):
                gradient_of_momenta = momenta_gradient[:, d].reshape(tuple(self.points_per_axis))
                
                # Compute haar transform of gradient
                gradient_of_coef = haar_backward_transpose(gradient_of_momenta, J = None) #before 13/12
                
                # Silence some coefficients
                gradient_of_coef = self.silence_fine_or_smooth_zones(gradient_of_coef, s)   
                subject_gradient.append(gradient_of_coef)

            haar_gradient.append(subject_gradient)
        gradient["haar_coef_momenta"] = haar_gradient

        return gradient

    def smooth_gradient(self, gradient):
        return gradient
    
        if "regression" in self.model_name.lower() and "kernel" not in self.model_name.lower():
            cp = torch.tensor(self.model.control_points, dtype=torch.float32, device='cuda:0')
            try:
                n_comp = len(self.model.fixed_effects['rupture_time']) + 1
            except:
                return gradient
                
            if n_comp > 1:
                for comp, t in enumerate(self.model.fixed_effects['rupture_time']):
                    if t < 30 and comp < n_comp - 1:
                        print("smooth comp", comp)
                        momenta = gradient["momenta"][comp]
                        ones = np.ones(momenta.shape)
                        if not isinstance(momenta, torch.Tensor):
                            momenta = torch.tensor(momenta, dtype=torch.float32, device='cuda:0')
                        smooth_gradient = self.smoothing_kernel.convolve(cp, cp, momenta)  
                        
                        # normalize 
                        ones = torch.from_numpy(ones).float()
                        sum_kernel_along_rows = self.smoothing_kernel.convolve(cp, cp, ones)
                        gradient["momenta"][comp] = smooth_gradient.cpu().numpy() / sum_kernel_along_rows.numpy()

        return gradient   

    
    def compute_haar_gradient(self, gradient):
        gradient = self.smooth_gradient(gradient)

        if self.ctf_is_happening():
            gradient = self.compute_haar_momenta_gradient(gradient)
        
        return gradient
    
    def gradient_ascent(self, parameters, new_parameters, gradient, step):
        """
            Wavelet transform the momenta
            Update the momenta with wavelet-transformed gradient
            Backward transform the momenta
        """
        haar_coef_momenta = self.compute_haar_transform(parameters['momenta'])

        if parameters["momenta"].shape[0] > 100:
            for c in range(self.n_momenta): 
                for d in range(self.dimension):
                    # Haar transform of momenta update
                    haar_coef_momenta[c][d].wc += gradient["haar_coef_momenta"][c][d].wc * step["haar_coef_momenta"]
                    
                    # Momenta update
                    momenta_recovered = haar_coef_momenta[c][d].haar_backward()
                    new_parameters['momenta'][c][d] = momenta_recovered.flatten()

        else:
            for c in range(self.n_momenta): 
                for d in range(self.dimension):
                    # Haar transform of momenta update
                    haar_coef_momenta[c][d].wc += gradient["haar_coef_momenta"][c][d].wc * step["haar_coef_momenta"]
                    
                    # Momenta update
                    momenta_recovered = haar_coef_momenta[c][d].haar_backward()
                    new_parameters['momenta'][c, :, d] = momenta_recovered.flatten()
        
        # def vect_norm(array, order = 2):
        #     return np.linalg.norm(array, ord = order, axis = 1)

        # if self.multiscale and self.scale > 1:
        #     logger.info("Smoothing momenta... ")

        #     tensor_scalar_type = utilities.get_torch_scalar_type('float32')
        #     device, _ = utilities.get_best_device(gpu_mode=GpuMode.KERNEL)

        #     mom = torch.from_numpy(new_parameters['momenta'])
        #     cp = torch.from_numpy(self.model.control_points)
        #     mom = utilities.move_data(mom, dtype=tensor_scalar_type, requires_grad=False, device=device)
        #     cp = utilities.move_data(cp, dtype=tensor_scalar_type, requires_grad=False, device=device)
        #     self.smoothing_kernel_ = kernel_factory.factory("keops", gpu_mode = GpuMode.KERNEL,
        #                                                kernel_width=10)
        #     filtered_mom = self.smoothing_kernel_.convolve(cp, cp, mom, mode = "gaussian_weighted")
        #     filtered_mom = filtered_mom.detach().cpu().numpy()
        #     old_norm = vect_norm(new_parameters['momenta'])
        #     new_norm = vect_norm(filtered_mom)
        #     ratio = np.expand_dims(old_norm/new_norm, axis = 1)

        #     new_parameters['momenta'] = filtered_mom * ratio

        # Update momenta
        # haar_coef_momenta = self.compute_haar_transform(parameters['momenta'])
        # updated_momenta = [np.zeros((self.n_cp, self.dimension))] * self.n_momenta
                
        # if self.n_momenta != 1:
        #     updated_momenta = np.array(updated_momenta)
        # elif updated_momenta[0].shape == parameters['momenta'].shape:
        #     updated_momenta = updated_momenta[0] 
        # else:
        #     updated_momenta = updated_momenta[0].reshape(parameters['momenta'].shape)

        # new_parameters['momenta'] = updated_momenta
        
        return new_parameters

    ####################################################################################################################
    ### Local adaptation
    ####################################################################################################################     

    def silence_fine_or_smooth_zones(self, gradient_of_coef, sujet):
        """
            Filters the values of the Haar coefficients of the gradient of the momenta
            Coefficients whose scale is lower than the current scale are set to 0

        zones: dictionary containing information about zone 
        silent_haar_coef_momenta_subject: zones to silence 
        gradient_of_coef: Haar coefficients of the gradient of the momenta
        """
        silent_haar_coef_momenta_subject = self.silent_haar_coef_momenta_subjects[sujet]
        
        indices_to_browse_along_dim = [list(range(e)) for e in list(gradient_of_coef.wc.shape)]

        for indices in itertools.product(*indices_to_browse_along_dim):
            position, type, scale = gradient_of_coef.haar_coeff_pos_type([i for i in indices])

            #silence finer zones that we haven't reached yet
            if scale < self.scale: #the higher the scale, the coarser
                gradient_of_coef.wc[indices] = 0 #fine scales : only ad, da, dd
                
            #silence smooth zones
            elif silent_haar_coef_momenta_subject and scale in silent_haar_coef_momenta_subject.keys():
                zones_to_silence_at_scale = silent_haar_coef_momenta_subject[scale]
                positions_to_silence_at_scale = [self.zones[scale][k]["position"] for k in zones_to_silence_at_scale]
                if position in positions_to_silence_at_scale and type != ['L', 'L']:
                    gradient_of_coef.wc[indices] = 0
                        
        return gradient_of_coef  

    ####################################################################################################################
    ### Optimization tools
    ####################################################################################################################     
    def enough(self, iteration):
        return iteration - self.iter[-1] > self.ctf_interval

    def too_much(self, iteration):
        return iteration - self.iter[-1] > self.ctf_max_interval

    def ctf_is_happening(self):
        """
        Check the algorithm still performs CTF
        (avoid useless computations when the end scale is reached)
        """
        return self.multiscale and self.scale > self.end_scale

    def convergence(self, iteration, cond):
        """
        Check that convergence conditions fulfilled
        """
        if not self.multiscale: return cond

        return self.scale <= self.end_scale and self.enough(iteration)
        
    def after_ctf(self, iteration):
        """
        Check that CTF just happened (to prevent convergence)
        """

        return self.multiscale and (self.iter[-1] == iteration)
    
    def folder_name(self, name):
        if not self.multiscale: return name

        return name + "scale_{}".format(self.scale)
    
