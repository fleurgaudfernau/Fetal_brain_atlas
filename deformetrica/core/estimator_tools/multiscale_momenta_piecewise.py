import copy
import logging
import itertools
import os.path as op
from ...support.utilities.tools import residuals_change
from ...support.utilities.wavelets import haar_backward_transpose

logger = logging.getLogger(__name__)

class MultiscaleMomentaPiecewise():
    def __init__(self, multiscale, multiscale_images, multiscale_meshes, model, 
                ctf_interval, ctf_max_interval, points_per_axis):
        # Model parameters
        self.model = model
        self.convergence_threshold = 1e-3 #0.001

        # Piecewise
        self.n_components = 1
        self.rupture_times = [0]
        if "Regression" in model.name and len(model.get_momenta().shape) > 2:
            self.n_components = model.get_momenta().shape[0]
            self.rupture_times = model.get_rupture_time().tolist()   

        self.n_sources = 0 
        if model.name in ["BayesianGeodesicRegression", "BayesianPiecewiseRegression"]:
            self.n_sources = self.model.number_of_sources

        # Model options
        self.freeze_template = model.freeze_template
        self.initial_cp_spacing = model.initial_cp_spacing
        self.dimension = model.dimension

        self.initial_convergence_threshold = 0.001
        self.convergence_threshold = 0.001
        self.ctf_interval = ctf_interval
        self.ctf_max_interval = ctf_max_interval

        self.multiscale = multiscale
        self.multiscale_images = multiscale_images
        self.multiscale_meshes = multiscale_meshes
        self.iter = [0] if multiscale else []

        # if we want to stop before scale 0
        self.end_scale = {c : 1 for c in range(self.n_components)}
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
        self.coarser_scale = haar_coef_momenta[0][0].J

        if self.multiscale:
            logger.info("\nInitialisation - piecewise coarse to fine on momenta")

            self.scale = {c : haar_coef_momenta[0][0].J for c in range(self.n_components)}
    
    def compute_haar_transform(self, momenta):
        if len(momenta.shape) == 2:
            momenta = list(momenta)
        
        haar_coef_momenta = [[haar_backward_transpose(momenta[s, :, d].reshape(tuple(self.points_per_axis))) \
                            for d in range(self.dimension)] for s in range(len(momenta))]
        
        return haar_coef_momenta
    
    ####################################################################################################################
    ### Coarse to fine on momenta
    ####################################################################################################################
    def coarse_to_fine_condition(self, component, iteration, residuals_components, end):
        """
        Authorizes coarse to fine if more than 2 iterations after previous CTF (or beginning of algo)
        and if residuals diminution is low
        """
        if self.multiscale_images: return False # dual multiscale handled later
        
        if self.multiscale_meshes: return False

        if not self.multiscale: return False

        if self.scale[component] == 1: self.scale[component] = 0

        if self.scale[component] == self.end_scale[component]: return False
        
        if end: return True
                
        if self.too_much(iteration): return True
            
        if self.enough(iteration): 
            return residuals_change(residuals_components[component]) < self.convergence_threshold
        
        return False

    def coarse_to_fine_step(self, component, iteration, step = 1):
        #we only go to scale 0 so that the local adaptation is not called again ... 
        self.scale[component] = max(1, self.scale[component] - step) 
        
        if step > 0:
            logger.info("Coarse to fine on momenta for component {}".format(component))
        else:
            logger.info("Reverse coarse to fine on momenta for component {}".format(component))
            
        logger.info("Scale {}".format(self.scale[component]))
        self.iter.append(iteration)
        
    def coarse_to_fine(self, new_parameters, current_dataset, iteration, 
                        residuals_components, end):
        
        for c in range(self.n_components):
            if self.coarse_to_fine_condition(c, iteration, residuals_components, end):
                self.coarse_to_fine_step(c, iteration)
        
        return current_dataset, new_parameters
    
    ####################################################################################################################
    ### Gradient tools
    ####################################################################################################################     
    def compute_haar_momenta_gradient(self, gradient):
        haar_gradient = []
        for c in range(self.n_components): #for each subject OR component
            subject_gradient = []

            for d in range(self.dimension):
                gradient_of_momenta = gradient['momenta'][c, :, d].reshape(tuple(self.points_per_axis))
                
                # Compute haar transform of gradient
                gradient_of_coef = haar_backward_transpose(gradient_of_momenta, J = None) #before 13/12
                
                # Silence some coefficients
                gradient_of_coef = self.silence_fine_or_smooth_zones(c, gradient_of_coef, c)   
                subject_gradient.append(gradient_of_coef)

            haar_gradient.append(subject_gradient)
        
        gradient["haar_coef_momenta"] = haar_gradient

        return gradient
    
    def compute_haar_gradient(self, gradient):
        if self.ctf_is_happening():
            return self.compute_haar_momenta_gradient(gradient)
        
        return gradient
    
    def gradient_ascent(self, parameters, new_parameters, gradient, step):
        # Update momenta
        haar_coef_momenta = self.compute_haar_transform(parameters['momenta'])

        for c in range(self.n_components): 
            for d in range(self.dimension):
                # Haar transform of momenta update
                haar_coef_momenta[c][d].wc += gradient["haar_coef_momenta"][c][d].wc * step["haar_coef_momenta"]
                
                # Momenta update
                momenta_recovered = haar_coef_momenta[c][d].haar_backward()
                new_parameters['momenta'][c, :, d] = momenta_recovered.flatten()
        
        return new_parameters
    
    def silence_fine_or_smooth_zones(self, c, gradient_of_coef, sujet):
        """
            Filters the values of the Haar coefficients of the gradient of the momenta
            Coefficients whose scale is lower than the current scale are set to 0

        zones: dictionary containing information about zone 
        silent_haar_coef_momenta_subject: zones to silence 
        gradient_of_coef: Haar coefficients of the gradient of the momenta
        """
        
        indices_to_browse_along_dim = [list(range(e)) for e in list(gradient_of_coef.wc.shape)]

        for indices in itertools.product(*indices_to_browse_along_dim):
            _, _, scale = gradient_of_coef.haar_coeff_pos_type([i for i in indices])

            #silence finer zones that we haven't reached yet
            if scale < self.scale[c]: #the higher the scale, the coarser
                gradient_of_coef.wc[indices] = 0 #fine scales : only ad, da, dd
                                        
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
        if self.multiscale:
            for c in range(self.n_components):
                if self.scale[c] > self.end_scale[c]:
                    return True

        return False
    
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
