import math
import torch
import os.path as op
from time import perf_counter
from ...support import kernels as kernel_factory
from ...core import default
from ...core.model_tools.deformations.piecewise_spatiotemporal_reference_frame import SpatiotemporalReferenceFrame
from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...core.models.model_functions import initialize_control_points, initialize_momenta,\
                                            initialize_modulation_matrix, initialize_sources,\
                                                initialize_covariance_momenta_inverse
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...in_out.array_readers_and_writers import *
from ...in_out.dataset_functions import create_template_metadata, create_mesh_attachements, compute_noise_dimension
from ...support import utilities
from ...support.probability_distributions.multi_scalar_inverse_wishart_distribution import \
    MultiScalarInverseWishartDistribution
from ...support.probability_distributions.multi_scalar_normal_distribution import MultiScalarNormalDistribution

torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)

class BayesianPiecewiseGeodesicRegression(AbstractStatisticalModel):
    """
    Geodesic regression object class.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, template_specifications,
                 dimension=default.dimension,
                 number_of_processes=default.number_of_processes,

                 deformation_kernel_width=default.deformation_kernel_width,

                 concentration_of_time_points=default.concentration_of_time_points, 
                 t0=default.t0, tR=[], t1 = default.tmax,
                 use_rk2_for_shoot=default.use_rk2_for_shoot, use_rk2_for_flow=default.use_rk2_for_flow,

                 freeze_template=default.freeze_template,
                 use_sobolev_gradient=default.use_sobolev_gradient,
                 smoothing_kernel_width=default.smoothing_kernel_width,

                 initial_control_points=default.initial_control_points,
                 initial_cp_spacing=default.initial_cp_spacing,
                 initial_momenta=default.initial_momenta,
                 initial_modulation_matrix = default.initial_modulation_matrix,
                 freeze_modulation_matrix = False,
                 freeze_rupture_time = default.freeze_rupture_time,
                 freeze_noise_variance = default.freeze_noise_variance,

                 gpu_mode=default.gpu_mode,

                 num_component = 2,
                 new_bounding_box = None, # ajout fg
                 number_of_observations = 1, # ajout fg 
                 number_of_sources = 2,
                 weights = None,
                 
                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='BayesianGeodesicRegression', gpu_mode=gpu_mode)

        # Global-like attributes.
        self.dimension = dimension
        self.number_of_processes = number_of_processes
        self.number_of_observations = number_of_observations
        self.freeze_template = freeze_template
        self.freeze_momenta = False
        self.freeze_rupture_time = freeze_rupture_time

        self.weights = weights

        # Declare model structure.
        self.t0 = t0 # t0 AND t1 must be provided to compute the tR
        self.t1 = t1
        self.nb_components = num_component

        self.is_frozen = {'template_data': False, 
                          'control_points': True,
                          'momenta': False, 
                          'modulation_matrix': freeze_modulation_matrix,
                          'rupture_time': freeze_rupture_time, 
                          'noise_variance': freeze_noise_variance}
    
        # Deformation.
        self.spt_reference_frame = SpatiotemporalReferenceFrame(
            kernel=kernel_factory.factory(gpu_mode=self.gpu_mode,
                                          kernel_width=deformation_kernel_width),
            concentration_of_time_points = concentration_of_time_points, 
            nb_components = self.nb_components, template_tR = None,
            use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow,
            transport_cp = False)
        self.spt_reference_frame_is_modified = True
        

        # Template TODO? several templates
        (object_list, self.objects_name, self.objects_name_extension, self.objects_noise_variance, 
        self.multi_object_attachment) = create_template_metadata(template_specifications, 
                                                        self.dimension, gpu_mode=gpu_mode)
        
        self.template = DeformableMultiObject(object_list)

        self.objects_noise_dimension = compute_noise_dimension(self.template, self.multi_object_attachment,
                                                                self.dimension, self.objects_name)
        self.number_of_objects = len(self.template.object_list)
        
        # Ajout fg: cost function with curvature matching term
        self.curvature = False

        self.use_sobolev_gradient = use_sobolev_gradient
        self.smoothing_kernel_width = smoothing_kernel_width
        self.deformation_kernel_width = deformation_kernel_width
        self.initial_cp_spacing = initial_cp_spacing
        self.number_of_sources = number_of_sources
        
        if self.use_sobolev_gradient:
            self.sobolev_kernel = kernel_factory.factory(gpu_mode=gpu_mode, kernel_width=smoothing_kernel_width)

        # Template data.
        self.set_template_data(self.template.get_data())
        
        # Control points.
        self.set_control_points(initialize_control_points(
            initial_control_points, self.template, initial_cp_spacing, deformation_kernel_width,
            self.dimension, new_bounding_box = new_bounding_box))

        # Momenta.
        self.set_momenta(initialize_momenta(initial_momenta, len(self.fixed_effects['control_points']), 
                        self.dimension, number_of_subjects = self.nb_components))

        # Modulation matrix. shape (ncp x dim)  x n_sources 
        self.fixed_effects['modulation_matrix'] = initialize_modulation_matrix(
            initial_modulation_matrix, len(self.fixed_effects['control_points']), self.number_of_sources, self.dimension)
        # self.fixed_effects['modulation_matrix'] = np.random.rand(self.fixed_effects['control_points'].shape[0] * self.dimension, 
        #                                                            self.number_of_sources) * 1e-7

        # Rupture time: a parameter but can also be considered a RE (exp model)
        self.fixed_effects['rupture_time'] = np.zeros((self.nb_components - 1))

        if not tR: # set tR at regular intervals
            segment = int((math.ceil(self.t1) - math.trunc(self.t0))/self.nb_components)
            for i in range(self.nb_components - 1):
                self.set_rupture_time(math.trunc(self.t0) + segment * (i+1), i)
        else: # the tR are provided by the user
            for i, t in enumerate(tR):
                self.set_rupture_time(t, i)
                if t == self.t0:
                    self.template_index = i      
                      
        
        # Noise variance.
        self.fixed_effects['noise_variance'] = np.array(self.objects_noise_variance)
        self.objects_noise_variance_prior_normalized_dof = [elt['noise_variance_prior_normalized_dof']
                                                            for elt in template_specifications.values()]
        self.objects_noise_variance_prior_scale_std = [elt['noise_variance_prior_scale_std']
                                                       for elt in template_specifications.values()]

        # Source random effect.
        self.individual_random_effects['sources'] = MultiScalarNormalDistribution()
        self.individual_random_effects['sources'].set_mean(np.zeros((self.number_of_sources,)))
        self.individual_random_effects['sources'].set_variance(1.0)

        # Priors on the population parameters
        self.priors['rupture_time'] = [MultiScalarNormalDistribution()] * (self.nb_components-1)
        self.priors['template_data'] = {}
        self.priors['momenta'] = MultiScalarNormalDistribution()
        self.priors['modulation_matrix'] = MultiScalarNormalDistribution()
        self.priors['noise_variance'] = MultiScalarInverseWishartDistribution()

        self.__initialize_template_data_prior()
        self.__initialize_momenta_prior()
        self.__initialize_modulation_matrix_prior()      
        self.__initialize_rupture_time_prior(initial_rupture_time_variance = 2)

        # ajout fg to prevent useless residuals computation
        self.current_residuals = None
    
    def get_template_index(self):
        for i, t in enumerate(self.get_rupture_time()):
            if t == self.t0:
                self.template_index = i      

    def initialize_random_effects_realization(self, number_of_subjects, 
                                            initial_sources=default.initial_sources,
                                            **kwargs):

        # Initialize the random effects realization.
        individual_RER = {
        'sources': initialize_sources(initial_sources, number_of_subjects, self.number_of_sources),
        }

        return individual_RER
        
    def initialize_noise_variance(self, dataset, individual_RER):
        # Prior on the noise variance (inverse Wishart: degrees of freedom parameter).
        for k, normalized_dof in enumerate(self.objects_noise_variance_prior_normalized_dof):
            dof = dataset.total_number_of_observations * normalized_dof * self.objects_noise_dimension[k]
            self.priors['noise_variance'].degrees_of_freedom.append(dof)

        # Useless when provided by user --> find sot around 200 !
        if np.min(self.fixed_effects['noise_variance']) < 0.0:
            # Prior on the noise variance (inverse Wishart: scale scalars parameters).
            template_data, template_points, cp, momenta, mod_matrix, tR, _ = \
            self._fixed_effects_to_torch_tensors(False)
            sources = self._individual_RER_to_torch_tensors(individual_RER, False)
            
            self._update_geodesic(dataset, template_points, cp, momenta, mod_matrix, tR)
            
            residuals = self.compute_residuals_(dataset.times, dataset.deformable_objects, template_data, sources)

            residuals_per_object = np.zeros((self.number_of_objects,))
            for j in range(self.number_of_objects):
                residuals_per_object[j] = np.sum([r[j].detach().cpu().numpy() for r in residuals])
            
            for k, scale_std in enumerate(self.objects_noise_variance_prior_scale_std):
                if scale_std is None:
                    self.priors['noise_variance'].scale_scalars.append(
                        0.01 * residuals_per_object[k] / self.priors['noise_variance'].degrees_of_freedom[k])
                else:
                    self.priors['noise_variance'].scale_scalars.append(scale_std ** 2)

            # New, more informed initial value for the noise variance.
            self.fixed_effects['noise_variance'] = np.array(self.priors['noise_variance'].scale_scalars)

        else:
            for k, object_noise_variance in enumerate(self.fixed_effects['noise_variance']):
                self.priors['noise_variance'].scale_scalars.append(object_noise_variance)

    def __initialize_template_data_prior(self):
        """
        Initialize the template data prior.
        """
        if not self.is_frozen['template_data']:
            template_data = self.get_template_data()

            for key, value in template_data.items():
                # Initialization.
                self.priors['template_data'][key] = MultiScalarNormalDistribution()

                # Set the template data prior mean as the initial template data.
                self.priors['template_data'][key].mean = value

                if key == 'landmark_points':
                    self.priors['template_data'][key].set_variance_sqrt(self.spt_reference_frame.get_kernel_width())
                elif key == 'image_intensities':
                    std = 0.5
                    logger.info('Template image intensities prior std parameter is ARBITRARILY set to %.3f.' % std)
                    self.priors['template_data'][key].set_variance_sqrt(std)

    def __initialize_momenta_prior(self):
        """
        Initialize the momenta prior.
        """
        if not self.is_frozen['momenta']:
            self.priors['momenta'].set_mean(self.get_momenta())
            # Set the momenta prior variance as the norm of the initial rkhs matrix.
            assert self.spt_reference_frame.get_kernel_width() is not None
            rkhs_matrix = initialize_covariance_momenta_inverse(
                self.fixed_effects['control_points'], self.spt_reference_frame.exponential.kernel,
                self.dimension)
            self.priors['momenta'].set_variance(1. / np.linalg.norm(rkhs_matrix))  # Frobenius norm.
            logger.info('>> Momenta prior std set to %.3E.' % self.priors['momenta'].get_variance_sqrt())

    def __initialize_modulation_matrix_prior(self):
        """
        Initialize the modulation matrix prior.
        """
        if not self.is_frozen['modulation_matrix']:
            # Set the modulation_matrix prior mean as the initial modulation_matrix.
            self.priors['modulation_matrix'].set_mean(self.get_modulation_matrix())
            # Set the modulation_matrix prior standard deviation to the deformation kernel width.
            self.priors['modulation_matrix'].set_variance_sqrt(self.spt_reference_frame.get_kernel_width())
    
    def __initialize_rupture_time_prior(self, initial_rupture_time_variance):
        """
        Initialize the reference time prior.
        """
        if not self.is_frozen['rupture_time']:
            for k in range(self.nb_components - 1):
                # Set the rupture_time prior mean as the initial reference_time.
                self.priors['rupture_time'][k].set_mean(np.zeros((1,)) + self.get_rupture_time()[k].copy())
                # Check that the reference_time prior variance has been set.
                self.priors['rupture_time'][k].set_variance(initial_rupture_time_variance)

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################
    # Template data ----------------------------------------------------------------------------------------------------
    def get_template_data(self):
        return self.fixed_effects['template_data']

    def set_template_data(self, td):
        self.fixed_effects['template_data'] = td
        self.template.set_data(td)

    # Control points ---------------------------------------------------------------------------------------------------
    def get_control_points(self):
        return self.fixed_effects['control_points']

    def set_control_points(self, cp):
        self.fixed_effects['control_points'] = cp

    # Momenta ----------------------------------------------------------------------------------------------------------
    def get_momenta(self):
        return self.fixed_effects['momenta']

    def set_momenta(self, mom): # mom = n_comp x n_cp x dim
        self.fixed_effects['momenta'] = mom

    # Reference time ---------------------------------------------------------------------------------------------------
    def get_rupture_time(self):
        return self.fixed_effects['rupture_time']

    def set_rupture_time(self, rt, index):
        self.fixed_effects['rupture_time'][index] = np.array(rt)
        
    # Modulation matrix ------------------------------------------------------------------------------------------------
    def get_modulation_matrix(self):
        return self.fixed_effects['modulation_matrix']

    def set_modulation_matrix(self, mm):
        self.fixed_effects['modulation_matrix'] = mm
        self.spt_reference_frame_is_modified = True
    
    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self):
        out = {}
        for k in self.fixed_effects.keys():
            if not self.is_frozen[k]:
                if k != "template_data":
                    out[k] = self.fixed_effects[k]
                else:
                    for key, value in self.fixed_effects[k].items():
                        out[key] = value

        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.is_frozen["momenta"]:
            self.set_momenta(fixed_effects['momenta'])
        if not self.is_frozen['template_data']:
            template_data = {key: fixed_effects[key] for key in self.fixed_effects['template_data'].keys()}
            self.set_template_data(template_data)
        if not self.is_frozen['rupture_time']:
            for i in range(self.nb_components - 1):
                self.set_rupture_time(fixed_effects['rupture_time'][i], i)
        if not self.is_frozen['modulation_matrix']:
            self.set_modulation_matrix(fixed_effects['modulation_matrix'])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def compute_gradients(self, attachment, regularity, template_data, template_points,
                           cp, momenta, mod_matrix, rupture_time, 
                           sources, mode, with_grad=False):
        # Compute gradient if needed -----------------------------------------------------------------------------------
        if with_grad:
            total = regularity + attachment
            total.backward()

            gradient = {}
            # Template data.
            if not self.is_frozen['template_data']:
                if 'landmark_points' in template_data.keys():
                    gradient['landmark_points'] = template_points['landmark_points'].grad
                if 'image_intensities' in template_data.keys():
                    gradient['image_intensities'] = template_data['image_intensities'].grad
                if self.use_sobolev_gradient and 'landmark_points' in gradient.keys():
                    gradient['landmark_points'] = self.sobolev_kernel.convolve(
                        template_data['landmark_points'].detach(), template_data['landmark_points'].detach(),
                        gradient['landmark_points'].detach())

            if not self.is_frozen['rupture_time']:
                gradient['rupture_time'] = rupture_time.grad
            if not self.is_frozen["momenta"]:
                gradient['momenta'] = momenta.grad
            if not self.is_frozen['modulation_matrix']:
                gradient["modulation_matrix"] = mod_matrix.grad
            
            if mode == 'complete':
                gradient['sources'] = sources.grad

            # Convert the gradient back to numpy.
            gradient = {key: value.data.cpu().numpy() for key, value in gradient.items()}

            return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient

        else:
            return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy()

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, population_RER, individual_RER, 
                                mode='complete', with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.

        :param dataset: LongitudinalDataset instance
        :param fixed_effects: Dictionary of fixed effects.
        :param population_RER: Dictionary of population random effects realizations.
        :param indRER: Dictionary of individual random effects realizations.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        """
        device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)

        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        template_data, template_points, cp, momenta, mod_matrix, tR, _ \
        = self._fixed_effects_to_torch_tensors(with_grad, device=device)
        sources = self._individual_RER_to_torch_tensors(individual_RER, with_grad and mode == 'complete', device=device)

        #Each variable has a .grad_fn attribute that references a function that has created a function 
        # (except for Tensors created by the user - these have None as .grad_fn).
        self._update_geodesic(dataset, template_points, cp, momenta, mod_matrix, tR)

        # Deform -------------------------------------------------------------------------------------------------------
        t2 = perf_counter()
        attachment, regularity = self._compute_attachment_and_regularity(
        dataset, template_data, template_points, momenta, mod_matrix, sources, mode)
        t3 = perf_counter()

        # Gradients -------------------------------------------------------------------------------------------------------
        return self.compute_gradients(attachment, regularity, template_data, template_points,
                                cp, momenta, mod_matrix, tR, sources, mode, with_grad=with_grad)

    def _compute_class1_priors_regularity(self, regularity):
        """
        Fully torch.
        Prior terms of the class 1 fixed effects, i.e. those for which we know a close-form update. 
        No derivative wrt those fixed effects will therefore be necessary.
        """
        if not self.is_frozen['rupture_time']:
            for k in range(self.fixed_effects['rupture_time'].__len__()):
                regularity += self.priors['rupture_time'][k].compute_log_likelihood(self.fixed_effects['rupture_time'][k])

        if not self.is_frozen['noise_variance']:
            regularity += self.priors['noise_variance'].compute_log_likelihood(self.fixed_effects['noise_variance'])

        return regularity

    def _compute_class2_priors_regularity(self, regularity, template_data, momenta, mod_matrix):
        """
        Fully torch.
        Prior terms of the class 2 fixed effects, i.e. those for which we do not know a close-form update. 
        Derivative wrt those fixed effects will therefore be necessary.
        """
        if not self.is_frozen['template_data']:
            for key, value in template_data.items():
                regularity += self.priors['template_data'][key].compute_log_likelihood_torch(
                    value)

        if not self.is_frozen['momenta']:
            regularity += self.priors['momenta'].compute_log_likelihood_torch(momenta)
            
        if not self.is_frozen['modulation_matrix']:
            regularity += self.priors['modulation_matrix'].compute_log_likelihood_torch(
                mod_matrix)        

        return regularity
    
    def _compute_random_effects_regularity(self, sources, device='cpu'):
        """
        Fully torch.
        """
        number_of_subjects = sources.shape[0]
        regularity = 0.0

        # Sources random effect.
        for i in range(number_of_subjects):
            regularity += self.individual_random_effects['sources'].compute_log_likelihood_torch(
                sources[i], device=device)

        return regularity

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################
    def compute_residuals(self, dataset, individual_RER = None, k = False, option = None):
        device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)

        template_data, template_points, cp, momenta, mod_matrix, tR, _ = \
        self._fixed_effects_to_torch_tensors(False, device = device)
        sources = self._individual_RER_to_torch_tensors(individual_RER, False, device = device)
        
        self._update_geodesic(dataset, template_points, cp, momenta, mod_matrix, tR)
        
        res = self.compute_residuals_(dataset.times, dataset.deformable_objects, template_data, sources)
        
        return sum(res, [])

    def compute_residuals_(self, target_times, target_objects, template_data, sources):
        device, _ = utilities.get_best_device(self.gpu_mode)
        residuals = [] 
        t1= perf_counter()

        if self.weights is None:
            self.weights = [1.] * len(target_objects)

        for i in range(len(target_objects)):
            residuals_i = []

            # Compute the distance between obs_i,j and Exp_y(tij)(s_i)
            for j, (time, target) in enumerate(zip(target_times[i], target_objects[i])):
                deformed_points = self.spt_reference_frame.get_template_points(time, sources[i], device=device)
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                residual = self.weights[i] * self.multi_object_attachment.compute_distances(deformed_data, self.template, target)
                residuals_i.append(residual.cpu())
            residuals.append(residuals_i)
        t2= perf_counter()

        self.current_residuals = sum(residuals, [])
                
        return residuals

    def _compute_batch_attachment_and_regularity(self, target_times, target_objects, template_data, template_points, 
                                                momenta, mod_matrix, sources, mode):        
        residuals = self.compute_residuals_(target_times, target_objects, template_data, sources)

        # Individual attachments
        device = residuals[0][0].device
        noise_variance = utilities.move_data(self.fixed_effects['noise_variance'], device=device)

        attachment = 0.0
        for i in range(len(residuals)):
            for j in range(len(residuals[i])):
                attachment -= 0.5 * torch.sum(residuals[i][j] / noise_variance)

        # Regularity  
        if mode == 'complete':
            regularity = self._compute_random_effects_regularity(sources, device=device)
            regularity += self._compute_class1_priors_regularity(regularity)
        if mode in ['complete', 'class2']:
            regularity += self._compute_class2_priors_regularity(regularity, template_data, momenta,
                                                                 mod_matrix)
        return attachment, regularity

    
    def _compute_attachment_and_regularity(self, dataset, template_data, template_points, 
                                            momenta, mod_matrix, sources, mode):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """
        # update the geodesic times again to fit the batch in mini batch GD
        tmin = math.trunc(float(min(dataset.times)))
        tmax = math.ceil(float(max(dataset.times)))
        
        self.spt_reference_frame.set_tmin(tmin)
        self.spt_reference_frame.set_tmax(tmax)
        self.spt_reference_frame.update()

        return self._compute_batch_attachment_and_regularity(dataset.times, dataset.deformable_objects, template_data, template_points, momenta, mod_matrix, sources, mode)
    
    def compute_mini_batch_gradient(self, batch, dataset, population_RER, individual_RER, mode = 'complete', with_grad=True):
        device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, cp, momenta, mod_matrix, tR, _ \
        = self._fixed_effects_to_torch_tensors(with_grad, device=device)

        sources = self._individual_RER_to_torch_tensors(individual_RER, with_grad and mode == 'complete', device=device)
        
        self._update_geodesic(dataset, template_points, cp, momenta, mod_matrix, tR)
        
        # get target times and objects from the batch
        target_times = [t[0] for t in batch]
        target_objects = [t[1] for t in batch]
                
        attachement, regularity = self._compute_batch_attachment_and_regularity(target_times, target_objects, 
                                                                                template_data, template_points, 
                                                                                momenta, mod_matrix)
        return self.compute_gradients(attachement, regularity, template_data, template_points,
                                        cp, momenta, mod_matrix, tR, sources, mode, with_grad=with_grad)
        

    def mini_batches(self, dataset, number_of_batches):
        batch_size = len(dataset.deformable_objects[0])//number_of_batches

        targets = [[t,target] for t, target in zip(dataset.times, dataset.deformable_objects)]
        targets_copy = targets.copy()
        np.random.shuffle(targets_copy)

        mini_batches = []
        n_minibatches = len(targets_copy) // batch_size    

        for i in range(0, n_minibatches):
            mini_batch = targets_copy[i * batch_size:(i + 1)*batch_size]
            mini_batches.append(mini_batch)
        if len(targets_copy) % batch_size != 0:
            mini_batch = targets_copy[i * batch_size:len(targets_copy)]
            mini_batches.append(mini_batch)
        
        return mini_batches
    
    def compute_objects_distances(self, dataset, j, individual_RER = None, dist = "current", deformed = True): 
        """
        Compute current distance between deformed template and object
        #obj = a deformablemultiobject
        """
        device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, _, _ = self.prepare_geodesic(dataset, device)
        sources = self._individual_RER_to_torch_tensors(individual_RER, False, device)
        
        deformed_points = self.spt_reference_frame.get_template_points(dataset.times[j][0], sources[j])
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)

        if dist in ["current", "varifold"]:
            return self.multi_object_attachment.compute_vtk_distance(deformed_data, self.template, dataset.deformable_objects[j][0], dist)
        elif dist in ["ssim", "mse"]:
            return self.multi_object_attachment.compute_ssim_distance(deformed_data, self.template, dataset.deformable_objects[j][0], dist)

    def compute_flow_curvature(self, dataset, time, curvature = "gaussian"):
        device, _ = utilities.get_best_device(gpu_mode = self.gpu_mode)
        template_data, _, _ = self.prepare_geodesic(dataset, device)

        deformed_points = self.spt_reference_frame.geodesic.get_template_points(time)
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        
        for i, obj1 in enumerate(self.template.object_list):
            obj1.polydata.points = deformed_data['landmark_points'].cpu().numpy()
            obj1.curvature_metrics(curvature)
        
        return self.template

    def compute_initial_curvature(self, dataset, j, curvature = "gaussian"):
        obj = dataset.deformable_objects[j][0] 
        for obj1 in (obj.object_list):
            obj1.curvature_metrics(curvature)
        
        return obj


    def compute_curvature(self, dataset, j, individual_RER = None, curvature = "gaussian", iter = None):
        """
            Compute object curvature (at iter 0) or deformed template to object curvature
        """     
        if j is None:
            data = self.template.get_data()
            for i, obj1 in enumerate(self.template.object_list):
                obj1.polydata.points = data['landmark_points'][0:obj1.get_number_of_points()]
                obj1.curvature_metrics(curvature)
            
                return self.template
             
        if iter == 0:
            return self.compute_initial_curvature(dataset, j, curvature)
          
        device, _ = utilities.get_best_device(gpu_mode = self.gpu_mode)
        template_data, _, _ = self.prepare_geodesic(dataset, device)
        sources = self._individual_RER_to_torch_tensors(individual_RER, False, device)
        
        deformed_points = self.spt_reference_frame.get_template_points(dataset.times[j][0], sources[j])
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        
        # Compute deformed template curvature
        for i, obj1 in enumerate(self.template.object_list):
            obj1.polydata.points = deformed_data['landmark_points'][0:obj1.get_number_of_points()].cpu().numpy()
            obj1.curvature_metrics(curvature)
        
        return self.template
    
    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################
    def _individual_RER_to_torch_tensors(self, individual_RER, with_grad, device='cpu'):
        """
        Convert the input individual_RER into torch tensors.
        """
        # Sources.
        sources = individual_RER['sources']
        sources = utilities.move_data(sources, requires_grad=with_grad, device=device)

        return sources

    def _fixed_effects_to_torch_tensors(self, with_grad, device='cpu'):
        """
        Convert the fixed_effects into torch tensors.
        """
        # Template data.
        template_data = self.fixed_effects['template_data']
        template_data = {key: utilities.move_data(value,
                                                  requires_grad=(not self.is_frozen['template_data'] and with_grad),
                                                  device=device)
                         for key, value in template_data.items()}

        # Template points.
        template_points = self.template.get_points()
        template_points = {key: utilities.move_data(value, 
                                                    requires_grad=(not self.is_frozen['template_data'] and with_grad),
                                                    device=device)
                           for key, value in template_points.items()}

        liste = [template_data, template_points]

        for k, v in self.fixed_effects.items():
            if not isinstance(v, dict):
                effect = utilities.move_data(v,
                                            requires_grad=(with_grad and not self.is_frozen[k]), device=device)
                liste.append(effect)

        return tuple(liste)
        

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################        
    def prepare_geodesic(self, dataset, device = "cpu"):
        template_data, template_points, cp, momenta, mod_matrix, tR, _ = \
        self._fixed_effects_to_torch_tensors(False, device=device)
        
        self._update_geodesic(dataset, template_points, cp, momenta, mod_matrix, tR)

        return template_data, dataset.times, dataset.deformable_objects
    
    def add_component(self, dataset, c, new_tR=None):
        
        self.nb_components += 1

        # Add rupture time
        tmin = min(math.trunc(min(dataset.times[0])), self.t0)
        tmax = math.ceil(max(dataset.times[0]))
        tR = self.get_rupture_time().tolist()

        rupture_times = [tmin] + tR + [tmax]
        if new_tR is None:
            new_tR = (rupture_times[c] + rupture_times[c + 1])/2
        
        tR.insert(c, new_tR)
        self.fixed_effects['rupture_time'] = np.array(tR)
        self.get_template_index()

        logger.info("Adding component at time {}".format(new_tR))
        print("new tR", tR)

        # Update momenta
        momenta = np.zeros((self.nb_components, self.get_control_points().shape[0], 
                                self.get_control_points().shape[1]))
        for i in range(self.nb_components):
            if i <= c:
                momenta[i] = self.get_momenta()[i]
            elif i == c+1:
                start_time_t = self.spt_reference_frame.nb_of_tp(tR[c+1] - tmin) -1            
                momenta[i] = self.spt_reference_frame.get_momenta_trajectory()[start_time_t].cpu().numpy()
            else:
                momenta[i] = self.get_momenta()[i-1]  
        
        self.set_momenta(momenta)

        # Get torch tensors
        device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)
        _, template_points, _, momenta, _, tR, _ = self._fixed_effects_to_torch_tensors(False, device=device)

        self.spt_reference_frame.set_tR(tR)
        self.spt_reference_frame.set_t0(tR[self.template_index]) #ajout fg
        self.spt_reference_frame.set_momenta_tR(momenta)
        
        # Update geodesic and spt
        self.spt_reference_frame.add_component()
        self.spt_reference_frame.add_exponential(c)

        self.spt_reference_frame.set_template_points_tR(template_points)
        self.spt_reference_frame.update

    def _update_geodesic(self, dataset, template_points, cp, momenta, mod_matrix, tR):
        """
        Tries to optimize the computations, by avoiding repetitions of shooting / flowing / parallel transporting.
        If modified_individual_RER is None or that self.spt_reference_frame_is_modified is True,
        no particular optimization is carried.
        In the opposite case, the spatiotemporal reference frame will be more subtly updated.
        """
        
        tmin = math.trunc(float(min(dataset.times)))
        tmax = math.ceil(float(max(dataset.times)))

        # no grad_fn in here. Normal
        if self.spt_reference_frame_is_modified:
            self.spt_reference_frame.set_template_points_tR(template_points)
            self.spt_reference_frame.set_control_points_tR(cp)
            self.spt_reference_frame.set_momenta_tR(momenta)
            self.spt_reference_frame.set_modulation_matrix_tR(mod_matrix)
            self.spt_reference_frame.set_tR(tR)
            self.spt_reference_frame.set_t0(tR[self.template_index]) #ajout fg
            self.spt_reference_frame.set_tmin(tmin)
            self.spt_reference_frame.set_tmax(tmax)
            self.spt_reference_frame.update()

        self.spt_reference_frame_is_modified = False


    def write(self, dataset, population_RER, individual_RER, output_dir, iteration, 
                write_adjoint_parameters=False, write_all = True):
        self._write_model_predictions(output_dir, individual_RER, dataset, write_adjoint_parameters, write_all)
        self._write_model_parameters(output_dir, iteration, write_all)

        #residuals = self.compute_residuals(dataset, individual_RER)

        #write_2D_list(residuals, output_dir, self.name + "__EstimatedParameters__Residuals.txt")

    def _write_model_predictions(self, output_dir, individual_RER, dataset=None, 
                                 write_adjoint_parameters=False, write_all = True):

        # Initialize ---------------------------------------------------------------------------------------------------
        device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, target_times, _ = self.prepare_geodesic(dataset, device) 
        sources = self._individual_RER_to_torch_tensors(individual_RER, False, device)  

        # Write --------------------------------------------------------------------------------------------------------
        # write Geometric modes + optionally writes flow of A0, 
        # + call geodesic.write (trajectory)
        self.spt_reference_frame.write(self.name, self.objects_name, self.objects_name_extension, self.template, template_data,
                            output_dir, write_adjoint_parameters = False, write_exponential_flow = True,
                            write_all = write_all)

        # Write sources
        name = '{}__EstimatedParameters__Sources.txt'.format(self.name) 
        write_2D_array(individual_RER['sources'], output_dir, name)

        # Model predictions.
        if dataset is not None: #and not write_all:
            for i, subject_id in enumerate(dataset.subject_ids):
                for j, time in enumerate(target_times[i]):
                    names = []
                    for k, (object_name, object_extension) in enumerate(zip(self.objects_name, self.objects_name_extension)):
                        name = '{}__Reconstruction__subject__{}_{}__tp_{}__age_{}{}'.format(self.name, subject_id, object_name, j, time, object_extension) 
                        names.append(name)
                    deformed_points = self.spt_reference_frame.get_template_points(time, sources[j], device = device)
                    deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                    self.template.write(output_dir, names,
                                        {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

    def output_path(self, output_dir, dataset):
        self.cp_path = op.join(output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")
        self.momenta_path = op.join(output_dir, self.name + "__EstimatedParameters__Momenta.txt")
        
        # if self.spt_reference_frame.geodesic.exponential[0].number_of_time_points is None:
        #     device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)
        #     self.prepare_geodesic(dataset, device)
            
        try:
            self.spt_reference_frame.geodesic.output_path(self.name, self.objects_name, self.objects_name_extension, output_dir)
        except:
            pass

    def _write_model_parameters(self, output_dir, iteration, write_all = True):

        if write_all:
            template_names = []
            for k in range(len(self.objects_name)):
                n = '__Fixed__Template_' if self.is_frozen['template_data'] else '__EstimatedParameters__Template_' 
                    
                aux = '{}{}{}__tp_{}__age_{}{}'.format(self.name, n, self.objects_name[k], 
                self.spt_reference_frame.exponential.number_of_time_points - 1,
                self.get_rupture_time()[0],self.objects_name_extension[k]) 
            template_names.append(aux)
            self.template.write(output_dir, template_names)

            write_2D_array(self.get_modulation_matrix(), output_dir,
                           "{}__EstimatedParameters__ModulationMatrix.txt".format(self.name))

            if self.objects_name_extension[0] != ".vtk":
                #Control points.
                write_2D_array(self.get_control_points(), output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")

                # Momenta.
                write_3D_array(self.get_momenta(), output_dir, self.name + "__EstimatedParameters__Momenta.txt")
            else:
                # Fuse control points and momenta for paraview display
                for c in range(self.nb_components):
                    if self.dimension == 3:
                        concatenate_for_paraview(self.get_momenta()[c], self.get_control_points(), output_dir, 
                                        "{}__EstimatedParameters__Fusion_CP_Momenta__component_{}_iter_.vtk".format(self.name, c, iteration))

