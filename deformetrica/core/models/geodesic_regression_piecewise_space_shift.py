import math
import torch
import os.path as op
from time import perf_counter
from ...support import kernels as kernel_factory
from ...core import default
from ...core.model_tools.deformations.spatial_piecewise_geodesic import SpatialPiecewiseGeodesic
from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...core.models.model_functions import initialize_control_points, initialize_momenta,\
                                            initialize_modulation_matrix, initialize_sources,\
                                            initialize_covariance_momenta_inverse, \
                                            initialize_rupture_time
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...in_out.array_readers_and_writers import *
from ...in_out.dataset_functions import create_template_metadata, compute_noise_dimension
from ...support.utilities import get_best_device, move_data
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

                 deformation_kernel_width=default.deformation_kernel_width,

                 concentration_of_time_points=default.concentration_of_time_points, 
                 t0=default.t0, tR=[], t1 = default.tmax,

                 freeze_template=default.freeze_template,

                 initial_control_points=default.initial_control_points,
                 initial_momenta=default.initial_momenta,
                 initial_modulation_matrix = default.initial_modulation_matrix,
                 freeze_modulation_matrix = False,
                 freeze_rupture_time = default.freeze_rupture_time,
                 freeze_noise_variance = default.freeze_noise_variance,

                 gpu_mode=default.gpu_mode,

                 num_component = 2,
                 new_bounding_box = None, # ajout fg
                 number_of_sources = 2,
                 weights = None,
                 
                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='BayesianGeodesicRegression', gpu_mode=gpu_mode)

        # Global-like attributes.
        self.dimension = dimension

        self.device, _ = get_best_device(gpu_mode=self.gpu_mode)

        # Declare model structure.
        self.t0 = t0 # t0 AND t1 must be provided to compute the tR
        self.t1 = t1
        self.nb_components = num_component

        self.is_frozen = {'template_data': freeze_template, 
                          'momenta': False, 
                          'modulation_matrix': freeze_modulation_matrix,
                          'rupture_time': freeze_rupture_time, 
                          'noise_variance': freeze_noise_variance}
    
        # Deformation.
        self.piecewise_geodesic = SpatialPiecewiseGeodesic(
            kernel=kernel_factory.factory(gpu_mode=self.gpu_mode,
                                          kernel_width=deformation_kernel_width),
                                        concentration_of_time_points = concentration_of_time_points, 
                                        nb_components = self.nb_components, template_tR = None,
                                        transport_cp = False)
        self.piecewise_geodesic_is_modified = True
        
        (object_list, self.objects_extension, self.objects_noise_variance, 
        self.multi_object_attachment) = create_template_metadata(template_specifications, 
                                                        self.dimension, gpu_mode=gpu_mode)
        
        self.template = DeformableMultiObject(object_list)

        self.objects_noise_dimension = compute_noise_dimension(self.template, self.multi_object_attachment,
                                                                self.dimension)
        self.number_of_objects = len(self.template.object_list)
        
        self.deformation_kernel_width = deformation_kernel_width
        self.n_sources = number_of_sources
        
        # Template data.
        self.set_template_data(self.template.get_data())
        self.points = self.get_points()
        
        # Control points.
        self.control_points = initialize_control_points(initial_control_points, self.template, 
                            deformation_kernel_width, self.dimension, new_bounding_box = new_bounding_box)

        # Momenta.
        self.set_momenta(initialize_momenta(initial_momenta, len(self.control_points), 
                        self.dimension, number_of_subjects = self.nb_components))

        # Modulation matrix. shape (ncp x dim)  x n_sources 
        self.fixed_effects['modulation_matrix'] = initialize_modulation_matrix(
            initial_modulation_matrix, len(self.control_points), self.n_sources, self.dimension)

        # Rupture time: a parameter but can also be considered a RE (exp model)
        logger.info("Setting rupture times...")
        tR = initialize_rupture_time(tR, t0, t1, nb_components)
        for i, t in enumerate(tR):
            print("Time", i, "set to", t)
            self.set_rupture_time(t, i)
        self.get_template_index()
        
        # Noise variance.
        self.fixed_effects['noise_variance'] = np.array(self.objects_noise_variance)
        self.objects_noise_variance_prior_normalized_dof = [elt['noise_variance_prior_normalized_dof']
                                                            for elt in template_specifications.values()]
        self.objects_noise_variance_prior_scale_std = [elt['noise_variance_prior_scale_std']
                                                       for elt in template_specifications.values()]

        # Source random effect.
        self.__initialize_source_random_effects()

        # Priors on the population parameters
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

    def __initialize_source_random_effects(self):
        self.individual_random_effects['sources'] = MultiScalarNormalDistribution()
        self.individual_random_effects['sources'].set_mean(np.zeros((self.n_sources,)))
        self.individual_random_effects['sources'].set_variance(1.0)

    def initialize_random_effects_realization(self, number_of_subjects, 
                                            initial_sources=default.initial_sources):
        individual_RER = {
        'sources': initialize_sources(initial_sources, number_of_subjects, self.n_sources)}

        return individual_RER
        
    def initialize_noise_variance(self, dataset, individual_RER):
        self.priors['noise_variance'] = MultiScalarInverseWishartDistribution()

        # Prior on the noise variance (inverse Wishart: degrees of freedom parameter).
        for k, normalized_dof in enumerate(self.objects_noise_variance_prior_normalized_dof):
            dof = dataset.total_number_of_observations * normalized_dof * self.objects_noise_dimension[k]
            self.priors['noise_variance'].degrees_of_freedom.append(dof)

        # Useless when provided by user --> find sot around 200 !
        if np.min(self.fixed_effects['noise_variance']) < 0.0:
            # Prior on the noise variance (inverse Wishart: scale scalars parameters).
            template_data, template_points, momenta, mod_matrix, tR, _ = \
            self._fixed_effects_to_torch_tensors()
            sources = self._individual_RER_to_torch_tensors(individual_RER)
            
            self._update_geodesic(dataset, template_points, momenta, mod_matrix, tR)
            
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
            self.priors['template_data'] = {}
            template_data = self.get_template_data()

            for key, value in template_data.items():
                # Initialization.
                self.priors['template_data'][key] = MultiScalarNormalDistribution()

                # Set the template data prior mean as the initial template data.
                self.priors['template_data'][key].mean = value

                if key == self.points:
                    self.priors['template_data'][key].set_variance_sqrt(self.deformation_kernel_width)
                elif key == 'image_intensities':
                    logger.info('Template image intensities prior std parameter is ARBITRARILY set to %.3f.' % 0.5)
                    self.priors['template_data'][key].set_variance_sqrt(0.5)

    def __initialize_momenta_prior(self):
        """
        Initialize the momenta prior.
        """
        if not self.is_frozen['momenta']:
            self.priors['momenta'] = MultiScalarNormalDistribution()
            self.priors['momenta'].set_mean(self.get_momenta())
            # Set the momenta prior variance as the norm of the initial rkhs matrix.
            rkhs_matrix = initialize_covariance_momenta_inverse(self.control_points, 
                                        self.piecewise_geodesic.exponential.kernel, self.dimension)
            self.priors['momenta'].set_variance(1. / np.linalg.norm(rkhs_matrix))  # Frobenius norm.
            logger.info('>> Momenta prior std set to %.3E.' % self.priors['momenta'].get_variance_sqrt())

    def __initialize_modulation_matrix_prior(self):
        """
        Initialize the modulation matrix prior.
        """
        if not self.is_frozen['modulation_matrix']:
            self.priors['modulation_matrix'] = MultiScalarNormalDistribution()
            # Set the modulation_matrix prior mean as the initial modulation_matrix.
            self.priors['modulation_matrix'].set_mean(self.get_modulation_matrix())
            # Set the modulation_matrix prior standard deviation to the deformation kernel width.
            self.priors['modulation_matrix'].set_variance_sqrt(self.deformation_kernel_width)
    
    def __initialize_rupture_time_prior(self, initial_rupture_time_variance):
        """
        Initialize the reference time prior.
        """
        if not self.is_frozen['rupture_time']:
            self.priors['rupture_time'] = [MultiScalarNormalDistribution()] * (self.nb_components-2)

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
    
    def get_points(self):
        return list(self.fixed_effects['template_data'].keys())[0]

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
        self.piecewise_geodesic_is_modified = True
    
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
                           momenta, mod_matrix, rupture_time, 
                           sources, mode, with_grad=False):
        # Compute gradient if needed -----------------------------------------------------------------------------------
        if with_grad:
            total = regularity + attachment
            total.backward()

            gradient = {}
            # Template data.
            if not self.is_frozen['template_data']:
                gradient[self.points] = template_points[self.points].grad
            if not self.is_frozen['rupture_time']:
                gradient['rupture_time'] = rupture_time.grad
            if not self.is_frozen["momenta"]:
                gradient['momenta'] = momenta.grad
            if not self.is_frozen['modulation_matrix']:
                gradient["modulation_matrix"] = mod_matrix.grad
            
            if mode == 'complete':
                gradient['sources'] = sources.grad

            gradient = {key: value.data.cpu().numpy() for key, value in gradient.items()}

            return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient

        else:
            return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy()

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, individual_RER, mode='complete', with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations

        :param dataset: LongitudinalDataset instance
        :param fixed_effects: Dictionary of fixed effects.
        :param indRER: Dictionary of individual random effects realizations.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        """
        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        template_data, template_points, momenta, mod_matrix, tR, _ \
        = self._fixed_effects_to_torch_tensors(with_grad)
        sources = self._individual_RER_to_torch_tensors(individual_RER, with_grad and mode == 'complete')

        #Each variable has a .grad_fn attribute that references a function that has created a function 
        self._update_geodesic(dataset, template_points, momenta, mod_matrix, tR)

        # Deform -------------------------------------------------------------------------------------------------------
        attachment, regularity = self._compute_attachment_and_regularity(
                                dataset, template_data, template_points, momenta, mod_matrix, sources, mode)

        # Gradients -------------------------------------------------------------------------------------------------------
        return self.compute_gradients(attachment, regularity, template_data, template_points,
                                momenta, mod_matrix, tR, sources, mode, with_grad=with_grad)

    def _compute_class1_priors_regularity(self):
        """
        Fully torch.
        Prior terms of the class 1 fixed effects, i.e. those for which we know a close-form update. 
        No derivative wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0
        if not self.is_frozen['rupture_time']:
            for k in range(self.nb_components - 1):
                regularity += self.priors['rupture_time'][k].compute_log_likelihood(self.fixed_effects['rupture_time'][k])

        if not self.is_frozen['noise_variance']:
            regularity += self.priors['noise_variance'].compute_log_likelihood(self.fixed_effects['noise_variance'])

        return regularity

    def _compute_class2_priors_regularity(self, template_data, momenta, mod_matrix):
        """
        Fully torch.
        Prior terms of the class 2 fixed effects, i.e. those for which we do not know a close-form update. 
        Derivative wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0
        if not self.is_frozen['template_data']:
            for key, value in template_data.items():
                regularity += self.priors['template_data'][key].compute_log_likelihood_torch(value)

        if not self.is_frozen['momenta']:
            regularity += self.priors['momenta'].compute_log_likelihood_torch(momenta)
            
        if not self.is_frozen['modulation_matrix']:
            regularity += self.priors['modulation_matrix'].compute_log_likelihood_torch(mod_matrix)        

        return regularity
    
    def _compute_random_effects_regularity(self, sources):
        """
        Fully torch.
        """
        number_of_subjects = sources.shape[0]
        regularity = 0.0

        for i in range(number_of_subjects):
            regularity += self.individual_random_effects['sources'].compute_log_likelihood_torch(sources[i])

        return regularity

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################
    def compute_residuals(self, dataset, individual_RER = None, k = False, option = None):

        template_data, template_points, momenta, mod_matrix, tR, _ = \
        self._fixed_effects_to_torch_tensors()
        sources = self._individual_RER_to_torch_tensors(individual_RER)
        
        self._update_geodesic(dataset, template_points, momenta, mod_matrix, tR)
        
        res = self.compute_residuals_(dataset.times, dataset.deformable_objects, template_data, sources)
        
        return sum(res, [])

    def compute_residuals_(self, target_times, target_objects, template_data, sources):
        residuals = [] 
        t1= perf_counter()

        for i in range(len(target_objects)):
            residuals_i = []

            # Compute the distance between obs_i,j and Exp_y(tij)(s_i)
            for j, (time, target) in enumerate(zip(target_times[i], target_objects[i])):
                deformed_points = self.piecewise_geodesic.get_template_points(time, sources[i], device=self.device)
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                residual = self.multi_object_attachment.compute_distances(deformed_data, self.template, target)
                residuals_i.append(residual.cpu())
            residuals.append(residuals_i)
        t2= perf_counter()

        self.current_residuals = sum(residuals, [])
                
        return residuals

    def _compute_batch_attachment_and_regularity(self, target_times, target_objects, template_data, template_points, 
                                                momenta, mod_matrix, sources, mode):        
        residuals = self.compute_residuals_(target_times, target_objects, template_data, sources)

        # Individual attachments
        noise_variance = move_data(self.fixed_effects['noise_variance'], device=self.device)

        attachment = 0.0
        for i in range(len(residuals)):
            for j in range(len(residuals[i])):
                attachment -= 0.5 * torch.sum(residuals[i][j] / noise_variance)

        # Regularity  
        if mode == 'complete':
            regularity = self._compute_random_effects_regularity(sources)
            regularity += self._compute_class1_priors_regularity()
        if mode in ['complete', 'class2']:
            regularity += self._compute_class2_priors_regularity(template_data, momenta, mod_matrix)

        return attachment, regularity

    
    def _compute_attachment_and_regularity(self, dataset, template_data, template_points, 
                                            momenta, mod_matrix, sources, mode):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """
        # update the geodesic times again to fit the batch in mini batch GD
        tmin = math.trunc(float(min(dataset.times)))
        tmax = math.ceil(float(max(dataset.times)))
        
        self.piecewise_geodesic.set_tmin(tmin)
        self.piecewise_geodesic.set_tmax(tmax)
        self.piecewise_geodesic.update()

        return self._compute_batch_attachment_and_regularity(dataset.times, dataset.deformable_objects, template_data, 
                                                            template_points, momenta, mod_matrix, sources, mode)
    
    def compute_mini_batch_gradient(self, batch, dataset, individual_RER, mode = 'complete', with_grad=True):
        template_data, template_points, momenta, mod_matrix, tR, _ \
        = self._fixed_effects_to_torch_tensors(with_grad)

        sources = self._individual_RER_to_torch_tensors(individual_RER, with_grad and mode == 'complete')
        
        self._update_geodesic(dataset, template_points, momenta, mod_matrix, tR)
        
        # get target times and objects from the batch
        target_times = [t[0] for t in batch]
        target_objects = [t[1] for t in batch]
                
        attachement, regularity = self._compute_batch_attachment_and_regularity(target_times, target_objects, 
                                                                                template_data, template_points, 
                                                                                momenta, mod_matrix)
        return self.compute_gradients(attachement, regularity, template_data, template_points,
                                        momenta, mod_matrix, tR, sources, mode, with_grad=with_grad)
        

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
        template_data, _, _ = self.prepare_geodesic(dataset)
        sources = self._individual_RER_to_torch_tensors(individual_RER)
        
        deformed_points = self.piecewise_geodesic.get_template_points(dataset.times[j][0], sources[j])
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)

        if dist in ["current", "varifold"]:
            return self.multi_object_attachment.compute_vtk_distance(deformed_data, self.template, dataset.deformable_objects[j][0], dist)
        elif dist in ["ssim", "mse"]:
            return self.multi_object_attachment.compute_ssim_distance(deformed_data, self.template, dataset.deformable_objects[j][0], dist)

    def compute_flow_curvature(self, dataset, time, curvature = "gaussian"):
        template_data, _, _ = self.prepare_geodesic(dataset)

        deformed_points = self.piecewise_geodesic.geodesic.get_template_points(time)
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        
        for i, obj1 in enumerate(self.template.object_list):
            obj1.polydata.points = deformed_data[self.points].cpu().numpy()
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
                obj1.polydata.points = data[self.points][0:obj1.get_number_of_points()]
                obj1.curvature_metrics(curvature)
            
                return self.template
             
        if iter == 0:
            return self.compute_initial_curvature(dataset, j, curvature)
          
        template_data, _, _ = self.prepare_geodesic(dataset)
        sources = self._individual_RER_to_torch_tensors(individual_RER)
        
        deformed_points = self.piecewise_geodesic.get_template_points(dataset.times[j][0], sources[j])
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        
        # Compute deformed template curvature
        for i, obj1 in enumerate(self.template.object_list):
            obj1.polydata.points = deformed_data[self.points][0:obj1.get_number_of_points()].cpu().numpy()
            obj1.curvature_metrics(curvature)
        
        return self.template
    
    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################
    def _individual_RER_to_torch_tensors(self, individual_RER, with_grad = False):
        """
        Convert the input individual_RER into torch tensors.
        """
        # Sources.
        sources = individual_RER['sources']
        sources = move_data(sources, requires_grad=with_grad, device=self.device)

        return sources

    def _fixed_effects_to_torch_tensors(self, with_grad = False):
        """
        Convert the fixed_effects into torch tensors.
        """
        # Template data.
        template_data = {key: move_data(value, device=self.device,
                                        requires_grad=(not self.is_frozen['template_data'] and with_grad))
                         for key, value in self.get_template_data().items()}

        # Template points.
        template_points = {key: move_data(value, device=self.device,
                                            requires_grad=(not self.is_frozen['template_data'] and with_grad)) 
                            for key, value in self.template.get_points().items()}

        liste = [template_data, template_points]

        for k, v in self.fixed_effects.items():
            if not isinstance(v, dict):
                effect = move_data(v, requires_grad=(with_grad and not self.is_frozen[k]), device=self.device)
                liste.append(effect)

        return tuple(liste)
        

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################        
    def prepare_geodesic(self, dataset):
        template_data, template_points, momenta, mod_matrix, tR, _ = \
        self._fixed_effects_to_torch_tensors()
        
        self._update_geodesic(dataset, template_points, momenta, mod_matrix, tR)

        return template_data, dataset.times, dataset.deformable_objects
    
    def _update_geodesic(self, dataset, template_points, momenta, mod_matrix, tR):
        """
        Tries to optimize the computations, by avoiding repetitions of shooting / flowing / parallel transporting.
        no particular optimization is carried.
        In the opposite case, the spatiotemporal reference frame will be more subtly updated.
        """
        cp = move_data(self.control_points, device=self.device)
        
        tmin = math.trunc(float(min(dataset.times)))
        tmax = math.ceil(float(max(dataset.times)))

        # no grad_fn in here. Normal
        if self.piecewise_geodesic_is_modified:
            self.piecewise_geodesic.set_template_points_tR(template_points)
            self.piecewise_geodesic.set_control_points_tR(cp)
            self.piecewise_geodesic.set_momenta_tR(momenta)
            self.piecewise_geodesic.set_modulation_matrix_tR(mod_matrix)
            self.piecewise_geodesic.set_tR(tR)
            self.piecewise_geodesic.set_t0(tR[self.template_index]) #ajout fg
            self.piecewise_geodesic.set_tmin(tmin)
            self.piecewise_geodesic.set_tmax(tmax)
            self.piecewise_geodesic.update()

        self.piecewise_geodesic_is_modified = False

    def write(self, dataset, individual_RER, output_dir, iteration, 
                write_adjoint_parameters=False, write_all = True):
        self._write_model_predictions(output_dir, individual_RER, dataset, write_adjoint_parameters, write_all)
        self._write_model_parameters(output_dir, iteration, write_all)

        #residuals = self.compute_residuals(dataset, individual_RER)

        #write_2D_list(residuals, output_dir, self.name + "__EstimatedParameters__Residuals.txt")

    def _write_model_predictions(self, output_dir, individual_RER, dataset=None, 
                                 write_adjoint_parameters=False, write_all = True):

        # Initialize ---------------------------------------------------------------------------------------------------
        template_data, target_times, _ = self.prepare_geodesic(dataset) 
        sources = self._individual_RER_to_torch_tensors(individual_RER, False)  

        # Write --------------------------------------------------------------------------------------------------------
        # write Geometric modes + optionally writes flow of A0, 
        # + call geodesic.write (trajectory)
        self.piecewise_geodesic.write(self.name, self.objects_extension, self.template, template_data,
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
                    for ext in self.objects_extension:
                        name = '{}__Reconstruction__subject__{}__tp_{}__age_{}{}'.format(self.name, subject_id, j, time, ext) 
                        names.append(name)
                    deformed_points = self.piecewise_geodesic.get_template_points(time, sources[j], device = device)
                    deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                    self.template.write(output_dir, names,
                                        {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

    def output_path(self, output_dir, dataset):
        self.cp_path = op.join(output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")
        self.momenta_path = op.join(output_dir, self.name + "__EstimatedParameters__Momenta.txt")
        
        # if self.piecewise_geodesic.geodesic.exponential[0].number_of_time_points is None:
        #     self.prepare_geodesic(dataset)
            
        try:
            self.piecewise_geodesic.geodesic.output_path(self.name, self.objects_extension, output_dir)
        except:
            pass

    def _write_model_parameters(self, output_dir, iteration, write_all = True):

        if write_all:
            template_names = []
            for ext in self.objects_extension:
                n = '__Fixed__Template_' if self.is_frozen['template_data'] else '__EstimatedParameters__Template_' 
                    
                aux = '{}{}__tp_{}__age_{}{}'.format(self.name, n,
                self.piecewise_geodesic.exponential.number_of_time_points - 1, self.get_rupture_time()[0], ext) 
            template_names.append(aux)
            self.template.write(output_dir, template_names)

            write_2D_array(self.get_modulation_matrix(), output_dir,
                           "{}__EstimatedParameters__ModulationMatrix.txt".format(self.name))

            if self.objects_extension[0] != ".vtk":
                #Control points.
                write_2D_array(self.control_points, output_dir, "{}__ControlPoints.txt".format(self.name))

                # Momenta.
                write_3D_array(self.get_momenta(), output_dir, "{}__EstimatedParameters__Momenta.txt".format(self.name))
            else:
                # Fuse control points and momenta for paraview display
                for c in range(self.nb_components):
                    if self.dimension == 3:
                        concatenate_for_paraview(self.get_momenta()[c], self.control_points, output_dir, 
                                        "{}__EstimatedParameters__Fusion_CP_Momenta__component_{}_iter_.vtk".format(self.name, c, iteration))

