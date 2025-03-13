import math

import torch

from ...support.kernels import factory
from ...core import default
from ...core.model_tools.deformations.exponential import Exponential
from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...core.models.model_functions import initialize_momenta, initialize_covariance_momenta_inverse, \
                                            initialize_cp
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...in_out.array_readers_and_writers import *
from ...in_out.dataset_functions import template_metadata, compute_noise_dimension
from ...support.utilities import get_best_device, move_data, detach
from ...support.probability_distributions.inverse_wishart_distribution import InverseWishartDistribution
from ...support.probability_distributions.multi_scalar_inverse_wishart_distribution import \
    MultiScalarInverseWishartDistribution
from ...support.probability_distributions.normal_distribution import NormalDistribution

import logging
logger = logging.getLogger(__name__)


class BayesianAtlas(AbstractStatisticalModel):
    """
    Bayesian atlas object class.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, template_specifications,
                 deformation_kernel_width=default.deformation_kernel_width,
                 n_time_points=default.n_time_points,
                 freeze_template=default.freeze_template,
                 initial_cp=None,
                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='BayesianAtlas')

        # Global-like attributes.
        self.deformation_kernel_width = deformation_kernel_width

        # Declare model structure.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['cp'] = None
        self.fixed_effects['covariance_momenta_inverse'] = None
        self.fixed_effects['noise_variance'] = None

        self.freeze_template = freeze_template
        self.freeze_momenta = False

        self.priors['covariance_momenta'] = InverseWishartDistribution()
        self.priors['noise_variance'] = MultiScalarInverseWishartDistribution()

        self.individual_random_effects['momenta'] = NormalDistribution()

        # Deformation.
        self.exponential = Exponential(kernel=factory(kernel_width=deformation_kernel_width),
                                        n_time_points=n_time_points)

        self.device = get_best_device()

        # Template.
        (object_list, self.extensions, objects_noise_variance, self.multi_object_attachment) = \
                                                    template_metadata(template_specifications)

        self.template = DeformableMultiObject(object_list)
        self.dimension = self.template.dimension
        self.objects_noise_dimension = compute_noise_dimension(self.template, self.multi_object_attachment,
                                                                self.objects_name)
        self.number_of_objects = len(self.template.object_list)

        # Template data.
        self.fixed_effects['template_data'] = self.template.get_data()

        # Control points.
        self.fixed_effects['cp'] = initialize_cp(initial_cp, self.template, deformation_kernel_width)
        self.number_of_cp = len(self.fixed_effects['cp'])

        # Covariance momenta.
        self.fixed_effects['covariance_momenta_inverse'] = initialize_covariance_momenta_inverse(
            self.fixed_effects['cp'], self.exponential.kernel, self.dimension)
        self.priors['covariance_momenta'].scale_matrix = np.linalg.inv(self.fixed_effects['covariance_momenta_inverse'])
	
        # Noise variance.
        self.fixed_effects['noise_variance'] = np.array(objects_noise_variance)
        self.objects_noise_variance_prior_norm_dof = [elt['noise_variance_prior_norm_dof']
                                                            for elt in template_specifications.values()]
        self.objects_noise_variance_prior_scale_std = [elt['noise_variance_prior_scale_std']
                                                       for elt in template_specifications.values()]

        # Momenta random effect.
        self.individual_random_effects['momenta'].mean = np.zeros((self.number_of_cp * self.dimension,))
        self.individual_random_effects['momenta'].set_covariance_inverse(
            self.fixed_effects['covariance_momenta_inverse'])

        # Ajouts fg
        self.number_of_subjects = 1

    def initialize_random_effects_realization(self, number_of_subjects,
                initial_momenta=default.initial_momenta,
                covariance_momenta_prior_norm_dof=default.covariance_momenta_prior_norm_dof,
                **kwargs):

        # Initialize the random effects realization.
        individual_RER = {'momenta': initialize_momenta(initial_momenta, self.number_of_cp, 
                                                        self.dimension, number_of_subjects)}

        # Initialize the corresponding priors.
        self.priors['covariance_momenta'].dof = \
            number_of_subjects * covariance_momenta_prior_norm_dof

        return individual_RER

    def initialize_noise_variance(self, dataset, individual_RER):
        # Prior on the noise variance (inverse Wishart: degrees of freedom parameter).
        for k, norm_dof in enumerate(self.objects_noise_variance_prior_norm_dof):
            dof = dataset.number_of_subjects * norm_dof * self.objects_noise_dimension[k]
            self.priors['noise_variance'].dof.append(dof)

        # Prior on the noise variance (inverse Wishart: scale scalars parameters).
        template_data, template_points, cp = self._fixed_effects_to_torch_tensors(False)
        momenta = self._individual_RER_to_torch_tensors(individual_RER, False)

        residuals_per_object = sum(self._compute_residuals(
            dataset, template_data, template_points, cp, momenta))
        for k, scale_std in enumerate(self.objects_noise_variance_prior_scale_std):
            if scale_std is None:
                self.priors['noise_variance'].scale_scalars.append(
                                    0.01 * detach(residuals_per_object[k])
                    / self.priors['noise_variance'].dof[k])
            else:
                self.priors['noise_variance'].scale_scalars.append(scale_std ** 2)

        # New, more informed initial value for the noise variance.
        self.fixed_effects['noise_variance'] = np.array(self.priors['noise_variance'].scale_scalars)

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
    def get_cp(self):
        return self.fixed_effects['cp']

    def set_cp(self, cp):
        self.fixed_effects['cp'] = cp
        self.number_of_cp = len(cp)
    
    def get_momenta(self):
        return self.individual_random_effects['momenta']

    # Covariance momenta inverse ---------------------------------------------------------------------------------------
    def get_covariance_momenta_inverse(self):
        return self.fixed_effects['covariance_momenta_inverse']

    def set_covariance_momenta_inverse(self, cmi):
        self.fixed_effects['covariance_momenta_inverse'] = cmi
        self.individual_random_effects['momenta'].set_covariance_inverse(cmi)

    def set_covariance_momenta(self, cm):
        self.set_covariance_momenta_inverse(np.linalg.inv(cm))

    # Noise variance ---------------------------------------------------------------------------------------------------
    def get_noise_variance(self):
        return self.fixed_effects['noise_variance']

    def set_noise_variance(self, nv):
        self.fixed_effects['noise_variance'] = nv

    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self, mode='class2'):
        out = {}

        if mode == 'class2':
            if not self.freeze_template:
                for key, value in self.fixed_effects['template_data'].items():
                    out[key] = value

        elif mode == 'all':
            for key, value in self.fixed_effects['template_data'].items():
                out[key] = value
            out['cp'] = self.fixed_effects['cp']
            out['covariance_momenta_inverse'] = self.fixed_effects['covariance_momenta_inverse']
            out['noise_variance'] = self.fixed_effects['noise_variance']

        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template:
            template_data = {key: fixed_effects[key] for key in self.fixed_effects['template_data'].keys()}
            self.set_template_data(template_data)

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def compute_log_likelihood(self, dataset, individual_RER, mode='complete', with_grad=False,
                               modified_individual_RER='all'):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
         and indRER.
        Start by updating the class 1 fixed effects.

        :param dataset: LongitudinalDataset instance
        :param individual_RER: Dictionary of individual random effects realizations.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """

        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        template_data, template_points, cp = self._fixed_effects_to_torch_tensors(with_grad)
        momenta = self._individual_RER_to_torch_tensors(individual_RER, with_grad and mode == 'complete')

        # Deform, update, compute metrics ------------------------------------------------------------------------------
        residuals = self._compute_residuals(dataset, template_data, template_points, cp, momenta)

        # Update the fixed effects only if the user asked for the complete log likelihood.
        if mode == 'complete':
            sufficient_statistics = self.compute_sufficient_statistics(dataset, individual_RER, residuals)
            self.update_fixed_effects(dataset, sufficient_statistics)

        # Compute the attachment, with the updated noise variance parameter in the 'complete' mode.
        attachments = self._compute_individual_attachments(residuals)
        attachment = torch.sum(attachments)

        # Compute the regularity terms according to the mode.
        regularity = torch.from_numpy(np.array(0.0)).type(default.tensor_scalar_type)
        if mode == 'complete':
            regularity = self._compute_random_effects_regularity(momenta)
            regularity += self._compute_class1_priors_regularity()
        if mode in ['complete', 'class2']:
            regularity += self._compute_class2_priors_regularity(template_data, cp)
        
        return self.compute_gradients(attachment, attachments, regularity, template_data, template_points,
                                        cp, momenta, mode, with_grad)
                             
    def compute_mini_batch_gradient(self, batch, dataset, individual_RER, mode = 'complete', with_grad=True):
        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        template_data, template_points, cp = self._fixed_effects_to_torch_tensors(with_grad)
        momenta = self._individual_RER_to_torch_tensors(individual_RER, with_grad and mode == 'complete')

        # Deform, update, compute metrics ------------------------------------------------------------------------------
        residuals = self._compute_batch_residuals(batch, template_data, template_points, cp, momenta)

        # Update the fixed effects only if the user asked for the complete log likelihood.
        if mode == 'complete':
            sufficient_statistics = self.compute_sufficient_statistics(dataset, individual_RER, residuals)
            self.update_fixed_effects(dataset, sufficient_statistics)

        # Compute the attachment, with the updated noise variance parameter in the 'complete' mode.
        attachments = self._compute_individual_attachments(residuals)
        attachment = torch.sum(attachments)

        # Compute the regularity terms according to the mode.
        regularity = torch.from_numpy(np.array(0.0)).type(default.tensor_scalar_type)
        if mode == 'complete':
            regularity = self._compute_random_effects_regularity_batch(batch, momenta)
            regularity += self._compute_class1_priors_regularity()
        if mode in ['complete', 'class2']:
            regularity += self._compute_class2_priors_regularity(template_data, cp)
        
        return self.compute_gradients(attachment, attachments, regularity, template_data, template_points,
                                        cp, momenta, mode, with_grad)

    def compute_gradients(self, attachment, attachments, regularity, template_data, template_points,
                           cp, momenta, mode, with_grad=False):
        # Compute gradient if needed -----------------------------------------------------------------------------------
        if with_grad:
            total = regularity + attachment
            total.backward()

            gradient = {}
            if not self.freeze_template:
                if 'landmark_points' in template_data.keys():
                    gradient['landmark_points'] = template_points['landmark_points'].grad
                if 'image_intensities' in template_data.keys():
                    gradient['image_intensities'] = template_data['image_intensities'].grad

            if mode == 'complete':
                gradient['momenta'] = momenta.grad

            gradient = {key: detach(value) for key, value in gradient.items()}   

            # Return as appropriate.
            if mode in ['complete', 'class2']:
                return detach(attachment), detach(regularity), gradient
            elif mode == 'model':
                return detach(attachments), gradient

        else:
            if mode in ['complete', 'class2']:
                return detach(attachments), detach(regularity)
            elif mode == 'model':
                return detach(attachments)

    def compute_sufficient_statistics(self, dataset, individual_RER, residuals=None, model_terms=None):
        """
        Compute the model sufficient statistics.
        """
        targets = [[i,target[0]] for i, target in enumerate(dataset.deformable_objects)]

        return self.compute_sufficient_statistics_batch(targets, individual_RER, residuals=None, model_terms=None)
    
    def compute_sufficient_statistics_batch(self, targets, individual_RER, residuals=None, model_terms=None):
        sufficient_statistics = {}

        # Empirical momenta covariance ---------------------------------------------------------------------------------
        momenta = individual_RER['momenta']
        sufficient_statistics['S1'] = np.zeros((momenta[0].size, momenta[0].size))
        for i, _ in targets:
            sufficient_statistics['S1'] += np.dot(momenta[i].reshape(-1, 1), momenta[i].reshape(-1, 1).transpose())

        # Empirical residuals variances, for each object ---------------------------------------------------------------
        sufficient_statistics['S2'] = np.zeros((self.number_of_objects,))

        # Trick to save useless computations. Could be extended to work in the multi-object case as well ...
        if model_terms is not None and self.number_of_objects == 1:
            sufficient_statistics['S2'][0] += - 2 * np.sum(model_terms) * self.get_noise_variance()
            return sufficient_statistics

        # Standard case.
        if residuals is None:
            template_data, template_points, cp = self._fixed_effects_to_torch_tensors(False)
            momenta = self._individual_RER_to_torch_tensors(individual_RER, False)
            residuals = self._compute_batch_residuals(targets, template_data, template_points, cp, momenta)
            residuals = [torch.sum(residuals_i) for residuals_i in residuals]

        for i in range(len(targets)):
            sufficient_statistics['S2'] += detach(residuals[i])

        return sufficient_statistics

    def update_fixed_effects(self, dataset, sufficient_statistics):
        """
        Updates the fixed effects based on the sufficient statistics, maximizing the likelihood.
        """
        # Covariance of the momenta update.
        prior_scale_matrix = self.priors['covariance_momenta'].scale_matrix
        prior_dof = self.priors['covariance_momenta'].dof
        covariance_momenta = (sufficient_statistics['S1'] + prior_dof * np.transpose(prior_scale_matrix)) \
                             / (dataset.number_of_subjects + prior_dof)
        self.set_covariance_momenta(covariance_momenta)

        # Variance of the residual noise update.
        noise_variance = np.zeros((self.number_of_objects,))
        prior_scale_scalars = self.priors['noise_variance'].scale_scalars
        prior_dofs = self.priors['noise_variance'].dof
        for k in range(self.number_of_objects):
            noise_variance[k] = (sufficient_statistics['S2'][k] + prior_scale_scalars[k] * prior_dofs[k]) \
                                / float(dataset.number_of_subjects * self.objects_noise_dimension[k] + prior_dofs[k])
        self.set_noise_variance(noise_variance)

    def initialize_template_attributes(self, template_specifications):
        """
        Sets the Template, TemplateObjectsName, TemplateObjectsNameExtension, TemplateObjectsNorm,
        TemplateObjectsNormKernelType and TemplateObjectsNormKernelWidth attributes.
        """

        t_list, t_name, t_name_extension, t_noise_variance, t_multi_object_attachment = \
            template_metadata(template_specifications)

        self.template.object_list = t_list
        self.objects_name = t_name
        self.extensions = t_name_extension
        self.multi_object_attachment = t_multi_object_attachment

        self.template.update(self.dimension)
        self.objects_noise_dimension = compute_noise_dimension(self.template, self.multi_object_attachment,
                                                               self.dimension)
    
    def mini_batches(self, dataset, number_of_batches):
        """
        Split randomly the dataset into batches of size batch_size
        """
        batch_size = len(dataset.deformable_objects)//number_of_batches
        print("batch_size", batch_size)
        targets = [[i,target[0]] for i, target in enumerate(dataset.deformable_objects)]
        targets_copy = targets.copy()
        np.random.shuffle(targets_copy)

        mini_batches = []
        n_minibatches = len(targets_copy) // batch_size    

        for i in range(n_minibatches):
            mini_batch = targets_copy[i * batch_size:(i + 1)*batch_size]
            mini_batches.append(mini_batch)
        if len(targets_copy) % batch_size != 0:
            mini_batch = targets_copy[i * batch_size:len(targets_copy)]
            if len(mini_batches) > batch_size/2: #if last batch big enough
                mini_batches.append(mini_batch)
            else:
                mini_batches[-1] += mini_batch
        
        return mini_batches

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _compute_attachment(self, residuals):
        """
        Fully torch.
        """
        return torch.sum(self._compute_individual_attachments(residuals))

    def _compute_individual_attachments(self, residuals):
        """
        Fully torch.
        """
        number_of_subjects = len(residuals)
        attachments = torch.zeros((number_of_subjects,)).type(default.tensor_scalar_type)
        for i in range(number_of_subjects):
            attachments[i] = - 0.5 * torch.sum(residuals[i] / move_data(
                            self.fixed_effects['noise_variance'], device=self.device))
        return attachments
    
    def _compute_random_effects_regularity(self, momenta):
        """
        Fully torch.
        """
        targets = [[i, None] for i in range(momenta.shape[0])]

        return self._compute_random_effects_regularity_batch(targets, momenta)
    
    def _compute_random_effects_regularity_batch(self, targets, momenta):
        """
        Fully torch.
        """
        number_of_subjects = len(targets)
        regularity = 0.0

        # Momenta random effect.
        for i, _ in targets:
            regularity += self.individual_random_effects['momenta'].compute_log_likelihood_torch(
                                                                momenta[i])

        # Noise random effect.
        for k in range(self.number_of_objects):
            regularity -= 0.5 * self.objects_noise_dimension[k] * number_of_subjects \
                          * math.log(self.fixed_effects['noise_variance'][k])

        return regularity
    
    def _compute_class1_priors_regularity(self):
        """
        Fully torch.
        Prior terms of the class 1 fixed effects, i.e. those for which we know a close-form update. No derivative
        wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0

        # Covariance momenta prior.
        regularity += self.priors['covariance_momenta'].compute_log_likelihood(
            self.fixed_effects['covariance_momenta_inverse'])

        # Noise variance prior.
        regularity += self.priors['noise_variance'].compute_log_likelihood(self.fixed_effects['noise_variance'])

        return regularity
    

    def _compute_class2_priors_regularity(self, template_data, cp):
        """
        Fully torch.
        Prior terms of the class 2 fixed effects, i.e. those for which we do not know a close-form update. Derivative
        wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0

        # Prior on template_data fixed effects (if not frozen). None implemented yet TODO.
        if not self.freeze_template:
            regularity += 0.0

        return regularity

    def _compute_residuals(self, dataset, template_data, template_points, cp, momenta):
        targets = [[i, target[0]] for i, target in enumerate(dataset.deformable_objects)]

        return self._compute_batch_residuals(targets, template_data, template_points, cp, momenta)
    
    def _compute_batch_residuals(self, targets, template_data, template_points, cp, momenta):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """
        # Deform -------------------------------------------------------------------------------------------------------
        residuals = []

        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_cp(cp)

        for i, target in targets:
            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.move_data_to_(device=self.device)
            self.exponential.update()
            deformed_points = self.exponential.get_template_points()
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)
            residuals.append(self.multi_object_attachment.compute_distances(deformed_data, self.template, target))

        return residuals


    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad):
        """
        Convert the input fixed_effects into torch tensors.
        """
        # Template data.
        template_data = self.fixed_effects['template_data']
        template_data = {key: torch.from_numpy(value).type(default.tensor_scalar_type).requires_grad_(
            not self.freeze_template and with_grad) for key, value in template_data.items()}

        # Template points.
        template_points = self.template.get_points()
        template_points = {key: torch.from_numpy(value).type(default.tensor_scalar_type).requires_grad_(
            not self.freeze_template and with_grad) for key, value in template_points.items()}

        # Control points.
        cp = self.fixed_effects['cp']
        cp = torch.from_numpy(cp).type(default.tensor_scalar_type).requires_grad_(with_grad)

        return template_data, template_points, cp

    def _individual_RER_to_torch_tensors(self, individual_RER, with_grad):
        """
        Convert the input individual_RER into torch tensors.
        """
        # Momenta.
        momenta = individual_RER['momenta']
        momenta = torch.from_numpy(momenta).type(default.tensor_scalar_type).requires_grad_(with_grad)
        return momenta


    ####################################################################################################################
    ### Printing and writing methods:
    ####################################################################################################################

    def print(self, individual_RER):
        pass

    def write(self, dataset, individual_RER, output_dir, update_fixed_effects=True,
              write_residuals=True):

        # Write the model predictions, and compute the residuals at the same time.
        residuals = self._write_model_predictions(dataset, individual_RER, output_dir,
                                                  compute_residuals=(update_fixed_effects or write_residuals))

        # Optionally update the fixed effects.
        if update_fixed_effects:
            sufficient_statistics = self.compute_sufficient_statistics(dataset, individual_RER, residuals)
            self.update_fixed_effects(dataset, sufficient_statistics)

        # Write residuals.
        if write_residuals:
            residuals_list = [[residuals_i_k for residuals_i_k in residuals_i]
                              for residuals_i in residuals]
            write_2D_list(residuals_list, output_dir, self.name + "__Estimated__Residuals.txt")

        # Write the model parameters.
        self._write_model_parameters(individual_RER, output_dir)

    def _write_model_predictions(self, dataset, individual_RER, output_dir, compute_residuals=True, write = True):

        # Initialize.
        template_data, template_points, cp = self._fixed_effects_to_torch_tensors(False)
        momenta = self._individual_RER_to_torch_tensors(individual_RER, False)

        # Deform, write reconstructions and compute residuals.
        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_cp(cp)

        residuals = []  # List of torch 1D tensors. Individuals, objects.
        for i, subject_id in enumerate(dataset.subject_ids):
            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.move_data_to_(device=self.device)
            self.exponential.update()

            deformed_points = self.exponential.get_template_points()
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)

            if compute_residuals:
                residuals.append(self.multi_object_attachment.compute_distances(
                    deformed_data, self.template, dataset.deformable_objects[i][0]))

            if write:
                names = []
                for k, (object_name, object_extension) \
                        in enumerate(zip(self.objects_name, self.extensions)):
                    name = self.name + '__Reconstruction__' + object_name + '__subject_' + subject_id + object_extension
                    names.append(name)
                self.template.write(output_dir, names, deformed_data)

        return residuals

    def _write_model_parameters(self, individual_RER, output_dir):
        # Template.
        template_names = [ template_name(self.name, ext = ext) for ext in self.extensions]
        self.template.write(output_dir, template_names)

        # Control points.
        write_2D_array(self.get_cp(), output_dir, self.name + "__Estimated__ControlPoints.txt")

        # Momenta.
        write_3D_array(individual_RER['momenta'], output_dir, self.name + "__Estimated__Momenta.txt")

        # Momenta covariance. modif fg
        #write_2D_array(self.get_covariance_momenta_inverse(), output_dir,
        #               self.name + "__Estimated__CovarianceMomentaInverse.txt")

        # Noise variance.
        write_2D_array(np.sqrt(self.get_noise_variance()), output_dir,
                       self.name + "__Estimated__NoiseStd.txt")
    

    def subject_residuals(self, subject, dataset, template_data, momenta, cp):
        """
        Compute residuals at each pixel/voxel between one subject and deformed template.
        """
        #deform template
        self.exponential.set_initial_cp(cp)
        self.exponential.set_initial_momenta(momenta[subject])
        self.exponential.update()

        deformed_points = self.exponential.get_template_points() #template points #tensor
        deformed_template = self.template.get_deformed_data(deformed_points, template_data) #dict containing tensor
        
        #get object intensities
        objet = dataset.deformable_objects[subject][0]
        objet_intensities = objet.get_data()["image_intensities"]
        target_intensities = move_data(objet_intensities, device=self.device, 
                                    dtype = next(iter(template_data.values())).dtype) #tensor not dict 
        residuals = (target_intensities - deformed_template['image_intensities']) ** 2
        
        return residuals, deformed_template
    
    def compute_residuals(self, dataset, current_iteration, save_every_n_iters, output_dir, individual_RER):
        """
        Compute residuals at each pixel/voxel between objects and deformed template.
        Save a heatmap of the residuals
        """
        #template_data, momenta = self.initialize_template_before_transformation()
        
        template_data, _, cp = self._fixed_effects_to_torch_tensors(False)
        momenta = self._individual_RER_to_torch_tensors(individual_RER, False)

        residuals_by_point = torch.zeros((template_data['image_intensities'].shape), 
                                        device=self.device, dtype=next(iter(template_data.values())).dtype)   #tensor not dict             

        for i, _ in enumerate(dataset.subject_ids):
            subject_residuals, _ = self.subject_residuals(i, dataset, template_data, momenta, cp)
            residuals_by_point += (1/dataset.number_of_subjects) * subject_residuals
                
        if current_iteration == 0:
            print("First residuals computed")
            self.initial_residuals = residuals_by_point.cpu().numpy()
        
        return residuals_by_point.cpu().numpy()
