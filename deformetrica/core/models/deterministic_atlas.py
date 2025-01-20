import math
import gc
import time
import sys
import os
import copy
import os.path as op
import warnings
import torch
from math import floor
import numpy as np
import cv2 as cv
import open3d as o3d
from random import sample
from ...support import kernels as kernel_factory
from ...core import default
from ...core.model_tools.deformations.exponential import Exponential
from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...core.models.model_functions import initialize_control_points, initialize_momenta, gaussian_kernel
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...in_out.array_readers_and_writers import *
from ...in_out.dataset_functions import create_template_metadata, create_mesh_attachements
from ...support import utilities

warnings.filterwarnings("ignore") #fg
logger = logging.getLogger(__name__)

def _subject_attachment_and_regularity(arg):
    """
    Auxiliary function for multithreading (cannot be a class method).
    """
    from .abstract_statistical_model import process_initial_data
    if process_initial_data is None:
        raise RuntimeError('process_initial_data is not set !')

    # Read arguments.
    (deformable_objects, multi_object_attachment, objects_noise_variance,
     freeze_template, freeze_momenta,
     exponential, sobolev_kernel, use_sobolev_gradient, gpu_mode) = process_initial_data
    (i, template, template_data, control_points, momenta, with_grad) = arg

    # start = time.perf_counter()
    device, device_id = utilities.get_best_device(gpu_mode=gpu_mode)
    # device, device_id = ('cpu', -1)
    if device_id >= 0:
        torch.cuda.set_device(device_id)

    # convert np.ndarrays to torch tensors. This is faster than transferring torch tensors to process.
    # template intensities
    template_data = {key: utilities.move_data(value, device=device, 
                                              requires_grad=with_grad and not freeze_template)
                     for key, value in template_data.items()}
    #template points
    template_points = {key: utilities.move_data(value, device=device, 
                                                requires_grad=with_grad and not freeze_template)
                       for key, value in template.get_points().items()}
    control_points = utilities.move_data(control_points, device=device, requires_grad=False)
    momenta = utilities.move_data(momenta, device=device, 
                                  requires_grad=with_grad and not freeze_momenta)

    assert torch.device(
        device) == control_points.device == momenta.device, 'control_points and momenta tensors must be on the same device. ' \
                                                            'device=' + device + \
                                                            ', control_points.device=' + str(control_points.device) + \
                                                            ', momenta.device=' + str(momenta.device)

    attachment, regularity = DeterministicAtlas._deform_and_compute_attachment_and_regularity(
        exponential, template_points, control_points, momenta,
        template, template_data, multi_object_attachment,
        deformable_objects[i], objects_noise_variance,
        device) #ajout fg

    res = DeterministicAtlas._compute_gradients(
        attachment, regularity, template_data,
        freeze_template, template_points,
        control_points,
        freeze_momenta, momenta,
        use_sobolev_gradient, sobolev_kernel,
        with_grad)

    
    return i, res 

class DeterministicAtlas(AbstractStatisticalModel):
    """
    Deterministic atlas object class.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, template_specifications, number_of_subjects,

                 dimension=default.dimension,
                 number_of_processes=default.number_of_processes,

                 deformation_kernel_width=default.deformation_kernel_width,

                 number_of_time_points=default.number_of_time_points,
                 use_rk2_for_shoot=default.use_rk2_for_shoot, 
                 use_rk2_for_flow=default.use_rk2_for_flow,

                 freeze_template=default.freeze_template,
                 use_sobolev_gradient=default.use_sobolev_gradient,
                 smoothing_kernel_width=default.smoothing_kernel_width,

                 initial_control_points=default.initial_control_points,
                 initial_cp_spacing=default.initial_cp_spacing,

                 initial_momenta=default.initial_momenta,
                 freeze_momenta=default.freeze_momenta,

                 gpu_mode=default.gpu_mode,
                 process_per_gpu=default.process_per_gpu,

                 kernel_regression = default.kernel_regression,
                 visit_ages = None,
                 time = None,

                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='DeterministicAtlas', number_of_processes=number_of_processes,
                                          gpu_mode=gpu_mode)
        
        # Global-like attributes.
        self.dimension = dimension

        # Declare model structure.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['control_points'] = None
        self.fixed_effects['momenta'] = None

        self.freeze_template = freeze_template
        self.freeze_momenta = freeze_momenta

        #ajout fg
        self.initial_cp_spacing = initial_cp_spacing #determines nb of points 
        self.gpu_mode = gpu_mode
        self.deformation_kernel_width = deformation_kernel_width

        # Template.
        (object_list, self.objects_name, self.objects_name_extension,
         self.objects_noise_variance, self.multi_object_attachment) = create_template_metadata(template_specifications,
                                                                                               self.dimension)

        self.template = DeformableMultiObject(object_list)
        self.number_of_objects = len(self.template.object_list)

        #self.multi_object_attachments_k = create_mesh_attachements(template_specifications, gpu_mode=gpu_mode)
        
        # Deformation.
        self.exponential = Exponential(kernel=kernel_factory.factory(gpu_mode=gpu_mode,
                                          kernel_width=deformation_kernel_width),
            number_of_time_points=number_of_time_points,
            use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)


        self.use_sobolev_gradient = use_sobolev_gradient
        self.smoothing_kernel_width = smoothing_kernel_width
        if self.use_sobolev_gradient:
            self.sobolev_kernel = kernel_factory.factory(gpu_mode=gpu_mode,
                                                         kernel_width=smoothing_kernel_width)

        # Template data.
        self.fixed_effects['template_data'] = self.template.get_data()

        self.points = 'image_intensities'
        if 'landmark_points' in self.fixed_effects['template_data'].keys():
            self.points = 'landmark_points'

        # Control points.
        self.fixed_effects['control_points'] = initialize_control_points(
            initial_control_points, self.template, self.initial_cp_spacing, deformation_kernel_width,
            self.dimension)
        self.number_of_control_points = len(self.fixed_effects['control_points'])

        # Momenta.
        self.fixed_effects['momenta'] = initialize_momenta(
            initial_momenta, self.number_of_control_points, self.dimension, number_of_subjects)
        self.number_of_subjects = number_of_subjects

        self.process_per_gpu = process_per_gpu
            
        self.kernel_regression = kernel_regression
        self.time = time
        self.visit_ages = visit_ages

        if self.kernel_regression:
            self.weights = [gaussian_kernel(self.time, age_s[0]) for age_s in self.visit_ages]
            self.total_weights = np.sum([gaussian_kernel(self.time, age_s[0]) for age_s in self.visit_ages])
            logger.info("Weights: {}".format([gaussian_kernel(self.time, age_s[0]) for age_s in self.visit_ages]))

        self.current_residuals = None

    def initialize_noise_variance(self, dataset, device='cpu'):
        if np.min(self.objects_noise_variance) < 0:
            template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(False,
                                                                                                           device=device)
            targets = dataset.deformable_objects
            targets = [target[0] for target in targets]

            residuals_torch = []
            self.exponential.set_initial_template_points(template_points)
            self.exponential.set_initial_control_points(control_points)
            for i, target in enumerate(targets):
                self.exponential.set_initial_momenta(momenta[i])
                self.exponential.update()
                deformed_points = self.exponential.get_template_points()
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                residuals_torch.append(self.multi_object_attachment.compute_distances(
                    deformed_data, self.template, target))

            residuals = np.zeros((self.number_of_objects,))
            for i in range(len(residuals_torch)):
                residuals += residuals_torch[i].detach().cpu().numpy()

            # Initialize the noise variance hyper-parameter as a 1/100th of the initial residual.
            for k, obj in enumerate(self.objects_name):
                if self.objects_noise_variance[k] < 0:
                    nv = 0.01 * residuals[k] / float(self.number_of_subjects)
                    self.objects_noise_variance[k] = nv
                    logger.info('>> Automatically chosen noise std: %.4f [ %s ]' % (math.sqrt(nv), obj))
        
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
        # self.number_of_control_points = len(cp)

    # Momenta ----------------------------------------------------------------------------------------------------------
    def get_momenta(self):
        return self.fixed_effects['momenta']

    def set_momenta(self, mom):
        self.fixed_effects['momenta'] = mom
        
    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self):
        out = {}
        if not self.freeze_template:
            for key, value in self.fixed_effects['template_data'].items():
                out[key] = value
        if not self.freeze_momenta:
            out['momenta'] = self.fixed_effects['momenta']
                
        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template:
            template_data = {key: fixed_effects[key] for key in self.fixed_effects['template_data'].keys()}
            self.set_template_data(template_data)
        if not self.freeze_momenta:
            self.set_momenta(fixed_effects['momenta'])


    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def setup_multiprocess_pool(self, dataset):
        self._setup_multiprocess_pool(initargs=([target[0] for target in dataset.deformable_objects],
                                                self.multi_object_attachment,
                                                self.objects_noise_variance,
                                                self.freeze_template, 
                                                self.freeze_momenta, 
                                                self.exponential, self.sobolev_kernel, self.use_sobolev_gradient,
                                                self.gpu_mode))

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, population_RER, individual_RER, mode='complete', with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.

        :param fixed_effects: Dictionary of fixed effects.
        :param population_RER: Dictionary of population random effects realizations.
        :param individual_RER: Dictionary of individual random effects realizations.
        :param mode: Indicates which log_likelihood should be computed, between 'complete', 'model', and 'class2'.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """
        if self.number_of_processes > 1: # For when there are multiple GPUs
            targets = [target[0] for target in dataset.deformable_objects]
            args = [(i, self.template,
                     self.fixed_effects['template_data'],
                     self.fixed_effects['control_points'],
                     self.fixed_effects['momenta'][i],
                     with_grad) for i in range(len(targets))]

            start = time.perf_counter()
            results = self.pool.map(_subject_attachment_and_regularity, args, chunksize=1)  # TODO: optimized chunk size
            logger.debug('time taken for deformations : ' + str(time.perf_counter() - start))

            # Sum and return.
            attachment = 0.0
            regularity = 0.0

            if with_grad:
                
                gradient = {}
                if not self.freeze_template:
                    for key, value in self.fixed_effects['template_data'].items():
                        gradient[key] = np.zeros(value.shape)
                if not self.freeze_momenta:
                    gradient['momenta'] = np.zeros(self.fixed_effects['momenta'].shape)

                for i, (attachment_i, regularity_i, gradient_i) in results:
                    attachment += attachment_i
                    regularity += regularity_i
                    for key, value in gradient_i.items():
                        if key == 'momenta':
                            gradient[key][i] = value
                        else:
                            gradient[key] += value
                return attachment, regularity, gradient
            
            else:
                for _, (attachment_i, regularity_i, _) in results:
                    attachment += attachment_i
                    regularity += regularity_i

                return attachment, regularity

        else:
            device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)
            template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(with_grad,
                                                                                                           device=device)
            return self._compute_attachment_and_regularity(dataset, template_data, template_points, control_points,
                                                           momenta, with_grad, device=device) 

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    @staticmethod
    def _deform_and_compute_attachment_and_regularity(exponential, template_points, control_points, momenta,
                                                      template, template_data,
                                                      multi_object_attachment, deformable_objects,
                                                      objects_noise_variance,
                                                      device='cpu'):
        # Deform.
        exponential.set_initial_template_points(template_points)
        
        exponential.set_initial_control_points(control_points)
        exponential.set_initial_momenta(momenta)
        exponential.move_data_to_(device=device)
        exponential.update() #flow the template points according to the momenta using kernel.convolve

        # Compute attachment and regularity. (-> increase memory)
        deformed_points = exponential.get_template_points() #template points
        deformed_data = template.get_deformed_data(deformed_points, template_data) #template intensities after deformation
        
        #(observation) deformable multi object -> image -> torch.interpolate
        attachment = -multi_object_attachment.compute_weighted_distance(deformed_data, template, deformable_objects,
                                                                        objects_noise_variance)
        regularity = -exponential.get_norm_squared()

        assert torch.device(device) == attachment.device == regularity.device, 'attachment and regularity tensors must be on the same device. ' \
                                                               'device=' + device + \
                                                               ', attachment.device=' + str(attachment.device) + \
                                                               ', regularity.device=' + str(regularity.device)
        
        return attachment, regularity
         
    @staticmethod
    def _compute_gradients(attachment, regularity, template_data,
                           freeze_template, template_points,
                           control_points,
                           freeze_momenta, momenta, 
                           use_sobolev_gradient, sobolev_kernel,
                           with_grad=False):
        if with_grad:
            total_for_subject = attachment + regularity #torch tensor
                        
            total_for_subject.backward() #compute gradient  -> the tensors stay in memory until this point
               
            gradient = {}
            if not freeze_template:
                if 'landmark_points' in template_data.keys():
                    assert template_points['landmark_points'].grad is not None, 'Gradients have not been computed'
                    if use_sobolev_gradient:
                        gradient['landmark_points'] = sobolev_kernel.convolve(
                            template_data['landmark_points'].detach(), 
                            template_data['landmark_points'].detach(),
                            template_points['landmark_points'].grad.detach()).cpu().numpy()
                    else:
                        gradient['landmark_points'] = template_points['landmark_points'].grad.detach().cpu().numpy()
                if 'image_intensities' in template_data.keys():
                    assert template_data['image_intensities'].grad is not None, 'Gradients have not been computed'
                    gradient['image_intensities'] = template_data['image_intensities'].grad.detach().cpu().numpy()
            
            if not freeze_momenta:
                assert momenta.grad is not None, 'Gradients have not been computed'
                gradient['momenta'] = momenta.grad.detach().cpu().numpy()
                                                                        
            res = attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient

        else:
            res = attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy()

        return res
    
    
    def _compute_batch_gradient(self, targets, template_data, template_points, control_points, momenta,
                                        with_grad=False, device='cpu'):
        """
        Core part of the ComputeLogLikelihood methods. Torch input, numpy output.
        Single-thread version.
        """
        attachment = 0.
        regularity = 0.
        
        for i, target in targets:
            new_attachment, new_regularity = DeterministicAtlas._deform_and_compute_attachment_and_regularity(
                                            self.exponential, template_points, control_points, momenta[i],
                                            self.template, template_data, self.multi_object_attachment,
                                            target, self.objects_noise_variance, device=device)
            self.current_residuals = new_attachment.cpu()

            if self.kernel_regression:
                weight = gaussian_kernel(self.time, self.visit_ages[i][0])
                attachment += (weight/self.total_weights) * new_attachment 
                regularity += (weight/self.total_weights) * new_regularity
            else:
                attachment += new_attachment
                regularity += new_regularity            
            
        #attachment and regularity still stored in memory (with intermediate tensors) to compute gradients
        gradients = self._compute_gradients(attachment, regularity, template_data,
                                       self.freeze_template, template_points,
                                       control_points,
                                       self.freeze_momenta, momenta, 
                                       self.use_sobolev_gradient, self.sobolev_kernel,
                                       with_grad)

        return gradients
    
    def _compute_attachment_and_regularity(self, dataset, template_data, template_points, control_points, momenta,
                                            with_grad=False, device='cpu'):
        # Initialize.
        targets = [[i, target[0]] for i, target in enumerate(dataset.deformable_objects)]

        return self._compute_batch_gradient(targets, template_data, template_points, control_points, momenta,
                                            with_grad=with_grad, device=device)
    
    def compute_mini_batch_gradient(self, batch, dataset, population_RER, individual_RER, with_grad=True):
        device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(with_grad, device=device)                                    
        
        return self._compute_batch_gradient(batch, template_data, template_points, control_points, momenta,
                                            with_grad=with_grad, device=device)
    
    def mini_batches(self, dataset, number_of_batches):
        """
        Split randomly the dataset into batches of size batch_size
        """
        batch_size = len(dataset.deformable_objects)//number_of_batches
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
    
    def prepare_exponential(self, i, template_points, control_points, momenta, device):
                
        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)

        self.exponential.set_initial_momenta(momenta[i])
        self.exponential.move_data_to_(device=device)
        self.exponential.update()

        return 

    def compute_curvature(self, dataset, j = None, individual_RER = None, curvature = "gaussian", iter = None):
        """
            Compute object curvature (at iter 0) or deformed template to object curvature
        """
        device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)

        template_data, template_points, control_points, momenta = \
        self._fixed_effects_to_torch_tensors(False, device=device)

        # template curvature
        if j is None:
            obj = self.template
            data = self.template.get_data()
            for i, obj1 in enumerate(obj.object_list):
                obj1.polydata.points = data['landmark_points'][0:obj1.get_number_of_points()]
                obj1.curvature_metrics(curvature)
            
                return obj

        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)
        self.exponential.set_initial_momenta(momenta[j])
        self.exponential.move_data_to_(device=device)
        self.exponential.update() 
        deformed_points = self.exponential.get_template_points()
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        
        # Compute deformed template curvature
        obj = dataset.deformable_objects[j][0] if iter == 0 else self.template 

        for i, obj1 in enumerate(obj.object_list):
            if iter != 0:
                obj1.polydata.points = deformed_data['landmark_points'][0:obj1.get_number_of_points()].cpu().numpy()

            obj1.curvature_metrics(curvature)
        
        return obj

    def compute_residuals(self, dataset, individual_RER = None):
        device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta = \
        self._fixed_effects_to_torch_tensors(False, device=device)

        residuals = []
        for i, target in enumerate(dataset.deformable_objects):
            self.prepare_exponential(i, template_points, control_points, momenta, device)

            # Compute attachment
            deformed_points = self.exponential.get_template_points() #template points
            deformed_data = self.template.get_deformed_data(deformed_points, template_data) #template intensities after deformation

            att = self.multi_object_attachment.compute_weighted_distance(deformed_data, self.template, target[0],
                                                                        self.objects_noise_variance)
            
            residuals.append(att)
        
        return residuals

    def compute_residuals_per_point(self, dataset, individual_RER = None):
        device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta = \
        self._fixed_effects_to_torch_tensors(False, device=device)
        
        residuals_by_point = torch.zeros((template_data[self.points].shape), 
                                        device=device, dtype=next(iter(template_data.values())).dtype) 
        
        for i, target in enumerate(dataset.deformable_objects):
            self.prepare_exponential(i, template_points, control_points, momenta, device)
        
            # Compute attachment
            deformed_points = self.exponential.get_template_points() #template points
            deformed_data = self.template.get_deformed_data(deformed_points, template_data) #template intensities after deformation

            objet_intensities = target[0].get_data()[self.points]
            target_intensities = utilities.move_data(objet_intensities, device=device, 
                                                    dtype = next(iter(template_data.values())).dtype) #tensor not dict 
            residuals = (target_intensities - deformed_data[self.points]) ** 2

            
            
            residuals_by_point += residuals
        
        return residuals_by_point.cpu().numpy().flatten()
    

    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad, device='cpu'):
        """
        Convert the fixed_effects into torch tensors.
        """
        # Template data.
        template_data = self.fixed_effects['template_data']
        template_data = {key: utilities.move_data(value, device=device,
                                                  requires_grad=(not self.freeze_template and with_grad))
                         for key, value in template_data.items()}


        # Template points.
        template_points = self.template.get_points()
        template_points = {key: utilities.move_data(value, device=device, 
                                                    requires_grad= (not self.freeze_template and with_grad))
                           for key, value in template_points.items()}

        # Control points.
        control_points = self.fixed_effects['control_points']
        control_points = utilities.move_data(control_points, device=device, requires_grad=False)

        # Momenta.
        momenta = self.fixed_effects['momenta']
        momenta = utilities.move_data(momenta, device=device, 
                                      requires_grad=(not self.freeze_momenta and with_grad))

        return template_data, template_points, control_points, momenta #ajout fg

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, dataset, population_RER, individual_RER, output_dir, current_iteration, write_residuals=True, write_all = True):

        # Write the model predictions, and compute the residuals at the same time.
        self._write_model_predictions(dataset, individual_RER, output_dir, current_iteration,
                                                  compute_residuals=write_residuals)

        # Write the model parameters.
        self._write_model_parameters(output_dir, str(current_iteration))

    def _write_model_predictions(self, dataset, individual_RER, output_dir, current_iteration, 
                                 compute_residuals=True):
        device, _ = utilities.get_best_device(self.gpu_mode)

        # Initialize.
        template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(False, device=device)

        # Deform, write reconstructions and compute residuals.
        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)

        residuals = []  # List of torch 1D tensors. Individuals, objects.
        for i, subject_id in enumerate(dataset.subject_ids):
            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.update()

            # Writing the whole flow. -> modif fg
            names = []
            for k, object_name in enumerate(self.objects_name):
                name = self.name + '__flow__' + object_name + '__subject_' + subject_id
                names.append(name)
            
            deformed_points = self.exponential.get_template_points()
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)

            names = []
            for k, (object_name, object_extension) \
                    in enumerate(zip(self.objects_name, self.objects_name_extension)):
                
                if self.kernel_regression:
                    name = self.name + '__Reconstruction__' + object_name + '__subject_' + subject_id + "_age_" + str(self.visit_ages[i][0]) + object_extension
                else:
                    name = '{}__Reconstruction__{}__subject_{}{}'.format(self.name, object_name, subject_id, object_extension)
                names.append(name)

            self.template.write(output_dir, names,
                                {key: value.data.cpu().numpy() for key, value in deformed_data.items()})
            
        return residuals

    def output_path(self, output_dir, dataset):
        self.cp_path = op.join(output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")
        self.momenta_path = op.join(output_dir, self.name + "__EstimatedParameters__Momenta.txt")

        if not self.freeze_template:
            for i in range(len(self.objects_name)):
                self.template_path = op.join(output_dir, self.name + "__EstimatedParameters__Template_"  + self.objects_name[i] + self.objects_name_extension[i])

    def _write_model_parameters(self, output_dir, current_iteration):

        template_names = []
        for i in range(len(self.objects_name)):
            if self.kernel_regression:
                aux = self.name + "__EstimatedParameters__Template_time" + str(self.time) + "_" + self.objects_name[i] + self.objects_name_extension[i]
                aux = self.name + "__EstimatedParameters__Template_"  + self.objects_name[i] + "_" + str(current_iteration) + self.objects_name_extension[i]
                template_names.append(aux)
            else:
                if not self.freeze_template:
                    aux = self.name + "__EstimatedParameters__Template_"  + self.objects_name[i] + "_" + str(current_iteration) + self.objects_name_extension[i]
                    template_names.append(aux)
        if template_names:
            self.template.write(output_dir, template_names)

        template_names = []
        for i in range(len(self.objects_name)):
            aux = self.name + "__EstimatedParameters__Template_"  + self.objects_name[i] + self.objects_name_extension[i]
            template_names.append(aux)
        self.template.write(output_dir, template_names)
        
        write_2D_array(self.get_control_points(), output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")
        
        # Momenta.
        write_3D_array(self.get_momenta(), output_dir, self.name + "__EstimatedParameters__Momenta.txt")#.format(current_iteration))
        
        concatenate_for_paraview(self.get_momenta(), self.get_control_points(), output_dir, 
                             "{}__EstimatedParameters__Fusion_CP_Momenta_{}.vtk".format(self.name, current_iteration))

        # #ajout fg: write zones
        # if self.multiscale_momenta and not self.naive:
        #     array = np.zeros((5000, 3))
        #     j = 0
        #     for scale in range(self.coarser_scale, max(self.current_scale -1, 0), -1):
        #         nombre_zones = len(self.zones[scale])
        #         if scale in self.silent_haar_coef_momenta.keys():
        #             nombre_zones_silencees = len(self.silent_haar_coef_momenta[scale])
        #             array[j] = scale, nombre_zones, nombre_zones_silencees
        #             for silent_zone in self.silent_haar_coef_momenta[scale]:
        #                 j += 1
        #                 array[j, 0] = silent_zone
        #             j += 2
        #     write_3D_array(array, output_dir, self.name + "_silenced_zones.txt")
    
        
    

    

    
    
    
        
    
