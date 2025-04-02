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
from ...support.kernels import factory
from ...core import default
from ...core.model_tools.deformations.exponential import Exponential
from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...core.models.model_functions import initialize_cp, initialize_momenta, gaussian_kernel
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...in_out.array_readers_and_writers import *
from ...in_out.dataset_functions import template_metadata
from ...support.utilities import get_best_device, move_data, detach, assert_same_device

warnings.filterwarnings("ignore") #fg
logger = logging.getLogger(__name__)

class DeformableTemplate(AbstractStatisticalModel):
    """
    DeformableTemplate object class.

    """
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, template_specifications, n_subjects,
                 deformation_kernel_width=None,

                 n_time_points=default.n_time_points,

                 initial_cp=None, initial_momenta=None,
                 freeze_momenta = default.freeze_momenta,
                 freeze_template = default.freeze_template,

                 kernel_regression = False,
                 visit_ages = None, time = None,
                 bounding_box = None,

                 **kwargs):

        name='DeformableTemplate' if not kernel_regression else "KernelRegression"
        AbstractStatisticalModel.__init__(self, name)
        
        # Declare model structure.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['momenta'] = None

        self.freeze_template = freeze_template
        self.freeze_momenta = freeze_momenta

        self.device = get_best_device()
        self.deformation_kernel_width = deformation_kernel_width

        # Template.
        (object_list, self.objects_noise_variance, self.attachment) = \
                                            template_metadata(template_specifications)

        self.template = DeformableMultiObject(object_list)
        self.number_of_objects = len(object_list)
        self.dimension = self.template.dimension 
        
        # Deformation.
        self.exponential = Exponential(kernel=factory(kernel_width=deformation_kernel_width),
                                        n_time_points=n_time_points)

        # Template data.
        self.fixed_effects['template_data'] = self.template.get_data()
        self.points = self.get_points() # image_intensities or landmark_points

        # Control points.
        self.cp = initialize_cp(initial_cp, self.template, deformation_kernel_width, bounding_box)
        cp = move_data(self.cp, device=self.device)                
        self.exponential.set_initial_cp(cp)

        # Momenta.
        self.fixed_effects['momenta'] = initialize_momenta(initial_momenta, len(self.cp), 
                                                            self.dimension, n_subjects)
        self.n_subjects = n_subjects
            
        self.kernel_regression = kernel_regression
        self.time = time
        self.visit_ages = visit_ages

        if self.kernel_regression:
            self.weights = [round(gaussian_kernel(self.time, age_s[0]),2) for age_s in self.visit_ages]
            self.total_weights = np.sum(self.weights)

        self.current_residuals = None

    def initialize_noise_variance(self, dataset):
        if np.min(self.objects_noise_variance) < 0:
            template_data, template_points, momenta = self._fixed_effects_to_torch_tensors()
            targets = [target[0] for target in dataset.objects]

            residuals_torch = []
            self.exponential.set_initial_template_points(template_points)

            for i, target in enumerate(targets):
                self.exponential.set_initial_momenta(momenta[i])
                self.exponential.update()
                deformed_points = self.exponential.get_template_points()
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                residuals_torch.append(self.attachment.compute_distances(
                                        deformed_data, self.template, target))

            residuals = np.zeros((self.number_of_objects,))
            for i in range(len(residuals_torch)):
                residuals += detach(residuals_torch[i])

            # Initialize the noise variance hyper-parameter as a 1/100th of the initial residual.
            for k in range(self.number_of_objects):
                if self.objects_noise_variance[k] < 0:
                    nv = 0.01 * residuals[k] / float(self.n_subjects)
                    self.objects_noise_variance[k] = nv
                    logger.info('>> Automatically chosen noise std for objet: %.4f [ %s ]' % (math.sqrt(nv), k))
        
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
        self._setup_multiprocess_pool(initargs=([target[0] for target in dataset.objects],
                                                self.attachment, self.objects_noise_variance,
                                                self.freeze_template, self.freeze_momenta, 
                                                self.exponential))

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, individual_RER, mode='complete', with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        :param fixed_effects: Dictionary of fixed effects.
        :param individual_RER: Dictionary of individual random effects realizations.
        :param mode: Indicates which log_likelihood should be computed, between 'complete', 'model', and 'class2'.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """
        template_data, template_points, momenta = self._fixed_effects_to_torch_tensors(with_grad)
        
        return self._compute_attachment_and_regularity(dataset, template_data, template_points, momenta, with_grad) 

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    @staticmethod
    def _deform_and_compute_attachment_and_regularity(exponential, template_points, momenta,
                                                      template, template_data, objects_attachment, 
                                                      objects, objects_noise_variance, device):
        # Deform.
        exponential.set_initial_template_points(template_points)        
        exponential.set_initial_momenta(momenta)
        exponential.move_data_to_(device=device)
        exponential.update() #flow the template points according to the momenta using kernel.convolve

        # Compute attachment and regularity. (-> increase memory)
        deformed_points = exponential.get_template_points() #template points
        deformed_data = template.get_deformed_data(deformed_points, template_data) #template intensities after deformation
        
        #(observation) deformable multi object -> image -> torch.interpolate
        attachment = -objects_attachment.compute_weighted_distance(deformed_data, template, objects,
                                                                    objects_noise_variance)
        regularity = -exponential.get_norm_squared()
        
        assert_same_device(attachment = attachment, regularity = regularity)
        
        return attachment, regularity
         
    @staticmethod
    def _compute_gradients(attachment, regularity, template_data, freeze_template, template_points,
                           freeze_momenta, momenta, points, with_grad=False):

        if with_grad:
            total_for_subject = attachment + regularity #torch tensor
                        
            total_for_subject.backward() #compute gradient  -> the tensors stay in memory until this point
               
            gradient = {}
            if not freeze_template:
                if points == 'landmark_points' :
                    gradient[points] = template_points[points].grad
                else:
                    gradient[points] = template_data[points].grad
            
            if not freeze_momenta:
                gradient['momenta'] = momenta.grad

            gradient = {key: detach(value) for key, value in gradient.items()}                                                         
            res = detach(attachment), detach(regularity), gradient

        else:
            res = detach(attachment), detach(regularity)

        return res
    
    
    def _compute_batch_gradient(self, targets, template_data, template_points, momenta, with_grad=False):
        """
        Core part of the ComputeLogLikelihood methods. Torch input, numpy output.
        Single-thread version.
        """
        attachment = 0.
        regularity = 0.
        
        for i, target in targets:
            new_attachment, new_regularity = DeformableTemplate._deform_and_compute_attachment_and_regularity(
                                            self.exponential, template_points, momenta[i],
                                            self.template, template_data, self.attachment,
                                            target, self.objects_noise_variance, self.device)
            self.current_residuals = new_attachment.cpu()

            if self.kernel_regression:
                weight = gaussian_kernel(self.time, self.visit_ages[i][0])
                attachment += (weight / self.total_weights) * new_attachment 
                regularity += (weight / self.total_weights) * new_regularity
            else:
                attachment += new_attachment
                regularity += new_regularity            
            
        gradients = self._compute_gradients(attachment, regularity, template_data, self.freeze_template, 
                                            template_points,  self.freeze_momenta, momenta, self.points,
                                            with_grad)

        return gradients
    
    def _compute_attachment_and_regularity(self, dataset, template_data, template_points, momenta,
                                            with_grad=False):
        # Initialize.
        targets = [[i, target[0]] for i, target in enumerate(dataset.objects)]

        return self._compute_batch_gradient(targets, template_data, template_points, momenta, with_grad)
    
    def compute_mini_batch_gradient(self, batch, dataset, individual_RER, with_grad=True):
        template_data, template_points, momenta = self._fixed_effects_to_torch_tensors(with_grad)                                    
        
        return self._compute_batch_gradient(batch, template_data, template_points, momenta, with_grad)
    
    def mini_batches(self, dataset, number_of_batches):
        """
        Split randomly the dataset into batches of size batch_size
        """
        batch_size = len(dataset.objects) // number_of_batches
        targets = [[i,target[0]] for i, target in enumerate(dataset.objects)]
        targets_copy = targets.copy()
        np.random.shuffle(targets_copy)

        n_minibatches = len(targets_copy) // batch_size    

        mini_batches = [targets_copy[i:i + batch_size]\
                        for i in range(0, len(targets_copy), batch_size)]
    
        if len(mini_batches) > 1 and len(mini_batches[-1]) < batch_size / 2:
            mini_batches[-2].extend(mini_batches.pop())
        
        return mini_batches
    
    def prepare_exponential(self, i, template_points, momenta):
                
        self.exponential.set_initial_template_points(template_points)

        self.exponential.set_initial_momenta(momenta[i])
        self.exponential.move_data_to_(device=self.device)
        self.exponential.update()

        return 

    def compute_curvature(self, dataset, j = None, individual_RER = None, curvature = "gaussian", iter = None):
        """
            Compute object curvature (at iter 0) or deformed template to object curvature
        """
        template_data, template_points, momenta = self._fixed_effects_to_torch_tensors()

        # template curvature
        if j is None:
            data = self.template.get_data()
            for i, obj1 in enumerate(self.template.object_list):
                obj1.polydata.points = data[self.points][0:obj1.n_points()]
                obj1.curvature_metrics(curvature)
                return self.template

        self.exponential.prepare_and_update(self.cp, momenta[j], template_points, device = self.device)
        deformed_points = self.exponential.get_template_points()
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        
        # Compute deformed template curvature
        obj = dataset.objects[j][0] if iter == 0 else self.template 

        for i, obj1 in enumerate(self.template.object_list):
            if iter != 0:
                obj1.polydata.points = deformed_data[self.points][0:obj1.n_points()].cpu().numpy()

            obj1.curvature_metrics(curvature)
        
        return obj

    def compute_residuals(self, dataset, individual_RER = None):
        template_data, template_points, momenta = self._fixed_effects_to_torch_tensors()

        residuals = []
        for i, target in enumerate(dataset.objects):
            self.prepare_exponential(i, template_points, momenta)

            # Compute attachment
            deformed_points = self.exponential.get_template_points() #template points
            deformed_data = self.template.get_deformed_data(deformed_points, template_data) #template intensities after deformation
            att = self.attachment.compute_weighted_distance(deformed_data, self.template, target[0],
                                                            self.objects_noise_variance)
            residuals.append(att)
        
        return residuals

    def compute_residuals_per_point(self, dataset, individual_RER = None):
        template_data, template_points, momenta = self._fixed_effects_to_torch_tensors()
        
        residuals_by_point = torch.zeros((template_data[self.points].shape), device=self.device) 
        
        for i, target in enumerate(dataset.objects):
            self.prepare_exponential(i, template_points, momenta)
        
            # Compute attachment
            deformed_points = self.exponential.get_template_points() #template points
            deformed_data = self.template.get_deformed_data(deformed_points, template_data) #template intensities after deformation

            objet_intensities = target[0].get_data()[self.points]
            target_intensities = move_data(objet_intensities, device=self.device, 
                                            dtype = next(iter(template_data.values())).dtype) #tensor not dict 
            residuals = (target_intensities - deformed_data[self.points]) ** 2
            
            residuals_by_point += residuals
        
        return residuals_by_point.cpu().numpy().flatten()
    
    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad = False, device=None):
        """
        Convert the fixed_effects into torch tensors.
        """
        # Template data.
        template_data = {key: move_data(value, device=self.device,
                                        requires_grad=(not self.freeze_template and with_grad))
                         for key, value in self.fixed_effects['template_data'].items()}

        # Template points.
        template_points = {key: move_data(value, device=self.device, 
                                                    requires_grad= (not self.freeze_template and with_grad))
                           for key, value in self.template.get_points().items()}

        # Momenta.
        momenta = move_data(self.fixed_effects['momenta'], device=self.device, 
                            requires_grad=(not self.freeze_momenta and with_grad))

        return template_data, template_points, momenta #ajout fg

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, dataset, individual_RER, output_dir, current_iteration, write_residuals=True, write_all = True):
        from time import perf_counter
        t1 = perf_counter()
        # Write the model predictions, and compute the residuals at the same time.
        self._write_model_predictions(dataset, individual_RER, output_dir, current_iteration,
                                        compute_residuals=write_residuals)

        # Write the model parameters.
        t2 = perf_counter()
        self._write_model_parameters(output_dir, str(current_iteration))
        t3 = perf_counter()
        logger.info("Time for writing: {} s (Predictions: {} s, Parameters: {} s)"\
                    .format(int(t3-t1), int(t2-t1), int(t3-t2)))

    def _write_model_predictions(self, dataset, individual_RER, output_dir, current_iteration, 
                                 compute_residuals=True):
        # Initialize.
        template_data, template_points, momenta = self._fixed_effects_to_torch_tensors()

        # Deform, write reconstructions and compute residuals.
        self.exponential.set_initial_template_points(template_points)

        for i, subject_id in enumerate(dataset.ids):
            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.update()
            deformed_points = self.exponential.get_template_points()
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)

            names = [ reconstruction_name(self.name, subject_id, age = self.visit_ages[i][0])\
                    if self.kernel_regression else reconstruction_name(self.name, subject_id) ]       
            self.template.write(output_dir, names, deformed_data)

            it = current_iteration if i == 0 else ""
            names = [reconstruction_name(self.name, subject_id, age = self.visit_ages[i][0], 
                    iteration = it) if self.kernel_regression else\
                    reconstruction_name(self.name, subject_id, iteration = it)]
            self.template.write_png(output_dir, names, deformed_data)
            
        return []

    def output_path(self, output_dir, dataset):
        self.cp_path = op.join(output_dir, cp_name(self.name))
        self.momenta_path = op.join(output_dir, momenta_name(self.name))

        if not self.freeze_template:
            self.template_path = op.join(output_dir, template_name(self.name))

    def _write_model_parameters(self, output_dir, current_iteration):
        time = self.time if self.kernel_regression else ""
        template_names = [template_name(self.name, time = time, 
                        iteration = current_iteration if not self.freeze_template else "", 
                        freeze_template = self.freeze_template)]

        self.template.write(output_dir, template_names)
        self.template.write_png(output_dir, template_names)
        
        # Momenta and cp.
        write_cp(self.cp, output_dir, self.name)
        write_momenta(self.get_momenta(), output_dir, self.name)
        
        # Write only the first subject 
        momenta = self.get_momenta()[0,:,:]
        concatenate_for_paraview(momenta, self.cp, output_dir, self.name, current_iteration)
    
    
    
        
    
