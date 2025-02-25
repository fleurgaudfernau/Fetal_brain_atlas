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
from ...core.models.model_functions import initialize_control_points, initialize_momenta, gaussian_kernel
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...in_out.array_readers_and_writers import *
from ...in_out.dataset_functions import create_template_metadata
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

    def __init__(self, template_specifications, number_of_subjects,

                 deformation_kernel_width=default.deformation_kernel_width,

                 number_of_time_points=default.number_of_time_points,

                 freeze_template=default.freeze_template,

                 initial_control_points=default.initial_control_points,

                 initial_momenta=default.initial_momenta,
                 freeze_momenta=default.freeze_momenta,

                 kernel_regression = default.kernel_regression,
                 visit_ages = None,
                 time = None,

                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='DeformableTemplate')
        
        # Global-like attributes.
        

        # Declare model structure.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['momenta'] = None

        self.freeze_template = freeze_template
        self.freeze_momenta = freeze_momenta

        self.device, _ = get_best_device()
        self.deformation_kernel_width = deformation_kernel_width

        # Template.
        (object_list, self.objects_extension, self.objects_noise_variance, self.attachment) = \
                                                create_template_metadata(template_specifications)

        self.template = DeformableMultiObject(object_list)
        self.number_of_objects = len(object_list)
        self.dimension = self.template.dimension 
        
        # Deformation.
        self.exponential = Exponential(kernel=factory(kernel_width=deformation_kernel_width),
                                                    number_of_time_points=number_of_time_points,)

        # Template data.
        self.fixed_effects['template_data'] = self.template.get_data()
        self.points = self.get_points() # image_intensities or landmark_points

        # Control points.
        self.control_points = initialize_control_points(initial_control_points, 
                                                    self.template, deformation_kernel_width)

        control_points = move_data(self.control_points, device=self.device)                
        self.exponential.set_initial_control_points(control_points)

        # Momenta.
        self.fixed_effects['momenta'] = initialize_momenta(
                initial_momenta, len(self.control_points), self.dimension, number_of_subjects)
        self.number_of_subjects = number_of_subjects
            
        self.kernel_regression = kernel_regression
        self.time = time
        self.visit_ages = visit_ages

        if self.kernel_regression:
            self.weights = [gaussian_kernel(self.time, age_s[0]) for age_s in self.visit_ages]
            self.total_weights = np.sum([gaussian_kernel(self.time, age_s[0]) for age_s in self.visit_ages])
            logger.info("Weights: {}".format([gaussian_kernel(self.time, age_s[0]) for age_s in self.visit_ages]))

        self.current_residuals = None

    def initialize_noise_variance(self, dataset):
        if np.min(self.objects_noise_variance) < 0:
            template_data, template_points, momenta = self._fixed_effects_to_torch_tensors()
            targets = dataset.deformable_objects
            targets = [target[0] for target in targets]

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
                    nv = 0.01 * residuals[k] / float(self.number_of_subjects)
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
        self._setup_multiprocess_pool(initargs=([target[0] for target in dataset.deformable_objects],
                                                self.attachment,
                                                self.objects_noise_variance,
                                                self.freeze_template, 
                                                self.freeze_momenta, 
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
                                                      deformable_objects, objects_noise_variance, device):
        # Deform.
        exponential.set_initial_template_points(template_points)        
        exponential.set_initial_momenta(momenta)
        exponential.move_data_to_(device=device)
        exponential.update() #flow the template points according to the momenta using kernel.convolve

        # Compute attachment and regularity. (-> increase memory)
        deformed_points = exponential.get_template_points() #template points
        deformed_data = template.get_deformed_data(deformed_points, template_data) #template intensities after deformation
        
        #(observation) deformable multi object -> image -> torch.interpolate
        attachment = -objects_attachment.compute_weighted_distance(deformed_data, template, deformable_objects,
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
        targets = [[i, target[0]] for i, target in enumerate(dataset.deformable_objects)]

        return self._compute_batch_gradient(targets, template_data, template_points, momenta, with_grad)
    
    def compute_mini_batch_gradient(self, batch, dataset, individual_RER, with_grad=True):
        template_data, template_points, momenta = self._fixed_effects_to_torch_tensors(with_grad)                                    
        
        return self._compute_batch_gradient(batch, template_data, template_points, momenta, with_grad)
    
    def mini_batches(self, dataset, number_of_batches):
        """
        Split randomly the dataset into batches of size batch_size
        """
        batch_size = len(dataset.deformable_objects) // number_of_batches
        targets = [[i,target[0]] for i, target in enumerate(dataset.deformable_objects)]
        targets_copy = targets.copy()
        np.random.shuffle(targets_copy)

        mini_batches = []
        n_minibatches = len(targets_copy) // batch_size    

        for i in range(n_minibatches):
            mini_batch = targets_copy[i * batch_size:(i + 1) * batch_size]
            mini_batches.append(mini_batch)
        if len(targets_copy) % batch_size != 0:
            mini_batch = targets_copy[i * batch_size:len(targets_copy)]
            if len(mini_batches) > batch_size / 2: #if last batch big enough
                mini_batches.append(mini_batch)
            else:
                mini_batches[-1] += mini_batch
        
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
            obj = self.template
            data = self.template.get_data()
            for i, obj1 in enumerate(obj.object_list):
                obj1.polydata.points = data[self.points][0:obj1.get_number_of_points()]
                obj1.curvature_metrics(curvature)
            
                return obj

        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_momenta(momenta[j])
        self.exponential.move_data_to_(device=self.device)
        self.exponential.update() 
        deformed_points = self.exponential.get_template_points()
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        
        # Compute deformed template curvature
        obj = dataset.deformable_objects[j][0] if iter == 0 else self.template 

        for i, obj1 in enumerate(obj.object_list):
            if iter != 0:
                obj1.polydata.points = deformed_data[self.points][0:obj1.get_number_of_points()].cpu().numpy()

            obj1.curvature_metrics(curvature)
        
        return obj

    def compute_residuals(self, dataset, individual_RER = None):
        template_data, template_points, momenta = self._fixed_effects_to_torch_tensors()

        residuals = []
        for i, target in enumerate(dataset.deformable_objects):
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
        
        for i, target in enumerate(dataset.deformable_objects):
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

        # Write the model predictions, and compute the residuals at the same time.
        self._write_model_predictions(dataset, individual_RER, output_dir, current_iteration,
                                    compute_residuals=write_residuals)

        # Write the model parameters.
        self._write_model_parameters(output_dir, str(current_iteration))

    def _write_model_predictions(self, dataset, individual_RER, output_dir, current_iteration, 
                                 compute_residuals=True):
        # Initialize.
        template_data, template_points, momenta = self._fixed_effects_to_torch_tensors()

        # Deform, write reconstructions and compute residuals.
        self.exponential.set_initial_template_points(template_points)

        residuals = []  # List of torch 1D tensors. Individuals, objects.
        for i, subject_id in enumerate(dataset.subject_ids):
            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.update()

            # Writing the whole flow. -> modif fg
            names = []
            for k in range(self.number_of_objects):
                name = self.name + '__flow____subject_' + subject_id
                names.append(name)
            
            deformed_points = self.exponential.get_template_points()
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)

            names = []
            for k, object_extension in enumerate(self.objects_extension):
                
                if self.kernel_regression:
                    name = self.name + '__Reconstruction__subject_' + subject_id + "_age_" + str(self.visit_ages[i][0]) + object_extension
                else:
                    name = '{}__Reconstruction__subject_{}{}'.format(self.name, subject_id, object_extension)
                names.append(name)

            self.template.write(output_dir, names,
                                {key: value.data.cpu().numpy() for key, value in deformed_data.items()})
            
        return residuals

    def output_path(self, output_dir, dataset):
        self.cp_path = op.join(output_dir, "{}__EstimatedParameters__ControlPoints.txt".format(self.name))
        self.momenta_path = op.join(output_dir, "{}__EstimatedParameters__Momenta.txt".format(self.name))

        if not self.freeze_template:
            for ext in self.objects_extension:
                self.template_path = op.join(output_dir, "{}__EstimatedParameters__Template_{}".format(self.name, ext))

    def _write_model_parameters(self, output_dir, current_iteration):

        template_names = None
        if self.kernel_regression:
            template_names = ["{}__EstimatedParameters__Template_time_{}{}".format(self.name, self.time, ext)\
                                for ext in self.objects_extension]

        elif not self.freeze_template:
            template_names = ["{}__EstimatedParameters__Template_{}{}".format(self.name, current_iteration, ext)\
                                for ext in self.objects_extension]
        self.template.write(output_dir, template_names)

        # template_names = ["{}__EstimatedParameters__Template{}".format(self.name, ext)\
        #                 for ext in self.objects_extension]
        # self.template.write(output_dir, template_names)
        
        write_2D_array(self.control_points, output_dir, op.join(output_dir, "{}__EstimatedParameters__ControlPoints.txt".format(self.name)))
        
        # Momenta.
        write_3D_array(self.get_momenta(), output_dir, self.name + "__EstimatedParameters__Momenta.txt")#.format(current_iteration))
        
        concatenate_for_paraview(self.get_momenta(), self.control_points, output_dir, 
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
    
        
    

    

    
    
    
        
    
