import math
import os.path as op
from ...support.kernels import factory
from ...core import default
from ...core.model_tools.deformations.geodesic import Geodesic
from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...core.models.model_functions import initialize_control_points, initialize_momenta
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...in_out.array_readers_and_writers import *
from ...in_out.dataset_functions import create_template_metadata
from ...support.utilities import get_best_device, move_data

logger = logging.getLogger(__name__)

class GeodesicRegression(AbstractStatisticalModel):
    """
    Geodesic regression object class.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, template_specifications,

                 deformation_kernel_width=default.deformation_kernel_width,

                 concentration_of_time_points=default.concentration_of_time_points, t0=default.t0,

                 freeze_template=default.freeze_template,

                 initial_control_points=default.initial_control_points,
                 initial_momenta=default.initial_momenta,

                 write_adjoint_parameters = False,
                 new_bounding_box = None,

                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='GeodesicRegression')
        
        self.write_adjoint_parameters = write_adjoint_parameters

        # Declare model structure.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['momenta'] = None

        self.freeze_template = freeze_template
        self.freeze_momenta = False

        self.t0 = t0
       
        # Template.
        (object_list, self.objects_extension, self.objects_noise_variance, self.objects_attachment) = \
                                                        create_template_metadata(template_specifications)
        
        self.template = DeformableMultiObject(object_list)
        self.dimension = self.template.dimension

        self.number_of_objects = len(self.template.object_list)

        self.deformation_kernel_width = deformation_kernel_width
        self.number_of_subjects = 1
        
        # Template data.
        self.set_template_data(self.template.get_data())
        self.points = self.get_points()

        # Control points.
        self.control_points = initialize_control_points(initial_control_points, self.template, 
                                                    deformation_kernel_width,  new_bounding_box = new_bounding_box)

        # Momenta.
        self.fixed_effects['momenta'] = initialize_momenta(initial_momenta, len(self.control_points), self.dimension)
    
        self.current_residuals = None

        # Deformation.
        self.device, _ = get_best_device()

        self.geodesic = Geodesic( kernel=factory(kernel_width=deformation_kernel_width),
                                t0=t0, concentration_of_time_points=concentration_of_time_points)
        
        control_points = move_data(self.control_points, device=self.device)
        self.geodesic.set_control_points_t0(control_points)
        
    def initialize_noise_variance(self, dataset):
        if np.min(self.objects_noise_variance) < 0: # only if not provided by user
            self.objects_noise_variance = [1] * self.number_of_objects
            residuals = np.sum(self.compute_residuals(dataset))

            # Initialize the noise variance hyper-parameter as a 1/100th of the initial residual.
            target_times = dataset.times[0]
            for k in range(self.number_of_objects):
                nv = 0.01 * residuals / float(len(target_times))
                self.objects_noise_variance[k] = nv

        logger.info('>> Chosen noise std {}'.format(math.sqrt(self.objects_noise_variance[0])))
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
        out['momenta'] = self.get_momenta()
        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template:
            template_data = {key: fixed_effects[key] for key in self.fixed_effects['template_data'].keys()}
            self.set_template_data(template_data)
        self.set_momenta(fixed_effects['momenta'])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def compute_gradients(self, attachment, regularity, template_data, template_points,
                           momenta, points, with_grad=False):
        # Compute gradient if needed -----------------------------------------------------------------------------------
        if with_grad:
            total = regularity + attachment
            total.backward()            

            gradient = {}
            # Template data.
            if not self.freeze_template:
                if points == 'landmark_points' :
                    gradient[points] = template_points[points].grad
                else:
                    gradient[points] = template_data[points].grad
            
            gradient['momenta'] = momenta.grad

            # Convert the gradient back to numpy.
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
        :return:
        """

        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        template_data, template_points, momenta = self._fixed_effects_to_torch_tensors(with_grad)

        # Deform -------------------------------------------------------------------------------------------------------
        attachment, regularity = self._compute_attachment_and_regularity(
                                                            dataset, template_data, template_points, momenta)
        
        # Gradients -------------------------------------------------------------------------------------------------------
        return self.compute_gradients(attachment, regularity, template_data, template_points,
                                        momenta, self.points, with_grad=with_grad)


    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _compute_batch_attachment_and_regularity(self, target_times, target_objects, template_data, 
                                                template_points, momenta):       

        # Deform -------------------------------------------------------------------------------------------------------
        self.geodesic.set_template_points_t0(template_points)
        self.geodesic.set_momenta_t0(momenta)
        self.geodesic.update()
        attachment = 0.

        self.current_residuals = []
        
        for time, obj in zip(target_times, target_objects):
            deformed_points = self.geodesic.get_template_points(time)
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)
            
            att = self.objects_attachment.compute_weighted_distance(
                        deformed_data, self.template, obj, self.objects_noise_variance)
            attachment -= att
            
            self.current_residuals.append(att.cpu())
                    
        regularity = - self.geodesic.get_norm_squared()

        return attachment, regularity
    
    def compute_objects_distances(self, dataset, j, individual_RER = None, dist = "current", deformed = True): 
        """
        Compute current distance between deformed template and object
        #obj = a deformablemultiobject
        """
        template_data, _, _ = self.prepare_geodesic(dataset)
        
        deformed_points = self.geodesic.get_template_points(dataset.times[0][j])
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        obj = dataset.deformable_objects[0][j]
        
        if dist in ["current", "varifold"]:
            return self.objects_attachment.compute_vtk_distance(deformed_data, self.template, obj, dist)
        elif dist in ["ssim", "mse"]:
            return self.objects_attachment.compute_ssim_distance(deformed_data, self.template, obj, dist)

    def compute_flow_curvature(self, dataset, time, curvature = "gaussian"):
        template_data, _, _ = self.prepare_geodesic(dataset)

        deformed_points = self.geodesic.get_template_points(time)
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        for i, obj1 in enumerate(self.template.object_list):
            obj1.polydata.points = deformed_data['landmark_points'].cpu().numpy()
            obj1.curvature_metrics(curvature)
        
        return self.template

    def compute_initial_curvature(self, dataset, j, curvature = "gaussian"):
        obj = dataset.deformable_objects[0][j] 
        for obj1 in (obj.object_list):
            obj1.curvature_metrics(curvature)
        
        return obj

    def compute_curvature(self, dataset, j = None, individual_RER = None, curvature = "gaussian", 
                          iter = None):
        """
        """
        if j is None:
            data = self.template.get_data()
            for i, obj1 in enumerate(self.template.object_list):
                obj1.polydata.points = data['landmark_points'][0:obj1.get_number_of_points()]
                obj1.curvature_metrics(curvature)
            
                return self.template
            
        if iter == 0:
            return self.compute_initial_curvature(dataset, j, curvature)

        template_data, _, _ = self.prepare_geodesic(dataset)
        
        deformed_points = self.geodesic.get_template_points(dataset.times[0][j])
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        
        for i, obj1 in enumerate(self.template.object_list):
            obj1.polydata.points = deformed_data['landmark_points'][0:obj1.get_number_of_points()].cpu().numpy()
            obj1.curvature_metrics(curvature)
        
        return self.template
        
    def _compute_attachment_and_regularity(self, dataset, template_data, template_points, 
                                            momenta):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """
        target_times = dataset.times[0]
        target_objects = dataset.deformable_objects[0]

        self.geodesic.set_tmin(min(dataset.tmin, self.t0))
        self.geodesic.set_tmax(dataset.tmax)

        return self._compute_batch_attachment_and_regularity(target_times, target_objects, 
                                                            template_data, template_points, 
                                                            momenta)
    
    def compute_mini_batch_gradient(self, batch, dataset, individual_RER, with_grad=True):
        # get target times and objects from the batch
        target_times = [t[0] for t in batch]
        target_objects = [t[1] for t in batch]

        # Careful: the batch selected might not go from tmin to tmax
        self.geodesic.set_tmin(min(dataset.tmin, self.t0))
        self.geodesic.set_tmax(dataset.tmax)

        template_data, template_points, momenta = self._fixed_effects_to_torch_tensors(with_grad)
                
        attachement, regularity = self._compute_batch_attachment_and_regularity(target_times, target_objects, 
                                                                                template_data, template_points, momenta)
        
        return self.compute_gradients(attachement, regularity, template_data, template_points,
                                        momenta, self.points, with_grad=with_grad)

    def mini_batches(self, dataset, number_of_batches):
        batch_size = len(dataset.deformable_objects[0]) // number_of_batches

        targets = list(zip(dataset.times[0], dataset.deformable_objects[0]))
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

    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad = False):
        """
        Convert the fixed_effects into torch tensors.
        """
        # Template data.
        template_data = {key: move_data(value, device=self.device,
                                        requires_grad=(not self.freeze_template and with_grad))
                         for key, value in self.get_template_data().items()}

        # Template points.
        template_points = {key: move_data(value, device=self.device,
                                        requires_grad=(not self.freeze_template and with_grad)) 
                            for key, value in self.template.get_points().items()}

        # Momenta.
        momenta = move_data(self.get_momenta(),requires_grad=with_grad, device=self.device)

        return template_data, template_points, momenta
        

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################        
    
    def prepare_geodesic(self, dataset):
        template_data, template_points, momenta = self._fixed_effects_to_torch_tensors()

        self.geodesic.set_tmin(min(dataset.tmin, self.t0))
        self.geodesic.set_tmax(dataset.tmax)
        self.geodesic.set_template_points_t0(template_points)
        self.geodesic.set_momenta_t0(momenta)
        self.geodesic.update()  

        return template_data, dataset.times[0], dataset.deformable_objects[0]

    def compute_residuals(self, dataset, individual_RER = None, option = None):
        """
        Compute distances with various kernel width (for current or varifold) 
        (- default = model width) or additional dist like KD tree
        """
        template_data, target_times, target_objects = self.prepare_geodesic(dataset)
        
        residuals = []
        for (time, target) in zip(target_times, target_objects):
            deformed_points = self.geodesic.get_template_points(time)
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)

            if not option:
                #residuals.append(self.objects_attachment.compute_distances(deformed_data, self.template, target).cpu().numpy())
                residuals.append(self.objects_attachment.compute_weighted_distance(
                deformed_data, self.template, target, self.objects_noise_variance).cpu().numpy())
            else:
                residuals.append(self.objects_attachment.compute_additional_distances(deformed_data, self.template, target, option).cpu().numpy())

        return residuals
    
    def write(self, dataset, individual_RER, output_dir, iteration, write_all = True):
        self._write_model_predictions(output_dir, dataset, write_all)
        self._write_model_parameters(output_dir, iteration, write_all)

    def _write_model_predictions(self, output_dir, dataset=None, write_all = True):

        # Initialize ---------------------------------------------------------------------------------------------------
        template_data, target_times, _ = self.prepare_geodesic(dataset)

        # Write --------------------------------------------------------------------------------------------------------
        # Geodesic flow.
        self.geodesic.write(self.name, self.objects_extension, self.template, template_data,
                            output_dir, write_adjoint_parameters = self.write_adjoint_parameters, write_all = write_all)

        # Model predictions.
        if dataset is not None and not write_all:
            for j, time in enumerate(target_times):
                names = ['{}__Reconstruction____tp_{}__age_{}'.format(self.name, j, time, ext)\
                        for ext in self.objects_extension]
                deformed_points = self.geodesic.get_template_points(time)
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                self.template.write(output_dir, names,
                                    {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

    def output_path(self, output_dir, dataset):
        self.cp_path = op.join(output_dir, "{}__ControlPoints.txt".format(self.name))
        self.momenta_path = op.join(output_dir, "{}__Estimated__Momenta.txt".format(self.name))
        
        if self.geodesic.fw_exponential.n_time_points is None:
            self.prepare_geodesic(dataset)

        try:
            self.geodesic.output_path(self.name, self.objects_extension, output_dir)
        except:
            pass

    def _write_model_parameters(self, output_dir, iteration, write_all = True):
        
        # Template.
        template_names = []
        template_names = None

        if not self.freeze_template:
            template_names = ['{}__Estimated__Template__tp_{}__age_{}{}'.format(self.name, 
                            self.geodesic.bw_exponential.n_time_points - 1, 
                            self.geodesic.t0, ext) for ext in self.objects_extension]
        else:
            template_names = ['{}__Fixed__Template__tp_{}__age_{}{}'.format(self.name, 
                                self.geodesic.bw_exponential.n_time_points - 1, 
                                self.geodesic.t0, ext) for ext in self.objects_extension]

        self.template.write(output_dir, template_names)

        if self.objects_extension[0] != ".vtk":
            write_2D_array(self.control_points, output_dir, "{}__ControlPoints.txt".format(self.name))
            write_3D_array(self.get_momenta(), output_dir, "{}__Estimated__Momenta.txt".format(self.name))

        else:
            # Fuse control points and momenta for paraview display
            concatenate_for_paraview(self.get_momenta(), self.control_points, output_dir, 
                                    "{}__Estimated__Fusion_CP_Momenta_iter_{}.vtk".format(self.name, iteration))