import math
import os.path as op
from ...support import kernels as kernel_factory
from ...core import default
from ...core.model_tools.deformations.geodesic import Geodesic
from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...core.models.model_functions import initialize_control_points, initialize_momenta
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...in_out.array_readers_and_writers import *
from ...in_out.dataset_functions import create_template_metadata, create_mesh_attachements
from ...support import utilities

logger = logging.getLogger(__name__)

class GeodesicRegression(AbstractStatisticalModel):
    """
    Geodesic regression object class.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, template_specifications,

                 dimension=default.dimension,
                 tensor_scalar_type=default.tensor_scalar_type,
                 tensor_integer_type=default.tensor_integer_type,
                 dense_mode=default.dense_mode,
                 number_of_processes=default.number_of_processes,

                 deformation_kernel_width=default.deformation_kernel_width,

                 concentration_of_time_points=default.concentration_of_time_points, t0=default.t0,
                 use_rk2_for_shoot=default.use_rk2_for_shoot, use_rk2_for_flow=default.use_rk2_for_flow,

                 freeze_template=default.freeze_template,
                 use_sobolev_gradient=default.use_sobolev_gradient,
                 smoothing_kernel_width=default.smoothing_kernel_width,

                 initial_control_points=default.initial_control_points,
                 freeze_control_points=default.freeze_control_points,
                 initial_cp_spacing=default.initial_cp_spacing,
                 initial_momenta=default.initial_momenta,

                 gpu_mode=default.gpu_mode,

                 nb_components = 1, # ajout fg
                 freeze_rupture_time = True,
                 write_adjoint_parameters = False,
                 new_bounding_box = None,

                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='GeodesicRegression', gpu_mode=gpu_mode)
        
        # Ajout fg for piecewise geodesic
        self.nb_components = nb_components

        # Global-like attributes.
        self.dimension = dimension
        self.tensor_scalar_type = tensor_scalar_type
        self.tensor_integer_type = tensor_integer_type
        self.dense_mode = dense_mode
        self.number_of_processes = number_of_processes

        self.write_adjoint_parameters = write_adjoint_parameters

        # Declare model structure.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['control_points'] = None
        self.fixed_effects['momenta'] = None
        self.fixed_effects['rupture_time'] = [None] * (self.nb_components - 1)

        self.freeze_template = freeze_template
        self.freeze_momenta = False
        self.freeze_control_points = freeze_control_points
        self.freeze_rupture_time = freeze_rupture_time

        self.t0 = t0

        # Deformation.
        self.geodesic = Geodesic(
            dense_mode=dense_mode,
            kernel=kernel_factory.factory(gpu_mode=gpu_mode, kernel_width=deformation_kernel_width),
            t0=t0, concentration_of_time_points=concentration_of_time_points,
            use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)

        # Template.
        (object_list, self.objects_name, self.objects_name_extension,
         self.objects_noise_variance, self.multi_object_attachment) = create_template_metadata(
            template_specifications, self.dimension, gpu_mode=gpu_mode)

        # k1 attachement for meshes
        # -> possibility to compute different residuals functions and attachements
        self.multi_object_attachments_k = create_mesh_attachements(template_specifications, gpu_mode=gpu_mode)
        
        # Ajout fg: multiscale cost function
        self.k = None
        # Ajout fg: cost function with curvature matching term
        self.curvature = False

        self.template = DeformableMultiObject(object_list)

        self.number_of_objects = len(self.template.object_list)

        self.use_sobolev_gradient = use_sobolev_gradient
        self.smoothing_kernel_width = smoothing_kernel_width
        self.deformation_kernel_width = deformation_kernel_width
        self.initial_cp_spacing = initial_cp_spacing
        self.number_of_subjects = 1
        
        if self.use_sobolev_gradient:
            self.sobolev_kernel = kernel_factory.factory(gpu_mode=gpu_mode, kernel_width=smoothing_kernel_width)

        # Template data.
        self.fixed_effects['template_data'] = self.template.get_data()

        # Control points.
        self.fixed_effects['control_points'] = initialize_control_points(initial_control_points, 
            self.template, initial_cp_spacing, deformation_kernel_width, self.dimension, 
            self.dense_mode, new_bounding_box = new_bounding_box)

        self.number_of_control_points = len(self.fixed_effects['control_points'])

        # Momenta.
        self.fixed_effects['momenta'] = initialize_momenta(
            initial_momenta, self.number_of_control_points, self.dimension)
    
        self.current_residuals = None
        
        
        
    def initialize_noise_variance(self, dataset):
        if np.min(self.objects_noise_variance) < 0:
            template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(False)
            target_times = dataset.times[0]
            target_objects = dataset.deformable_objects[0]

            
            tmin = min(math.trunc(min(target_times)), self.t0)
            self.geodesic.set_tmin(tmin)
            self.geodesic.set_tmax(math.ceil(max(target_times)))
            self.geodesic.set_template_points_t0(template_points)
            self.geodesic.set_control_points_t0(control_points)
            self.geodesic.set_momenta_t0(momenta)
            self.geodesic.update()

            residuals = np.zeros((self.number_of_objects,))

            for (time, target) in zip(target_times, target_objects):
                deformed_points = self.geodesic.get_template_points(time)
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                residuals += self.multi_object_attachment.compute_distances(
                    deformed_data, self.template, target).data.numpy()

            # Initialize the noise variance hyper-parameter as a 1/100th of the initial residual.
            for k, obj in enumerate(self.objects_name):
                if self.objects_noise_variance[k] < 0:
                    nv = 0.01 * residuals[k] / float(len(target_times))
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
        if not self.freeze_control_points:
            out['control_points'] = self.fixed_effects['control_points']
        out['momenta'] = self.fixed_effects['momenta']
        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template:
            template_data = {key: fixed_effects[key] for key in self.fixed_effects['template_data'].keys()}
            self.set_template_data(template_data)
        if not self.freeze_control_points:
            self.set_control_points(fixed_effects['control_points'])
        self.set_momenta(fixed_effects['momenta'])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def compute_gradients(self, attachment, regularity, template_data, template_points,
                           control_points, momenta,  with_grad=False):
        # Compute gradient if needed -----------------------------------------------------------------------------------
        if with_grad:
            total = regularity + attachment
            total.backward()            

            gradient = {}
            # Template data.
            if not self.freeze_template:
                if 'landmark_points' in template_data.keys():
                    gradient['landmark_points'] = template_points['landmark_points'].grad
                if 'image_intensities' in template_data.keys():
                    gradient['image_intensities'] = template_data['image_intensities'].grad

                if self.use_sobolev_gradient and 'landmark_points' in gradient.keys():
                    gradient['landmark_points'] = self.sobolev_kernel.convolve(
                        template_data['landmark_points'].detach(), template_data['landmark_points'].detach(),
                        gradient['landmark_points'].detach())

            # Control points and momenta.
            if not self.freeze_control_points: gradient['control_points'] = control_points.grad
            
            gradient['momenta'] = momenta.grad

            # Convert the gradient back to numpy.
            gradient = {key: value.data.cpu().numpy() for key, value in gradient.items()}

            return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient

        else:
            return attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy()

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, population_RER, individual_RER, mode='complete', with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.

        :param dataset: LongitudinalDataset instance
        :param fixed_effects: Dictionary of fixed effects.
        :param population_RER: Dictionary of population random effects realizations.
        :param indRER: Dictionary of individual random effects realizations.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """

        device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)

        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(with_grad, device=device)

        # Deform -------------------------------------------------------------------------------------------------------
        attachment, regularity = self._compute_attachment_and_regularity(
            dataset, template_data, template_points, control_points, momenta)
        
        # Gradients -------------------------------------------------------------------------------------------------------
        return self.compute_gradients(attachment, regularity, template_data, template_points,
                           control_points, momenta,  with_grad=with_grad)


    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _compute_batch_attachment_and_regularity(self, target_times, target_objects, template_data, 
                                                template_points, control_points, momenta): # ajout fg        

        # Deform -------------------------------------------------------------------------------------------------------
        self.geodesic.set_template_points_t0(template_points)
        self.geodesic.set_control_points_t0(control_points)
        self.geodesic.set_momenta_t0(momenta)
        self.geodesic.update()
        attachment = 0.
        attachment_curv = 0.
        if self.curvature: w = 0.5
        self.current_residuals = []

        # The user can ask to compute the cost function with a different kernel (multiscale strategy)
        #attachment_fct =self.multi_object_attachment if not self.k else self.multi_object_attachments_k[self.k]
        
        for j, (time, obj) in enumerate(zip(target_times, target_objects)):
            deformed_points = self.geodesic.get_template_points(time)
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)
            
            att = self.multi_object_attachment.compute_weighted_distance(
                        deformed_data, self.template, obj, self.objects_noise_variance)
            attachment -= att
            
            self.current_residuals.append(att.cpu())
            
            # ajout fg: compute curvature distance
            #attachment_curv = w*self.multi_object_attachment.compute_weighted_curvature_distance(
            #            deformed_data, self.template, obj, self.objects_noise_variance)
            #attachment -= attachment_curv
        
        regularity = - self.geodesic.get_norm_squared()

        return attachment, regularity
    
    def compute_objects_distances(self, dataset, j, individual_RER = None, dist = "current", deformed = True): 
        """
        Compute current distance between deformed template and object
        #obj = a deformablemultiobject
        """
        device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, _, _ = self.prepare_geodesic(dataset, device)
        
        deformed_points = self.geodesic.get_template_points(dataset.times[0][j])
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        obj = dataset.deformable_objects[0][j]
        
        if dist in ["current", "varifold"]:
            return self.multi_object_attachment.compute_vtk_distance(deformed_data, self.template, obj, dist)
        elif dist in ["ssim", "mse"]:
            return self.multi_object_attachment.compute_ssim_distance(deformed_data, self.template, obj, dist)

    def compute_flow_curvature(self, dataset, time, curvature = "gaussian"):
        device, _ = utilities.get_best_device(gpu_mode = self.gpu_mode)
        template_data, _, _ = self.prepare_geodesic(dataset, device)

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

        device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, _, _ = self.prepare_geodesic(dataset, device)
        
        deformed_points = self.geodesic.get_template_points(dataset.times[0][j])
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        
        for i, obj1 in enumerate(self.template.object_list):
            obj1.polydata.points = deformed_data['landmark_points'][0:obj1.get_number_of_points()].cpu().numpy()
            obj1.curvature_metrics(curvature)
        
        return self.template

    
    def _compute_attachment_and_regularity(self, dataset, template_data, template_points, 
                                            control_points, momenta):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """
        target_times = dataset.times[0]
        target_objects = dataset.deformable_objects[0]

        tmin = min(math.trunc(min(target_times)), self.t0)
        self.geodesic.set_tmin(tmin)
        self.geodesic.set_tmax(math.ceil(max(target_times)))

        return self._compute_batch_attachment_and_regularity(target_times, target_objects, 
                                                            template_data, template_points, 
                                                            control_points, momenta)
    
    def compute_mini_batch_gradient(self, batch, dataset, population_RER, individual_RER, with_grad=True):
        # get target times and objects from the batch
        target_times = [t[0] for t in batch]
        target_objects = [t[1] for t in batch]

        # Careful: the batch selected might not go from tmin to tmax
        tmin = min(math.trunc(min(dataset.times[0])), self.t0)
        self.geodesic.set_tmin(tmin)
        self.geodesic.set_tmax(math.ceil(max(dataset.times[0])))

        device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(with_grad, device=device)
                
        attachement, regularity = self._compute_batch_attachment_and_regularity(target_times, target_objects, 
                                                                                template_data, template_points, control_points, momenta)
        
        return self.compute_gradients(attachement, regularity, template_data, template_points,
                                        control_points, momenta,  with_grad=with_grad)
        

    def mini_batches(self, dataset, number_of_batches):
        batch_size = len(dataset.deformable_objects[0])//number_of_batches

        targets = [[t,target] for t, target in zip(dataset.times[0], dataset.deformable_objects[0])]
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

    def _fixed_effects_to_torch_tensors(self, with_grad, device='cpu'):
        """
        Convert the fixed_effects into torch tensors.
        """
        # Template data.
        template_data = self.fixed_effects['template_data']
        template_data = {key: utilities.move_data(value,
                                                  dtype=self.tensor_scalar_type,
                                                  requires_grad=(not self.freeze_template and with_grad),
                                                  device=device)
                         for key, value in template_data.items()}

        # Template points.
        template_points = self.template.get_points()
        template_points = {key: utilities.move_data(value,
                                                    dtype=self.tensor_scalar_type,
                                                    requires_grad=(not self.freeze_template and with_grad),
                                                    device=device)
                           for key, value in template_points.items()}

        # Control points.
        if self.dense_mode:
            assert (('landmark_points' in self.template.get_points().keys()) and
                    ('image_points' not in self.template.get_points().keys())), \
                'In dense mode, only landmark objects are allowed. One at least is needed.'
            control_points = template_points['landmark_points']
        else:
            control_points = self.fixed_effects['control_points']
            control_points = utilities.move_data(control_points,
                                                 dtype=self.tensor_scalar_type,
                                                 requires_grad=(not self.freeze_control_points and with_grad),
                                                 device=device)

        # Momenta.
        momenta = self.fixed_effects['momenta']
        momenta = utilities.move_data(momenta,
                                      dtype=self.tensor_scalar_type,
                                      requires_grad=with_grad,
                                      device=device)

        return template_data, template_points, control_points, momenta
        

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################        
    
    def prepare_geodesic(self, dataset, device = "cpu"):
        template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(False, device)
        target_times = dataset.times[0]
        target_objects = dataset.deformable_objects[0]

        tmin = min(math.trunc(min(target_times)), self.t0)
        self.geodesic.set_tmin(tmin)
        self.geodesic.set_tmax(math.ceil(max(target_times)))
        self.geodesic.set_template_points_t0(template_points)
        self.geodesic.set_control_points_t0(control_points)
        self.geodesic.set_momenta_t0(momenta)
        self.geodesic.update()  

        return template_data, target_times, target_objects

    def compute_residuals(self, dataset, individual_RER = None, k = False, option = None):
        """
        Compute distances with various kernel width (for current or varifold) 
        (- default = model width) or additional dist like KD tree
        """
        template_data, target_times, target_objects = self.prepare_geodesic(dataset)
        
        if not k:
            residuals = []
            for (time, target) in zip(target_times, target_objects):
                deformed_points = self.geodesic.get_template_points(time)
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)

                if not option:
                    #residuals.append(self.multi_object_attachment.compute_distances(deformed_data, self.template, target).cpu().numpy())
                    residuals.append(self.multi_object_attachment.compute_weighted_distance(
                    deformed_data, self.template, target, self.objects_noise_variance).cpu().numpy())
                else:
                    residuals.append(self.multi_object_attachment.compute_additional_distances(deformed_data, self.template, target, option).cpu().numpy())
        else:
            residuals = {i : list() for i in k}
            for i in k:
                for (time, target) in zip(target_times, target_objects):
                    deformed_points = self.geodesic.get_template_points(time)
                    deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                    residuals[i].append(self.multi_object_attachments_k[i].compute_distances(deformed_data, self.template, target, residuals = True).cpu().numpy())

        return residuals
    
    def compute_residuals_per_point(self, dataset, k = False):
        # residuals: if True, kernel=1 in distance computation for meshes

        template_data, target_times, target_objects = self.prepare_geodesic(dataset)     

        residuals = []
        #ajout fg multiscale meshes
        attachment =self.multi_object_attachment if not k else self.multi_object_attachments_k[k]

        for (time, target) in zip(target_times, target_objects):
            deformed_points = self.geodesic.get_template_points(time)
            
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)
            residuals.append(attachment.compute_distances(deformed_data, self.template, target).cpu().numpy())
        
        return residuals


    def write(self, dataset, population_RER, individual_RER, output_dir, iteration, 
              write_all = True):
        self._write_model_predictions(output_dir, dataset, write_all)
        self._write_model_parameters(output_dir, iteration, write_all)

        residuals = self.compute_residuals(dataset)

        write_2D_list(residuals, output_dir, self.name + "__EstimatedParameters__Residuals.txt")

    def _write_model_predictions(self, output_dir, dataset=None, write_all = True):

        # Initialize ---------------------------------------------------------------------------------------------------
        template_data, target_times, _ = self.prepare_geodesic(dataset)

        # Write --------------------------------------------------------------------------------------------------------
        # Geodesic flow.
        self.geodesic.write(self.name, self.objects_name, self.objects_name_extension, self.template, template_data,
                            output_dir, write_adjoint_parameters = self.write_adjoint_parameters, write_all = write_all)

        # Model predictions.
        if dataset is not None: #and not write_all:
            for j, time in enumerate(target_times):
                names = []
                for k, (object_name, object_extension) in enumerate(
                        zip(self.objects_name, self.objects_name_extension)):
                    name = self.name + '__Reconstruction__' + object_name + '__tp_' + str(j) + ('__age_%.2f' % time) \
                           + object_extension
                    names.append(name)
                deformed_points = self.geodesic.get_template_points(time)
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                self.template.write(output_dir, names,
                                    {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

    def output_path(self, output_dir, dataset):
        self.cp_path = op.join(output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")
        self.momenta_path = op.join(output_dir, self.name + "__EstimatedParameters__Momenta.txt")
        
        if self.geodesic.forward_exponential.number_of_time_points is None:
            device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)
            self.prepare_geodesic(dataset, device)

        try:
            self.geodesic.output_path(self.name, self.objects_name, self.objects_name_extension, output_dir)
        except:
            pass
    def _write_model_parameters(self, output_dir, iteration, write_all = True):
        
        # Template.
        template_names = []
        for k in range(len(self.objects_name)):
            if not self.freeze_template:
                aux = self.name + '__EstimatedParameters__Template_' + self.objects_name[k] + '__tp_' \
                  + str(self.geodesic.backward_exponential.number_of_time_points - 1) \
                  + ('__age_%.2f' % self.geodesic.t0) + "_iter_"+str(iteration)+ self.objects_name_extension[k]
            else:
                aux = self.name + '__Fixed__Template_' + self.objects_name[k] + '__tp_' \
                  + str(self.geodesic.backward_exponential.number_of_time_points - 1) \
                  + ('__age_%.2f' % self.geodesic.t0) + self.objects_name_extension[k]
            template_names.append(aux)
        self.template.write(output_dir, template_names)

        if self.objects_name_extension[0] != ".vtk":
            #Control points.
            write_2D_array(self.get_control_points(), output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")

            # Momenta.
            write_3D_array(self.get_momenta(), output_dir, self.name + "__EstimatedParameters__Momenta.txt")
        else:
            # Fuse control points and momenta for paraview display
            concatenate_for_paraview(self.get_momenta(), self.get_control_points(), output_dir, 
                            self.name + "__EstimatedParameters__Fusion_CP_Momenta_iter_{}.vtk".format(iteration))