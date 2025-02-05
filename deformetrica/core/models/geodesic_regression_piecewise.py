import math
from ...support import kernels as kernel_factory
from ...core import default
from ...core.model_tools.deformations.piecewise_geodesic import PiecewiseGeodesic
from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...core.models.model_functions import initialize_control_points, initialize_momenta
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...in_out.array_readers_and_writers import *
from ...in_out.dataset_functions import create_template_metadata
from ...support.utilities import get_best_device, move_data
import os.path as op

logger = logging.getLogger(__name__)

class PiecewiseGeodesicRegression(AbstractStatisticalModel):
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
                 freeze_rupture_time = default.freeze_rupture_time,
                 freeze_t0 = default.freeze_rupture_time, # ajout fg

                 gpu_mode=default.gpu_mode,

                 num_component = 2,
                 new_bounding_box = None, # ajout fg
                 weights = None,
                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='GeodesicRegression', gpu_mode=gpu_mode)

        # Global-like attributes.
        self.dimension = dimension

        # Declare model structure.
        self.t0 = t0 # t0 AND t1 must be provided to compute the tR
        self.t1 = t1
        self.tR = tR
        self.nb_components = num_component

        self.freeze_template = freeze_template
        self.freeze_momenta = False
        self.freeze_rupture_time = freeze_rupture_time

        self.weights = weights

        # Deformation.
        self.geodesic = PiecewiseGeodesic(t0 = t0, nb_components = self.nb_components, template_tR=None,
                kernel=kernel_factory.factory(gpu_mode=gpu_mode, kernel_width=deformation_kernel_width),
                concentration_of_time_points=concentration_of_time_points)
        self.geodesic.set_tR(self.tR)
        #self.geodesic.set_t0(self.t0)

        self.device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)

        # Template TODO? several templates
        (object_list, self.objects_extension, self.objects_noise_variance, self.multi_object_attachment) = \
        create_template_metadata(template_specifications, self.dimension, gpu_mode=gpu_mode)
        
        self.template = DeformableMultiObject(object_list)

        self.number_of_objects = len(self.template.object_list)
        
        self.deformation_kernel_width = deformation_kernel_width
        self.number_of_subjects = 1
        
        # Template data.
        self.fixed_effects['template_data'] = self.template.get_data()        

        # Make the template bouding box wider in order to array dim x 2 (max and min)
        # Control points and Momenta.
        self.control_points = initialize_control_points(initial_control_points, 
            self.template, deformation_kernel_width, self.dimension, new_bounding_box = new_bounding_box)

        self.fixed_effects['momenta']= initialize_momenta(
            initial_momenta, len(self.control_points), self.dimension, number_of_subjects = self.nb_components)
                
        # Rupture times: FIXED at regular intervals
        self.fixed_effects['rupture_time'] = np.zeros((self.nb_components - 1))
        
        # We store ONLY the rupture times
        # in piecewise_geodesic, tmin and tmax are also stored with the tR
        if not self.tR:
            segment = int((math.ceil(self.t1) - math.trunc(self.t0))/self.nb_components)
            for i in range(self.nb_components - 1):
                self.set_rupture_time(math.trunc(self.t0) + segment * (i+1), i)
        else:
            for i, t in enumerate(self.tR):
                self.set_rupture_time(t, i)
    
        self.current_residuals = None
    
    def get_template_index(self):
        for i, t in enumerate(self.get_rupture_time()):
            if t == self.t0:
                self.template_index = i     
        
    def initialize_noise_variance(self, dataset):
        # only if not provided by user
        if np.min(self.objects_noise_variance) < 0:
            target_times = dataset.times[0]
            residuals = np.sum(self.compute_residuals(dataset))

            # Initialize the noise variance hyper-parameter as a 1/100th of the initial residual.
            for k in range(self.objects_extension): #careful nb objects
                if self.objects_noise_variance[k] < 0:
                    nv = 0.01 * residuals / float(len(target_times))
                    self.objects_noise_variance[k] = nv
                    logger.info('>> Automatically chosen noise std', math.sqrt(nv))

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################
    # Template data ----------------------------------------------------------------------------------------------------
    def get_template_data(self):
        return self.fixed_effects['template_data']

    def set_template_data(self, td):
        self.fixed_effects['template_data'] = td
        self.template.set_data(td)

    def set_weights(self, w):
        # Ajout fg: to weight subjects in cost function
        logger.info("Setting weights to {}".format(w)) 
        self.weights = w

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

    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self):
        out = {}
        if not self.freeze_template:
            for key, value in self.fixed_effects['template_data'].items():
                out[key] = value
        if not self.freeze_rupture_time:
            out['rupture_time'] = self.fixed_effects['rupture_time']

        out['momenta'] = self.fixed_effects['momenta']

        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template:
            template_data = {key: fixed_effects[key] for key in self.fixed_effects['template_data'].keys()}
            self.set_template_data(template_data)
        if not self.freeze_rupture_time:
            for i in range(self.nb_components - 1):
                self.set_rupture_time(fixed_effects['rupture_time'], i)

        self.set_momenta(fixed_effects['momenta'])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def compute_gradients(self, attachment, regularity, template_data, template_points,
                           momenta, rupture_time, with_grad=False):
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

            # Control points and momenta.
            if not self.freeze_rupture_time:
                gradient['rupture_time'] = rupture_time.grad
            if not self.freeze_momenta:
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

        :param dataset: LongitudinalDataset instance
        :param fixed_effects: Dictionary of fixed effects.
        :param indRER: Dictionary of individual random effects realizations.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """
        # Initialize: conversion from numpy to torch -------------------------------------------------------------------
        template_data, template_points, momenta, rupture_time = self._fixed_effects_to_torch_tensors(with_grad)

        # Deform -------------------------------------------------------------------------------------------------------
        att, reg = self._compute_attachment_and_regularity(dataset, template_data, template_points, momenta)
        
        # Gradients -------------------------------------------------------------------------------------------------------
        return self.compute_gradients(att, reg, template_data, template_points,
                                    momenta, rupture_time, with_grad=with_grad)


    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _compute_batch_attachment_and_regularity(self, target_times, target_objects, 
                                                 template_data, template_points, momenta):        

        # Deform -------------------------------------------------------------------------------------------------------
        self.geodesic.set_template_points_tR(template_points)
        #self.geodesic.set_control_points_tR(control_points)
        self.geodesic.set_momenta_tR(momenta)
        self.geodesic.update()

        if self.weights is None:
            self.weights = [1.] * len(target_objects)

        attachment = 0.
        self.current_residuals = []
        for j, (time, obj, w) in enumerate(zip(target_times, target_objects, self.weights)):
            deformed_points = self.geodesic.get_template_points(time)
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)
            att = self.multi_object_attachment.compute_weighted_distance(
                deformed_data, self.template, obj, self.objects_noise_variance)
            attachment -= w * att
            self.current_residuals.append(att.cpu())
        
        for i in range(self.nb_components):
            regularity = - self.geodesic.get_norm_squared(i)
        
        return attachment, regularity
    
    def _compute_attachment_and_regularity(self, dataset, template_data, template_points, momenta):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """
        target_times = dataset.times[0]
        target_objects = dataset.deformable_objects[0]

        tmin = min(math.trunc(min(dataset.times[0])), self.t0)
        self.geodesic.set_tmin(tmin) # arrondi à entier inf
        self.geodesic.set_tmax(math.ceil(max(dataset.times[0]))) #entier sup

        return self._compute_batch_attachment_and_regularity(target_times, target_objects, template_data, template_points, momenta)
    
    def compute_mini_batch_gradient(self, batch, dataset, individual_RER, with_grad=True):
        # get target times and objects from the batch
        target_times = [t[0] for t in batch]
        target_objects = [t[1] for t in batch]

        # Careful: the batch selected might not go from tmin to tmax
        tmin = min(math.trunc(min(dataset.times[0])), self.t0)
        self.geodesic.set_tmin(tmin)
        self.geodesic.set_tmax(math.ceil(max(dataset.times[0])))

        
        template_data, template_points, momenta, rupture_time \
        = self._fixed_effects_to_torch_tensors(with_grad)
                
        attachement, regularity = self._compute_batch_attachment_and_regularity(target_times, target_objects, 
                                                                                template_data, template_points, 
                                                                                momenta)
        
        return self.compute_gradients(attachement, regularity, template_data, template_points,
                                        momenta, rupture_time, with_grad=with_grad)
        
    def mini_batches(self, dataset, number_of_batches):
        batch_size = len(dataset.deformable_objects[0]) // number_of_batches

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
            return self.multi_object_attachment.compute_vtk_distance(deformed_data, self.template, obj, dist)
        elif dist in ["ssim", "mse"]:
            return self.multi_object_attachment.compute_ssim_distance(deformed_data, self.template, obj, dist)


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

    def compute_curvature(self, dataset, j = None, individual_RER = None, curvature = "gaussian", iter = None):
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

        template_data, _, _ = self.prepare_geodesic(dataset)
        
        deformed_points = self.geodesic.get_template_points(dataset.times[0][j])
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        
        for i, obj1 in enumerate(self.template.object_list):
            obj1.polydata.points = deformed_data['landmark_points'][0:obj1.get_number_of_points()].cpu().numpy()
            obj1.curvature_metrics(curvature)
        
        return self.template

    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad = False):
        """
        Convert the fixed_effects into torch tensors.
        """
        # Template data.
        template_data = {key: move_data(value, device = self.device,
                                        requires_grad = (not self.freeze_template and with_grad))
                         for key, value in self.get_template_data().items()}

        # Template points.
        template_points = {key: move_data(value, device = self.device,
                                        requires_grad = (not self.freeze_template and with_grad)) 
                            for key, value in self.template.get_points().items()}        

        # Momenta.
        momenta = move_data(self.get_momenta(),requires_grad = with_grad, device=self.device)
        
        # Rupture time
        rupture_time = move_data(self.get_rupture_time(), requires_grad = with_grad, device = self.device)

        return template_data, template_points, momenta, rupture_time
        

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################            
    def prepare_geodesic(self, dataset, tmax = None):
        control_points = move_data(self.control_points, device = self.device)
        template_data, template_points, momenta, rupture_time = self._fixed_effects_to_torch_tensors()
        target_times = dataset.times[0]
        target_objects = dataset.deformable_objects[0]

        tmin = min(math.trunc(min(target_times)), self.t0)
        self.geodesic.set_tmin(tmin)
        if tmax is None:
            self.geodesic.set_tmax(math.ceil(max(target_times)))
        else:
            self.geodesic.set_tmax(tmax)

        self.geodesic.set_tR(rupture_time)
        self.geodesic.set_t0(self.t0) #ajout fg
        self.geodesic.set_template_points_tR(template_points) # a list
        self.geodesic.set_control_points_tR(control_points) # a list
        self.geodesic.set_momenta_tR(momenta) # a list
        self.geodesic.update()

        return template_data, target_times, target_objects

    def compute_residuals(self, dataset, individual_RER = None, option = None):
        template_data, target_times, target_objects = self.prepare_geodesic(dataset)
        
        residuals = []
        for (time, target) in zip(target_times, target_objects):
            deformed_points = self.geodesic.get_template_points(time)
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)
            if not option:
                residuals.append(self.multi_object_attachment.compute_weighted_distance(
                deformed_data, self.template, target, self.objects_noise_variance).cpu().numpy())
            else:
                residuals.append(self.multi_object_attachment.compute_additional_distances(deformed_data, self.template, target, option).cpu().numpy())
            
        return residuals


    def write(self, dataset, individual_RER, output_dir, iteration, 
              write_adjoint_parameters=True, write_all = True):
        self._write_model_predictions(output_dir, dataset, write_adjoint_parameters, write_all)
        self._write_model_parameters(output_dir, iteration, write_all)

    def _write_model_predictions(self, output_dir, dataset=None, write_adjoint_parameters=False, write_all = True):

        # Initialize ---------------------------------------------------------------------------------------------------
        template_data, target_times, _ = self.prepare_geodesic(dataset, tmax = self.t1)   

        # Write --------------------------------------------------------------------------------------------------------
        # Geodesic flow.
        self.geodesic.write(self.name, self.objects_extension, self.template, template_data,
                            output_dir, write_adjoint_parameters, write_all = write_all)

        # Model predictions.
        if dataset is not None and not write_all:
            for j, time in enumerate(target_times):
                names = []
                for ext in self.objects_extension:
                    name = '{}__Reconstruction____tp_{}__age_{}'.format(self.name, j, time, ext)
                    names.append(name)
                deformed_points = self.geodesic.get_template_points(time)
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                self.template.write(output_dir, names,
                                    {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

    def output_path(self, output_dir, dataset):
        self.cp_path = op.join(output_dir, "{}__ControlPoints.txt".format(self.name))
        self.momenta_path = op.join(output_dir, "{}__Estimated__Momenta.txt".format(self.name))

        if self.geodesic.exponential[0].number_of_time_points is None:
            self.prepare_geodesic(dataset)

        self.geodesic.output_path(self.name, self.objects_extension, output_dir)
        
    def _write_model_parameters(self, output_dir, iteration, write_all = True):
        if write_all:
            template_names = []
            for ext in self.objects_extension:
                tp = 0
                l= 0
                while self.geodesic.tR[l] < self.t0:
                    l += 1
                    tp += self.geodesic.exponential[l].number_of_time_points

                if not self.freeze_template:   
                    aux =  '{}__Estimated__Template__tp_{}__age_{}{}'.format(self.name, tp, self.t0, ext)
                else:
                    aux =  '{}__Fixed__Template__tp_{}__age_{}{}'.format(self.name, tp, self.t0, ext)
            template_names.append(aux)
            self.template.write(output_dir, template_names)
            
            #Control points.
            write_2D_array(self.control_points, output_dir, "{}___ControlPoints.txt".format(self.name))
            write_3D_array(self.get_momenta(), output_dir, self.name + "__Estimated__Momenta.txt")

            for c in range(self.nb_components):
                if self.dimension == 3:
                    concatenate_for_paraview(self.get_momenta()[c], self.control_points, output_dir, 
                                    self.name + "__Estimated__Fusion_CP_Momenta__component_{}_iter_.vtk".format(c, iteration))
    
