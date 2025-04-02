import math
import os.path as op
from ...support.kernels import factory
from ...core import default
from ...core.model_tools.deformations.piecewise_geodesic import PiecewiseGeodesic
from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...core.models.model_functions import initialize_cp, initialize_momenta
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...in_out.array_readers_and_writers import *
from ...in_out.dataset_functions import template_metadata
from ...support.utilities import get_best_device, move_data, detach

logger = logging.getLogger(__name__)

class PiecewiseGeodesicRegression(AbstractStatisticalModel):
    """
    Geodesic regression object class.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, template_specifications, deformation_kernel_width=None,
                 time_concentration=default.time_concentration, 
                 t0=default.t0, tR=[], t1 = default.tmax,
                 freeze_template=default.freeze_template,
                 initial_cp=None, initial_momenta=None,
                 freeze_rupture_time = default.freeze_rupture_time,
                 freeze_t0 = default.freeze_rupture_time, 
                 num_component = 2, bounding_box = None, # ajout fg
                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='GeodesicRegression')

        # Declare model structure.
        self.t0 = t0 # t0 AND t1 must be provided to compute the tR
        self.t1 = t1
        self.tR = tR
        self.nb_components = num_component

        self.freeze_template = freeze_template
        self.freeze_momenta = False
        self.freeze_rupture_time = freeze_rupture_time
       
        self.device = get_best_device()

        object_list, self.objects_noise_variance, self.attachment = \
                                                template_metadata(template_specifications)
        
        self.template = DeformableMultiObject(object_list)

        self.number_of_objects = len(self.template.object_list)
        self.dimension = self.template.dimension 
        
        self.deformation_kernel_width = deformation_kernel_width
        self.n_subjects = 1
        
        # Template data.
        self.set_template_data(self.template.get_data())
        self.points = self.get_points()      

        # Make the template bouding box wider in order to array dim x 2 (max and min)
        # Control points and Momenta.
        self.cp = initialize_cp(initial_cp, self.template, deformation_kernel_width, bounding_box)

        self.fixed_effects['momenta']= initialize_momenta(
            initial_momenta, len(self.cp), self.dimension, n_subjects = self.nb_components)
                
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

        # Deformation.
        self.geodesic = PiecewiseGeodesic(t0 = t0, nb_components = self.nb_components, template_tR=None,
                                        kernel=factory(kernel_width=deformation_kernel_width),
                                        time_concentration=time_concentration, extensions = self.extensions,
                                        root_name = self.name)
        self.geodesic.set_tR(self.tR)
        #self.geodesic.set_t0(self.t0)

        cp = move_data(self.cp, device = self.device)
        self.geodesic.set_cp_tR(cp) # a list
    
        self.current_residuals = None
    
    def get_template_index(self):
        for i, t in enumerate(self.get_rupture_time()):
            if t == self.t0:
                self.template_index = i     
        
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
                           momenta, rupture_time, points, with_grad=False):
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

            # Control points and momenta.
            if not self.freeze_rupture_time:
                gradient['rupture_time'] = rupture_time.grad
            if not self.freeze_momenta:
                gradient['momenta'] = momenta.grad

            # Convert the gradient back to numpy.
            gradient = {key: value.data.cpu().numpy() for key, value in gradient.items()}

            return detach(attachment), detach(regularity), gradient

        else:
            return detach(attachment), detach(regularity)

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, individual_RER, mode='complete', with_grad=False):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations

        :param dataset: LongitudinalDataset instance
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
                                    momenta, rupture_time, self.points, with_grad=with_grad)


    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _compute_batch_attachment_and_regularity(self, target_times, target_objects, 
                                                 template_data, template_points, momenta):        

        # Deform -------------------------------------------------------------------------------------------------------
        self.geodesic.set_template_points_tR(template_points)
        self.geodesic.set_momenta_tR(momenta)
        self.geodesic.update()

        attachment = 0.
        self.current_residuals = []
        for time, obj in zip(target_times, target_objects):
            deformed_points = self.geodesic.get_template_points(time)
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)
            att = self.attachment.compute_weighted_distance(deformed_data, self.template, 
                                                            obj, self.objects_noise_variance)
            attachment -= att
            self.current_residuals.append(att.cpu())
        
        for i in range(self.nb_components):
            regularity = - self.geodesic.get_norm_squared(i)
        
        return attachment, regularity
    
    def _compute_attachment_and_regularity(self, dataset, template_data, template_points, momenta):
        """
        Core part of the ComputeLogLikelihood methods. Fully torch.
        """
        target_times = dataset.times[0]
        target_objects = dataset.objects[0]

        self.geodesic.set_tmin(min(dataset.tmin, self.t0)) 
        self.geodesic.set_tmax(dataset.tmax) 

        return self._compute_batch_attachment_and_regularity(target_times, target_objects, 
                                                    template_data, template_points, momenta)
    
    def compute_mini_batch_gradient(self, batch, dataset, individual_RER, with_grad=True):
        # get target times and objects from the batch
        target_times = [t[0] for t in batch]
        target_objects = [t[1] for t in batch]

        # Careful: the batch selected might not go from tmin to tmax
        self.geodesic.set_tmin(min(dataset.tmin, self.t0))
        self.geodesic.set_tmax(dataset.tmax)

        template_data, template_points, momenta, rupture_time \
        = self._fixed_effects_to_torch_tensors(with_grad)
                
        attachement, regularity = self._compute_batch_attachment_and_regularity(target_times, target_objects, 
                                                                                template_data, template_points, 
                                                                                momenta)
        
        return self.compute_gradients(attachement, regularity, template_data, template_points,
                                        momenta, rupture_time, self.points, with_grad=with_grad)
        
    def mini_batches(self, dataset, number_of_batches):
        targets = list(zip(dataset.times[0], dataset.objects[0]))
        np.random.shuffle(targets)

        batch_size = len(targets) // number_of_batches
        mini_batches = [targets[i:i + batch_size] for i in range(0, len(targets), batch_size)]
        
        return mini_batches
    
    def compute_objects_distances(self, dataset, j, individual_RER = None, dist = "current", deformed = True): 
        """
        Compute current distance between deformed template and object
        #obj = a deformablemultiobject
        """
        template_data, _, _ = self.prepare_geodesic(dataset)
        
        deformed_points = self.geodesic.get_template_points(dataset.times[0][j])
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        obj = dataset.objects[0][j]
        if dist in ["current", "varifold"]:
            return self.attachment.compute_vtk_distance(deformed_data, self.template, obj, dist)
        elif dist in ["ssim", "mse"]:
            return self.attachment.compute_ssim_distance(deformed_data, self.template, obj, dist)


    def compute_flow_curvature(self, dataset, time, curvature = "gaussian"):
        template_data, _, _ = self.prepare_geodesic(dataset)

        deformed_points = self.geodesic.get_template_points(time)
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        for i, obj1 in enumerate(self.template.object_list):
            obj1.polydata.points = deformed_data['landmark_points'].cpu().numpy()
            obj1.curvature_metrics(curvature)
        
        return self.template

    def compute_initial_curvature(self, dataset, j, curvature = "gaussian"):
        obj = dataset.objects[0][j] 
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
                obj1.polydata.points = data['landmark_points'][0:obj1.n_points()]
                obj1.curvature_metrics(curvature)
            
                return self.template
            
        if iter == 0:
            return self.compute_initial_curvature(dataset, j, curvature)

        template_data, _, _ = self.prepare_geodesic(dataset)
        
        deformed_points = self.geodesic.get_template_points(dataset.times[0][j])
        deformed_data = self.template.get_deformed_data(deformed_points, template_data)
        
        for i, obj1 in enumerate(self.template.object_list):
            obj1.polydata.points = deformed_data['landmark_points'][0:obj1.n_points()].cpu().numpy()
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
        template_data, template_points, momenta, rupture_time = \
                                                        self._fixed_effects_to_torch_tensors()

        self.geodesic.set_tmin(min(dataset.tmin, self.t0))
        if tmax is None:
            self.geodesic.set_tmax(dataset.tmax)
        else:
            self.geodesic.set_tmax(tmax)

        self.geodesic.set_tR(rupture_time)
        self.geodesic.set_t0(self.t0) #ajout fg
        self.geodesic.set_template_points_tR(template_points) # a list
        self.geodesic.set_momenta_tR(momenta) # a list
        self.geodesic.update()

        return template_data, dataset.times[0], dataset.objects[0]

    def compute_residuals(self, dataset, individual_RER = None, option = None):
        template_data, target_times, target_objects = self.prepare_geodesic(dataset)
        
        residuals = []
        for (time, target) in zip(target_times, target_objects):
            deformed_points = self.geodesic.get_template_points(time)
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)
            if not option:
                residuals.append(self.attachment.compute_weighted_distance(
                deformed_data, self.template, target, self.objects_noise_variance).cpu().numpy())
            else:
                residuals.append(self.attachment.compute_additional_distances(deformed_data, self.template, target, option).cpu().numpy())
            
        return residuals

    def write(self, dataset, individual_RER, output_dir, iteration, write_all = True): 
        self._write_model_predictions(output_dir, dataset, write_all)
        self._write_model_parameters(output_dir, iteration, write_all)

    def _write_model_predictions(self, output_dir, dataset=None, write_all = True):

        # Initialize ---------------------------------------------------------------------------------------------------
        template_data, target_times, _ = self.prepare_geodesic(dataset, tmax = self.t1)   

        # Write --------------------------------------------------------------------------------------------------------
        # Geodesic flow.
        self.geodesic.write(self.template, template_data, output_dir, write_all = write_all)

        # Model predictions.
        if dataset is not None and not write_all:
            for j, time in enumerate(target_times):
                names = [reconstruction_name(self.name, j, time)]
                deformed_points = self.geodesic.get_template_points(time)
                deformed_data = self.template.get_deformed_data(deformed_points, template_data)
                self.template.write(output_dir, names, deformed_data)

    def output_path(self, output_dir, dataset):
        self.cp_path = op.join(output_dir, cp_name(name))
        self.momenta_path = op.join(output_dir, momenta_name(name))

        if self.geodesic.exponential[0].n_time_points is None:
            self.prepare_geodesic(dataset)

        self.geodesic.output_path(output_dir)
        
    def _write_model_parameters(self, output_dir, iteration, write_all = True):
                
        if write_all:
            time_points = sum([self.geodesic.exponential[l].n_time_points \
                                for l in range(self.geodesic.template_index()) ])
            template_names = [template_name(self.name, time = time_points, t0 = self.t0, 
                                        freeze_template = self.freeze_template)]
            self.template.write(output_dir, template_names)
            
            #Control points.
            write_cp(self.cp, output_dir, self.name)
            write_momenta(self.get_momenta(), output_dir, self.name)

            for c in range(self.nb_components):
                if self.dimension == 3:
                    concatenate_for_paraview(self.get_momenta()[c], self.cp, output_dir, self.name, iteration, c)
    
