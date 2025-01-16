import math
import time

import torch
import numpy as np

from ...support import kernels as kernel_factory
from ...core import default
from ...core.model_tools.deformations.exponential import Exponential
from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...core.models.model_functions import initialize_control_points, initialize_momenta
from ...core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ...in_out.array_readers_and_writers import *
from ...in_out.dataset_functions import create_template_metadata
from ...support import utilities

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
     freeze_template, freeze_control_points, freeze_momenta,
     exponential, sobolev_kernel, use_sobolev_gradient, tensor_scalar_type, gpu_mode) = process_initial_data
    (i, template, template_data, control_points, momenta, with_grad) = arg

    # start = time.perf_counter()
    device, device_id = utilities.get_best_device(gpu_mode=gpu_mode)
    # device, device_id = ('cpu', -1)
    if device_id >= 0:
        torch.cuda.set_device(device_id)

    # convert np.ndarrays to torch tensors. This is faster than transferring torch tensors to process.
    template_data = {key: utilities.move_data(value, device=device, dtype=tensor_scalar_type,
                                              requires_grad=with_grad and not freeze_template)
                     for key, value in template_data.items()}
    template_points = {key: utilities.move_data(value, device=device, dtype=tensor_scalar_type,
                                                requires_grad=with_grad and not freeze_template)
                       for key, value in template.get_points().items()}
    control_points = utilities.move_data(control_points, device=device, dtype=tensor_scalar_type,
                                         requires_grad=with_grad and not freeze_control_points)
    momenta = utilities.move_data(momenta, device=device, dtype=tensor_scalar_type,
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
        freeze_control_points, control_points,
        freeze_momenta, momenta,
        use_sobolev_gradient, sobolev_kernel,
        with_grad)
    # elapsed = time.perf_counter() - start
    # logger.info('pid=' + str(os.getpid()) + ', ' + torch.multiprocessing.current_process().name +
    #       ', device=' + device + ', elapsed=' + str(elapsed))
    
    return i, res #ajout fg


class DeterministicAtlas(AbstractStatisticalModel):
    """
    Deterministic atlas object class.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, template_specifications, number_of_subjects,

                 dimension=default.dimension,
                 tensor_scalar_type=default.tensor_scalar_type,
                 tensor_integer_type=default.tensor_integer_type,
                 dense_mode=default.dense_mode,
                 number_of_processes=default.number_of_processes,

                 deformation_kernel_type=default.deformation_kernel_type,
                 deformation_kernel_width=default.deformation_kernel_width,
                 deformation_kernel_device=default.deformation_kernel_device,

                 shoot_kernel_type=default.shoot_kernel_type,
                 number_of_time_points=default.number_of_time_points,
                 use_rk2_for_shoot=default.use_rk2_for_shoot, use_rk2_for_flow=default.use_rk2_for_flow,

                 freeze_template=default.freeze_template,
                 use_sobolev_gradient=default.use_sobolev_gradient,
                 smoothing_kernel_width=default.smoothing_kernel_width,

                 initial_control_points=default.initial_control_points,
                 freeze_control_points=default.freeze_control_points,
                 optimize_nb_control_points = default.optimize_nb_control_points, #ajout fg
                 max_spacing = default.max_spacing, #ajout fg
                 initial_cp_spacing=default.initial_cp_spacing,

                 initial_momenta=default.initial_momenta,
                 freeze_momenta=default.freeze_momenta,

                 gpu_mode=default.gpu_mode,
                 process_per_gpu=default.process_per_gpu,

                 **kwargs):

        AbstractStatisticalModel.__init__(self, name='DeterministicAtlas', number_of_processes=number_of_processes,
                                          gpu_mode=gpu_mode)

        # Global-like attributes.
        self.dimension = dimension
        self.tensor_scalar_type = tensor_scalar_type
        self.tensor_integer_type = tensor_integer_type
        self.dense_mode = dense_mode

        # Declare model structure.
        self.fixed_effects['template_data'] = None
        self.fixed_effects['control_points'] = None
        self.fixed_effects['momenta'] = None

        self.freeze_template = freeze_template
        self.freeze_control_points = freeze_control_points
        self.optimize_nb_control_points = optimize_nb_control_points #ajout fg
        self.max_spacing = max_spacing #ajout fg
        self.maximum_spacing = 0.5 #ajout fg
        self.freeze_momenta = freeze_momenta

        #ajout fg
        self.coarse_to_fine_count = 0
        self.original_cp_spacing = initial_cp_spacing
        self.initial_cp_spacing = initial_cp_spacing #will change if optimize_nb_control_points is True
        self.gpu_mode = gpu_mode
        self.deformation_kernel_width = deformation_kernel_width
        self.deformation_kernel_type = deformation_kernel_type
        self.initial_residuals = 0

        if self.optimize_nb_control_points:
            self.fixed_effects['coarse_momenta'] = None


        # Template.
        (object_list, self.objects_name, self.objects_name_extension,
         self.objects_noise_variance, self.multi_object_attachment) = create_template_metadata(template_specifications,
                                                                                               self.dimension)

        self.template = DeformableMultiObject(object_list)
        # self.template.update()
        self.number_of_objects = len(self.template.object_list)
        
        #ajout fg
        #initialize grid of control points
        if self.optimize_nb_control_points:
            intensities = self.template.get_data()["image_intensities"]
            width = min([intensities.shape[k] for k in range(self.dimension)]) - 1
            if self.dimension == 2: #to have 8 control points
                width /= 2
            self.original_cp_spacing = width
            self.initial_cp_spacing = width
            self.deformation_kernel_width = width 
            #parameters ignored if initial_control_points given

        # Deformation.
        self.exponential = Exponential(
            dense_mode=dense_mode,
            kernel=kernel_factory.factory(deformation_kernel_type,
                                          gpu_mode=gpu_mode,
                                          kernel_width=deformation_kernel_width),
            shoot_kernel_type=shoot_kernel_type,
            number_of_time_points=number_of_time_points,
            use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)


        self.use_sobolev_gradient = use_sobolev_gradient
        self.smoothing_kernel_width = smoothing_kernel_width
        if self.use_sobolev_gradient:
            self.sobolev_kernel = kernel_factory.factory(deformation_kernel_type,
                                                         gpu_mode=gpu_mode,
                                                         kernel_width=smoothing_kernel_width)

        # Template data.
        self.fixed_effects['template_data'] = self.template.get_data()

        # Control points.
        self.fixed_effects['control_points'] = initialize_control_points(
            initial_control_points, self.template, self.initial_cp_spacing, deformation_kernel_width,
            self.dimension, self.dense_mode)
        self.number_of_control_points = len(self.fixed_effects['control_points'])

        # Momenta.
        self.fixed_effects['momenta'] = initialize_momenta(
            initial_momenta, self.number_of_control_points, self.dimension, number_of_subjects)
        self.number_of_subjects = number_of_subjects

        self.process_per_gpu = process_per_gpu

        #ajout fg
        if self.optimize_nb_control_points:
            self.fixed_effects['coarse_momenta'] = self.fixed_effects['momenta']


    def initialize_noise_variance(self, dataset, device='cpu'):
        if np.min(self.objects_noise_variance) < 0:
            template_data, template_points, control_points, momenta, coarse_momenta = self._fixed_effects_to_torch_tensors(False,
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

            print("(initial ?) residuals", residuals)
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
    
    #ajout fg
    def get_coarse_momenta(self):
        return self.fixed_effects['coarse_momenta']
    
    def set_coarse_momenta(self, mom):
        self.fixed_effects['coarse_momenta'] = mom

    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self):
        out = {}
        if not self.freeze_template:
            for key, value in self.fixed_effects['template_data'].items():
                out[key] = value
        if not self.freeze_control_points:
            out['control_points'] = self.fixed_effects['control_points']
        if not self.freeze_momenta:
            out['momenta'] = self.fixed_effects['momenta']
        
        #ajout fg
        if self.optimize_nb_control_points:
            out['coarse_momenta'] = self.fixed_effects['coarse_momenta']
        
        return out

    def set_fixed_effects(self, fixed_effects):
        if not self.freeze_template:
            template_data = {key: fixed_effects[key] for key in self.fixed_effects['template_data'].keys()}
            self.set_template_data(template_data)
        if not self.freeze_control_points:
            self.set_control_points(fixed_effects['control_points'])
        if not self.freeze_momenta:
            self.set_momenta(fixed_effects['momenta'])
        #ajout fg
        if self.optimize_nb_control_points:
            self.set_coarse_momenta(fixed_effects['coarse_momenta'])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def setup_multiprocess_pool(self, dataset):
        self._setup_multiprocess_pool(initargs=([target[0] for target in dataset.deformable_objects],
                                                self.multi_object_attachment,
                                                self.objects_noise_variance,
                                                self.freeze_template, self.freeze_control_points, 
                                                self.freeze_momenta, 
                                                self.exponential, self.sobolev_kernel, self.use_sobolev_gradient,
                                                self.tensor_scalar_type, self.gpu_mode))

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

        if self.number_of_processes > 1:
            targets = [target[0] for target in dataset.deformable_objects]
            args = [(i, self.template,
                     self.fixed_effects['template_data'],
                     self.fixed_effects['control_points'],
                     self.fixed_effects['momenta'][i],
                     with_grad) for i in range(len(targets))]

            start = time.perf_counter()
            results = self.pool.map(_subject_attachment_and_regularity, args, chunksize=1)  # TODO: optimized chunk size
            # results = self.pool.imap_unordered(_subject_attachment_and_regularity, args, chunksize=1)
            # results = self.pool.imap(_subject_attachment_and_regularity, args, chunksize=int(len(args)/self.number_of_processes))
            logger.debug('time taken for deformations : ' + str(time.perf_counter() - start))

            # Sum and return.
            if with_grad:
                attachment = 0.0
                regularity = 0.0
                #additione le gradient et l'attachement de chaque sujet
                gradient = {}
                if not self.freeze_template:
                    for key, value in self.fixed_effects['template_data'].items():
                        gradient[key] = np.zeros(value.shape)
                if not self.freeze_control_points:
                    gradient['control_points'] = np.zeros(self.fixed_effects['control_points'].shape)
                if not self.freeze_momenta:
                    gradient['momenta'] = np.zeros(self.fixed_effects['momenta'].shape)
                #ajout fg
                if self.optimize_nb_control_points:
                    gradient['coarse_momenta'] = np.zeros(self.fixed_effects['coarse_momenta'].shape)

                for result in results:
                    i, (attachment_i, regularity_i, gradient_i) = result
                    #ajout fg
                    i, (attachment_i, regularity_i, gradient_i) = result
                    attachment += attachment_i
                    regularity += regularity_i
                    #residus += residus_i #ajout fg
                    for key, value in gradient_i.items():
                        if key == 'momenta':
                            gradient[key][i] = value
                        else:
                            gradient[key] += value
                return attachment, regularity, gradient
            else:
                attachment = 0.0
                regularity = 0.0
                for result in results:
                    #i, (attachment_i, regularity_i) = result
                    #ajout fg
                    i, (attachment_i, regularity_i, gradient_i) = result
                    attachment += attachment_i
                    regularity += regularity_i
                return attachment, regularity

        else:
            device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
            template_data, template_points, control_points, momenta, coarse_momenta = self._fixed_effects_to_torch_tensors(with_grad,
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

        # Compute attachment and regularity.
        #print("template_points", template_points['image_points'], template_points['image_points'].shape)
        deformed_points = exponential.get_template_points() #template points
        #print("deformed_points", deformed_points['image_points'], deformed_points['image_points'].shape)
        deformed_data = template.get_deformed_data(deformed_points, template_data) #template intensities after deformation
        #(observation) deformable multi object -> image -> torch.interpolate
        attachment = -multi_object_attachment.compute_weighted_distance(deformed_data, template, deformable_objects,
                                                                        objects_noise_variance)
        #print("attachment", attachment)
        regularity = -exponential.get_norm_squared()

        assert torch.device(
            device) == attachment.device == regularity.device, 'attachment and regularity tensors must be on the same device. ' \
                                                               'device=' + device + \
                                                               ', attachment.device=' + str(attachment.device) + \
                                                               ', regularity.device=' + str(regularity.device)
        
        return attachment, regularity

    @staticmethod
    def _compute_gradients(attachment, regularity, template_data,
                           freeze_template, template_points,
                           freeze_control_points, control_points,
                           freeze_momenta, momenta,
                           use_sobolev_gradient, sobolev_kernel,
                           with_grad=False):
        if with_grad:
            total_for_subject = attachment + regularity #torch tensor
            #print("total_for_subject", total_for_subject)
            total_for_subject.backward() #compute gradient 
            #print("template_data['image_intensities'].grad", template_data['image_intensities'].grad)

            gradient = {}
            #print("_compute_gradients")
            #print("momenta[0]", momenta[0])
            #print("momenta.grad[0]", momenta.grad[0])
            if not freeze_template:
                if 'landmark_points' in template_data.keys():
                    assert template_points['landmark_points'].grad is not None, 'Gradients have not been computed'
                    if use_sobolev_gradient:
                        gradient['landmark_points'] = sobolev_kernel.convolve(
                            template_data['landmark_points'].detach(), template_data['landmark_points'].detach(),
                            template_points['landmark_points'].grad.detach()).cpu().numpy()
                    else:
                        gradient['landmark_points'] = template_points['landmark_points'].grad.detach().cpu().numpy()
                if 'image_intensities' in template_data.keys():
                    assert template_data['image_intensities'].grad is not None, 'Gradients have not been computed'
                    gradient['image_intensities'] = template_data['image_intensities'].grad.detach().cpu().numpy()
            if not freeze_control_points:
                assert control_points.grad is not None, 'Gradients have not been computed'
                gradient['control_points'] = control_points.grad.detach().cpu().numpy()
            if not freeze_momenta:
                assert momenta.grad is not None, 'Gradients have not been computed'
                gradient['momenta'] = momenta.grad.detach().cpu().numpy()

            res = attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy(), gradient

        else:
            res = attachment.detach().cpu().numpy(), regularity.detach().cpu().numpy()

        return res

    def _compute_attachment_and_regularity(self, dataset, template_data, template_points, control_points, momenta,
                                           with_grad=False, device='cpu'):
        """
        Core part of the ComputeLogLikelihood methods. Torch input, numpy output.
        Single-thread version.
        """

        # Initialize.
        targets = [target[0] for target in dataset.deformable_objects]
        attachment = 0.
        regularity = 0.
        #residus = torch.zeros((template_data['image_intensities'].shape), device=device, dtype=dtype) #ajout fg

        # loop for every deformable object
        # deform and update attachment and regularity
        for i, target in enumerate(targets):
            new_attachment, new_regularity = DeterministicAtlas._deform_and_compute_attachment_and_regularity(
                self.exponential, template_points, control_points, momenta[i],
                self.template, template_data, self.multi_object_attachment,
                target, self.objects_noise_variance,
                device=device)

            attachment += new_attachment
            regularity += new_regularity

        # Compute gradient.
        return self._compute_gradients(attachment, regularity, template_data,
                                       self.freeze_template, template_points,
                                       self.freeze_control_points, control_points,
                                       self.freeze_momenta, momenta,
                                       self.use_sobolev_gradient, self.sobolev_kernel,
                                       with_grad)

    ####################################################################################################################
    ### Private utility methods:
    ####################################################################################################################

    def _fixed_effects_to_torch_tensors(self, with_grad, device='cpu'):
        """
        Convert the fixed_effects into torch tensors.
        """
        # Template data.
        template_data = self.fixed_effects['template_data']
        template_data = {key: utilities.move_data(value, device=device, dtype=self.tensor_scalar_type,
                                                  requires_grad=(not self.freeze_template and with_grad))
                         for key, value in template_data.items()}
        # template_data = {key: Variable(torch.from_numpy(value).type(self.tensor_scalar_type),
        #                                requires_grad=(not self.freeze_template and with_grad))
        #                  for key, value in template_data.items()}

        # Template points.
        template_points = self.template.get_points()
        template_points = {key: utilities.move_data(value, device=device, dtype=self.tensor_scalar_type,
                                                    requires_grad=(not self.freeze_template and with_grad))
                           for key, value in template_points.items()}
        # template_points = {key: Variable(torch.from_numpy(value).type(self.tensor_scalar_type),
        #                                  requires_grad=(not self.freeze_template and with_grad))
        #                    for key, value in template_points.items()}

        # Control points.
        if self.dense_mode:
            assert (('landmark_points' in self.template.get_points().keys()) and
                    ('image_points' not in self.template.get_points().keys())), \
                'In dense mode, only landmark objects are allowed. One at least is needed.'
            control_points = template_points['landmark_points']
        else:
            control_points = self.fixed_effects['control_points']
            control_points = utilities.move_data(control_points, device=device, dtype=self.tensor_scalar_type,
                                                 requires_grad=(not self.freeze_control_points and with_grad))
            # control_points = Variable(torch.from_numpy(control_points).type(self.tensor_scalar_type),
            #                           requires_grad=(not self.freeze_control_points and with_grad))
        # Momenta.
        momenta = self.fixed_effects['momenta']
        momenta = utilities.move_data(momenta, device=device, dtype=self.tensor_scalar_type,
                                      requires_grad=(not self.freeze_momenta and with_grad))
        # momenta = Variable(torch.from_numpy(momenta).type(self.tensor_scalar_type),
        #                    requires_grad=(not self.freeze_momenta and with_grad))

        #ajout fg
        coarse_momenta = self.fixed_effects['momenta']
        coarse_momenta = utilities.move_data(momenta, device=device, dtype=self.tensor_scalar_type,
                                      requires_grad=(self.optimize_nb_control_points and with_grad))

        return template_data, template_points, control_points, momenta, coarse_momenta #ajout fg

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, dataset, population_RER, individual_RER, output_dir, write_residuals=True):

        # Write the model predictions, and compute the residuals at the same time.
        residuals = self._write_model_predictions(dataset, individual_RER, output_dir,
                                                  compute_residuals=write_residuals)

        # Write residuals.
        if write_residuals:
            residuals_list = [[residuals_i_k.data.cpu().numpy() for residuals_i_k in residuals_i]
                              for residuals_i in residuals] #moyenne des résidus pour chaque sujet
            #ajout fg: moyenne des résidus
            moyenne = np.mean(np.asarray(residuals_list).flatten())
            initial_residuals_sum = np.sum(self.initial_residuals.flatten())
            #last_residuals_sum = np.sum(np.asarray(residuals_list).flatten()) #faux: trop élevé
            #residuals_ratio = 1 - last_residuals_sum/initial_residuals_sum
            residuals_ratio = 1 - moyenne/initial_residuals_sum


            residuals_list.append([0])
            residuals_list.append([moyenne])
            residuals_list.append([initial_residuals_sum]) #good
            #residuals_list.append([last_residuals_sum])
            residuals_list.append([residuals_ratio])

            write_2D_list(residuals_list, output_dir, self.name + "__EstimatedParameters__Residuals.txt")

        # Write the model parameters.
        self._write_model_parameters(output_dir)

    def _write_model_predictions(self, dataset, individual_RER, output_dir, compute_residuals=True):
        device, _ = utilities.get_best_device(self.gpu_mode)

        # Initialize.
        template_data, template_points, control_points, momenta, _ = self._fixed_effects_to_torch_tensors(False, device=device)

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
            self.exponential.write_flow(names, self.objects_name_extension, self.template, template_data, output_dir)
            
            deformed_points = self.exponential.get_template_points()
            deformed_data = self.template.get_deformed_data(deformed_points, template_data)

            if compute_residuals:
                residuals.append(self.multi_object_attachment.compute_distances(
                    deformed_data, self.template, dataset.deformable_objects[i][0]))

            names = []
            for k, (object_name, object_extension) \
                    in enumerate(zip(self.objects_name, self.objects_name_extension)):
                name = self.name + '__Reconstruction__' + object_name + '__subject_' + subject_id + object_extension
                names.append(name)
            self.template.write(output_dir, names,
                                {key: value.data.cpu().numpy() for key, value in deformed_data.items()})

        return residuals

    def _write_model_parameters(self, output_dir):

        # Template.
        template_names = []
        for i in range(len(self.objects_name)):
            aux = self.name + "__EstimatedParameters__Template_" + self.objects_name[i] + self.objects_name_extension[i]
            template_names.append(aux)
        self.template.write(output_dir, template_names)

        # Control points.
        write_2D_array(self.get_control_points(), output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")
        
        #save file name
        #self.last_control_points = os.path.join(output_dir, self.name + "__EstimatedParameters__ControlPoints.txt")

        # Momenta.
        write_3D_array(self.get_momenta(), output_dir, self.name + "__EstimatedParameters__Momenta.txt")


    ####################################################################################################################
    ### Coarse to fine 
    ###ajouts fg
    ####################################################################################################################
    
    def compute_residuals(self, dataset, current_iteration, save_every_n_iters, output_dir):
        """
        Compute residuals at each pixel/voxel between objects and deformed template.
        Save a heatmap of the residuals
        """

        #print("template_data", template_data) #template_data['image_intensities]
        #print("template_points", template_points) #dico ['image_points]
        #print("self.fixed_effects.keys()", self.fixed_effects.keys())
        #print("new_parameters", new_parameters.keys())
        #print("gradient.keys()", gradient.keys())

        # Deform template
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta, _ = \
            self._fixed_effects_to_torch_tensors(False, device = device)
        
        #####compute residuals
        self.exponential.set_initial_template_points(template_points)
        self.exponential.set_initial_control_points(control_points)
        residuals_by_point = torch.zeros((template_data['image_intensities'].shape), 
                                    device=next(iter(template_data.values())).device, 
                                    dtype=next(iter(template_data.values())).dtype)   #tensor not dict             

        for i, subject_id in enumerate(dataset.subject_ids):
            #deform template
            self.exponential.set_initial_momenta(momenta[i])
            self.exponential.update()

            deformed_points = self.exponential.get_template_points() #template points #tensor
            deformed_template = self.template.get_deformed_data(deformed_points, template_data) #dict containing tensor
            
            #get object intensities
            objet = dataset.deformable_objects[i][0]
            objet_intensities = objet.get_data()["image_intensities"]
            target_intensities = utilities.move_data(objet_intensities, device=next(iter(template_data.values())).device, 
                                    dtype = next(iter(template_data.values())).dtype) #tensor not dict 
            #compute residuals
            residuals_by_point += (1/dataset.number_of_subjects) * (target_intensities - deformed_template['image_intensities']) ** 2

        #residuals heat map
        if (not current_iteration % save_every_n_iters) or current_iteration in [0, 1]:
            names = "Heat_map_" + str(current_iteration) + self.objects_name_extension[0]
            deformed_template['image_intensities'] = residuals_by_point
            self.template.write(output_dir, [names], 
            {key: value.data.cpu().numpy() for key, value in deformed_template.items()})
        
        if current_iteration == 0:
            self.initial_residuals = residuals_by_point.cpu().numpy()
        
        return residuals_by_point.cpu().numpy()
    
    def number_of_pixels(self, dataset):
        objet_intensities = dataset.deformable_objects[0][0].get_data()["image_intensities"]
        number_of_pixels = 1
        for k in range(self.dimension):
            number_of_pixels = number_of_pixels * objet_intensities.shape[k]
        
        return number_of_pixels

    def add_points_linearly(self, control_points, taux = 0.3):
        self.coarse_to_fine_count += 1
        new_control_points, new_spacing = [], self.original_cp_spacing * np.exp((-1) * taux * self.coarse_to_fine_count)
        #si new_spacing < max_spacing on renvoie un mauvais spacing
        while len(new_control_points) <= len(control_points) and new_spacing > self.maximum_spacing:
            print("try spacing", new_spacing)
            new_control_points = initialize_control_points(None, self.template, new_spacing, self.deformation_kernel_width,
                                                    self.dimension, self.dense_mode)
            new_spacing = self.original_cp_spacing * np.exp((-1) * taux * self.coarse_to_fine_count)
            self.coarse_to_fine_count += 1
        
        return new_spacing, new_control_points

    def add_points_regularly(self, current_iteration):
        
        #add points 3 times
        new_spacing = None
        if current_iteration == 1:
            new_spacing = self.max_spacing*3
        if current_iteration == 2 or (new_spacing is not None and self.initial_cp_spacing < new_spacing):
            new_spacing = self.max_spacing*2
        if current_iteration == 3 or (new_spacing is not None and self.initial_cp_spacing < new_spacing):
            new_spacing = self.max_spacing
            
        if new_spacing is not None and self.initial_cp_spacing > new_spacing:
            new_control_points = initialize_control_points(None, self.template, new_spacing, self.deformation_kernel_width,
                                                    self.dimension, self.dense_mode)
            
            return new_spacing, new_control_points
        else:
            return self.maximum_spacing, []

    def add_points_slowly(self, current_iteration, nb_points_origin):
        new_spacing = None
        n = 10 - current_iteration
        if current_iteration > 0 and n > 0:
            new_spacing = self.max_spacing*n
            new_control_points = initialize_control_points(None, self.template, new_spacing, self.deformation_kernel_width,
                                                    self.dimension, self.dense_mode)
            while n > 1 and (self.initial_cp_spacing <= new_spacing or len(new_control_points) <= nb_points_origin):
                n = n-1
                new_spacing = self.max_spacing*n
                new_control_points = initialize_control_points(None, self.template, new_spacing, self.deformation_kernel_width,
                                                    self.dimension, self.dense_mode)            
            if n > 0:
                return new_spacing, new_control_points
        
        return self.maximum_spacing, []
        
    def add_points_only_once(self, current_iteration):
        if current_iteration == 1:
            new_control_points = initialize_control_points(None, self.template, self.max_spacing, self.deformation_kernel_width,
                                                    self.dimension, self.dense_mode)
            
            return self.max_spacing, new_control_points
        else:
            return self.maximum_spacing, []
        
    def add_points_same_spacing(self, current_iteration):
        if current_iteration < 15:
            new_control_points = initialize_control_points(None, self.template, self.max_spacing, self.deformation_kernel_width,
                                                    self.dimension, self.dense_mode)
            
            return self.max_spacing, new_control_points
        else:
            return self.maximum_spacing, []

    def save_points(self, current_iteration, control_points, dataset, output_dir):
        """
        Save control points on blank image
        """
        names = "Points_" + str(current_iteration) + "_" + str(len(control_points))+ self.objects_name_extension[0]
        objet_intensities = dataset.deformable_objects[0][0].get_data()["image_intensities"]
        shape = tuple([objet_intensities.shape[k] for k in range(self.dimension)])
        new_objet = np.zeros(shape)
        for point in control_points:
            new_objet[tuple([int(t) for t in point])] = 1
        

        #add kernel width
        """
        if self.deformation_kernel_width.shape[0] == len(control_points):
            for (i,point) in enumerate(control_points):
                new_objet[tuple([int(t) for t in point])] = 1
                sigma = self.deformation_kernel_width[i]
        """

        self.template.write(output_dir, [names],  {"image_intensities":new_objet})

    def closest_neighbors(self, new_control_points, old_control_points):
        """
        Compute the 4 closest controls points for each point in new_control_points.
        """
        closest_points_list = []
        for point in new_control_points:
            distance_to_pts = [(np.sqrt(np.sum((point-old_control_points[k])**2, axis=0)), k) \
                for k in range(len(old_control_points))]                    
            #if the point existed before, we keep its momenta
            #else we average 4 closest neighbors
            closest_points_indices = [c[1] for c in sorted(distance_to_pts)[:3]]
            same_point = [d[1] for d in distance_to_pts if d[0] == 0]
            if len(same_point) > 0:
                closest_points_indices = same_point
            closest_points_list.append(closest_points_indices)
        
        return closest_points_list

    def set_new_momenta(self, new_control_points, old_momenta, closest_points_list):
        """
        Update the momenta for each new point and each subject
        The momenta of a new point =  average momenta of its 4 closest neighbors
        """
        new_moments = np.zeros((old_momenta.shape[0], new_control_points.shape[0],
                                new_control_points.shape[1])) #n sujets x n points x dimension
        #for each subject -> for each new point -> add a new momenta
        for ind, old_momenta_subject in enumerate(old_momenta):
            new_momenta_subject = np.zeros((new_control_points.shape[0],new_control_points.shape[1]))

            for (i,new_point) in enumerate(new_control_points):            
                #average of closest neighbours momenta
                new_coordinates = []
                for k in range(self.dimension):
                    new_coordinates.append(np.mean([old_momenta_subject[c][k] for c in closest_points_list[i]]))
                new_momenta_subject[i] = np.asarray(new_coordinates)

            new_moments[ind] = new_momenta_subject
        
        return new_moments
    
    def compute_new_vector_field(self, old_control_points, new_control_points, old_momenta, new_kernel_width):
        """
        Update momenta values by resolving system of equations to preserve vector field
        Old moments of each new ctrl points = coefficients (convolution between new cp) x new moments
        """
        #print("old kernel", self.deformation_kernel_width)
        new_control_points = torch.tensor(new_control_points, dtype = torch.float) #21 x 3
        old_control_points = torch.tensor(old_control_points, dtype = torch.float) #8x3
        old_momenta = torch.tensor(old_momenta, dtype = torch.float) #11 x 8 x 3

        #compute old vector field
        old_vect_field = np.zeros((old_momenta.shape[0], len(new_control_points), old_momenta.shape[2]))
        for ind in range(old_momenta.shape[0]):
            vect_field = self.exponential.kernel.convolve(new_control_points, old_control_points, old_momenta[ind])
            old_vect_field[ind] = vect_field.cpu().numpy()

        #print("old_vect_field", old_vect_field.shape, old_vect_field[0])
        
        #compute coefficient (kernel convolution between new control points) (same for everyone)
        coef_new_vect_field = np.zeros((len(new_control_points), len(new_control_points)))
        for (i, point_i) in enumerate(new_control_points):
            for (j, point_j) in enumerate(new_control_points):
                square_distance = np.linalg.norm(point_i-point_j)**2
                coefficient = np.exp((-1/new_kernel_width[j]** 2) * square_distance) 
                coef_new_vect_field[i, j] = coefficient
        #print("coef_new_vect_field", coef_new_vect_field)

        #solve equations
        new_momenta = np.zeros((old_momenta.shape[0], len(new_control_points), old_momenta.shape[2])) 
        for ind in range(old_momenta.shape[0]):
            x = np.linalg.solve(coef_new_vect_field, old_vect_field[ind]) #AX = B solve(A, B) -> B old moments, A new moments B (len(new ctl points)) A 
            new_momenta[ind] = x
            #print("old_momenta", old_momenta[ind])
            #print("new_momenta", x)

        #print("new_momenta", new_momenta)
        
        return new_momenta


    def save_new_parameters(self, new_parameters, new_moments, new_control_points):
        """
            Save new control points and momenta in the model.
        """
        self.set_control_points(new_control_points)
        new_parameters['momenta'] = new_moments
        fixed_effects = {key: new_parameters[key] for key in self.get_fixed_effects().keys()}
        self.set_fixed_effects(fixed_effects)

        return new_parameters

    def compute_new_kernel_width(self, nb_points_origin = None):
        """
            Compute new kernels for each point. Width = distance to closest neighbor.
        """
        control_points = self.fixed_effects['control_points']
        new_kernel_width = np.full((control_points.shape[0], 1), 5)
        for (i, point) in enumerate(control_points):
            if nb_points_origin is not None and i < nb_points_origin:
                if isinstance(self.deformation_kernel_width, int):
                    new_kernel_width[i] = int(self.deformation_kernel_width)
                else:
                    new_kernel_width[i] = int(self.deformation_kernel_width[i])
            else:
                distance_to_pts = [(np.sqrt(np.sum((point-control_points[k])**2, axis=0))) \
                    for k in range(len(control_points)) if (k != i)]
                min_dist_to_another_point = [c for c in sorted(distance_to_pts) if c != 0][0]
                new_kernel_width[i] = max(int(min_dist_to_another_point), 1)
                
        
        print("new_kernel_width", new_kernel_width.shape)
        #print("new_kernel_width", new_kernel_width)
        return new_kernel_width

    def adapt_kernel(self, new_kernel_width):
        """
            Save new kernel width
        """
        self.deformation_kernel_width = new_kernel_width
        
        self.exponential = Exponential(
            dense_mode=self.dense_mode,
            kernel=kernel_factory.factory(self.deformation_kernel_type,
                                        gpu_mode=self.gpu_mode,
                                        kernel_width=new_kernel_width),
            shoot_kernel_type=self.exponential.shoot_kernel_type,
            number_of_time_points=self.exponential.number_of_time_points,
            use_rk2_for_shoot=self.exponential.use_rk2_for_shoot, 
            use_rk2_for_flow=self.exponential.use_rk2_for_flow)

    
    def naive_coarse_to_fine(self, new_parameters, current_iteration, output_dir, dataset):

        print("Naive coarse to fine")
        #get control points
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta = \
            self._fixed_effects_to_torch_tensors(False, device = device)
        control_points = control_points.cpu().numpy()
        
        new_spacing, new_control_points = self.add_points_regularly(current_iteration)
        if (self.dimension == 2 and new_spacing > 1) or (self.dimension == 3 and new_spacing > self.maximum_spacing):
            self.initial_cp_spacing = new_spacing
            print("new_spacing", new_spacing)
            print("old kernel width", self.deformation_kernel_width)
            print("new_control_points", len(new_control_points))
            
            #closest neighbours of the new points
            closest_points_list = self.closest_neighbors(new_control_points, control_points)

            #save new points
            self.save_points(current_iteration, new_control_points, dataset, output_dir)

            #update momenta
            new_moments = self.set_new_momenta(new_control_points, new_parameters["momenta"], closest_points_list)
            
            new_parameters = self.save_new_parameters(new_parameters, new_moments, new_control_points)
            print("after adding points", np.shape(self.fixed_effects['control_points']),
            np.shape(self.fixed_effects['momenta']))

            #adapt kernel
            new_kernel_width = 10
            #new_kernel_width = np.array([10] * len(new_control_points))
            #new_kernel_width = new_kernel_width.reshape((new_kernel_width.shape[0], 1))
            self.adapt_kernel(new_kernel_width = new_kernel_width)  

            #test for sum of kernels /!\
            """
            new_kernel_width = np.full((new_control_points.shape[0], 1), new_spacing * 1.3)
            new_kernel_width[0] = new_spacing * 2
            #new_kernel_width[0] = new_kernel_width[1]/2
            self.adapt_kernel(new_kernel_width = new_kernel_width)   
            print("new_kernel_width[0:8]", new_kernel_width[0:8])
            """     
                
        return new_parameters

    def naive_coarse_to_fine_v2(self, new_parameters, current_iteration, output_dir, dataset):
        """
        Naive coarse to fine.
        Add regular grid of points with reduced spacing - keep previous grid of points
        """
        print("Naive coarse to finev2")
        
        #get control points
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta = \
            self._fixed_effects_to_torch_tensors(False, device = device)
        control_points = control_points.cpu().numpy()
        old_spacing = self.initial_cp_spacing
        
        new_spacing, new_control_points = self.add_points_regularly(current_iteration)
        #new_spacing, new_control_points = self.add_points_slowly(current_iteration, len(control_points))
        if (self.dimension == 2 and new_spacing > 1) or (self.dimension == 3 and new_spacing > self.maximum_spacing and new_spacing < old_spacing):
            self.initial_cp_spacing = new_spacing
            print("new_spacing", new_spacing)
            print("new_control_points", len(new_control_points))
            
            #keep old points
            new_control_points = np.concatenate((control_points, new_control_points)) 

            #save new points
            self.save_points(current_iteration, new_control_points, dataset, output_dir)
            
            #update momenta (old momenta = same, new momenta = 0)
            new_moments = np.zeros((new_parameters["momenta"].shape[0], len(new_control_points), new_parameters["momenta"].shape[2]))

            for ind, old_momenta_subject in enumerate(new_parameters["momenta"]):
                new_moments[ind][:len(control_points)] = new_parameters["momenta"][ind]
            print("new_moments",new_moments.shape)
            
            new_parameters = self.save_new_parameters(new_parameters, new_moments, new_control_points)
            print("after adding points", np.shape(self.fixed_effects['control_points']))      

            #adapt kernel (sigma = old spacing, new spacing)
            new_kernel_width = np.full((new_control_points.shape[0], 1), 1)
            new_kernel_width[:len(control_points)] = old_spacing
            new_kernel_width[len(control_points):] = new_spacing
            self.adapt_kernel(new_kernel_width)

                
        return new_parameters

    
    def residus_moyens_voisins(self, coord, new_spacing, residuals_by_point, objet_intensities):
        if self.dimension == 2:
            limite_inf1 = max(coord[0]-int(new_spacing/2), 0)
            limite_sup1 = min(coord[0]+ int(new_spacing/2) + 1, objet_intensities.shape[0])
            limite_inf2 = max(coord[1]-int(new_spacing/2), 0)
            limite_sup2 = min(coord[1]+ int(new_spacing/2) + 1, objet_intensities.shape[1])
            zone = residuals_by_point[limite_inf1:limite_sup1, limite_inf2:limite_sup2]
        elif self.dimension == 3:
            limite_inf1 = max(coord[0]-int(new_spacing/2), 0)
            limite_sup1 = min(coord[0]+ int(new_spacing/2) + 1, objet_intensities.shape[0])
            limite_inf2 = max(coord[1]-int(new_spacing/2), 0)
            limite_sup2 = min(coord[1]+ int(new_spacing/2) + 1, objet_intensities.shape[1])
            limite_inf3 = max(coord[2]-int(new_spacing/2), 0)
            limite_sup3 = min(coord[2]+ int(new_spacing/2) + 1, objet_intensities.shape[2])
            zone = residuals_by_point[limite_inf1:limite_sup1, limite_inf2:limite_sup2, limite_inf3:limite_sup3]
        
        return np.mean(zone)

    """
    def residus_moyens_voisins_ponderes(self, coord, new_spacing, residuals_by_point, objet_intensities):
        
        liste_coord_voisins, listes_coord_dimensions = [], []
        sum_weighted_res = 0

        #liste des coordonnées des points voisins
        for k in range(self.dimension):
            listes_coord_dimensions.append([x for x in range(max(coord[k]-int(new_spacing), 0), 
            min(coord[k]+ int(new_spacing) + 1, objet_intensities.shape[k]))])
        for x in listes_coord_dimensions[0]:
            for y in listes_coord_dimensions[1]:
                if self.dimension == 2:
                    liste_coord_voisins.append([x, y])
                else:
                    for z in listes_coord_dimensions[2]:
                        liste_coord_voisins.append([x, y, z])
        
        #gaussian convolution of the point neighboring residuals
        sum_ponderations = 0
        for voisin in liste_coord_voisins:
            distance = np.linalg.norm(np.asarray(voisin)-np.asarray(coord))  
            residu = residuals_by_point[tuple(voisin)]
            exp = np.exp((-1) *distance**2 / (2*(new_spacing/2)**2))
            sum_ponderations += exp
            sum_weighted_res += exp * residu

        moyenne = sum_weighted_res/sum_ponderations
        #print("sum_weighted_res", sum_weighted_res,"moyenne", moyenne)
        
        return moyenne
        """

    def residus_moyens_voisins2(self, residuals_by_point, output_dir):
        """
        Gaussian filter of the residuals to account for neighborhood
        """
        dimensions = (residuals_by_point.shape)
        voxels = np.zeros((int(dimensions[0]), int(dimensions[1]), int(dimensions[2]), 3))
        residuals_by_point2 = np.zeros((int(dimensions[0]), int(dimensions[1]), int(dimensions[2]), 3))
        for i in range(int(dimensions[0])):
            for j in range(int(dimensions[1])):
                for k in range(int(dimensions[2])):
                    voxels[i, j, k] = [i, j, k]
                    residuals_by_point2[i, j, k, :] = residuals_by_point[i, j, k]
        voxels = torch.tensor(voxels, dtype = torch.float)
        residuals_by_point2 = torch.tensor(residuals_by_point2, dtype = torch.float)

        old_kernel_width = self.deformation_kernel_width
        self.adapt_kernel(new_kernel_width = 1.5)
        #/!\ convolution prévue pour des vecteurs de dimension 3 et non pas 1 -> redimensionner
        convole_res = self.exponential.kernel.convolve(voxels.contiguous().view(-1, 3), voxels.contiguous().view(-1, 3), #voxels x 3
                                                    residuals_by_point2.contiguous().view(-1, 3)) #voxels x 3
        convole_res = convole_res.contiguous().view(residuals_by_point.shape[0], residuals_by_point.shape[1], residuals_by_point.shape[2], 3).cpu().numpy()
        convole_res = convole_res[:, :, :, 0]
        #print("convole_res", convole_res.shape, convole_res[50, 30])
        #print("residuals_by_point", residuals_by_point[50, 30])
        self.adapt_kernel(new_kernel_width = old_kernel_width)

        #save residus
        dico = {}
        dico['image_intensities'] = torch.tensor(convole_res, dtype = torch.float)
        self.template.write(output_dir, ["Heat_map_test.nii"], {key: value.data.cpu().numpy() for key, value in dico.items()})

        return convole_res


    def add_former_points(self, last_control_points, new_parameters, current_iteration):
        """Function that initializes CTF with the control points used in the former CTF
        """
        if current_iteration == 1 and last_control_points is not None:
            print("Initialize CTF with former control points")
            print(len(last_control_points))
            #get current control points
            device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
            template_data, template_points, control_points, momenta = \
                self._fixed_effects_to_torch_tensors(False, device = device)
            control_points = control_points.cpu().numpy()

            #closest neighbours of the new points
            closest_points_list = self.closest_neighbors(last_control_points, control_points)

            #upate momenta
            new_moments = self.set_new_momenta(last_control_points, new_parameters["momenta"], closest_points_list)
            
            #replace current control points by last control points
            new_parameters = self.save_new_parameters(new_parameters, new_moments, last_control_points)
            print("after adding points", np.shape(self.fixed_effects['control_points']))      

            #adapt kernel only for new points
            new_kernel_width= self.compute_new_kernel_width()
            self.adapt_kernel(new_kernel_width)
        
        return new_parameters

    def coarse_to_fine_v2(self, residuals_by_point, new_parameters, current_iteration, output_dir, dataset):
        """Coarse to fine
        Add regular grid of points. Keep only points on high residuals
        """
        print("Coarse to fine v2")
        #get control points
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta = \
            self._fixed_effects_to_torch_tensors(False, device = device)
        control_points = control_points.cpu().numpy()

        #number_of_pixels = self.number_of_pixels(dataset)
        nb_points_origin = len(control_points)
        objet_intensities = dataset.deformable_objects[0][0].get_data()["image_intensities"]
        
        #new_spacing, new_control_points = self.add_points_linearly(control_points, max_spacing, taux = 0.3)
        #new_spacing, new_control_points = self.add_points_only_once(current_iteration)
        #new_spacing, new_control_points = self.add_points_regularly(current_iteration)
        new_spacing, new_control_points = self.add_points_slowly(current_iteration, nb_points_origin)

        if new_spacing > self.maximum_spacing: #limite le nombre total de points
            
            print("current_iteration", current_iteration)
            print("ancien nb de points", np.shape(control_points)[0])
            print("new_spacing", new_spacing)

            self.initial_cp_spacing = new_spacing #besoin que pour initialiser modèle...
            #keep only points where the residuals are high
            
            percentile = np.percentile(residuals_by_point.flatten(), 80) #99.5)   
            
            #select points to keep
            for (i,point) in enumerate(new_control_points):
                coord = tuple([int(point[k]) for k in range(self.dimension)])
                residus_voisins_moy = residuals_by_point[coord]

                if residus_voisins_moy >= percentile: 
                    if point.tolist() not in control_points.tolist():
                        control_points = np.vstack([control_points, point])
            print("out of", len(new_control_points), "points", "we keep", len(control_points[nb_points_origin:]), "points")
            
            #save new points
            self.save_points(current_iteration, control_points, dataset, output_dir)

            #closest neighbours of the new points
            closest_points_list = self.closest_neighbors(control_points, control_points[:nb_points_origin])

            #update momenta
            new_moments = self.set_new_momenta(control_points, new_parameters["momenta"], closest_points_list)
            print("new_moments", new_moments)
            new_parameters = self.save_new_parameters(new_parameters, new_moments, control_points)
            print("after adding points", np.shape(self.fixed_effects['control_points']))      

            #adapt kernel only for new points
            new_kernel_width = self.compute_new_kernel_width()
            
            #self.adapt_kernel(new_kernel_width = new_spacing)
            self.adapt_kernel(new_kernel_width)
            print("new_kernel_width", new_kernel_width)

        
        return new_parameters

    def coarse_to_fine_v3(self, residuals_by_point, new_parameters, current_iteration, output_dir, dataset):
        """Coarse to fine
        Add regular grid of points. Keep only points on high residuals
        Keep old momenta for old points. New points have null momenta.
        """
        print("Coarse to fine v3")
        #get control points
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta = \
            self._fixed_effects_to_torch_tensors(False, device = device)
        control_points = control_points.cpu().numpy()

        nb_points_origin = len(control_points)
        objet_intensities = dataset.deformable_objects[0][0].get_data()["image_intensities"]
        
        #initialize new control points

        #new_spacing, new_control_points = self.add_points_linearly(control_points, max_spacing, taux = 0.3)
        #new_spacing, new_control_points = self.add_points_only_once(current_iteration)
        #new_spacing, new_control_points = self.add_points_regularly(current_iteration)
        #new_spacing, new_control_points = self.add_points_slowly(current_iteration, nb_points_origin)   
        
        new_spacing = self.max_spacing
        new_control_points = initialize_control_points(None, self.template, self.max_spacing, self.deformation_kernel_width,
                                                    self.dimension, self.dense_mode)
        if nb_points_origin < 0.2 * len(new_control_points):
            if new_spacing > self.maximum_spacing: #limite le nombre total de points                
                print("current_iteration", current_iteration)
                #print("new_spacing", new_spacing)

                self.initial_cp_spacing = new_spacing #besoin que pour initialiser modèle...

                #avoid multiplying the current nb of points by 2 or more (Gradient descent issue)                
                seuil = 80
                while (1 - seuil*0.01) * len(new_control_points) > nb_points_origin and seuil < 99.4:
                    seuil += 0.1
                print('seuil', seuil)
                percentile = np.percentile(residuals_by_point.flatten(), seuil) 
                
                for (i,point) in enumerate(new_control_points):
                    coord = tuple([int(point[k]) for k in range(self.dimension)])
                    residus_voisins_moy = residuals_by_point[coord]

                    #Keep only voxels > 90th percentile
                    if residus_voisins_moy >= percentile: 
                        if point.tolist() not in control_points.tolist():
                            control_points = np.vstack([control_points, point])
                print("out of", len(new_control_points), "points", "we keep", len(control_points[nb_points_origin:]), "points")
                
                #save new points
                self.save_points(current_iteration, control_points, dataset, output_dir)

                #update momenta (old momenta = same, new momenta = 0)
                new_moments = np.zeros((new_parameters["momenta"].shape[0], len(control_points),
                                    new_parameters["momenta"].shape[2]))

                for ind, old_momenta_subject in enumerate(new_parameters["momenta"]):
                    add_moments_sub = np.zeros((len(control_points[nb_points_origin:]), control_points.shape[1]))
                    new_momenta_subject = np.concatenate((old_momenta_subject, add_moments_sub)) #old points x 3, new points x 3
                    new_moments[ind] = new_momenta_subject

                new_parameters = self.save_new_parameters(new_parameters, new_moments, control_points)
                print("after adding points", np.shape(self.fixed_effects['control_points']))      

                #adapt kernel (sigma = distance to closest neighbor)
                new_kernel_width = self.compute_new_kernel_width()
                
                self.adapt_kernel(new_kernel_width)
                #print("new_kernel_width", new_kernel_width)

        
        return new_parameters

    def coarse_to_fine_v4(self, residuals_by_point, new_parameters, current_iteration, output_dir, dataset):
        """Coarse to fine
        Add regular grid of points. Keep only points on high residuals (gaussien filtered)
        Adapt vector field to avoid mistakes.
        """
        print("Coarse to fine v4")

        #get control points
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta = \
            self._fixed_effects_to_torch_tensors(False, device = device)
        control_points = control_points.cpu().numpy()

        nb_points_origin = len(control_points)
        objet_intensities = dataset.deformable_objects[0][0].get_data()["image_intensities"]
        
        #new_spacing, new_control_points = self.add_points_linearly(control_points, max_spacing, taux = 0.3)
        #new_spacing, new_control_points = self.add_points_only_once(current_iteration)
        #new_spacing, new_control_points = self.add_points_regularly(current_iteration)
        #new_spacing, new_control_points = self.add_points_slowly(current_iteration, nb_points_origin)
        new_spacing, new_control_points = self.add_points_same_spacing(current_iteration)
        #seuil = 90
        seuil = 99 - current_iteration
        
        if new_spacing > self.maximum_spacing: #limite le nombre total de points            
            print("current_iteration", current_iteration)
            print("new_spacing", new_spacing)

            self.initial_cp_spacing = new_spacing #besoin que pour initialiser modèle...
            """
            percentile = np.percentile(residuals_by_point.flatten(), seuil) #99.5)   
            
            for (i,point) in enumerate(new_control_points):
                coord = tuple([int(point[k]) for k in range(self.dimension)])
                residus_voisins_moy = residuals_by_point[coord]
                
                if residus_voisins_moy >= percentile: 
                    if point.tolist() not in control_points.tolist():
                        control_points = np.vstack([control_points, point])
            print("out of", len(new_control_points), "points", "we keep", len(control_points[nb_points_origin:]), "points")
            
            #save new control points
            self.save_points(current_iteration, control_points, dataset, output_dir)"""

            #####version 2 (filtre gaussien des résidus pour prendre en compte voisins)
            #points_v1 = control_points[nb_points_origin:]
            #control_points = control_points[:nb_points_origin]
            convole_residus = self.residus_moyens_voisins2(residuals_by_point, output_dir)
            percentile = np.percentile(convole_residus.flatten(), seuil)
            for (i,point) in enumerate(new_control_points):
                coord = tuple([int(point[k]) for k in range(self.dimension)])
                residus_voisins_moy = convole_residus[coord]
                
                if residus_voisins_moy >= percentile: 
                    if point.tolist() not in control_points.tolist():
                        control_points = np.vstack([control_points, point])
            print("out of", len(new_control_points), "points", "we keep", len(control_points[nb_points_origin:]), "points")
            
            #sum = len([point for point in control_points[nb_points_origin:].tolist() if point in points_v1.tolist()])
            #print("points communs", sum)
            
            #save new control points
            self.save_points(current_iteration, control_points, dataset, output_dir)

            #save momenta (update needed to adapt kernel)
            old_momenta = new_parameters["momenta"]
            #print("old_momenta", old_momenta[0])
            new_moments = np.zeros((new_parameters["momenta"].shape[0], len(control_points), new_parameters["momenta"].shape[2]))

            new_parameters = self.save_new_parameters(new_parameters, new_moments, control_points)

            #adapt kernel only for new points
            new_kernel_width = self.compute_new_kernel_width()

            #adapt vector fields / momenta
            new_moments = self.compute_new_vector_field(control_points[:nb_points_origin], control_points, old_momenta, 
                                                        new_kernel_width)
            new_parameters = self.save_new_parameters(new_parameters, new_moments, control_points)
            
            print("after adding points", np.shape(self.fixed_effects['control_points'])) 
            #print("new_moments", new_moments[0])
            
            self.adapt_kernel(new_kernel_width)
            
        return new_parameters

    def coarse_to_fine_v5(self, residuals_by_point, new_parameters, current_iteration, output_dir, dataset):
        """Coarse to fine
        Add regular grid of points. Keep only points on high residuals (gaussien filtered)
        Adapt vector field to avoid mistakes.
        Kernel width adapted only for new points.
        """
        print("Coarse to fine v5")

        #get control points
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta = \
            self._fixed_effects_to_torch_tensors(False, device = device)
        control_points = control_points.cpu().numpy()

        nb_points_origin = len(control_points)
        
        #new_spacing, new_control_points = self.add_points_linearly(control_points, max_spacing, taux = 0.3)
        #new_spacing, new_control_points = self.add_points_only_once(current_iteration)
        #new_spacing, new_control_points = self.add_points_regularly(current_iteration)
        #new_spacing, new_control_points = self.add_points_slowly(current_iteration, nb_points_origin)
        seuil = 90
        new_spacing, new_control_points = self.add_points_same_spacing(current_iteration)
        seuil = 99 - current_iteration
        
        if new_spacing > self.maximum_spacing: #limite le nombre total de points            
            print("current_iteration", current_iteration)
            print("new_spacing", new_spacing)

            self.initial_cp_spacing = new_spacing #besoin que pour initialiser modèle...

            convole_residus = self.residus_moyens_voisins2(residuals_by_point, output_dir)
            percentile = np.percentile(convole_residus.flatten(), seuil)
            for (i,point) in enumerate(new_control_points):
                coord = tuple([int(point[k]) for k in range(self.dimension)])
                residus_voisins_moy = convole_residus[coord]
                
                if residus_voisins_moy >= percentile: 
                    if point.tolist() not in control_points.tolist():
                        control_points = np.vstack([control_points, point])
            print("out of", len(new_control_points), "points", "we keep", len(control_points[nb_points_origin:]), "points")

            #save momenta (update needed to adapt kernel)
            old_momenta = new_parameters["momenta"]
            
            new_moments = np.zeros((new_parameters["momenta"].shape[0], len(control_points), new_parameters["momenta"].shape[2]))
            new_parameters = self.save_new_parameters(new_parameters, new_moments, control_points)

            #adapt kernel only for new points
            #new_kernel_width = self.compute_new_kernel_width()
            #new_kernel_width = np.array([5] * len(control_points))

            #adapt vector fields / momenta
            new_moments = self.compute_new_vector_field(control_points[:nb_points_origin], control_points, old_momenta, 
                                                        new_kernel_width)
            new_parameters = self.save_new_parameters(new_parameters, new_moments, control_points)
            
            print("after adding points", np.shape(self.fixed_effects['control_points'])) 
            
            #print("old_momenta", old_momenta[0])
            #print("new_moments", new_moments[0])
            
            self.adapt_kernel(new_kernel_width = 10)

            #save new control points
            self.save_points(current_iteration, control_points, dataset, output_dir)
            
        return new_parameters

    def coarse_to_fine_v6(self, residuals_by_point, new_parameters, current_iteration, output_dir, dataset):
        """Coarse to fine
        Add regular grid of points. Keep only points on high residuals (gaussien filtered)
        Adapt vector field to avoid mistakes.
        Same at CTF v5 but Allows several points at same position. Kernel width adapted only for new points.
        """
        print("Coarse to fine v6")

        #get control points
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta = \
            self._fixed_effects_to_torch_tensors(False, device = device)
        control_points = control_points.cpu().numpy()

        nb_points_origin = len(control_points)
        
        #new_spacing, new_control_points = self.add_points_linearly(control_points, max_spacing, taux = 0.3)
        #new_spacing, new_control_points = self.add_points_only_once(current_iteration)
        #new_spacing, new_control_points = self.add_points_regularly(current_iteration)
        #new_spacing, new_control_points = self.add_points_slowly(current_iteration, nb_points_origin)
        new_spacing, new_control_points = self.add_points_same_spacing(current_iteration)
        seuil = 99.5 - current_iteration
        
        if new_spacing > self.maximum_spacing: #limite le nombre total de points            
            print("current_iteration", current_iteration)
            print("new_spacing", new_spacing)

            self.initial_cp_spacing = new_spacing #besoin que pour initialiser modèle...

            convole_residus = self.residus_moyens_voisins2(residuals_by_point, output_dir)
            percentile = np.percentile(convole_residus.flatten(), seuil)
            for (i,point) in enumerate(new_control_points):
                coord = tuple([int(point[k]) for k in range(self.dimension)])
                residus_voisins_moy = convole_residus[coord]
                
                if residus_voisins_moy >= percentile: 
                    if point.tolist() in control_points.tolist():
                        point = np.array([point[k]+0.01 for k in range(self.dimension)])
                        print("point", point)
                    control_points = np.vstack([control_points, point])
            print("out of", len(new_control_points), "points", "we keep", len(control_points[nb_points_origin:]), "points")
                        
            #save new control points
            self.save_points(current_iteration, control_points, dataset, output_dir)

            #save momenta (update needed to adapt kernel)
            old_momenta = new_parameters["momenta"]
            
            new_moments = np.zeros((new_parameters["momenta"].shape[0], len(control_points), new_parameters["momenta"].shape[2]))
            new_parameters = self.save_new_parameters(new_parameters, new_moments, control_points)

            #adapt kernel only for new points
            new_kernel_width = self.compute_new_kernel_width(nb_points_origin)

            #adapt vector fields / momenta
            new_moments = self.compute_new_vector_field(control_points[:nb_points_origin], control_points, old_momenta, 
                                                        new_kernel_width)
            new_parameters = self.save_new_parameters(new_parameters, new_moments, control_points)
            
            print("after adding points", np.shape(self.fixed_effects['control_points'])) 
            
            #print("old_momenta", old_momenta[0])
            #print("new_moments", new_moments[0])
            
            self.adapt_kernel(new_kernel_width)
            
        return new_parameters

    def coarse_to_fine_v7(self, residuals_by_point, new_parameters, current_iteration, output_dir, dataset):
        """Coarse to fine
        Add regular grid of points. Keep only points on high residuals (gaussien filtered)
        New moments = 0.
        Kernel width adapted only for new points.
        """
        print("Coarse to fine v7")

        #get control points
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta = \
            self._fixed_effects_to_torch_tensors(False, device = device)
        control_points = control_points.cpu().numpy()

        nb_points_origin = len(control_points)
        
        #new_spacing, new_control_points = self.add_points_linearly(control_points, max_spacing, taux = 0.3)
        #new_spacing, new_control_points = self.add_points_only_once(current_iteration)
        #new_spacing, new_control_points = self.add_points_regularly(current_iteration)
        #new_spacing, new_control_points = self.add_points_slowly(current_iteration, nb_points_origin)
        new_spacing, new_control_points = self.add_points_same_spacing(current_iteration)
        seuil = 99 - current_iteration
        
        if new_spacing > self.maximum_spacing: #limite le nombre total de points            
            print("current_iteration", current_iteration)
            print("new_spacing", new_spacing)

            self.initial_cp_spacing = new_spacing #besoin que pour initialiser modèle...

            convole_residus = self.residus_moyens_voisins2(residuals_by_point, output_dir)
            percentile = np.percentile(convole_residus.flatten(), seuil)
            for (i,point) in enumerate(new_control_points):
                coord = tuple([int(point[k]) for k in range(self.dimension)])
                residus_voisins_moy = convole_residus[coord]
                
                if residus_voisins_moy >= percentile: 
                    if point.tolist() not in control_points.tolist():
                        control_points = np.vstack([control_points, point])
            print("out of", len(new_control_points), "points", "we keep", len(control_points[nb_points_origin:]), "points")
                        
            #save new control points
            self.save_points(current_iteration, control_points, dataset, output_dir)

            #update momenta (old momenta = same, new momenta = 0)
            new_moments = np.zeros((new_parameters["momenta"].shape[0], len(control_points),
                                new_parameters["momenta"].shape[2]))

            for ind, old_momenta_subject in enumerate(new_parameters["momenta"]):
                add_moments_sub = np.zeros((len(control_points[nb_points_origin:]), control_points.shape[1]))
                new_momenta_subject = np.concatenate((old_momenta_subject, add_moments_sub)) #old points x 3, new points x 3
                new_moments[ind] = new_momenta_subject

            new_parameters = self.save_new_parameters(new_parameters, new_moments, control_points)            
            print("after adding points", np.shape(self.fixed_effects['control_points'])) 
            
            #print("old_momenta", old_momenta[0])
            #print("new_moments", new_moments[0])

            #adapt kernel only for new points
            new_kernel_width = self.compute_new_kernel_width(nb_points_origin)
            
            self.adapt_kernel(new_kernel_width)
            
        return new_parameters

    def coarse_to_fine_v11(self, residuals_by_point, new_parameters, current_iteration, output_dir, dataset):
        
        if current_iteration == 2:
            #get control points
            device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
            template_data, template_points, control_points, momenta = \
                self._fixed_effects_to_torch_tensors(False, device = device)
            control_points = control_points.cpu().numpy()

            print("nb de points", np.shape(control_points)[0])
            nb_points_origin = control_points.shape[0]
            spacing = self.initial_cp_spacing
            
            #by computing residuals and adding a point in the corresponding sub element
            percentile = np.percentile(residuals_by_point.flatten(), 99.5)   

            #select the highest residuals
                
            indices = np.where(residuals_by_point > percentile)
            min_spacing = 1
            for p in range(indices[0].shape[0]): #nb de points à ajouter
                point = np.array([indices[d][p] for d in range(self.dimension)])

                distance_to_pts = sorted([(np.sqrt(np.sum((point-control_points[k])**2, axis=0)), k) \
                    for k in range(len(control_points))])
                #check that there are no points too close
                if distance_to_pts[0][0] > min_spacing:
                    control_points = np.vstack([control_points, point])
            print("out of", indices[0].shape[0], "we keep", len(control_points[nb_points_origin:]))            

            #save new points
            self.save_points(current_iteration, control_points, dataset, output_dir)

            #closest neighbours of the new points
            closest_points_list = self.closest_neighbors(control_points, control_points[:nb_points_origin])

            #update momenta
            new_moments = self.set_new_momenta(control_points, new_parameters["momenta"], closest_points_list)

            new_parameters = self.save_new_parameters(new_parameters, new_moments, control_points)
            print("after adding points", np.shape(self.fixed_effects['control_points']))      

            #adapt kernel only for new points
            new_kernel_width = self.compute_new_kernel_width()
            self.adapt_kernel(new_kernel_width)
            print("new_kernel_width", new_kernel_width)

        return new_parameters
    
    def coarse_to_fine_v1(self, residuals_by_point, new_parameters, current_iteration, output_dir, dataset):
        
        #get control points
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta = \
            self._fixed_effects_to_torch_tensors(False, device = device)
        control_points = control_points.cpu().numpy()
        print("nb de points", np.shape(control_points)[0])
        nb_points_origin = control_points.shape[0]
        spacing = self.initial_cp_spacing
        
        
        #####
        #by computing avg residuals inside sub elements
        """
        length = self.template.bounding_box[dimension, 1] \
            - self.template.bounding_box[dimension, 0] 
        print("self.template.bounding_box", self.template.bounding_box)
        print("length", length)
        #we enumerate all the sub elements (cubes or 2D) inside the point grid 
        dico_cubes = dict()
        for (i, coord) in enumerate(control_points):
            #ne marche que pour une grille régulière !!
            cube = [coord]
            cube += [coord[0] + spacing, coord[1], coord[2]]
            cube += [coord[0], coord[1] + spacing, coord[2]]
            cube += [coord[0], coord[1], coord[2] + spacing]
            cube += [coord[0] + spacing, coord[1] + spacing, coord[2]]
            cube += [coord[0], coord[1] + spacing, coord[2] + spacing]
            cube += [coord[0] + spacing, coord[1], coord[2] + spacing]
            cube += [coord[0] + spacing, coord[1]  + spacing, coord[2] + spacing]
            valid_cube = True
            for point in cube:
                if point[0] > length or point[1] > length or point[2] > length:
                    valid_cube = False
            if valid_cube:
                dico_cubes[coord] = cube
        #for each cube, compute mean residual values inside
        print("number of cubes", len(dico_cubes.keys()))
        for cube in dico_cubes.keys():
            coord = cube[0]
            residuals_inside_x = np.arange(coord[0],coord+spacing)
            residuals_inside_y = np.arange(coord[1],coord+spacing)
            residuals_inside_z = np.arange(coord[2],coord+spacing)
            residuals_inside = residuals_by_point[residuals_inside_x, residuals_inside_y, residuals_inside_z]
            average = np.mean(residuals_inside)"""

        #####
        #by computing residuals and adding a point in the corresponding sub element

        print()
        nb_of_pixels = self.number_of_pixels(dataset)
        max_nb_of_points_allowed = min(nb_of_pixels, 500)

        longueur = template_data['image_intensities'].shape[0]
        min_spacing = max(min(spacing/1.5, longueur/5), 0.5)
        min_spacing = 1
        if self.dimension == 2:
            min_spacing = 0.9
        print("old spacing", spacing, "new spacing", min_spacing)
        self.initial_cp_spacing = min_spacing

        #ajout de points si < nb max de points et 
        if control_points.shape[0] < max_nb_of_points_allowed and np.amax(residuals_by_point) > 0:

            #select the highest residuals
            closest_points_list = [] #points already present to keep same momenta
                                
            print("augmentation du nb de points", (len(control_points) + min(nb_points_origin, 30))/len(control_points))
            
            while len(control_points[nb_points_origin:]) < min(nb_points_origin, 50): #double nb de points
                
                #coordinates of max value
                maximum = np.amax(residuals_by_point)
                indices = np.where(residuals_by_point == np.amax(residuals_by_point))
                
                point = np.array([indices[d][0] for d in range(self.dimension)])
                residuals_by_point[tuple(point)] = 0
                #print("highest residual location", point)
                distance_to_pts = sorted([(np.sqrt(np.sum((point-control_points[k])**2, axis=0)), k) \
                    for k in range(len(control_points))])
                #check that there are no points too close
                if distance_to_pts[0][0] > min_spacing:
                    closest_points_indices = [c[1] for c in distance_to_pts[:4]]
                    closest_points_list.append(closest_points_indices) #save closest points for new momenta
                    control_points = np.vstack([control_points, point])
                
                if control_points.shape[0] == max_nb_of_points_allowed:
                    break
            
            print("new points", np.shape(control_points[nb_points_origin:]))
            #save new points
            self.save_points(current_iteration, control_points, dataset, output_dir)
            
            #closest neighbours of the new points
            closest_points_list = self.closest_neighbors(control_points, control_points[:nb_points_origin])

            #update momenta
            new_moments = self.set_new_momenta(control_points, new_parameters["momenta"], closest_points_list)
            #print("new_moments", new_moments)
            
            self.set_control_points(control_points)
            new_parameters['momenta'] = new_moments
            fixed_effects = {key: new_parameters[key] for key in self.get_fixed_effects().keys()}
            self.set_fixed_effects(fixed_effects)
            
            print("after adding points", np.shape(self.fixed_effects['control_points']),
            np.shape(self.fixed_effects['momenta']))      

            #adapt kernel only for new points
            
            control_points = self.fixed_effects['control_points']
            new_kernel_width = np.full((control_points.shape[0], 1), 1)
            for (i, point) in enumerate(control_points):
                distance_to_pts = [(np.sqrt(np.sum((point-control_points[k])**2, axis=0))) \
                    for k in range(len(control_points)) if (k != i)]
                min_dist_to_another_point = [c for c in sorted(distance_to_pts) if c != 0][0]
                new_kernel_width[i] = max(int(min_dist_to_another_point*2), 1)
            print("min_dist_to_another_point", min_dist_to_another_point)
            #other try
            #new_kernel_width = 1 #ne règle pas pb non cvg
            #new_kernel_width = np.mean(new_kernel_width)*2
            print("new_kernel_width", new_kernel_width)

            #then save new kernel width in stat model and new exponential
            self.adapt_kernel(new_kernel_width)
            print("self.kernel.gamma", self.exponential.kernel.gamma)

        return new_parameters                       

#############################################################################################
def haarMatrix(n, normalized=False):
    # Allow only size n of power 2
    n = 2**np.ceil(np.log2(n))
    if n > 2:
        h = haarMatrix(n / 2)
    else:
        return np.array([[1, 1], [1, -1]])

    # calculate upper haar part
    h_n = np.kron(h, [1, 1])
    # calculate lower haar part 
    if normalized:
        h_i = np.sqrt(n/2)*np.kron(np.eye(len(h)), [1, -1])
    else:
        h_i = np.kron(np.eye(len(h)), [1, -1])
    # combine parts
    h = np.vstack((h_n, h_i))
    
    print("haar matrix", h)

    return h

def initialize_coarse_to_fine(self, new_parameters, current_iteration, output_dir, dataset):
    """

    """
    #get control points
    device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
    template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(False, device = device)
    
    #momenta = 0 at iteration 0
    control_points = control_points.cpu().numpy()
    momenta = new_parameters["momenta"]

    haar_matrix = haarMatrix(len(control_points))
    coarse_momenta = momenta 

    return 


def new_coarse_to_fine(self, new_parameters, current_iteration, output_dir, dataset):
        
        print("Naive coarse to fine")
        #get control points
        device, device_id = utilities.get_best_device(gpu_mode=self.gpu_mode)
        template_data, template_points, control_points, momenta = self._fixed_effects_to_torch_tensors(False, device = device)
        
        old_control_points = control_points.cpu().numpy()
        old_momenta = new_parameters["momenta"]
        #old_coarse_momenta = ####!!
        #add new control points
        new_spacing, new_control_points = self.add_points_regularly(current_iteration)
        new_haar_matrix = haarMatrix(len(new_control_points))

        #update coarse momenta

        #update momenta

        #update kernel

        if (self.dimension == 2 and new_spacing > 1) or (self.dimension == 3 and new_spacing > self.maximum_spacing):
            
            #save new points
            self.save_points(current_iteration, new_control_points, dataset, output_dir)

            #update COARSE momenta by preserving vector field
            #print("old kernel", self.deformation_kernel_width)
            new_control_points = torch.tensor(new_control_points, dtype = torch.float) #21 x 3
            old_control_points = torch.tensor(old_control_points, dtype = torch.float) #8x3
            old_momenta = torch.tensor(old_momenta, dtype = torch.float) #11 x 8 x 3

            #compute old vector field values at the new control points (using old kernel)
            old_vect_field = np.zeros((old_momenta.shape[0], len(new_control_points), old_momenta.shape[2]))
            for sujet in range(old_momenta.shape[0]):
                vect_field = self.exponential.kernel.convolve(new_control_points, old_control_points, old_momenta[sujet])
                old_vect_field[sujet] = vect_field.cpu().numpy()

            print("old_vect_field", old_vect_field.shape, old_vect_field[0])
            
            #compute coefficient (kernel convolution between new control points using new kernel)
            new_kernel_width = new_spacing
            coef_new_vect_field = np.zeros((len(new_control_points), len(new_control_points)))
            for (i, point_i) in enumerate(new_control_points):
                for (j, point_j) in enumerate(new_control_points):
                    square_distance = np.linalg.norm(point_i-point_j)**2
                    coefficient = np.exp((-1/new_kernel_width[j]** 2) * square_distance) 
                    coef_new_vect_field[i, j] = coefficient
            print("coef_new_vect_field", coef_new_vect_field.shape, coef_new_vect_field)

            #multiply these coefficients by the new haar matrix
            coef_new_vect_field = np.matmul(coef_new_vect_field, new_haar_matrix)

            #solve equations
            new_coarse_momenta = np.zeros((old_momenta.shape[0], len(new_control_points), old_momenta.shape[2])) 
            for sujet in range(old_momenta.shape[0]):
                x = np.linalg.solve(coef_new_vect_field, old_vect_field[sujet]) #AX = B solve(A, B) -> B old moments, A new moments B (len(new ctl points)) A 
                new_coarse_momenta[sujet] = x
            print("new_coarse_momenta", new_coarse_momenta)

            #update momenta
            new_momenta = np.matmul(new_haar_matrix, new_coarse_momenta)
            print("new_momenta", new_momenta)


            #save new parameters            
            new_parameters = self.save_new_parameters(new_parameters, new_moments, new_control_points)
            print("after adding points", np.shape(self.fixed_effects['control_points']),
            np.shape(self.fixed_effects['momenta']))

            #save new kernel
            self.adapt_kernel(new_kernel_width = new_kernel_width)  

                
        return new_parameters