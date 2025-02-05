import logging
import os
import warnings
import xml.etree.ElementTree as et

from ..core import default, GpuMode
from ..support import utilities

logger = logging.getLogger(__name__)

def get_dataset_specifications(xml_parameters):
    specifications = {}
    specifications['visit_ages'] = xml_parameters.visit_ages
    specifications['dataset_filenames'] = xml_parameters.dataset_filenames
    specifications['subject_ids'] = xml_parameters.subject_ids

    return specifications

def get_estimator_options(xml_parameters):
    options = {}

    print("!!!", xml_parameters.optimization_method, "\n")

    if xml_parameters.optimization_method.lower() in ['GradientAscent'.lower(), 'StochasticGradientAscent'.lower()]:
        options['initial_step_size'] = xml_parameters.initial_step_size
        options['scale_initial_step_size'] = xml_parameters.scale_initial_step_size
        options['line_search_shrink'] = xml_parameters.line_search_shrink
        options['line_search_expand'] = xml_parameters.line_search_expand
        options['max_line_search_iterations'] = xml_parameters.max_line_search_iterations
        options['optimized_log_likelihood'] = xml_parameters.optimized_log_likelihood

    elif xml_parameters.optimization_method.lower() == 'ScipyLBFGS'.lower():
        options['freeze_template'] = xml_parameters.freeze_template
        options['max_line_search_iterations'] = xml_parameters.max_line_search_iterations
        options['optimized_log_likelihood'] = xml_parameters.optimized_log_likelihood

    # common options
    options['optimization_method'] = xml_parameters.optimization_method.lower()
    options['max_iterations'] = xml_parameters.max_iterations
    options['convergence_tolerance'] = xml_parameters.convergence_tolerance
    options['print_every_n_iters'] = xml_parameters.print_every_n_iters
    options['save_every_n_iters'] = xml_parameters.save_every_n_iters
    options['gpu_mode'] = xml_parameters.gpu_mode
    options['state_file'] = xml_parameters.state_file
    options['load_state_file'] = xml_parameters.load_state_file

    options['multiscale_momenta'] = xml_parameters.multiscale_momenta #ajout fg
    options['multiscale_images'] = xml_parameters.multiscale_images
    options['multiscale_meshes'] = xml_parameters.multiscale_meshes
    options['naive'] = xml_parameters.naive #ajout fg
    options["start_scale"] = xml_parameters.start_scale
    options['multiscale_strategy'] =  xml_parameters.multiscale_strategy #ajout fg
    options['gamma'] = xml_parameters.gamma #ajout fg

    return options

def get_model_options(xml_parameters):
    options = {
        'deformation_kernel_width': xml_parameters.deformation_kernel_width,
        'deformation_kernel_device': xml_parameters.deformation_kernel_device,
        'number_of_time_points': xml_parameters.number_of_time_points,
        'concentration_of_time_points': xml_parameters.concentration_of_time_points,
        'freeze_template': xml_parameters.freeze_template,
        'freeze_momenta': xml_parameters.freeze_momenta,
        'freeze_noise_variance': xml_parameters.freeze_noise_variance,
        'initial_control_points': xml_parameters.initial_control_points,
        'initial_momenta': xml_parameters.initial_momenta,
        'downsampling_factor': xml_parameters.downsampling_factor,
        'dimension': xml_parameters.dimension,
        'gpu_mode': xml_parameters.gpu_mode,
        'perform_shooting':xml_parameters.perform_shooting, #ajout fg
        'interpolation':xml_parameters.interpolation #ajout fg
    }

    if xml_parameters.model_type.lower() in ['BayesianGeodesicRegression'.lower()]:
        options['t0'] = xml_parameters.t0
        options['tR'] = xml_parameters.tR
        options['tmin'] = xml_parameters.tmin
        options['tmax'] = xml_parameters.tmax
        options['number_of_sources'] = xml_parameters.number_of_sources
        options['initial_modulation_matrix'] = xml_parameters.initial_modulation_matrix
        options['initial_sources'] = xml_parameters.initial_sources
        options['freeze_modulation_matrix'] = xml_parameters.freeze_modulation_matrix
        options['freeze_reference_time'] = xml_parameters.freeze_reference_time
        options['freeze_rupture_time'] = xml_parameters.freeze_rupture_time

    elif xml_parameters.model_type.lower() in ['Regression'.lower(), "KernelRegression".lower()]:
        options['t0'] = xml_parameters.t0
        options['tmin'] = xml_parameters.tmin
        options['tmax'] = xml_parameters.tmax

    elif xml_parameters.model_type.lower() in ["PiecewiseRegression".lower()]:#ajout fg
        options['num_component'] = xml_parameters.num_component
        options['t0'] = xml_parameters.t0
        options['tR'] = xml_parameters.tR
        options['t1'] = xml_parameters.t1
        options['freeze_reference_time'] = xml_parameters.freeze_reference_time
        options['freeze_rupture_time'] = xml_parameters.freeze_rupture_time

    elif xml_parameters.model_type.lower() == 'ParallelTransport'.lower():
        options['t0'] = xml_parameters.t0
        options['t1'] = xml_parameters.t1 #ajout fg
        options['tmin'] = xml_parameters.tmin
        options['perform_shooting'] = xml_parameters.perform_shooting
        options['tmax'] = xml_parameters.tmax
        options['initial_momenta_to_transport'] = xml_parameters.initial_momenta_to_transport
        options['initial_control_points_to_transport'] = xml_parameters.initial_control_points_to_transport
    
    return options


class XmlParameters:
    """
    XmlParameters object class.
    Parses input xmls and stores the given parameters.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.model_type = default.model_type
        self.template_specifications = default.template_specifications
        self.deformation_kernel_width = 0
        self.deformation_kernel_device = default.deformation_kernel_device
        self.number_of_time_points = default.number_of_time_points
        self.concentration_of_time_points = default.concentration_of_time_points
        self.number_of_sources = default.number_of_sources
        self.t0 = None
        self.tR = [] # ajout fg
        self.t1 = None #ajout fg
        self.num_component = None # ajout fg
        self.tmin = default.tmin
        self.tmax = default.tmax
        self.dimension = default.dimension
        self.covariance_momenta_prior_normalized_dof = default.covariance_momenta_prior_normalized_dof

        self.dataset_filenames = default.dataset_filenames
        self.visit_ages = default.visit_ages
        self.subject_ids = default.subject_ids

        self.optimization_method = default.optimization_method
        self.optimized_log_likelihood = default.optimized_log_likelihood
        self.max_iterations = default.max_iterations
        self.max_line_search_iterations = default.max_line_search_iterations
        self.save_every_n_iters = default.save_every_n_iters
        self.print_every_n_iters = default.print_every_n_iters
        self.smoothing_kernel_width = default.smoothing_kernel_width
        self.initial_step_size = default.initial_step_size
        self.line_search_shrink = default.line_search_shrink
        self.line_search_expand = default.line_search_expand
        self.convergence_tolerance = default.convergence_tolerance
        self.scale_initial_step_size = default.scale_initial_step_size
        self.downsampling_factor = default.downsampling_factor
        self.interpolation = default.interpolation #ajout fg

        self.gpu_mode = default.gpu_mode
        self._cuda_is_used = default._cuda_is_used  # true if at least one operation will use CUDA.
        self._keops_is_used = default._keops_is_used  # true if at least one keops kernel operation will take place.

        self.state_file = None
        self.load_state_file = False

        self.freeze_template = default.freeze_template
        self.multiscale_momenta = default.multiscale_momenta #ajout fg
        self.multiscale_images = default.multiscale_images #ajout fg
        self.multiscale_meshes = default.multiscale_meshes
        self.naive = default.naive #ajout fg
        self.start_scale = None
        self.multiscale_strategy = default.multiscale_strategy #ajout fg
        self.gamma = default.gamma
        self.perform_shooting = default.perform_shooting #ajout fg
        self.freeze_momenta = default.freeze_momenta
        self.freeze_principal_directions = default.freeze_principal_directions
        self.freeze_modulation_matrix = default.freeze_modulation_matrix
        self.freeze_reference_time = default.freeze_reference_time
        self.freeze_rupture_time = default.freeze_rupture_time
        self.freeze_noise_variance = default.freeze_noise_variance

        self.freeze_translation_vectors = False
        self.freeze_rotation_angles = False
        self.freeze_scaling_ratios = False

        self.initial_control_points = default.initial_control_points
        self.initial_momenta = default.initial_momenta
        self.initial_principal_directions = default.initial_principal_directions
        self.initial_modulation_matrix = default.initial_modulation_matrix
        self.initial_sources = default.initial_sources
        self.initial_sources_mean = default.initial_sources_mean
        self.initial_sources_std = default.initial_sources_std

        self.initial_control_points_to_transport = default.initial_control_points_to_transport

        self.momenta_proposal_std = default.momenta_proposal_std
        self.sources_proposal_std = default.sources_proposal_std

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def read_all_xmls(self, model_xml_path, dataset_xml_path, optimization_parameters_xml_path):
        self._read_model_xml(model_xml_path)
        self._read_dataset_xml(dataset_xml_path)
        self._read_optimization_parameters_xml(optimization_parameters_xml_path)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    # Read the parameters from the model xml.
    def _read_model_xml(self, model_xml_path):
        model_xml_level0 = et.parse(model_xml_path).getroot()

        for model_xml_level1 in model_xml_level0:
            if model_xml_level1.tag.lower() == 'model-type':
                self.model_type = model_xml_level1.text.lower()
            
            elif model_xml_level1.tag.lower() == 'num-component':
                self.num_component = int(model_xml_level1.text)

            elif model_xml_level1.tag.lower() == 'dimension':
                self.dimension = int(model_xml_level1.text)

            elif model_xml_level1.tag.lower() == 'initial-control-points':
                self.initial_control_points = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-momenta':
                self.initial_momenta = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-principal-directions':
                self.initial_principal_directions = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-modulation-matrix':
                self.initial_modulation_matrix = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-sources':
                self.initial_sources = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-sources-mean':
                self.initial_sources_mean = model_xml_level1.text

            elif model_xml_level1.tag.lower() == 'initial-sources-std':
                self.initial_sources_std = model_xml_level1.text

            elif model_xml_level1.tag.lower() == 'initial-momenta-to-transport':
                self.initial_momenta_to_transport = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-control-points-to-transport':
                self.initial_control_points_to_transport = os.path.normpath(
                    os.path.join(os.path.dirname(model_xml_path), model_xml_level1.text))

            elif model_xml_level1.tag.lower() == 'initial-noise-std':
                self.initial_noise_variance = float(model_xml_level1.text) ** 2

            elif model_xml_level1.tag.lower() == 'latent-space-dimension':
                self.latent_space_dimension = int(model_xml_level1.text)

            elif model_xml_level1.tag.lower() == 'template':
                for model_xml_level2 in model_xml_level1:

                    if model_xml_level2.tag.lower() == 'object':

                        template_object = self._initialize_template_object_xml_parameters()
                        for model_xml_level3 in model_xml_level2:
                            if model_xml_level3.tag.lower() in['deformable-object-type', 'object-type']:
                                template_object['object_type'] = model_xml_level3.text.lower()
                            elif model_xml_level3.tag.lower() == 'attachment-type':
                                template_object['attachment_type'] = model_xml_level3.text.lower()
                            elif model_xml_level3.tag.lower() == 'kernel-width':
                                template_object['kernel_width'] = float(model_xml_level3.text)
                            elif model_xml_level3.tag.lower() == 'kernel-device':
                                template_object['kernel_device'] = model_xml_level3.text
                            elif model_xml_level3.tag.lower() == 'noise-std':
                                template_object['noise_std'] = float(model_xml_level3.text)
                            elif model_xml_level3.tag.lower() == 'filename':
                                template_object['filename'] = os.path.normpath(
                                    os.path.join(os.path.dirname(model_xml_path), model_xml_level3.text))
                            elif model_xml_level3.tag.lower() == 'noise-variance-prior-scale-std':
                                template_object['noise_variance_prior_scale_std'] = float(model_xml_level3.text)
                            elif model_xml_level3.tag.lower() == 'noise-variance-prior-normalized-dof':
                                template_object['noise_variance_prior_normalized_dof'] = float(model_xml_level3.text)
                            else:
                                msg = 'Unknown entry while parsing the template > ' + model_xml_level2.attrib['id'] + \
                                      ' object section of the model xml: ' + model_xml_level3.tag
                                warnings.warn(msg)

                        self.template_specifications[model_xml_level2.attrib['id']] = template_object
                    else:
                        msg = 'Unknown entry while parsing the template section of the model xml: ' \
                              + model_xml_level2.tag
                        warnings.warn(msg)

            elif model_xml_level1.tag.lower() == 'deformation-parameters':
                for model_xml_level2 in model_xml_level1:
                    if model_xml_level2.tag.lower() == 'kernel-width':
                        self.deformation_kernel_width = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'exponential-type':
                        self.exponential_type = model_xml_level2.text
                    elif model_xml_level2.tag.lower() == 'kernel-device':
                        self.deformation_kernel_device = model_xml_level2.text
                    elif model_xml_level2.tag.lower() == 'number-of-timepoints':
                        self.number_of_time_points = int(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'concentration-of-timepoints':
                        self.concentration_of_time_points = int(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'number-of-sources':
                        self.number_of_sources = int(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 't0':
                        self.t0 = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'tr':
                        self.tR.append(float(model_xml_level2.text))
                    elif model_xml_level2.tag.lower() == 't1':
                        self.t1 = float(model_xml_level2.text) #ajout fg
                    elif model_xml_level2.tag.lower() == 'perform-shooting': #ajout fg
                        self.perform_shooting = self._on_off_to_bool(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'tmin':
                        self.tmin = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'tmax':
                        self.tmax = float(model_xml_level2.text)
                    elif model_xml_level2.tag.lower() == 'p0':
                        self.p0 = model_xml_level2.text
                    elif model_xml_level2.tag.lower() == 'v0':
                        self.v0 = model_xml_level2.text
                    elif model_xml_level2.tag.lower() == 'covariance-momenta-prior-normalized-dof':
                        self.covariance_momenta_prior_normalized_dof = float(model_xml_level2.text)
                    else:
                        msg = 'Unknown entry while parsing the deformation-parameters section of the model xml: ' \
                              + model_xml_level2.tag
                        warnings.warn(msg)
            else:
                msg = 'Unknown entry while parsing root of the model xml: ' + model_xml_level1.tag
                warnings.warn(msg)

    # Read the parameters from the dataset xml.
    def _read_dataset_xml(self, dataset_xml_path):
        if dataset_xml_path is not None and dataset_xml_path != 'None':

            dataset_xml_level0 = et.parse(dataset_xml_path).getroot()
            data_set_xml_dirname = os.path.dirname(dataset_xml_path)

            dataset_filenames = []
            visit_ages = []
            subject_ids = []
            for dataset_xml_level1 in dataset_xml_level0:
                if dataset_xml_level1.tag.lower() == 'subject':
                    subject_ids.append(dataset_xml_level1.attrib['id'])

                    subject_filenames = []
                    subject_ages = []
                    for dataset_xml_level2 in dataset_xml_level1:
                        if dataset_xml_level2.tag.lower() == 'visit':

                            visit_filenames = {}
                            for dataset_xml_level3 in dataset_xml_level2:
                                if dataset_xml_level3.tag.lower() == 'filename':
                                    visit_filenames[dataset_xml_level3.attrib['object_id']] = os.path.normpath(
                                        os.path.join(data_set_xml_dirname, dataset_xml_level3.text))
                                elif dataset_xml_level3.tag.lower() == 'age':
                                    subject_ages.append(float(dataset_xml_level3.text))
                            subject_filenames.append(visit_filenames)
                    #a list of n_subjects lists containing dictionaries of filenames
                    dataset_filenames.append(subject_filenames)
                    
                    visit_ages.append(subject_ages)

            self.dataset_filenames = dataset_filenames
            self.visit_ages = visit_ages
            self.subject_ids = subject_ids

    # Read the parameters from the optimization_parameters xml.
    def _read_optimization_parameters_xml(self, optimization_parameters_xml_path):
        if optimization_parameters_xml_path is not None and optimization_parameters_xml_path != 'None':

            optimization_parameters_xml_level0 = et.parse(optimization_parameters_xml_path).getroot()

            for optimization_parameters_xml_level1 in optimization_parameters_xml_level0:
                if optimization_parameters_xml_level1.tag.lower() == 'optimization-method-type':
                    self.optimization_method = optimization_parameters_xml_level1.text.lower()
                elif optimization_parameters_xml_level1.tag.lower() == 'optimized-log-likelihood':
                    self.optimized_log_likelihood = optimization_parameters_xml_level1.text.lower()
                elif optimization_parameters_xml_level1.tag.lower() == 'max-iterations':
                    self.max_iterations = int(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'convergence-tolerance':
                    self.convergence_tolerance = float(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'downsampling-factor':
                    self.downsampling_factor = int(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'interpolation':
                    self.interpolation = str(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'save-every-n-iters':
                    self.save_every_n_iters = int(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'print-every-n-iters':
                    self.print_every_n_iters = int(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'initial-step-size':
                    self.initial_step_size = float(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'freeze-template':
                    self.freeze_template = self._on_off_to_bool(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'multiscale-momenta': #ajout fg
                    self.multiscale_momenta = self._on_off_to_bool(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'multiscale-images': #ajout fg
                    self.multiscale_images = self._on_off_to_bool(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'multiscale-meshes': #ajout fg
                    self.multiscale_meshes = self._on_off_to_bool(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'naive': #ajout fg
                    self.naive = self._on_off_to_bool(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'multiscale-strategy': #ajout fg
                    self.multiscale_strategy = str(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'start-scale': #ajout fg
                    self.start_scale = float(optimization_parameters_xml_level1.text)  
                elif optimization_parameters_xml_level1.tag.lower() == 'gamma': #ajout fg
                    self.gamma = float(optimization_parameters_xml_level1.text)  
                elif optimization_parameters_xml_level1.tag.lower() == 'freeze-principal-directions':
                    self.freeze_principal_directions = self._on_off_to_bool(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'gpu-mode':
                    self.gpu_mode = GpuMode[optimization_parameters_xml_level1.text.upper()]
                elif optimization_parameters_xml_level1.tag.lower() == 'max-line-search-iterations':
                    self.max_line_search_iterations = int(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'state-file':
                    self.state_file = os.path.join(os.path.dirname(optimization_parameters_xml_path),
                                                   optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'momenta-proposal-std':
                    self.momenta_proposal_std = float(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'sources-proposal-std':
                    self.sources_proposal_std = float(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'scale-initial-step-size':
                    self.scale_initial_step_size = self._on_off_to_bool(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'initialization-heuristic':
                    self.initialization_heuristic = self._on_off_to_bool(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'freeze-v0':
                    self.freeze_v0 = self._on_off_to_bool(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'freeze-p0':
                    self.freeze_p0 = self._on_off_to_bool(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'freeze-modulation-matrix':
                    self.freeze_modulation_matrix = self._on_off_to_bool(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'freeze-reference-time':
                    self.freeze_reference_time = self._on_off_to_bool(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'freeze-rupture-time':
                    self.freeze_rupture_time = self._on_off_to_bool(optimization_parameters_xml_level1.text)
                elif optimization_parameters_xml_level1.tag.lower() == 'freeze-noise-variance':
                    self.freeze_noise_variance = self._on_off_to_bool(optimization_parameters_xml_level1.text)

                else:
                    msg = 'Unknown entry while parsing the optimization_parameters xml: ' \
                          + optimization_parameters_xml_level1.tag
                    warnings.warn(msg)

    # Default xml parameters for any template object.
    @staticmethod
    def _initialize_template_object_xml_parameters():
        template_object = {}
        template_object['object_type'] = 'undefined'
        template_object['kernel_width'] = 0.0
        template_object['kernel_device'] = default.deformation_kernel_device
        template_object['noise_std'] = -1
        template_object['filename'] = 'undefined'
        template_object['noise_variance_prior_scale_std'] = None
        template_object['noise_variance_prior_normalized_dof'] = 0.01
        template_object["interpolation"] = "linear" #fg
        return template_object

    def _on_off_to_bool(self, s):
        if s.lower() == "on":
            return True
        elif s.lower() == "off":
            return False
        else:
            raise RuntimeError("Please give a valid flag (on, off)")
