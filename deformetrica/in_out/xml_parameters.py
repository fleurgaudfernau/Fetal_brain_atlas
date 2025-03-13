import logging
import os
import os.path as op 
import warnings
import xml.etree.ElementTree as et

from ..core import default, GpuMode
from ..support import utilities

logger = logging.getLogger(__name__)

def floatif(tag, value, variable, level):
    if tag == value:
        variable = float(level.text)

    return variable

def get_dataset_specifications(xml_parameters):
    print(len(xml_parameters.template_specifications))

    specifications = {}
    specifications['visit_ages'] = xml_parameters.visit_ages
    specifications['filenames'] = xml_parameters.dataset_filenames
    specifications['subject_ids'] = xml_parameters.subject_ids
    specifications['interpolation'] = xml_parameters.interpolation
    specifications['kernel_width'] = xml_parameters.template_specifications["Object_1"]["kernel_width"]
    specifications['n_subjects'] = len(xml_parameters.subject_ids)
    n_observations = sum(len(visit) for subject in xml_parameters.dataset_filenames for visit in subject)
    specifications['n_observations'] = n_observations

    specifications['n_objects'] = len(xml_parameters.template_specifications) 

    return specifications

def get_estimator_options(xml_parameters):
    options = {}

    optimization_method = xml_parameters.optimization_method.lower()

    if optimization_method in ['GradientAscent'.lower(), 'StochasticGradientAscent'.lower()]:
        options['initial_step_size'] = xml_parameters.initial_step_size
        options['max_line_search_iterations'] = xml_parameters.max_line_search_iterations
        options['optimized_log_likelihood'] = xml_parameters.optimized_log_likelihood

    elif optimization_method == 'ScipyLBFGS'.lower():
        options['freeze_template'] = xml_parameters.freeze_template
        options['max_line_search_iterations'] = xml_parameters.max_line_search_iterations
        options['optimized_log_likelihood'] = xml_parameters.optimized_log_likelihood

    # common options
    options['optimization_method'] = optimization_method
    options['max_iterations'] = xml_parameters.max_iterations
    options['convergence_tolerance'] = xml_parameters.convergence_tolerance
    options['print_every_n_iters'] = xml_parameters.print_every_n_iters
    options['save_every_n_iters'] = xml_parameters.save_every_n_iters
    options['state_file'] = xml_parameters.state_file
    options['load_state_file'] = xml_parameters.load_state_file

    # Multiscale options
    options['multiscale_momenta'] = xml_parameters.multiscale_momenta #ajout fg
    options['multiscale_images'] = xml_parameters.multiscale_images
    options['multiscale_meshes'] = xml_parameters.multiscale_meshes
    options['multiscale_strategy'] =  xml_parameters.multiscale_strategy #ajout fg

    return options

def get_model_options(xml_parameters):
    options = { 'deformation_kernel_width': xml_parameters.deformation_kernel_width,
                'n_time_points': xml_parameters.n_time_points,
                'time_concentration': xml_parameters.time_concentration,
                'freeze_template': xml_parameters.freeze_template,
                'freeze_momenta': xml_parameters.freeze_momenta,
                'freeze_noise_variance': xml_parameters.freeze_noise_variance,
                'initial_cp': xml_parameters.initial_cp,
                'initial_momenta': xml_parameters.initial_momenta,
                'downsampling_factor': xml_parameters.downsampling_factor,
                'perform_shooting':xml_parameters.perform_shooting, #ajout fg
                'interpolation':xml_parameters.interpolation,
                "rambouilli": xml_parameters.number_of_sources }

    model_type = xml_parameters.model_type.lower()
    if model_type in ['bayesiangeodesicregression']:
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

    elif model_type in ['regression', 'kernelregression']:
        options['t0'] = xml_parameters.t0
        options['tmin'] = xml_parameters.tmin
        options['tmax'] = xml_parameters.tmax

    elif model_type in ["piecewiseregression"]: #ajout fg
        options['num_component'] = xml_parameters.num_component
        options['t0'] = xml_parameters.t0
        options['tR'] = xml_parameters.tR
        options['t1'] = xml_parameters.t1
        options['freeze_reference_time'] = xml_parameters.freeze_reference_time
        options['freeze_rupture_time'] = xml_parameters.freeze_rupture_time

    elif model_type == 'paralleltransport':
        options['t0'] = xml_parameters.t0
        options['t1'] = xml_parameters.t1 #ajout fg
        options['start_time'] = xml_parameters.t1
        options['tmin'] = xml_parameters.tmin
        options['tmax'] = xml_parameters.tmax
        options['initial_momenta_to_transport'] = xml_parameters.initial_momenta_to_transport
        options['initial_cp_to_transport'] = xml_parameters.initial_cp_to_transport
        options['perform_shooting'] = xml_parameters.perform_shooting
    elif model_type == 'piecewiseparalleltransport':
        options['num_component'] = xml_parameters.num_component
        options['t0'] = xml_parameters.t0
        options['tR'] = xml_parameters.tR
        options['t1'] = xml_parameters.t1 #ajout fg
        options['start_time'] = xml_parameters.t1
        options['tmin'] = xml_parameters.tmin
        options['tmax'] = xml_parameters.tmax
        options['initial_momenta_to_transport'] = xml_parameters.initial_momenta_to_transport
        options['initial_cp_to_transport'] = xml_parameters.initial_cp_to_transport
        options['perform_shooting'] = xml_parameters.perform_shooting
    
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
        self.n_time_points = default.n_time_points
        self.time_concentration = default.time_concentration
        self.number_of_sources = default.number_of_sources
        self.t0 = None
        self.tR = [] # ajout fg
        self.t1 = None #ajout fg
        self.start_time = None #ajout fg
        self.num_component = None # ajout fg
        self.tmin = default.tmin
        self.tmax = default.tmax
        self.covariance_momenta_prior_norm_dof = default.covariance_momenta_prior_norm_dof

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
        self.convergence_tolerance = default.convergence_tolerance
        self.downsampling_factor = default.downsampling_factor
        self.interpolation = default.interpolation #ajout fg

        #self.gpu_mode = default.gpu_mode
        self._cuda_is_used = default._cuda_is_used  # true if at least one operation will use CUDA.
        self._keops_is_used = default._keops_is_used  # true if at least one keops kernel operation will take place.

        self.state_file = None
        self.load_state_file = False

        self.freeze_template = default.freeze_template
        self.multiscale_momenta = default.multiscale_momenta #ajout fg
        self.multiscale_images = default.multiscale_images #ajout fg
        self.multiscale_meshes = default.multiscale_meshes
        self.multiscale_strategy = default.multiscale_strategy #ajout fg
        self.perform_shooting = default.perform_shooting #ajout fg
        self.freeze_momenta = default.freeze_momenta
        self.freeze_modulation_matrix = default.freeze_modulation_matrix
        self.freeze_reference_time = default.freeze_reference_time
        self.freeze_rupture_time = default.freeze_rupture_time
        self.freeze_noise_variance = default.freeze_noise_variance

        self.initial_cp = default.initial_cp
        self.initial_momenta = default.initial_momenta
        self.initial_modulation_matrix = default.initial_modulation_matrix
        self.initial_sources = default.initial_sources
        self.initial_sources_mean = default.initial_sources_mean
        self.initial_sources_std = default.initial_sources_std

        self.initial_cp_to_transport = default.initial_cp_to_transport

        self.momenta_proposal_std = default.momenta_proposal_std
        self.sources_proposal_std = default.sources_proposal_std

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def read_all_xmls(self, model_xml_path, dataset_xml_path, parameters_xml_path):
        self._read_model_xml(model_xml_path)
        self._read_dataset_xml(dataset_xml_path)
        self._read_parameters_xml(parameters_xml_path)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    # Read the parameters from the model xml.
    def _read_model_xml(self, model_xml_path):
        root = et.parse(model_xml_path).getroot()
        directory = op.dirname(model_xml_path)

        for model in root:
            model_tag, text = model.tag.lower(), model.text

            if model_tag == 'model-type':
                self.model_type = model.text.lower()

            elif model_tag == 'num-component':
                self.num_component = int(model.text)

            elif model_tag == 'initial-control-points':
                self.initial_cp = op.join(directory, text)

            elif model_tag == 'initial-momenta':
                self.initial_momenta = op.join(directory, text)

            elif model_tag == 'initial-modulation-matrix':
                self.initial_modulation_matrix = op.join(directory, text)

            elif model_tag == 'initial-sources':
                self.initial_sources = op.join(directory, text)

            elif model_tag == 'initial-sources-mean':
                self.initial_sources_mean = model.text

            elif model_tag == 'initial-sources-std':
                self.initial_sources_std = model.text

            elif model_tag == 'initial-momenta-to-transport':
                self.initial_momenta_to_transport = op.join(directory, text)

            elif model_tag == 'initial-control-points-to-transport':
                self.initial_cp_to_transport = op.join(directory, text)

            elif model_tag == 'template':
                n_objects = 0
                for properties in model:

                    tag, text = properties.tag.lower(), properties.text

                    if tag == 'filename':
                        n_objects += 1
                        template_object = self._initialize_template_object_xml_parameters()
                        template_object['filename'] = op.join(directory, text)
                    elif tag == 'attachment-type':
                        template_object['attachment_type'] = text.lower()
                    elif tag == 'kernel-width':
                        template_object['kernel_width'] = float(text)
                    elif tag == 'noise-std':
                        template_object['noise_std'] = round(float(text) ** 2, 5)
                    elif tag == 'noise-variance-prior-scale-std':
                        template_object['noise_variance_prior_scale_std'] = float(text)
                    elif tag == 'noise-variance-prior-normalized-dof':
                        template_object['noise_variance_prior_normalized_dof'] = float(text)
                    else:
                        warnings.warn('Unknown entry while parsing model xml:{}'.format(tag))

                self.template_specifications["Object_{}".format(n_objects)] = template_object

            elif model_tag == 'deformation-parameters':
                for deformation in model:
                    tag = deformation.tag.lower()
                    # self.deformation_kernel_width = floatif(tag, 'kernel-width', 
                    #                         self.deformation_kernel_width, deformation)
                    if tag == 'kernel-width':
                        self.deformation_kernel_width = float(deformation.text)
                    elif tag == 'number-of-timepoints':
                        self.n_time_points = int(deformation.text)
                    elif tag == 'concentration-of-timepoints':
                        self.time_concentration = int(deformation.text)
                    elif tag == 'number-of-sources':
                        self.number_of_sources = int(deformation.text)
                    elif tag == 't0':
                        self.t0 = float(deformation.text)
                    elif tag == 'tr':
                        self.tR.append(float(deformation.text))
                    elif tag == 't1':
                        self.t1 = float(deformation.text) #ajout fg
                    elif tag == 'perform-shooting': #ajout fg
                        self.perform_shooting = self._on_off_to_bool(deformation.text)
                    elif tag == 'tmin':
                        self.tmin = float(deformation.text)
                    elif tag == 'tmax':
                        self.tmax = float(deformation.text)
                    elif tag == 'covariance-momenta-prior-normalized-dof':
                        self.covariance_momenta_prior_norm_dof = float(deformation.text)
                    else:
                        msg = 'Unknown entry while parsing the deformation-parameters section of the model xml: {}'.format(deformation.tag)
                        warnings.warn(msg)
            else:
                msg = 'Unknown entry while parsing root of the model xml: {}'.format(model.tag)
                warnings.warn(msg)

    # Read the parameters from the dataset xml.
    
    def _read_dataset_xml(self, dataset_xml_path):
        if dataset_xml_path is not None and dataset_xml_path != 'None':

            root = et.parse(dataset_xml_path).getroot()
            xml_dirname = op.dirname(dataset_xml_path)

            self.dataset_filenames = [[] for _ in range(len(root))]
            self.visit_ages = [[] for _ in range(len(root))]
            self.subject_ids = []

            for i, subject in enumerate(root):
                if subject.tag.lower() == 'subject':
                    
                    self.subject_ids.append(subject.attrib['id'])
                    
                    self.dataset_filenames[i] = [ { f"Object_{k}": op.join(xml_dirname, obj.text)
                                                    for k, obj in enumerate(visit) \
                                                    if obj.tag.lower() == 'filename'} \
                                                    for visit in subject if visit.tag.lower() == 'visit' ]
                    
                    self.visit_ages[i] = [  float(obj.text) for visit in subject \
                                            if visit.tag.lower() == 'visit'
                                            for obj in visit if obj.tag.lower() == 'age' ]

    # Read the parameters from the optimization_parameters xml.
    def _read_parameters_xml(self, parameters_xml_path):
        if parameters_xml_path is not None and parameters_xml_path != 'None':
            
            root = et.parse(parameters_xml_path).getroot()

            for level1 in root:
                tag = level1.tag.lower()
                if tag == 'optimization-method-type':
                    self.optimization_method = level1.text.lower()
                elif tag == 'optimized-log-likelihood':
                    self.optimized_log_likelihood = level1.text.lower()
                elif tag == 'max-iterations':
                    self.max_iterations = int(level1.text)
                elif tag == 'convergence-tolerance':
                    self.convergence_tolerance = float(level1.text)
                elif tag == 'downsampling-factor':
                    self.downsampling_factor = int(level1.text)
                elif tag == 'interpolation':
                    self.interpolation = str(level1.text)
                elif tag == 'save-every-n-iters':
                    self.save_every_n_iters = int(level1.text)
                elif tag == 'print-every-n-iters':
                    self.print_every_n_iters = int(level1.text)
                elif tag == 'initial-step-size':
                    self.initial_step_size = float(level1.text)
                elif tag == 'freeze-template':
                    self.freeze_template = self._on_off_to_bool(level1.text)
                elif tag == 'multiscale-momenta': #ajout fg
                    self.multiscale_momenta = self._on_off_to_bool(level1.text)
                elif tag == 'multiscale-images': #ajout fg
                    self.multiscale_images = self._on_off_to_bool(level1.text)
                elif tag == 'multiscale-meshes': #ajout fg
                    self.multiscale_meshes = self._on_off_to_bool(level1.text)
                elif tag == 'multiscale-strategy': #ajout fg
                    self.multiscale_strategy = str(level1.text)
                elif tag == 'max-line-search-iterations':
                    self.max_line_search_iterations = int(level1.text)
                elif tag == 'state-file':
                    self.state_file = op.join(op.dirname(parameters_xml_path), level1.text)
                elif tag == 'momenta-proposal-std':
                    self.momenta_proposal_std = float(level1.text)
                elif tag == 'sources-proposal-std':
                    self.sources_proposal_std = float(level1.text)
                elif tag == 'initialization-heuristic':
                    self.initialization_heuristic = self._on_off_to_bool(level1.text)
                elif tag == 'freeze-modulation-matrix':
                    self.freeze_modulation_matrix = self._on_off_to_bool(level1.text)
                elif tag == 'freeze-reference-time':
                    self.freeze_reference_time = self._on_off_to_bool(level1.text)
                elif tag == 'freeze-rupture-time':
                    self.freeze_rupture_time = self._on_off_to_bool(level1.text)
                elif tag == 'freeze-noise-variance':
                    self.freeze_noise_variance = self._on_off_to_bool(level1.text)
                else:
                    msg = 'Unknown entry while parsing the optimization_parameters xml: {}'.format(level1.tag)
                    warnings.warn(msg)

    # Default xml parameters for any template object.
    @staticmethod
    def _initialize_template_object_xml_parameters():
        template_object = {}
        template_object['kernel_width'] = 0.0
        template_object['filename'] = 'undefined'
        template_object['noise_std'] = -1
        template_object['noise_variance_prior_scale_std'] = None
        template_object['noise_variance_prior_normalized_dof'] = 0.01
        template_object["interpolation"] = "linear"
        
        return template_object

    def _on_off_to_bool(self, s):
        if s.lower() == "on":
            return True
        elif s.lower() == "off":
            return False
        else:
            raise RuntimeError("Please give a valid flag (on, off)")
