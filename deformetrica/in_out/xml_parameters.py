import logging
import os
import os.path as op 
import warnings
import xml.etree.ElementTree as et

from ..core import default, GpuMode
from ..support import utilities

logger = logging.getLogger(__name__)

def get_dataset_specifications(xml_parameters):

    specifications = { k : getattr(xml_parameters, k) for k in [
                                    "ids", "visit_ages", "filenames", "interpolation"] }
    specifications.update({ 
    'kernel_width': xml_parameters.template_specifications["Object_1"]["kernel_width"],
    'noise_std': xml_parameters.template_specifications["Object_1"]["noise_std"],
    'n_subjects': len(xml_parameters.ids),
    'n_observations': sum(len(visit) for subject in xml_parameters.filenames for visit in subject),
    'n_objects': len(xml_parameters.template_specifications) })

    return specifications

def get_estimator_options(xml_parameters):

    xml_params_keys = [ 'optimization_method', "optimized_log_likelihood",
                        'initial_step_size', 'max_iterations', 
                        'convergence_tolerance', 'print_every_n_iters', 'save_every_n_iters', 
                        'multiscale_momenta', 'multiscale_objects', "multiscale_strategy",
                        "state_file", "load_state_file" ]
    
    return {k : getattr(xml_parameters, k) for k in xml_params_keys }

def get_model_options(xml_parameters):

    model_type = xml_parameters.model_type.lower()

    keys = [ 'deformation_kernel_width', 'n_time_points', 'time_concentration',
            'freeze_momenta', 'freeze_noise_variance',
            'initial_cp', 'initial_momenta', 'downsampling_factor',
            'interpolation', 'perform_shooting',
            "momenta_proposal_std", "sources_proposal_std", 
            "freeze_reference_time", "freeze_rupture_time", "freeze_noise_variance",
            't0', 't1', 'tmin', 'tmax',
            'tR', "num_component", 'number_of_sources']

    options = {k : getattr(xml_parameters, k) for k in keys }

    options.update({"freeze_template" : xml_parameters.freeze_template\
                    if model_type != "registration" else False})
                    
    options.update({"kernel_regression" : False if model_type != "kernelregression"\
                                        else True})
                                        
    if "regression" in xml_parameters.model_type and "kernel" not in xml_parameters.model_type:
        if options['t0'] is None:
            ages = [ a[0] for a in xml_parameters.visit_ages ]   
            options.update({ 't0' : min(ages) })

    model_specific_options = {
        'bayesiangeodesicregression': { k : getattr(xml_parameters, k) for k in [
                                        'number_of_sources', 'initial_modulation_matrix',
                                        'initial_sources', 'freeze_modulation_matrix',
                                        'freeze_reference_time', 'freeze_rupture_time'] },

        'piecewiseregression': { k : getattr(xml_parameters, k) \
                            for k in ["freeze_reference_time", "freeze_rupture_time"] },

        'paralleltransport': { k : getattr(xml_parameters, k) for k in [
                                        'start_time',
                                        'initial_momenta_to_transport',
                                        'initial_cp_to_transport'] },

        'piecewiseparalleltransport': { k : getattr(xml_parameters, k) for k in [
                                        'start_time',
                                        'initial_momenta_to_transport',
                                        'initial_cp_to_transport'] } }
    if model_type in model_specific_options:
        options.update(model_specific_options[model_type])

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
        self.deformation_kernel_width = None

        self.n_time_points = default.n_time_points
        self.time_concentration = default.time_concentration
        self.number_of_sources = default.number_of_sources
        self.num_component = None

        self.t0 = None
        self.tR = [] 
        self.t1 = None 
        self.tmin = default.tmin
        self.tmax = default.tmax
        self.start_time = None 
        
        self.template_specifications = {}
        self.filenames = []
        self.visit_ages = []
        self.ids = []

        self.kernel_regression = False

        self.optimization_method = default.optimization_method
        self.optimized_log_likelihood = default.optimized_log_likelihood
        self.max_iterations = default.max_iterations
        self.save_every_n_iters = default.save_every_n_iters
        self.print_every_n_iters = default.print_every_n_iters
        self.smoothing_kernel_width = default.smoothing_kernel_width
        self.initial_step_size = default.initial_step_size
        self.convergence_tolerance = default.convergence_tolerance
        self.downsampling_factor = default.downsampling_factor
        self.interpolation = default.interpolation 

        self.state_file = None
        self.load_state_file = False

        self.freeze_template = default.freeze_template
        self.multiscale_momenta = False 
        self.multiscale_objects = False 
        self.multiscale_strategy = "stairs" 
        self.perform_shooting = default.perform_shooting 
        self.freeze_momenta = default.freeze_momenta
        self.freeze_modulation_matrix = False
        self.freeze_reference_time = default.freeze_reference_time
        self.freeze_rupture_time = default.freeze_rupture_time
        self.freeze_noise_variance = default.freeze_noise_variance

        self.initial_cp = default.initial_cp
        self.initial_cp_to_transport = default.initial_cp_to_transport
        self.initial_momenta = default.initial_momenta
        self.initial_momenta_to_transport = None

        self.initial_modulation_matrix = default.initial_modulation_matrix
        self.initial_sources = default.initial_sources
        self.initial_sources_mean = default.initial_sources_mean
        self.initial_sources_std = default.initial_sources_std

        self.momenta_proposal_std = default.momenta_proposal_std
        self.sources_proposal_std = default.sources_proposal_std
        self.covariance_momenta_prior_norm_dof = default.covariance_momenta_prior_norm_dof

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def read_all_xmls(self, args):
        try:
            self.directory = op.dirname(args.dataset)
        except:
            self.directory = None

        self._read_dataset_xml(args.dataset)
        self._read_args(args)
        self._set_output(args)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################
    def _set_output(self, args):
        self.output_dir = None
        try:
            if args.output is None:
                if args.command == 'initialize':
                    self.output_dir = dfca.default.preprocessing_dir
                else:
                    self.output_dir = "Output_k{}".format(args.deformation_kernel_width)
                    self.output_dir += "_noise_{}".format(args.noise_std) if args.noise_std else ""
                    self.output_dir += "_t0_{}".format(args.t0) if args.t0 else ""
                    self.output_dir += "_comp_{}".format(args.num_component) if args.num_component else ""
                    self.output_dir += "_ctf_momenta" if args.multiscale_momenta else ""
                    self.output_dir += "_ctf_objects" if args.multiscale_objects else ""
            else:
                self.output_dir = args.output
                
            os.makedirs(self.output_dir)
        except FileExistsError:
            pass

    def _read_args(self, args):
        """
            Read arguments provided by user
        """
        for arg_name in ["deformation_kernel_width", "initial_step_size", 
                        't0', 't1', 'tmin', 'tmax', "tR",
                        "num_component", "time_concentration", "number_of_sources",
                        'save_every_n_iters', 'max_iterations', 'downsampling_factor', 
                        'convergence_tolerance',
                        
                        "initial_control_points", "initial_momenta",
                        "initial_control_points_to_transport", "initial_momenta_to_transport",
                        "initial_sources", "initial_modulation_matrix"]:
            
            if getattr(args, arg_name):
                setattr(self, arg_name, getattr(args, arg_name))

        for arg_name in ["model_type", "interpolation", 'attachment_type']:
            if getattr(args, arg_name):
                setattr(self, arg_name, str(getattr(args, arg_name).lower()))

        for arg_name in ["multiscale_momenta", "multiscale_objects", "freeze_template"]:
            if getattr(args, arg_name):
                setattr(self, arg_name, True)
        
        template_object = self._initialize_template_object_xml_parameters()
        
        if args.template:
            template_object['filename'] = op.join(self.directory, args.template)

        if args.attachment_type:
            template_object['attachment_type'] = args.attachment_type.lower()

        if args.attachment_kernel:
            template_object['kernel_width'] = args.attachment_kernel
        else:
            template_object['kernel_width'] = self.deformation_kernel_width

        if args.noise_std:
            template_object['noise_std'] = round(float(args.noise_std) ** 2, 5)
        
        self.template_specifications["Object_1"] = template_object

    def _read_dataset_xml(self, dataset_xml_path):
        # filenames [ [{'Object_1'}:path] ]
        if dataset_xml_path not in [None, 'None']:

            root = et.parse(dataset_xml_path).getroot()

            self.filenames = [[] for _ in range(len(root))]
            self.visit_ages = [[] for _ in range(len(root))]
            self.ids = []

            for i, subject in enumerate(root):
                if subject.tag.lower() == 'subject':
                    
                    self.ids.append(subject.attrib['id'])
                    
                    self.filenames[i] = [ { f"Object_{k}": op.join(self.directory, obj.text)
                                                    for k, obj in enumerate(visit) \
                                                    if obj.tag.lower() == 'filename'} \
                                                    for visit in subject if visit.tag.lower() == 'visit' ]
                    
                    self.visit_ages[i] = [  float(obj.text) for visit in subject \
                                            if visit.tag.lower() == 'visit'
                                            for obj in visit if obj.tag.lower() == 'age' ]

    # Default xml parameters for any template object.
    @staticmethod
    def _initialize_template_object_xml_parameters():
        return { 'kernel_width': 0.0, 'filename': 'undefined',
                'noise_std': -1,
                'noise_variance_prior_scale_std': None,
                'noise_variance_prior_normalized_dof': 0.01,
                "interpolation": "linear" }
        
    def _on_off_to_bool(self, s):
        if s.lower() == "on":
            return True
        elif s.lower() == "off":
            return False
        else:
            raise RuntimeError("Please give a valid flag (on, off)")
