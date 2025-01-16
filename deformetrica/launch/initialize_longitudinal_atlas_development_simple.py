import os
import copy
import torch
import os.path as op
from os.path import join
import nibabel as nib
from imageio import imsave, imread
from PIL import Image
from ..launch.compute_shooting import compute_shooting
from torch.autograd import Variable
import warnings
import shutil
import math
from sklearn.decomposition import PCA, FastICA
import xml.etree.ElementTree as et
from xml.dom.minidom import parseString
from scipy.stats import norm, truncnorm
from ..core import default
from ..in_out.xml_parameters import XmlParameters, get_dataset_specifications, get_estimator_options, get_model_options
from ..in_out.dataset_functions import create_template_metadata, create_dataset
from ..core.model_tools.deformations.exponential import Exponential
from ..core.model_tools.deformations.geodesic import Geodesic
from ..in_out.array_readers_and_writers import *
from ..support import kernels as kernel_factory
from ..core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ..in_out.deformable_object_reader import DeformableObjectReader
from ..api.deformetrica import Deformetrica
from .deformetrica_functions import *
from .compute_parallel_transport import compute_parallel_transport

warnings.filterwarnings("ignore")

#############################################################################""


def scalar_product(kernel, cp, mom1, mom2):
    return torch.sum(mom1 * kernel.convolve(cp, cp, mom2))

def get_norm_squared(cp, momenta):
    return scalar_product(cp, momenta, momenta) 

def orthogonal_projection(cp, momenta_to_project, momenta):
    sp = scalar_product(cp, momenta_to_project, momenta) / get_norm_squared(cp, momenta)
    orthogonal_momenta = momenta_to_project - sp * momenta

    return orthogonal_momenta


def compute_RKHS_matrix(global_cp_nb, dimension, kernel_width, global_initial_cp):
    K = np.zeros((global_cp_nb * dimension, global_cp_nb * dimension))
    for i in range(global_cp_nb):
        for j in range(global_cp_nb):
            cp_i = global_initial_cp[i, :]
            cp_j = global_initial_cp[j, :]
            kernel_distance = math.exp(- np.sum((cp_j - cp_i) ** 2) / (kernel_width ** 2))
            for d in range(dimension):
                K[dimension * i + d, dimension * j + d] = kernel_distance
                K[dimension * j + d, dimension * i + d] = kernel_distance
    return K

#############################################################################""


class CrossSectionalLongitudinalAtlasInitializer():
    def __init__(self, model_xml_path, dataset_xml_path, optimization_parameters_xml_path):
        
        self.model_xml_path = model_xml_path
        self.dataset_xml_path = dataset_xml_path
        self.optimization_parameters_xml_path = optimization_parameters_xml_path
        
        #create preprocessing folder
        cwd = os.getcwd()
        self.output_dir = join(cwd, "Preprocessing")
        self.path_to_initialization = join(cwd, 'Initialization')

        if not op.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        if not op.isdir(self.path_to_initialization):
            os.mkdir(self.path_to_initialization)

        # Read original longitudinal model xml parameters.        
        self.xml_parameters = XmlParameters()
        self.xml_parameters._read_model_xml(self.model_xml_path)
        self.xml_parameters._read_dataset_xml(self.dataset_xml_path)
        self.xml_parameters._read_optimization_parameters_xml(self.optimization_parameters_xml_path)

        self.xml_parameters.max_iterations = 1000 ## TODO 
        self.xml_parameters.max_line_search_iterations = 10
        self.xml_parameters.print_every_n_iters = 10

        self.global_deformetrica = Deformetrica(output_dir=self.output_dir)

        # Save some global parameters.
        self.dimension = self.xml_parameters.dimension

        #global_dataset_filenames:list of lists: 1 list per subject of dict (1 dic = 1 obs)
        self.global_dataset_filenames = self.xml_parameters.dataset_filenames
        self.global_visit_ages = self.xml_parameters.visit_ages #list of lists: 1 list of visit ages/subject
        self.global_subject_ids = self.xml_parameters.subject_ids #list of ids
        self.objects_name, self.objects_ext = create_template_metadata(self.xml_parameters.template_specifications)[1:3]
        #xml_parameters.template_specifications: [deformable object Image], [object_name], [ext], [noise_std], [attachment]
       
        self.dataset = create_dataset(self.xml_parameters.template_specifications, 
                                    dimension=self.dimension, 
                                    visit_ages=copy.deepcopy(self.global_visit_ages), #to avoid modification
                                    dataset_filenames=self.global_dataset_filenames, 
                                    subject_ids=self.global_subject_ids)
        self.deformable_objects_dataset = self.dataset.deformable_objects #List of DeformableMultiObjects
        self.global_subjects_nb = len(self.global_dataset_filenames)
        self.global_observations_nb = sum([len(elt) for elt in self.global_visit_ages])

        # Classify dataset type
        self.nb_of_longitudinal_subjects = len([s for s in self.global_dataset_filenames if len(s) > 1])
        self.longitudinal_subjects_ind = [i for i in range(len(self.global_dataset_filenames))\
                                             if len(self.global_dataset_filenames[i]) > 1]
        self.single_subjects_ind = [i for i in range(len(self.global_dataset_filenames))\
                                             if len(self.global_dataset_filenames[i]) == 1]
        self.dataset_type = "single_points"

        # Deformation parameters
        self.dense_mode = self.xml_parameters.dense_mode #landmark points or not
        self.tensor_scalar_type = default.tensor_scalar_type #the type of float...
        self.global_kernel_type = self.xml_parameters.deformation_kernel_type #torch or keops
        self.global_kernel_width = self.xml_parameters.deformation_kernel_width
        self.global_kernel_device = self.xml_parameters.deformation_kernel_device
        self.kernel = kernel_factory.factory(self.global_kernel_type, kernel_width=self.global_kernel_width)
        
        # Times
        self.concentration_of_tp = self.xml_parameters.concentration_of_time_points
        self.global_t0 = self.xml_parameters.t0
        self.global_tmin = np.mean([e[0] for e in self.global_visit_ages])
        self.global_tmax = np.max([e[-1] for e in self.global_visit_ages])
        self.global_nb_of_tp = self.xml_parameters.number_of_time_points
        self.geodesic_nb_of_tp = int(1 + (self.global_t0 - self.global_tmin) * self.concentration_of_tp)
        
        self.number_of_sources = 4
        if self.xml_parameters.number_of_sources:
            self.number_of_sources = self.xml_parameters.number_of_sources            
    
    def to_torch_tensor(self, array):
        return Variable(torch.from_numpy(array).type(self.tensor_scalar_type), requires_grad=False)

    def create_folders(self):
        self.atlas_output_path = join(self.output_dir, '1_bayesian_atlas_all_subjects')
        self.regression_output = join(self.output_dir, '2_longitudinal_subjects_geodesic_regressions')
        self.registration_output = join(self.output_dir, '3_atlas_registration_to_subjects')
        
        self.registration_subjects_paths = [join(self.registration_output, 
                                            'Registration__subject_'+ self.global_subject_ids[i])
                                            for i in range(len(self.global_dataset_filenames))]

        self.shooting_output = join(self.output_dir, '4_subjects_shootings_to_t0')

        self.shooted_subjects_paths = [join(self.shooting_output, 
                                        'Shooting__subject_'+ self.global_subject_ids[i])
                                        for i in range(len(self.global_dataset_filenames))]
        
        self.atlas_output_path_2 = join(self.output_dir, '5_bayesian_atlas_all_subjects')

        self.age_error_output = join(self.output_dir, '6_age_correction')
        
        self.age_error_output_subjects = [join(self.age_error_output, 
                                            'Registrations__subject_'+ self.global_subject_ids[i])
                                            for i in range(len(self.global_dataset_filenames))]
        self.longitudinal_atlas_output_path = join(self.output_dir,'7_longitudinal_atlas_with_gradient_ascent')

                
        for path in [self.atlas_output_path, self.regression_output, self.registration_output, \
                    self.shooting_output, self.atlas_output_path_2, self.age_error_output, 
                    self.longitudinal_atlas_output_path]\
                    + self.registration_subjects_paths + self.shooted_subjects_paths:
            #if os.path.isdir(path): 
            #    shutil.rmtree(path)
            if not os.path.isdir(path):
                os.mkdir(path)

    def initialize_outputs(self):
        self.global_initial_cp_path = None
        self.global_initial_momenta_path = None
        self.global_initial_template_path = None
        self.global_initial_mod_matrix_path = None
        self.global_initial_sources_path = None
        self.global_initial_onset_ages_path = None 
        self.global_initial_accelerations_path = None 

        self.global_initial_noise_std_string = None
        self.global_initial_reference_time = None
        self.global_time_shift_std = 0
        self.global_acceleration_std = 0
    
    def insert_xml(self):
        model_xml_0 = et.parse(self.model_xml_path).getroot()
        self.model_xml_path = join(os.path.dirname(self.output_dir), 'initialized_model.xml')

        if self.global_initial_template_path:
            model_xml_0 = insert_model_xml_template(model_xml_0, 'filename', self.global_initial_template_path)
        if self.global_initial_noise_std_string:
            model_xml_0 = insert_model_xml_template(model_xml_0, 'noise-std', self.global_initial_noise_std_string)

        names = ['initial-control-points', 'initial-momenta', 'initial-onset-ages', 'initial-accelerations'\
                'initial-modulation-matrix', 'initial-sources']
        paths = [self.global_initial_cp_path, self.global_initial_momenta_path, self.global_initial_onset_ages_path,
                self.global_initial_accelerations_path, self.global_initial_mod_matrix_path, self.global_initial_sources_path]
        
        for variable, path in zip(names, paths):
            if path:
                model_xml_0 = insert_model_xml_level1_entry(model_xml_0, variable, path)
        
        if self.global_time_shift_std:
            model_xml_0 = insert_model_xml_level1_entry(model_xml_0, 'initial-time-shift-std', '%.4f' % self.global_time_shift_std)
        if self.global_acceleration_std:
            model_xml_0 = insert_model_xml_level1_entry(model_xml_0, 'initial-acceleration-std', '%.4f' % self.global_acceleration_std)
        if self.global_initial_reference_time:
            model_xml_0 = insert_model_xml_deformation_parameters(model_xml_0, 't0', '%.4f' % self.global_initial_reference_time)        

        # save the xml file
        doc = parseString((et.tostring(model_xml_0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
        np.savetxt(self.model_xml_path, [doc], fmt='%s')

    def mean_img(self, files):
        if ".nii" in files[0][0][self.objects_name[0]]:
            self.template_for_atlas = op.join(self.atlas_output_path, "mean_image.nii")
            data_list = [nib.load(f[0]["img"]).get_fdata() for f in files]
            mean = np.zeros((data_list[0].shape))
            for f in data_list:
                mean += f/len(data_list)
            image_new = nib.nifti1.Nifti1Image(mean, nib.load(files[0][0]["img"]).affine, nib.load(files[0][0]["img"]).header)
            nib.save(image_new, self.template_for_atlas)

        elif ".png" in files[0][0][self.objects_name[0]]:
            self.template_for_atlas = op.join(self.atlas_output_path, "mean_image.png")

            file_list = [np.array(Image.open(f[0])) for f in files]
            mean = np.zeros((file_list[0].shape))
            for f in file_list:
                mean += f/len(file_list)

            imsave(self.template_for_atlas, mean)  

    def define_prefix(self):
        ba = "BayesianAtlas__EstimatedParameters__"
        if self.global_subjects_nb > 30:
            ba = "DeterministicAtlas__EstimatedParameters__"
        return ba

    def define_atlas_outputs(self):
        ba = self.define_prefix()
        self.estimated_momenta_path = join(self.atlas_output_path, ba + 'Momenta.txt') #needed for ICA!!
        self.estimated_cp_path = join(self.atlas_output_path, ba + 'ControlPoints.txt')
        self.estimated_template_path = [join(self.atlas_output_path, '{}Template_{}{}'.format(ba, self.objects_name[0], self.objects_ext[0]))]
                
        # Path to where we will store the initialized parameters/variables
        fi = 'ForInitialization__'
        self.global_initial_template_path = [join(self.path_to_initialization, 
                                            '{}Template_{}__FromAtlas_AllSubjects{}'.format(fi, self.objects_name[0], self.objects_ext[0]))]   
        self.global_initial_cp_path = join(self.path_to_initialization, '{}ControlPoints__FromAtlas.txt'.format(fi))
        self.global_subjects_momenta_path = join(self.path_to_initialization, '{}Momenta__FromAtlas.txt'.format(fi))

    def set_xml_atlas(self, xml_parameters, all = True):
        
        xml_parameters.deformation_kernel_width = 10
    
        if self.global_subjects_nb > 30:
            xml_parameters.optimization_method_type = 'StochasticGradientAscent'.lower()
            estimate_atlas = estimate_deterministic_atlas
            xml_parameters.dataset_filenames = []
            xml_parameters.subject_ids = []
            xml_parameters.visit_ages = []

            if not all:
                for i in range(self.global_subjects_nb):    
                    if np.abs(self.global_t0 - self.global_visit_ages[i][0]) < 3:
                        xml_parameters.dataset_filenames.append(self.global_dataset_filenames[i])
                        xml_parameters.subject_ids.append(self.global_subject_ids[i])
                        xml_parameters.visit_ages.append([self.global_t0])
            else:
                xml_parameters.dataset_filenames = [[{self.objects_name[0] : sujet[0]}] for sujet in self.shooted_subjects_for_atlas]
                xml_parameters.visit_ages = [[self.global_t0]] * self.global_subjects_nb
                xml_parameters.subject_ids = self.global_subject_ids
        else:
            xml_parameters.optimization_method_type = 'GradientAscent'.lower()
            xml_parameters.visit_ages = [[self.global_t0]] * self.global_subjects_nb
            xml_parameters.subject_ids = self.global_subject_ids
            xml_parameters.dataset_filenames = self.global_dataset_filenames
            estimate_atlas = estimate_bayesian_atlas
        
        return xml_parameters, estimate_atlas


    def compute_atlas_all_subjects(self):
        """
        1]. Compute Bayesian atlas with Gradient Ascent 
        """
        logger.info('\n[ estimate an atlas from longitudinal subjects ] \n')

        self.define_atlas_outputs()

        if not op.exists(self.global_initial_template_path[0]):
            xml_parameters = copy.deepcopy(self.xml_parameters)

            xml_parameters, estimate_atlas = self.set_xml_atlas(xml_parameters, all = False)
            
            # Set the template image
            self.mean_img(xml_parameters.dataset_filenames)
            xml_parameters.template_specifications["img"]['filename'] = self.template_for_atlas
            
            # Launch and save the outputted noise standard deviation, for later use ----------------------------------------
            self.global_deformetrica.output_dir = self.atlas_output_path
            model, _ = estimate_atlas(self.global_deformetrica, xml_parameters)

            # Save the template
            shutil.copyfile(self.estimated_template_path[0], self.global_initial_template_path[0])
            np.savetxt(self.global_initial_cp_path, model.get_control_points())
        
            self.global_initial_template = model.template #deformable multi object
            self.global_initial_template_data = model.get_template_data() #intensities
        
        self.global_atlas_momenta = read_3D_array(self.estimated_momenta_path)
        self.global_initial_cp = read_3D_array(self.global_initial_cp_path)
        self.global_cp_nb = self.global_initial_cp.shape[0]

        # Modify and write the model.xml file accordingly.
        self.insert_xml()
    
    def define_atlas_outputs_2(self):
        ba = self.define_prefix()
        self.estimated_momenta_path = join(self.atlas_output_path_2, ba + 'Momenta.txt') #needed for ICA!!
        self.estimated_cp_path = join(self.atlas_output_path_2, ba + 'ControlPoints.txt')
        self.estimated_template_path = [join(self.atlas_output_path_2, '{}Template_{}{}'.format(ba, self.objects_name[0], self.objects_ext[0]))]
                
        # Path to where we will store the initialized parameters/variables
        fi = 'ForInitialization__'
        self.global_initial_template_path_2 = [join(self.path_to_initialization, 
                                            '{}Template_{}__FromAtlas_AllSubjects_2{}'.format(fi, self.objects_name[0], self.objects_ext[0]))]
        self.global_subjects_momenta_path = join(self.path_to_initialization, '{}Momenta__FromAtlas_2.txt'.format(fi))   

    def compute_atlas_all_subjects_2(self):
        """
        1]. Compute Bayesian atlas with Gradient Ascent This is NEEDED for tangent space ICA...
        """
        logger.info('\n[ estimate an atlas from longitudinal subjects 2 ] \n')

        self.define_atlas_outputs_2()

        if not op.exists(self.global_initial_template_path_2[0]):
            xml_parameters = copy.deepcopy(self.xml_parameters)
            xml_parameters, estimate_atlas = self.set_xml_atlas(xml_parameters)
            xml_parameters.print_every_n_iters = 10

            # Set the template image as the former atlas
            xml_parameters.template_specifications["img"]['filename'] = self.global_initial_template_path[0]
            
            # Launch and save the outputted noise standard deviation, for later use ----------------------------------------
            self.global_deformetrica.output_dir = self.atlas_output_path_2
            model, _ = estimate_atlas(self.global_deformetrica, xml_parameters)

            # Save the template
            shutil.copyfile(self.estimated_template_path[0], self.global_initial_template_path_2[0])
            np.savetxt(self.global_initial_cp_path, model.get_control_points())
            np.savetxt(self.global_subjects_momenta_path_2, model.get_momenta())
        
            self.global_initial_template = model.template #deformable multi object
            self.global_initial_template_data = model.get_template_data() #intensities
        
        self.global_atlas_momenta = read_3D_array(self.estimated_momenta_path)

        # Modify and write the model.xml file accordingly.
        self.insert_xml()

    def define_regression_outputs(self):
        self.regression_momenta_path = join(self.regression_output, 'Regression_EstimatedParameters__Momenta.txt')
        self.regression_cp_path = join(self.atlas_output_path, 'Regression_EstimatedParameters__ControlPoints.txt')

        fi = 'ForInitialization__'
        self.global_initial_momenta_path = join(self.path_to_initialization, '{}Momenta_from_regression.txt'.format(fi))
    
    def compute_geodesic_regression(self):
        """
        2]. Compute individual geodesic regression
        """
        self.define_regression_outputs()

        logger.info('\n [ Geodesic regression on all subjects ]')

        if not op.exists(self.global_initial_momenta_path):
            xml_parameters = copy.deepcopy(self.xml_parameters)

            xml_parameters.deformation_kernel_width = self.global_kernel_width

            # Read the current model xml parameters.            
            xml_parameters = set_xml_for_regression(xml_parameters)

            # set the  template as the Bayesian template
            xml_parameters.template_specifications["img"]['filename'] = self.global_initial_template_path[0]
            xml_parameters.initial_control_points = self.global_initial_cp_path

            # Adapt the specific xml parameters and update
            #join filenames as if from a single subject = [[{obs 1}] [{obs2}]] -> [[{obs 1} {obs2}]]
            xml_parameters.dataset_filenames = [sum(self.global_dataset_filenames, [])]
            xml_parameters.visit_ages = [sum(self.global_visit_ages, [])]
            xml_parameters.subject_ids = [self.global_subject_ids[0]]
            xml_parameters.t0 = self.global_t0   

            if self.global_subjects_nb > 30:
                xml_parameters.optimization_method_type = "StochasticGradientAscent".lower()         

            self.global_deformetrica.output_dir = self.regression_output
            model = estimate_geodesic_regression(self.global_deformetrica, xml_parameters)

            # Save results                
            np.savetxt(self.regression_momenta_path, model.get_momenta())
            np.savetxt(self.regression_cp_path, model.get_control_points())
            np.savetxt(self.global_initial_momenta_path, model.get_momenta())
        
        self.global_initial_momenta = read_3D_array(self.global_initial_momenta_path)
    
    def define_registration_outputs(self):
        self.template_shot_to_subjects = []
        self.momenta_shot_to_subjects = []
        self.registration_momenta = []

        templates = [f for f in os.listdir(self.regression_output) if self.objects_ext[0] in f and "Flow" in f]

        accepted_difference = (1/self.concentration_of_tp)/2+0.01
        for i in range(len(self.global_subject_ids)):
            self.registration_momenta.append(join(self.registration_subjects_paths[i],"DeterministicAtlas__EstimatedParameters__Momenta.txt")) 
            age = round(self.global_visit_ages[i][0], 2)
            # Get the template closest to subject age
            for f in templates:
                template_age = round(float(f.split("age_")[-1].split(self.objects_ext[0])[0]), 2)
                if np.abs(template_age - age)<=accepted_difference:
                    self.template_shot_to_subjects.append(join(self.regression_output, f))
                    self.momenta_shot_to_subjects.append(join(self.regression_output, f.replace("img", "Momenta").replace(self.objects_ext[0], ".txt")))
                    break
            # t = "%.2f" % round(self.global_visit_ages[i][0], 2)
            # max_time = max(self.global_t0, self.global_visit_ages[i][0])
            # min_time = min(self.global_t0, self.global_visit_ages[i][0])
            # time = round((max_time - min_time) * 1)
            
    def define_shooting_outputs(self):
        gs = 'Shooting__GeodesicFlow__'
        a = "Subject_shot_for_atlas_"

        # Momenta from PT
        gr = 'GeodesicRegression__'
        self.transported_regression_momenta_path = []
        self.shooted_subjects_files = []
        self.shooted_subjects_for_atlas = []

        for i in range(len(self.global_subject_ids)):
            time = max(0, round((self.global_t0 - self.global_visit_ages[i][0]) * self.concentration_of_tp)) #self.concentration_of_tp)
            
            # Output files
            self.transported_regression_momenta_path.append(join(self.registration_subjects_paths[i], 
                                                             'Transported_Momenta_tp_5__age_1.00.txt'))
            #self.transported_regression_momenta_path.append(join(self.shooted_subjects_paths[i], 
            #                                                 '%ssubject_%sEstimatedParameters__TransportedMomenta.txt'%(gr, self.global_subject_ids[i])))
            self.shooted_subjects_files.append([join(self.shooted_subjects_paths[i], 
                                    "{}{}__tp_{}__age_{}{}".format(gs, self.objects_name[0], time, self.global_t0, self.objects_ext[0]))])
            
            # Where to copy output files
            self.shooted_subjects_for_atlas.append([join(self.atlas_output_path_2, "{}{}_{}__tp_{}__age_{}{}"\
                                    .format(a, self.global_subject_ids[i], self.objects_name[0], time, self.global_t0, self.objects_ext[0]))])
    
    def set_registration_xml(self, i, xml_parameters):
        xml_parameters.dataset_filenames = [self.global_dataset_filenames[i]]
        xml_parameters.visit_ages = [self.global_visit_ages[i]]
        xml_parameters.subject_ids = [self.global_subject_ids[i]]
        xml_parameters.template_specifications["img"]['filename'] = self.template_shot_to_subjects[i]
        xml_parameters.print_every_n_iters = 50
        self.global_deformetrica.output_dir = self.registration_subjects_paths[i]
        

        return xml_parameters

    def compute_shootings_to_t0(self):
        """
        3]. Shoot subjects to t0 along the geodesic regression curve
        """
        self.define_registration_outputs()
        self.define_shooting_outputs()
        xml_parameters = copy.deepcopy(self.xml_parameters)

        for i in range(len(self.global_subject_ids)):
            if not op.exists(self.shooted_subjects_for_atlas[i][0]):
                logger.info('\nRegister shooted template to subject {} of age {}'.format(self.global_subject_ids[i], self.global_visit_ages[i][0]))

                # Set the target (subject) and the source (template at subject age)
                xml_parameters = self.set_registration_xml(i, xml_parameters)
                
                estimate_registration(self.global_deformetrica, xml_parameters)

                logger.info('\nParallel transport regression momenta to subject space')
                                
                # set tmin to 0 and tmax to 1
                # if shooting to True, equivalent to shoot template to subject age + 1 (parallel curve at t0)
                # and shoot subject to subject age + 1 using transported momenta
                
                compute_parallel_transport(xml_parameters.template_specifications, self.dimension, 
                                            self.tensor_scalar_type, self.global_kernel_type, self.global_kernel_width,
                                            None, self.regression_cp_path, self.registration_momenta[i],
                                            self.regression_cp_path, self.momenta_shot_to_subjects[i],
                                            tmin=0, tmax=1, t0 = 0, dense_mode=self.dense_mode,
                                            concentration_of_time_points=self.concentration_of_tp,
                                            number_of_time_points=self.global_nb_of_tp,
                                            output_dir=self.registration_subjects_paths[i], perform_shooting = False)
                
                logger.info("Shoot subject{} to t0".format(self.global_subject_ids[i]))
                xml_parameters.template_specifications["img"]['filename'] = self.global_dataset_filenames[i][0]["img"]

                compute_shooting(xml_parameters.template_specifications, dimension=self.dimension,
                                deformation_kernel_width=self.global_kernel_width,
                                initial_control_points=self.regression_cp_path, 
                                initial_momenta=self.transported_regression_momenta_path[i], 
                                concentration_of_time_points=self.concentration_of_tp, t0=self.global_visit_ages[i][0], 
                                tmin=min([self.global_t0, self.global_visit_ages[i][0]]), 
                                tmax=max([self.global_t0, self.global_visit_ages[i][0]]), dense_mode=self.dense_mode,
                                output_dir=self.shooted_subjects_paths[i], write_adjoint_parameters = False)  

                # Copy shooting outputs to the atlas folder
                shutil.copyfile(self.shooted_subjects_files[i][0], self.shooted_subjects_for_atlas[i][0])

        self.insert_xml()
    
    
    
    def define_age_error_outputs(self):
        self.ages_corrected_path = join(self.path_to_initialization, "Global_visit_ages_corrected.txt")

        self.neighboring_templates = []
        templates = [f for f in os.listdir(self.regression_output) if self.objects_ext[0] in f and "Flow" in f]

        for i in range(len(self.global_subject_ids)):            
            age = round(self.global_visit_ages[i][0], 1)
            age_moins_1 = age - 0.5
            age_plus_1 = age + 0.5
            
            # Get the template closest to subject age
            neighboring_templates = []
            for a in [age_moins_1, age, age_plus_1]:
                for f in templates:
                    template_age = round(float(f.split("age_")[-1].split(self.objects_ext[0])[0]), 1)
                    if template_age == a:
                        neighboring_templates.append(join(self.regression_output, f))
            self.neighboring_templates.append(neighboring_templates)

    def compute_age_error(self):
        self.define_age_error_outputs()

        logger.info('\n[ Compute age errors for subjects with registration ] \n')

        self.global_visit_ages_corrected = []

        if not op.exists(self.ages_corrected_path):
            for i in range(len(self.global_subject_ids)):                
                subject_array = nib.load(self.global_dataset_filenames[i][0]["img"]).get_fdata()
                ssd = []
                for template in self.neighboring_templates[i]:
                    template_array = nib.load(template).get_fdata()
                    ssd.append(np.sum((subject_array - template_array)**2))

                corrected_age = self.global_visit_ages[i][0]
                min_index = ssd.index(np.min(ssd)) # 0 1 2
                if min_index == 0:
                    corrected_age = self.global_visit_ages[i][0] - 0.5
                elif min_index == 2:
                    corrected_age = self.global_visit_ages[i][0] + 0.5
                
                self.global_visit_ages_corrected.append([corrected_age-self.global_visit_ages[i][0]])
            np.savetxt(self.ages_corrected_path, np.array(self.global_visit_ages_corrected))
        
        self.global_visit_ages_corrected = read_2D_array(self.ages_corrected_path)
                        

    def define_ica_outputs(self):
        fi = 'ForInitialization__'
        self.global_initial_mod_matrix_path = join(self.path_to_initialization, '{}ModulationMatrix__FromICA.txt'.format(fi))
        self.global_initial_sources_path = join(self.path_to_initialization, '{}Sources__FromICA.txt'.format(fi))

    def tangent_space_ica(self):
        """
        5]. Tangent-space ICA on the individual momenta outputted by the atlas estimation.
        ----------------------------------------------------------------------------------
            Those momenta are first projected on the space orthogonal to the initial (longitudinal) momenta.
            Skipped if initial control points and modulation matrix were specified.
        """
        self.define_ica_outputs()

        logger.info('\n[ tangent-space ICA on the projected individual momenta ]')

        if not op.exists(self.global_initial_sources_path):

            # Compute RKHS matrix.
            K = compute_RKHS_matrix(self.global_cp_nb, self.dimension, self.global_kernel_width, self.global_initial_cp)

            # Project.
            Km = np.dot(K, self.global_initial_momenta.ravel())
            mKm = np.dot(self.global_initial_momenta.ravel().transpose(), Km)

            w = []
            for i in range(self.global_atlas_momenta.shape[0]):
                w.append(self.global_atlas_momenta[i].ravel() - np.dot(self.global_atlas_momenta[i].ravel(), Km) / mKm * self.global_initial_momenta.ravel())
            w = np.array(w)

            number_of_sources = 4
            ica = FastICA(n_components=number_of_sources, max_iter=50000)
            global_initial_sources = ica.fit_transform(w)
            global_initial_mod_matrix = ica.mixing_

            # Rescale.
            for s in range(number_of_sources):
                std = np.std(global_initial_sources[:, s])
                global_initial_sources[:, s] /= std
                global_initial_mod_matrix[:, s] *= std

            # Print.
            residuals = []
            for i in range(self.global_subjects_nb):
                residuals.append(w[i] - np.dot(global_initial_mod_matrix, global_initial_sources[i]))
            mean_relative_residual = np.mean(np.absolute(np.array(residuals))) / np.mean(np.absolute(w))
            logger.info('>> Mean relative residual: %.3f %%.' % (100 * mean_relative_residual))

            np.savetxt(self.global_initial_mod_matrix_path, global_initial_mod_matrix)
            np.savetxt(self.global_initial_sources_path, global_initial_sources)

            self.insert_xml()

            logger.info('>> Estimated random effect statistics:')
            logger.info('\t\t sources =\t%.3f\t[ mean ]\t+/-\t%.4f\t[std]' %
                        (np.mean(global_initial_sources), np.std(global_initial_sources)))

    def define_longitudinal_atlas_outputs(self):
        la = "LongitudinalAtlas__EstimatedParameters__"
        fa = "LongitudinalAtlas__FrozenParameters__"
        
        self.estimated_cp_path = join(self.longitudinal_atlas_output_path, '{}ControlPoints.txt'.format(la))
        self.estimated_momenta_path = join(self.longitudinal_atlas_output_path, '{}Momenta.txt'.format(la))
        self.estimated_mod_matrix_path = join(self.longitudinal_atlas_output_path, '{}ModulationMatrix.txt'.format(la))
        self.estimated_onset_ages_path = join(self.longitudinal_atlas_output_path, '{}OnsetAges.txt'.format(fa))
        self.estimated_sources_path = join(self.longitudinal_atlas_output_path,'{}Sources.txt'.format(la))
        
        time = round((self.global_tmax - self.global_tmin) * self.concentration_of_tp)
        #for (object_name, ext) in zip(self.objects_name, self.objects_ext):
        self.estimated_template_path = [join(self.longitudinal_atlas_output_path, '{}Template_{}__tp_{}__age_{}{}' \
                                                .format(la, self.objects_name[0], time, self.global_t0, self.objects_ext[0]))]

        # Path to where we will store the initialized parameters/variables
        fi = 'ForInitialization__'
        self.global_initial_cp_path_2 = join(self.path_to_initialization, '{}ControlPoints__FromLongitudinalAtlas.txt'.format(fi))
        self.global_initial_momenta_path_2 = join(self.path_to_initialization, '{}Momenta__FromLongitudinalAtlas.txt'.format(fi))
        self.global_initial_mod_matrix_path_2 = join(self.path_to_initialization, '{}ModulationMatrix__FromLongitudinalAtlas.txt'.format(fi))
        self.global_initial_onset_ages_path_2 = join(self.path_to_initialization, '{}OnsetAges__FromLongitudinalAtlas.txt'.format(fi))
        self.global_initial_sources_path_2 = join(self.path_to_initialization, '{}Sources__FromLongitudinalAtlas.txt'.format(fi))
        self.global_initial_template_path_3 = [join(self.path_to_initialization,
                                    '{}Template_{}__FromLongitudinalAtlas{}'.format(fi, self.objects_name[0], self.objects_ext[0]))]
        
    
    def longitudinal_atlas(self):
        self.define_longitudinal_atlas_outputs()

        logger.info('[ longitudinal atlas estimation with the GradientAscent optimizer ]')

        if not op.exists(self.global_initial_template_path_3[0]):
            xml_parameters = copy.deepcopy(self.xml_parameters)
                        
            xml_parameters.print_every_n_iters = 1
            xml_parameters.optimized_log_likelihood = 'class2'.lower()
            xml_parameters.template_specifications["img"]['filename'] = self.global_initial_template_path_2[0]
            xml_parameters.initial_momenta = self.global_initial_momenta_path
            xml_parameters.initial_control_points = self.global_initial_cp_path
            xml_parameters.initial_modulation_matrix = self.global_initial_mod_matrix_path
            xml_parameters.initial_sources = self.global_initial_sources_path
            xml_parameters.number_of_sources = self.number_of_sources

            self.global_deformetrica.output_dir = self.longitudinal_atlas_output_path
            estimate_longitudinal_atlas(self.global_deformetrica, xml_parameters)

            shutil.copyfile(self.estimated_cp_path, self.global_initial_cp_path_2)
            shutil.copyfile(self.estimated_momenta_path, self.global_initial_momenta_path_2)
            shutil.copyfile(self.estimated_mod_matrix_path, self.global_initial_mod_matrix_path_2)
            shutil.copyfile(self.estimated_onset_ages_path, self.global_initial_onset_ages_path_2)
            shutil.copyfile(self.estimated_sources_path, self.global_initial_sources_path_2)
            shutil.copyfile(self.estimated_template_path[0], self.global_initial_template_path_3[0])

            self.global_initial_cp_path_2 = self.global_initial_cp_path_2
            self.global_initial_momenta_path = self.global_initial_momenta_path_2
            self.global_initial_mod_matrix_path = self.global_initial_mod_matrix_path_2
            self.global_initial_onset_ages_path = self.global_initial_onset_ages_path_2
            self.global_initial_sources_path = self.global_initial_sources_path_2
            self.global_initial_template_path = self.global_initial_template_path_3
            self.insert_xml()



