import os
import sys
import copy
import os.path as op
from os.path import join
import nibabel as nib
from imageio import imsave, imread
from PIL import Image

from ..launch.compute_shooting import compute_shooting
from ..launch.compute_parallel_transport import launch_parallel_transport
from torch.autograd import Variable

import warnings
from decimal import Decimal
import shutil
import math
from sklearn.decomposition import PCA, FastICA
import torch
import xml.etree.ElementTree as et
from xml.dom.minidom import parseString
from scipy.stats import norm, truncnorm
from ..core import default
from ..in_out.xml_parameters import XmlParameters, get_dataset_specifications, get_estimator_options, get_model_options
from ..in_out.dataset_functions import template_metadata, create_dataset
from ..core.model_tools.deformations.exponential import Exponential
from ..core.model_tools.deformations.geodesic import Geodesic
from ..in_out.array_readers_and_writers import *
from ..support import kernels as kernel_factory
from ..core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ..in_out.deformable_object_reader import ObjectReader
from ..api.deformetrica import Deformetrica
from .deformetrica_functions import *
from ..support import utilities
from .tools import *
from . initialize_longitudinal_atlas_development_simple import CrossSectionalLongitudinalAtlasInitializer
from .initialize_piecewise_geodesic_regression_with_space_shift import BayesianRegressionInitializer

import warnings
warnings.filterwarnings("ignore")

class LongitudinalAtlasInitializer():
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

        self.xml_parameters.max_iterations = 100 ## TODO 
        self.xml_parameters.max_line_search_iterations = 10
        self.xml_parameters.print_every_n_iters = 10

        self.global_deformetrica = Deformetrica(output_dir=self.output_dir)

        # Save some global parameters.
        self.dimension = self.xml_parameters.dimension

        #global_dataset_filenames:list of lists: 1 list per subject of dict (1 dic = 1 obs)
        self.global_dataset_filenames = self.xml_parameters.dataset_filenames
        self.global_visit_ages = self.xml_parameters.visit_ages #list of lists: 1 list of visit ages/subject
        self.global_subject_ids = self.xml_parameters.subject_ids #list of ids
        self.objects_name, self.objects_ext = template_metadata(self.xml_parameters.template_specifications)[1:3]
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
        self.dataset_type = "mixed" if self.nb_of_longitudinal_subjects > 0 else "single_points"

        # Deformation parameters
        self.global_kernel_width = self.xml_parameters.deformation_kernel_width
        self.global_kernel_device = self.xml_parameters.deformation_kernel_device
        self.kernel = kernel_factory.factory(kernel_width=self.global_kernel_width)
        
        # Times
        self.concentration_of_tp = self.xml_parameters.time_concentration
        self.global_t0 = self.xml_parameters.t0
        self.global_tmin = np.mean([e[0] for e in self.global_visit_ages])
        self.global_tmax = np.max([e[-1] for e in self.global_visit_ages])
        self.global_nb_of_tp = self.xml_parameters.n_time_points
        self.geodesic_nb_of_tp = int(1 + (self.global_t0 - self.global_tmin) * self.concentration_of_tp)
        
        self.number_of_sources = 4
        if self.xml_parameters.number_of_sources:
            self.number_of_sources = self.xml_parameters.number_of_sources            
  
    def to_torch_tensor(self, array):
        return Variable(utilities.move_data(array), requires_grad=False)

    def create_folders(self):
        self.regressions_output = join(self.output_dir, '1_longitudinal_subjects_geodesic_regressions')
        self.longitudinal_shooting_output = join(self.output_dir, '2_longitudinal_subjects_shootings_to_t0')
        self.atlas_output_path = join(self.output_dir, '3_atlas_on_longitudinal_data')
        self.atlas_trajectory_output = join(self.output_dir, '4_longitudinal_atlas_trajectory')
        self.atlas_trajectory_output_2 = join(self.output_dir, '4_longitudinal_atlas_trajectory_2')
        self.single_registration_output = join(self.output_dir, '5_single_subjects_registration_to_atlas')
        self.single_shooting_output = join(self.output_dir, '6_single_subjects_shooting_to_t0')
        self.atlas_output_path_2 = join(self.output_dir, '7_atlas_on_all_data')
        
        self.registration_output_path = join(self.output_dir, '_longitudinal_registration')
        self.longitudinal_atlas_output = join(self.output_dir, '_longitudinal_atlas_with_gradient_ascent')

        self.longitudinal_shooted_subjects_paths = [join(self.longitudinal_shooting_output, 
                                                    'Shooting__subject_'+ self.global_subject_ids[i])
                                                    for i in self.longitudinal_subjects_ind]
        self.single_registration_subjects_paths = [join(self.single_registration_output, 
                                                'Registration__subject_'+ self.global_subject_ids[i])
                                                for i in self.single_subjects_ind]

        self.single_shooted_subjects_paths = [join(self.single_shooting_output, 
                                                'Shooting__subject_'+ self.global_subject_ids[i])
                                                for i in self.single_subjects_ind]
        
        for path in [self.atlas_output_path, self.atlas_trajectory_output, self.atlas_trajectory_output_2, self.regressions_output,
                    self.longitudinal_shooting_output, self.single_registration_output, self.single_shooting_output,
                    self.registration_output_path, self.longitudinal_atlas_output, self.atlas_output_path_2]\
                    + self.longitudinal_shooted_subjects_paths + self.single_shooted_subjects_paths +\
                    self.single_registration_subjects_paths:
            if os.path.isdir(path): 
                shutil.rmtree(path)
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

    def define_regression_outputs(self):
        gr = 'GeodesicRegression__'
        self.regression_momenta_path = [join(self.regressions_output, 
                                        '%ssubject_%s_EstimatedParameters__Momenta.txt'%(gr, self.global_subject_ids[i]))\
                                        for i in self.longitudinal_subjects_ind]
        self.regression_cp_path = [join(self.atlas_output_path, 
                                    '%ssubject_%s_EstimatedParameters__ControlPoints.txt'%(gr, self.global_subject_ids[i]))\
                                    for i in self.longitudinal_subjects_ind]
    
    def get_regression_outputs(self):
        # Get subject at atlas age
        self.shooted_subjects_for_atlas = []
        self.shooted_subjects_for_atlas_2 = []
        self.shooted_subjects_regression_momenta = []
        
        for i, ind in enumerate(self.single_subjects_ind):
            regression_output = self.regressions_output
            files = [f for f in os.listdir(regression_output) if ".nii" in f]            
            for f in files:
                file_age = float(f.split("age_")[-1].split(".nii")[0])
                if file_age == round(self.global_t0, 1):
                    self.shooted_subjects_for_atlas.append([join(self.atlas_output_path, f)])
                    self.shooted_subjects_for_atlas_2.append([join(self.atlas_output_path_2, f)])
                    shutil.copyfile(join(regression_output, f), self.shooted_subjects_for_atlas[i][0])
                    shutil.copyfile(join(regression_output, f.replace("img", "Momenta").replace("nii", "txt")), self.shooted_subjects_for_atlas_2[i][0])

    def define_longitudinal_shooting_outputs(self):
        gs = 'Shooting__GeodesicFlow__'
        a = "Subject_shot_for_atlas_"
        m = "Momenta_shot_for_atlas_subject_"
        self.shooted_subjects_regression_momenta, self.shooted_momenta_for_atlas = [], []
        self.shooted_subjects_files = []
        self.shooted_subjects_for_atlas, self.shooted_subjects_for_atlas_2 = [], []

        for i, ind in enumerate(self.longitudinal_subjects_ind):
            time = max(0, round((self.global_t0 - self.global_visit_ages[ind][0]) * 1)) #self.concentration_of_tp)
            
            # Output files
            self.shooted_subjects_regression_momenta.append(join(self.longitudinal_shooted_subjects_paths[i], 
                                    "{}Momenta__tp_{}__age_{}.txt".format(gs, time, self.global_t0))) 
            self.shooted_subjects_files.append([join(self.longitudinal_shooted_subjects_paths[i], 
                                    "{}{}__tp_{}__age_{}{}".format(gs, self.objects_name[0], time, self.global_t0, self.objects_ext[0]))])
            
            # Where to copy output files
            self.shooted_momenta_for_atlas.append(join(self.atlas_trajectory_output, "{}_{}__tp_{}__age_{}.txt"\
                                                .format(m, self.global_subject_ids[ind], time, self.global_t0)))
            self.shooted_subjects_for_atlas.append([join(self.atlas_output_path, "{}{}_{}__tp_{}__age_{}{}"\
                                    .format(a, self.global_subject_ids[ind], self.objects_name[0], time, self.global_t0, self.objects_ext[0]))])
            self.shooted_subjects_for_atlas_2.append([join(self.atlas_output_path_2, "{}{}_{}__tp_{}__age_{}{}"\
                                    .format(a, self.global_subject_ids[ind], self.objects_name[0], time, self.global_t0, self.objects_ext[0]))])

    def compute_longitudinal_geodesic_regressions(self):
        """
        1]. Compute individual geodesic regressions for longitudinal observations
            that guide brain growth. 
        2] Shoot the longitudinal subjects (and their momenta to t0)
        """
        self.define_regression_outputs()
        self.define_longitudinal_shooting_outputs()

        # Read the current model xml parameters.
        xml_parameters = copy.deepcopy(self.xml_parameters)
        xml_parameters.time_concentration = 1
    
        for i, ind in enumerate(self.longitudinal_subjects_ind):
            logger.info('\n Geodesic regression for subject ' + self.global_subject_ids[ind] + '\n')
            
            xml_parameters.template_specifications["img"]['filename'] = self.global_dataset_filenames[ind][0]["img"]
            
            self.regression_control_points, regression_momenta = estimate_subject_geodesic_regression(
                ind, self.global_deformetrica, xml_parameters, self.regressions_output,
                self.global_dataset_filenames, self.global_visit_ages, self.global_subject_ids,
                t0 = self.global_visit_ages[ind][0])
                                
            np.savetxt(self.regression_momenta_path[i], regression_momenta)
            np.savetxt(self.regression_cp_path[i], self.regression_control_points) 

            logger.info('\nShoot subject {} from {} to {}'.format(self.global_subject_ids[ind], self.global_visit_ages[ind][0], self.global_t0))  

            compute_shooting(xml_parameters.template_specifications, dimension=self.dimension,
                            deformation_kernel_width=self.global_kernel_width,
                            initial_cp=self.regression_cp_path[i], initial_momenta=self.regression_momenta_path[i], 
                            time_concentration=1,
                            t0=self.global_visit_ages[ind][0], tmin=min([self.global_t0, self.global_visit_ages[ind][0]]), 
                            tmax=max([self.global_t0, self.global_visit_ages[ind][0]]),
                            output_dir=self.longitudinal_shooted_subjects_paths[i])  

            # Copy shooting outputs to the atlas folder
            shutil.copyfile(self.shooted_subjects_files[i][0], self.shooted_subjects_for_atlas[i][0])
            shutil.copyfile(self.shooted_subjects_files[i][0], self.shooted_subjects_for_atlas_2[i][0])
            shutil.copyfile(self.shooted_subjects_regression_momenta[i], self.shooted_momenta_for_atlas[i])

        self.insert_xml()                

    def define_atlas_outputs(self):
        ba = "BayesianAtlas__EstimatedParameters__"
        self.estimated_momenta_path = join(self.atlas_output_path, ba + 'Momenta.txt') #needed for ICA!!
        self.estimated_template_path = [join(self.atlas_output_path, '{}Template_{}{}'.format(ba, self.objects_name[0], self.objects_ext[0]))]
                
        # Path to where we will store the initialized parameters/variables
        fi = 'ForInitialization__'
        self.global_initial_template_path = [join(self.path_to_initialization, 
                                            '{}Template_{}__FromAtlas_LongitudinalSubjects{}'.format(fi, self.objects_name[0], self.objects_ext[0]))]
    
    def mean_img(self, files):
        print(files[0])
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

    def compute_atlas_with_longitudinal_subjects(self):
        """
        3]. Compute Bayesian atlas with Gradient Ascent on longitudinal subjects 
        """
        logger.info('\n[ estimate an atlas from longitudinal subjects ] \n')

        self.define_atlas_outputs()
        xml_parameters = copy.deepcopy(self.xml_parameters)

        # Initialization -----------------------------------------------------------------------------------------------        
        # Retrieve longitudinal subjects at t0 from regression    
        xml_parameters.dataset_filenames = [[{self.objects_name[0]:obj_list[0]}]\
                                             for obj_list in self.shooted_subjects_for_atlas]
        xml_parameters.visit_ages = [[self.global_t0]] * self.nb_of_longitudinal_subjects
        xml_parameters.subject_ids = [self.global_subject_ids[i] for i in self.longitudinal_subjects_ind]
        
        # Set the template image
        self.mean_img(xml_parameters.dataset_filenames)
        xml_parameters.template_specifications["img"]['filename'] = self.template_for_atlas
        
        # Use the regression cp
        xml_parameters.initial_cp = self.regression_cp_path[-1]
        xml_parameters.max_iterations = 10 #TODO
        
        # Launch and save the outputted noise standard deviation, for later use ----------------------------------------
        self.global_deformetrica.output_dir = self.atlas_output_path
        model, self.global_atlas_momenta = estimate_bayesian_atlas(self.global_deformetrica, xml_parameters)
        
        self.global_initial_template = model.template #deformable multi object
        self.global_initial_template_data = model.get_template_data() #intensities
        self.global_initial_cp = model.get_control_points()
        self.global_cp_nb = self.global_initial_cp.shape[0]

        # Correct the template 

        # Save the template
        shutil.copyfile(self.estimated_template_path[0], self.global_initial_template_path[0])

        # Modify and write the model.xml file accordingly.
        self.insert_xml()
    
    def define_momenta_output(self):
        gr = 'GeodesicRegression__'
        self.transported_regression_momenta_path = [join(self.atlas_trajectory_output, 
                                                '%ssubject_%sEstimatedParameters__TransportedMomenta.txt'%(gr, self.global_subject_ids[i]))\
                                                for i in range(len(self.global_subject_ids))]
        self.global_initial_momenta_for_atlas = join(self.atlas_trajectory_output, "Global_average_momenta.txt")
                                         
        # Path to where we will store the initialized parameters/variables
        fi = 'ForInitialization__'
        self.global_initial_momenta_path = join(self.path_to_initialization, '{}AverageMomenta__FromLongitudinalRegressions.txt'.format(fi))
        

    def compute_average_trajectory(self):
        """
        4]. Compute average trajectory based on longitudinal subjects growth
        """

        logger.info('\nCompute average momenta\n')

        self.define_momenta_output()

        self.global_initial_momenta = torch.zeros((self.global_initial_cp.shape), device = "cuda")
        self.global_initial_momenta = np.zeros((self.global_initial_cp.shape))
        
        for j, i in enumerate(self.longitudinal_subjects_ind):
            logger.info('\nParallel transport regression momenta of subject {} to atlas space\n'.format(self.global_subject_ids[i]))

            regression_momenta = read_3D_array(self.shooted_subjects_regression_momenta[j])
            registration_to_atlas_momenta = self.global_atlas_momenta[j]

            # Parallel transport of the regression momenta to the template space
            _, transported_regression_momenta = parallel_transport(
                self.regression_control_points, regression_momenta, - registration_to_atlas_momenta, 
                self.global_kernel_width, self.global_kernel_device, 
                self.global_nb_of_tp)
            
            # Increment the global initial momenta.
            self.global_initial_momenta += transported_regression_momenta / float(self.nb_of_longitudinal_subjects)
            
            np.savetxt(self.transported_regression_momenta_path[i], transported_regression_momenta)

        np.savetxt(self.global_initial_momenta_path, self.global_initial_momenta)
        np.savetxt(self.global_initial_momenta_for_atlas, self.global_initial_momenta)

        logger.info('\nCompute average template trajectory\n')

        geodesic = Geodesic(self.kernel, t0=self.global_t0, time_concentration=10)

        geodesic.set_tmin(min([self.global_t0, self.global_tmin]))
        geodesic.set_tmax(max([self.global_t0, self.global_tmax]))

        control_points_t0 = self.to_torch_tensor(self.global_initial_cp)
        template_points_t0 = {k: self.to_torch_tensor(v) for k, v in self.global_initial_template.get_points().items()}            
        template_intensities_t0 = {k: self.to_torch_tensor(v) for k, v in self.global_initial_template_data.items()}
        global_initial_momenta = self.to_torch_tensor(self.global_initial_momenta)

        geodesic.set_template_points_t0(template_points_t0)
        geodesic.set_control_points_t0(control_points_t0)
        geodesic.set_momenta_t0(global_initial_momenta)
        geodesic.update()

        geodesic.write('Shooting', self.objects_name, self.objects_ext, 
                    template = self.global_initial_template, template_data = template_intensities_t0, 
                    output_dir = self.atlas_trajectory_output, write_adjoint_parameters = True)

        #### bis
        # the results of the two shootings are different !!!!
        # this one is bad!
        # xml_parameters = copy.deepcopy(self.xml_parameters)
        # xml_parameters.template_specifications["img"]['filename'] = self.global_initial_template_path[0]

        # compute_shooting(xml_parameters.template_specifications, dimension=3,
        #                 deformation_kernel_width=self.global_kernel_width,
        #                 initial_cp=self.regression_cp_path[0], 
        #                 initial_momenta=self.global_initial_momenta_for_atlas, 
        #                 time_concentration=10, t0=self.global_t0, 
        #                 tmin=min([self.global_t0, self.global_tmin]), 
        #                 tmax=max([self.global_t0, self.global_tmax]),
        #                 output_dir=self.atlas_trajectory_output_2)

    def define_registration_outputs(self):
        self.template_shot_to_subjects = []
        self.template_shot_to_single_subjects = []
        self.momenta_shot_to_subjects = []
        self.momenta_for_single_subjects = []

        templates = [f for f in os.listdir(self.atlas_trajectory_output) if ".nii" in f]
        for i, ind in enumerate(self.single_subjects_ind):
            age = round(self.global_visit_ages[ind][0], 1)
            
            # Get the template closest to subject age
            for f in templates:
                template_age = float(f.split("age_")[-1].split(".nii")[0])
                if template_age == age:
                    self.template_shot_to_subjects.append([join(self.atlas_trajectory_output, f)])
                    self.template_shot_to_single_subjects.append([join(self.single_registration_output, f)])
                    self.momenta_shot_to_subjects.append([join(self.atlas_trajectory_output, f.replace("img", "Momenta").replace("nii", "txt"))])

            t = "%.2f" % round(self.global_visit_ages[ind][0], 2)
            max_time = max(self.global_t0, self.global_visit_ages[ind][0])
            min_time = min(self.global_t0, self.global_visit_ages[ind][0])
            time = round((max_time - min_time) * 1)
            
            self.momenta_for_single_subjects.append([join(self.single_shooted_subjects_paths[i], 
                                            "TransportedMomenta__tp_{}__age_{}.txt".format(time, t))])
            
    def register_atlas_to_single_subjects(self):
        """
        5]. Compute average trajectory based on longitudinal subjects growth
        """

        xml_parameters = copy.deepcopy(self.xml_parameters)

        logger.info('\n Register atlas to single subjects')

        self.define_registration_outputs()

        for i, ind in enumerate(self.single_subjects_ind[1:]):
            logger.info('Register shooted template to subject {}'.format(ind))
            shutil.copyfile(self.template_shot_to_subjects[i][0], self.template_shot_to_single_subjects[i][0])

            # Set the target (subject) and the source (template at subject age)
            xml_parameters.dataset_filenames = [self.global_dataset_filenames[ind]]
            xml_parameters.visit_ages = [self.global_visit_ages[ind]]
            xml_parameters.subject_ids = [self.global_subject_ids[ind]]
            xml_parameters.template_specifications["img"]['filename'] = self.template_shot_to_subjects[i][0]

            self.global_deformetrica.output_dir = self.single_registration_subjects_paths[i]
            model = estimate_registration(self.global_deformetrica, xml_parameters)

            logger.info('Parallel transport average regression momenta to subject space')
            
            momenta_to_transport = read_3D_array(self.momenta_shot_to_subjects[i][0])
            registration_momenta = model.get_momenta()

            _, transported_momenta = parallel_transport(
                self.global_initial_cp, momenta_to_transport, registration_momenta, 
                self.global_kernel_width, self.global_kernel_device, 
                self.global_nb_of_tp)
                
                # transported_regression_momenta = parallel_transport(
                # self.regression_control_points, regression_momenta, - registration_to_atlas_momenta, 
                # self.global_kernel_width, self.global_kernel_device, 
                # self.global_nb_of_tp)

            # Save
            np.savetxt(self.momenta_for_single_subjects[i], transported_momenta)
    
    def define_single_shooting_outputs(self):
        gs = 'Shooting__GeodesicFlow__'
        a = "Subject_shot_for_atlas_"
        self.single_shooted_subjects_files = []
        self.single_shooted_subjects_for_atlas_2 = []

        for i, ind in enumerate(self.single_subjects_ind):
            t0 = np.round(self.global_t0, 2)
            time = 0
            if self.global_visit_ages[ind][0] < self.global_t0:
                time = round((self.global_t0 - self.global_visit_ages[ind][0]) * self.concentration_of_tp)                
            
            self.single_shooted_subjects_files.append([join(self.single_shooted_subjects_paths[i], 
                                    "{}{}__tp_{}__age_{}{}".format(gs, self.objects_name[0], time, t0, self.objects_ext[0]))])
            self.single_shooted_subjects_for_atlas_2.append([join(self.atlas_output_path_2, 
                                    "{}{}_{}__tp_{}__age_{}{}"\
                                    .format(a, self.global_subject_ids[i], self.objects_name[0], time, t0, self.objects_ext[0]))])
            
    def shoot_single_subjects_to_t0(self):
        logger.info('\Compute cross sectional subjects trajectories\n')

        self.define_single_shooting_outputs()

        control_points_t0 = self.to_torch_tensor(self.global_initial_cp)
        
        for i in self.single_subjects_ind:
            logger.info('\Compute trajectory of subject {}\n'.format(self.global_subject_ids[i]))
            
            # Instantiate a geodesic.
            geodesic = Geodesic(self.kernel, t0=self.global_tmin, 
                                time_concentration=self.concentration_of_tp)

            geodesic.set_tmin(min([self.global_t0, self.global_visit_ages[i][0]]))
            geodesic.set_tmax(max([self.global_t0, self.global_visit_ages[i][0]]))

            # Set the template, control points and momenta and update.
            subject_objects_t0 = self.deformable_objects_dataset[i][0] #subject i obs 0 
            subjects_points_t0 = subject_objects_t0.get_points()
            subjects_points_t0 = {k: self.to_torch_tensor(v) for k, v in subjects_points_t0.items()}
            
            # Subject intensities
            subjects_intensities_t0 = subject_objects_t0.get_data()
            subjects_intensities_t0 = {k: self.to_torch_tensor(v) for k, v in subjects_intensities_t0.items()}

            initial_momenta = read_3D_array(self.momenta_for_single_subjects[i])
            
            geodesic.set_template_points_t0(subjects_points_t0)
            geodesic.set_control_points_t0(control_points_t0)
            geodesic.set_momenta_t0(initial_momenta)
            geodesic.update()

            geodesic.write('Shooting', self.objects_name, self.objects_ext, 
                        template = subject_objects_t0, template_data = subjects_intensities_t0, 
                        output_dir = self.single_shooted_subjects_paths[i], write_adjoint_parameters = True)            
            
        # Export results -----------------------------------------------------------------------------------------------        
        # Copy shooting outputs to the atlas folder
        for s, shooted_subject_files in enumerate(self.single_shooted_subjects_files):
            for o in range(len(shooted_subject_files)):
                shutil.copyfile(self.single_shooted_subjects_files[o], self.single_shooted_subjects_for_atlas_2[s][o])
    
    def define_atlas_outputs_2(self):
        ba = "BayesianAtlas__EstimatedParameters__"
        self.estimated_momenta_path = join(self.atlas_output_path_2, ba + 'Momenta.txt') #needed for ICA!!
        self.estimated_template_path = [join(self.atlas_output_path_2, '{}Template_{}{}'.format(ba, obj, ext))\
                                        for obj, ext in zip(self.objects_name, self.objects_ext)]
                
        # Path to where we will store the initialized parameters/variables
        fi = 'ForInitialization__'
        self.global_initial_template_path_2 = [[join(self.path_to_initialization, '{}Template_{}__FromAtlas_All_Subjects{}'\
                                                .format(fi, self.objects_name[0], self.objects_ext[0]))]]

    def compute_atlas_with_all_subjects(self):
        logger.info('\n[ estimate an atlas from all subjects at t0] \n')

        self.define_atlas_outputs_2()
        xml_parameters = copy.deepcopy(self.xml_parameters)

        # Initialization -----------------------------------------------------------------------------------------------        
        # Retrieve all subjects shot at t0    
        xml_parameters.dataset_filenames = [[{self.objects_name[0] : sujet[0]}]\
                                            for sujet in self.shooted_subjects_for_atlas + self.single_shooted_subjects_for_atlas_2]
        xml_parameters.visit_ages = [[self.global_t0]] * self.global_subjects_nb

        print("\n xml_parameters.dataset_filenames", xml_parameters.dataset_filenames)
        
        # Use the regression cp
        xml_parameters.initial_cp = self.regression_cp_path[0]

        # Use the previously estimated template for initialization
        xml_parameters.template_specifications[self.objects_name[0]]['filename'] = self.global_initial_template_path[0]

        # Launch and save the outputted noise standard deviation, for later use ----------------------------------------
        self.global_deformetrica.output_dir = self.atlas_output_path_2
        model, self.global_atlas_momenta = estimate_bayesian_atlas(self.global_deformetrica, xml_parameters)
        
        self.global_initial_template = model.template #deformable multi object
        self.global_initial_template_data = model.get_template_data() #intensities

        # Save the template
        shutil.copyfile(self.estimated_template_path[0], self.global_initial_template_path_2[0])

        # Modify and write the model.xml file accordingly.
        self.insert_xml()

    def define_ica_outputs(self):
        fi = 'ForInitialization__'
        self.global_initial_mod_matrix_path = join(self.path_to_initialization, '{}ModulationMatrix__FromICA.txt'.format(fi))
        self.global_initial_sources_path = join(self.path_to_initialization, '{}Sources__FromICA.txt'.format(fi))

    def define_longitudinal_registration_outputs(self):
        # Registration outputs
        lr = 'LongitudinalRegistration__EstimatedParameters__'
        self.estimated_onset_ages_path = join(self.registration_output_path, lr + 'OnsetAges.txt')
        self.estimated_accelerations_path = join(self.registration_output_path, lr + 'Accelerations.txt')
        self.estimated_sources_path = join(self.registration_output_path, lr +  'Sources.txt')

        # Path to where we will store the initialized parameters/variables
        fi = 'ForInitialization__'
        self.global_initial_momenta_path = join(self.path_to_initialization, '{}Momenta__RescaledWithLongitudinalRegistration.txt'.format(fi))
        self.global_initial_onset_ages_path = join(self.path_to_initialization, '{}OnsetAges__FromLongitudinalRegistration.txt'.format(fi))
        self.global_initial_accelerations_path = join(self.path_to_initialization, '{}Accelerations__FromLongitudinalRegistration.txt'.format(fi))
        self.global_initial_sources_path = join(self.path_to_initialization, '{}Sources__FromLongitudinalRegistration.txt'.format(fi))

    def define_longitudinal_atlas_outputs(self):
        la = "LongitudinalAtlas__EstimatedParameters__"
        
        self.estimated_cp_path = join(self.longitudinal_atlas_output, '{}ControlPoints.txt'.format(la))
        self.estimated_momenta_path = join(self.longitudinal_atlas_output, '{}Momenta.txt'.format(la))
        self.estimated_mod_matrix_path = join(self.longitudinal_atlas_output, '{}ModulationMatrix.txt'.format(la))
        self.estimated_reference_time_path = join(self.longitudinal_atlas_output, '{}ReferenceTime.txt'.format(la))
        self.estimated_time_shift_std_path = join(self.longitudinal_atlas_output, '{}TimeShiftStd.txt'.format(la))
        self.estimated_acceleration_std_path = join(self.longitudinal_atlas_output, '{}AccelerationStd.txt'.format(la))
        self.estimated_onset_ages_path = join(self.longitudinal_atlas_output, '{}OnsetAges.txt'.format(la))
        self.estimated_accelerations_path = join(self.longitudinal_atlas_output, '{}Accelerations.txt'.format(la))
        self.estimated_sources_path = join(self.longitudinal_atlas_output,'{}Sources.txt'.format(la))
        
        for (object_name, ext) in zip(self.objects_name, self.objects_ext):
            self.estimated_template_path = join(self.longitudinal_atlas_output,
                                    '{}Template_%s__tp_%d__age_%.2f%s' %
                                    (object_name, self.global_t0, ext))
                                    #model.spatiotemporal_reference_frame.geodesic.backward_exponential.n_time_points - 1,
                                    #model.get_reference_time(), ext))
        #LongitudinalRegistration__EstimatedParameters__Template_right_hippocampus__tp_23__age_76.28

        # Path to where we will store the initialized parameters/variables
        fi = 'ForInitialization__'
        self.global_initial_cp_path = join(self.path_to_initialization, '{}ControlPoints__FromLongitudinalAtlas.txt'.format(fi))
        self.global_initial_momenta_path = join(self.path_to_initialization, '{}Momenta__FromLongitudinalAtlas.txt'.format(fi))
        self.global_initial_mod_matrix_path = join(self.path_to_initialization, '{}ModulationMatrix__FromLongitudinalAtlas.txt'.format(fi))
        self.global_initial_onset_ages_path = join(self.path_to_initialization, '{}OnsetAges__FromLongitudinalAtlas.txt'.format(fi))
        self.global_initial_accelerations_path = join(self.path_to_initialization, '{}Accelerations__FromLongitudinalAtlas.txt'.format(fi))
        self.global_initial_sources_path = join(self.path_to_initialization, '{}Sources__FromLongitudinalAtlas.txt'.format(fi))
        for k, (object_name, ext) in enumerate(zip(self.objects_name, self.objects_ext)):
            self.global_initial_template_path[k] = join(self.path_to_initialization,
                                    '{}Template_%s__FromLongitudinalAtlas%s' % (fi, object_name, ext))

def initialize_longitudinal_atlas_development(model_xml_path, dataset_xml_path, optimization_parameters_xml_path):
    initializer = LongitudinalAtlasInitializer(model_xml_path, dataset_xml_path, optimization_parameters_xml_path)
    
    if initializer.dataset_type != "mixed":
        initializer = CrossSectionalLongitudinalAtlasInitializer(model_xml_path, dataset_xml_path, optimization_parameters_xml_path)

    initializer.create_folders()
    initializer.initialize_outputs()

    if initializer.dataset_type == "mixed":
        initializer.compute_longitudinal_geodesic_regressions()
        initializer.compute_atlas_with_longitudinal_subjects()
        initializer.compute_average_trajectory()
        initializer.register_atlas_to_single_subjects()
        initializer.shoot_single_subjects_to_t0()
        initializer.compute_atlas_with_all_subjects()

    else:
        initializer.compute_atlas_all_subjects()
        initializer.compute_geodesic_regression()
        initializer.compute_shootings_to_t0()
        initializer.compute_atlas_all_subjects_2()
        initializer.compute_age_error()
        initializer.tangent_space_ica()
        initializer.longitudinal_atlas()
    
def initialize_piecewise_geodesic_regression_with_space_shift(model_xml_path, dataset_xml_path, optimization_parameters_xml_path):
    initializer = BayesianRegressionInitializer(model_xml_path, dataset_xml_path, optimization_parameters_xml_path)
    initializer.create_folders()

    initializer.compute_geodesic_regression()
    initializer.compute_shootings_to_t0()
    initializer.tangent_space_ica()



