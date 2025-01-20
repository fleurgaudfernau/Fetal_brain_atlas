import os
import sys
import copy
import os.path as op
from os.path import join

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
from ..in_out.dataset_functions import create_template_metadata
from ..core.model_tools.deformations.exponential import Exponential
from ..core.model_tools.deformations.geodesic import Geodesic
from ..in_out.array_readers_and_writers import *
from ..support import kernels as kernel_factory
from ..core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ..in_out.deformable_object_reader import DeformableObjectReader
from ..api.deformetrica import Deformetrica
from .deformetrica_functions import *

def get_acceleration_std(accelerations):  # Fixed-point algorithm.
    ss = np.sum((accelerations - 1.0) ** 2)
    nb_of_subjects = len(accelerations)
    std_old, std_new = math.sqrt(ss / float(nb_of_subjects)), math.sqrt(ss / float(nb_of_subjects))
    for iteration in range(100):
        phi = norm.pdf(- 1.0 / std_old)
        Phi = norm.cdf(- 1.0 / std_old)
        std_new = 1.0 / math.sqrt(nb_of_subjects * (1 - (phi / std_old) / (1 - Phi)) / ss)
        difference = math.fabs(std_new - std_old)
        if difference < 1e-5:
            break
        std_old = std_new
            
    return std_new

def rescaling_factor(accelerations):
    acceleration_std = get_acceleration_std(accelerations)

    expected_mean_acceleration = float(truncnorm.stats(- 1.0 / acceleration_std, float('inf'),
                                                        loc=1.0, scale=acceleration_std, moments='m'))
    mean_acceleration = np.mean(accelerations)

    return expected_mean_acceleration/mean_acceleration


class LongitudinalAtlasInitializer():
    def __init__(self, model_xml_path, dataset_xml_path, optimization_parameters_xml_path):
        
        self.model_xml_path = model_xml_path
        self.dataset_xml_path = dataset_xml_path
        self.optimization_parameters_xml_path = optimization_parameters_xml_path
        
        #create preprocessing folder
        self.output_dir = "Preprocessing"
        self.path_to_initialization = join(op.dirname(self.output_dir), 'Initialization')

        if not op.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        if not op.isdir(self.path_to_initialization):
            os.mkdir(self.path_to_initialization)

        # Read original longitudinal model xml parameters.        
        self.xml_parameters = XmlParameters()
        self.xml_parameters._read_model_xml(self.model_xml_path)
        self.xml_parameters._read_dataset_xml(self.dataset_xml_path)
        self.xml_parameters._read_optimization_parameters_xml(self.optimization_parameters_xml_path)

        self.global_deformetrica = Deformetrica(output_dir=self.output_dir)

        # Save some global parameters.
        #global_dataset_filenames:list of lists: 1 list per subject of disctionaries (1 dic = 1 obs)
        self.global_dataset_filenames = self.xml_parameters.dataset_filenames
        self.global_visit_ages = self.xml_parameters.visit_ages #list of lists: 1 list of visit ages/subject
        self.global_subject_ids = self.xml_parameters.subject_ids #list of ids
        self.global_objects_name, self.global_objects_ext = create_template_metadata(self.xml_parameters.template_specifications)[1:3]
        #xml_parameters.template_specifications: [deformable object Image], [object_name], [ext], [noise_std], [attachment]
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
        self.concentration_of_tp = self.xml_parameters.concentration_of_time_points
        self.global_t0 = np.round(np.mean(sum(self.global_visit_ages, [])), 2)
        self.global_tmin = np.round(np.mean([e[0] for e in self.global_visit_ages]), 2)
        self.global_nb_of_tp = self.xml_parameters.number_of_time_points
        self.geodesic_nb_of_tp = int(1 + (self.global_t0 - self.global_tmin) * self.concentration_of_tp) 

        self.number_of_sources = 4
        if self.xml_parameters.number_of_sources:
            self.number_of_sources = self.xml_parameters.number_of_sources            

        self.create_folders()
        self.initialize_outputs()

    def create_folders(self):
        self.atlas_output_path = join(self.output_dir, '1_atlas_on_baseline_data')
        regression = '2_global_geodesic_regression' if self.dataset_type == "single_points" \
                    else '2_individual_geodesic_regressions'
        self.regressions_output = join(self.output_dir, regression)
        self.regression_tmp_path = join(self.regressions_output, 'tmp')
        self.shooting_output_path = join(self.output_dir, '3_shooting_from_baseline_to_average')
        self.registration_output_path = join(self.output_dir, '4_longitudinal_registration')
        self.longitudinal_atlas_output = join(self.output_dir, '5_longitudinal_atlas_with_gradient_ascent')

        for path in [self.atlas_output_path, self.regressions_output, self.regression_tmp_path, 
                    self.shooting_output_path, self.registration_output_path, self.longitudinal_atlas_output]:
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


    def define_atlas_outputs(self):
        ba = "BayesianAtlas__EstimatedParameters__"
        br = "BayesianAtlas__Reconstruction__"

        self.estimated_cp_path = join(self.atlas_output_path, ba + 'ControlPoints.txt')
        self.estimated_momenta_path = join(self.atlas_output_path, ba + 'Momenta.txt')

        #path to the template(s)
        self.estimated_template_path = [join(self.atlas_output_path, '{}Template_{}{}'.format(ba, obj_name, ext))\
                                        for (obj_name, ext) in (zip(self.global_objects_name, self.global_objects_ext))]
        
        # path to subject reconstructions
        self.estimated_atlas_reconstructions = []        
        for i in range(self.global_subjects_nb):
            liste = [join(self.atlas_output_path, '%s%s__subject_%s%s'%(br, obj_name, self.global_subject_ids[i], ext))\
                    for (obj_name, ext) in zip(self.global_objects_name, self.global_objects_ext)]
            self.estimated_atlas_reconstructions.append(liste)
        
        # Path to where we will store the initialized parameters/variables
        fi = 'ForInitialization__'

        self.global_initial_template_path = []
        for _, (obj_name, ext) in enumerate(zip(self.global_objects_name, self.global_objects_ext)):
            self.global_initial_template_path.append(join(self.path_to_initialization, 
                                                        '{}Template_{}__FromAtlas{}'.format(fi, obj_name, ext)))
        self.global_initial_cp_path = join(self.path_to_initialization, '{}ControlPoints__FromAtlas.txt'.format(fi))
    
    def define_regression_outputs(self):
        
        # Regression outputs
        gr = 'GeodesicRegression__'
        self.transported_regression_momenta_path = [join(self.regressions_output, 
                                                '%ssubject_%sEstimatedParameters__TransportedMomenta.txt'%(gr, self.global_subject_ids[i]))\
                                                for i in range(len(self.global_subject_ids))]
        self.path_to_regression_cp = [join(self.regression_tmp_path, 'regression_cp__%s.txt' % self.global_subject_ids[i]) \
                                        for i in range(len(self.global_subject_ids))]
        
        # Path to where we will store the initialized parameters/variables
        fi = 'ForInitialization__'
        self.global_initial_momenta_path = join(self.path_to_initialization, '{}Momenta__FromRegressions.txt'.format(fi))
    
    def define_heuristics_outputs(self):
        fi = 'ForInitialization__'
        self.global_initial_momenta_path = join(self.path_to_initialization, '{}Momenta__RescaledWithHeuristics.txt'.format(fi))
        self.global_initial_onset_ages_path = join(self.path_to_initialization, '{}OnsetAges__FromHeuristic.txt'.format(fi))
        self.global_initial_accelerations_path = join(self.path_to_initialization, '{}Accelerations__FromHeuristic.txt'.format(fi))

    def define_shooting_outputs(self):
        # Shooting outputs
        sg = "Shooting__GeodesicFlow__"
        self.shooted_template_path = []
        if self.dataset_type != "single_point":
            for (obj_name, ext) in zip(self.global_objects_name, self.global_objects_ext):
                self.shooted_template_path.append(join(self.shooting_output_path, 
                "{}{}__tp_{}__age_{}{}".format(sg, obj_name, self.geodesic_nb_of_tp, self.global_t0, ext)))
            self.shooted_cp_path = join(self.shooting_output_path, 
                                    '{}ControlPoints__tp_{}__age_{}.txt'.format(sg, self.geodesic_nb_of_tp, self.global_t0))
            self.shooted_momenta_path = join(self.shooting_output_path, 
                                        '{}Momenta__tp_{}__age_{}.txt'.format(sg, self.geodesic_nb_of_tp, self.global_t0))
        
        # Path to where we will store the initialized parameters/variables
        fi = 'ForInitialization__'
        if self.dataset_type != "single_point":
            for k, (obj_name, ext) in enumerate(zip(self.global_objects_name, self.global_objects_ext)):
                self.global_initial_template_path[k] = join(self.path_to_initialization, 
                '{}Template_{}__FromAtlasAndShooting{}'.format(fi, obj_name, ext))
        
        self.global_initial_cp_path = join(self.path_to_initialization, '{}ControlPoints__FromAtlasAndShooting.txt'.format(fi))
        self.global_initial_momenta_path = join(self.path_to_initialization, '{}Momenta__FromRegressionsAndShooting.txt'.format(fi))
    
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
        
        for (object_name, ext) in zip(self.global_objects_name, self.global_objects_ext):
            self.estimated_template_path = join(self.longitudinal_atlas_output,
                                    '{}Template_%s__tp_%d__age_%.2f%s' %
                                    (object_name, self.global_t0, ext))
                                    #model.spatiotemporal_reference_frame.geodesic.backward_exponential.number_of_time_points - 1,
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
        for k, (object_name, ext) in enumerate(zip(self.global_objects_name, self.global_objects_ext)):
            self.global_initial_template_path[k] = join(self.path_to_initialization,
                                    '{}Template_%s__FromLongitudinalAtlas%s' % (fi, object_name, ext))   

    def initialize_template_image_with_atlas(self):
        """
        1]. Compute an atlas on the baseline data.
        ------------------------------------------
        Atlas from 1st time point of the subjects
        Unchanged: works for single and not single time points data
        TO DO: MAYBE SELECT ONLY YOUNG SUBJECTS?
        """
        logger.info('[ estimate an atlas from baseline data ] \n')

        self.define_atlas_outputs()

        # Initialization -----------------------------------------------------------------------------------------------
        self.global_deformetrica.output_dir = self.atlas_output_path

        # Select the first observation of each subject
        xml_parameters = copy.deepcopy(self.xml_parameters)
        xml_parameters.dataset_filenames = [[self.global_dataset_filenames[i][0]] for i in self.longitudinal_subjects_ind]
        xml_parameters.visit_ages = [[self.global_visit_ages[i][0]] for i in self.longitudinal_subjects_ind]

        # Launch and save the outputted noise standard deviation, for later use ----------------------------------------
        model, self.global_atlas_momenta = estimate_bayesian_atlas(self.global_deformetrica, xml_parameters)
        global_objects_noise_std = [math.sqrt(elt) for elt in model.get_noise_variance()]
        
        # Save the template
        self.global_initial_template = model.template #deformable multi object
        self.global_initial_template_data = model.get_template_data() #intensities
        self.global_initial_cp = model.get_control_points()
        self.global_cp_nb = self.global_initial_cp.shape[0]
        self.dimension = self.global_initial_cp.shape[1]

        original_objects_noise_variance = create_template_metadata(xml_parameters.template_specifications)[3]

        shutil.copyfile(self.estimated_cp_path, self.global_initial_cp_path)
        for k, (original_obj_noise_var) in enumerate(original_objects_noise_variance):            
            shutil.copyfile(self.estimated_template_path[k], self.global_initial_template_path[k])

            # Override the obtained noise standard deviation values, if it was already given by the user.
            if original_obj_noise_var > 0:
                global_objects_noise_std[k] = math.sqrt(original_obj_noise_var)

        # Convert the noise std float values to formatted strings.
        self.global_initial_noise_std_string = ['{:.4f}'.format(e) for e in global_objects_noise_std]

        # Modify and write the model.xml file accordingly.
        self.insert_xml()
        
    def compute_individual_geodesic_regressions(self):
        """
        2]. Compute individual geodesic regressions.
        --------------------------------------------
            The time t0 is chosen as the baseline age for every subject.
            The control points are the one outputted by the atlas estimation, and are frozen.
            Skipped if an initial control points and (longitudinal) momenta are specified.
        """

        self.define_regression_outputs()

        # Read the current model xml parameters.
        xml_parameters = copy.deepcopy(self.xml_parameters)

        # Check if the computations have been done already.
        if os.path.isdir(self.global_initial_momenta_path):
            self.global_initial_momenta = read_3D_array(self.global_initial_momenta_path)
        else:
            logger.info('\n[ Compute individual geodesic regressions ]')

            self.global_initial_momenta = np.zeros(self.global_initial_cp.shape)

            if self.dataset_type == "single_points":
                regression_cp, self.global_initial_momenta = estimate_geodesic_regression_single_point(
                                                                self.global_deformetrica, xml_parameters, self.regressions_output,
                                                                self.global_dataset_filenames,self.global_visit_ages, self.global_subject_ids,
                                                                self.global_tmin, self.global_initial_template_path)
            else:                    
                # Regression for subjects with longitudinal observations
                for i in self.longitudinal_subjects_ind:

                    # Set the initial template as the reconstructed object after the atlas.
                    for k, (_, object_specs) in enumerate(xml_parameters.template_specifications.items()):
                        object_specs['filename'] = self.estimated_atlas_reconstructions[i][k]

                    # Find the control points and momenta that transforms the previously computed template into the individual.
                    registration_cp, registration_momenta = shoot(self.global_initial_cp, self.global_atlas_momenta[i],
                                                                    self.global_kernel_width, 
                                                                    self.global_nb_of_tp)

                    # Use those cp for the regression.
                    np.savetxt(self.path_to_regression_cp[i], registration_cp)
                    xml_parameters.initial_control_points = self.path_to_regression_cp[i]

                    # Regression. (outputed regression cp ~ registration_cp)
                    regression_cp, regression_momenta = estimate_subject_geodesic_regression(
                        i, self.global_deformetrica, xml_parameters, self.regressions_output,
                        self.global_dataset_filenames, self.global_visit_ages, self.global_subject_ids)

                    # Parallel transport of the estimated momenta.
                    _, transported_regression_momenta = parallel_transport(
                        regression_cp, regression_momenta, - registration_momenta, self.global_kernel_width, 
                         self.global_kernel_device, self.global_nb_of_tp)
                    
                    # Increment the global initial momenta.
                    self.global_initial_momenta += transported_regression_momenta/float(self.nb_of_longitudinal_subjects)
                    
                    np.savetxt(self.transported_regression_momenta_path[i], transported_regression_momenta)                
            
            np.savetxt(self.global_initial_momenta_path, self.global_initial_momenta)

            # Modify and write the model.xml file accordingly.
            self.insert_xml()

    def compute_onset_ages(self):
        """
        3]. Initializing heuristics for accelerations and onset ages.
        -----------------------------------------------------------------
        """

        logger.info('\n[ initializing heuristics for individual accelerations and onset ages ]\n')

        self.define_heuristics_outputs()

        global_initial_cp_torch = torch.from_numpy(self.global_initial_cp).type(default.tensor_scalar_type)
        global_initial_momenta_torch = torch.from_numpy(self.global_initial_momenta).type(default.tensor_scalar_type)
        global_initial_momenta_norm_squared = torch.dot(global_initial_momenta_torch.view(-1), 
                                                        self.kernel.convolve(
                                                        global_initial_cp_torch, global_initial_cp_torch,
                                                        global_initial_momenta_torch).view(-1)).detach().cpu().numpy()
        
        heuristic_onset_ages = []
        heuristic_accelerations = []
        
        for i in range(self.global_subjects_nb):
            # Heuristic for the initial onset age.
            heuristic_onset_ages.append(np.mean(self.global_visit_ages[i]))

            # Heuristic for the initial acceleration.
            if i not in self.longitudinal_subjects_ind:
                heuristic_accelerations.append(1.0)
            else:
                regression_momenta = read_3D_array(self.transported_regression_momenta_path[i])
                regression_momenta_torch = torch.from_numpy(regression_momenta).type(default.tensor_scalar_type)
                sc_product_with_pop_momenta = torch.dot(global_initial_momenta_torch.view(-1), self.kernel.convolve(
                                                    global_initial_cp_torch, global_initial_cp_torch, 
                                                    regression_momenta_torch).view(-1)).detach().cpu().numpy()

                if sc_product_with_pop_momenta <= 0.0:
                    heuristic_accelerations.append(1.0)
                else:
                    heuristic_accelerations.append(float(np.sqrt(sc_product_with_pop_momenta / global_initial_momenta_norm_squared)))            

        heuristic_onset_ages = np.array(heuristic_onset_ages)
        heuristic_accelerations = np.array(heuristic_accelerations)

        # Standard deviations.
        if self.dataset_type != "single_points":                        
            # Computations only on longitudinal subjects
            onset_ages_longitudinal = np.take(heuristic_onset_ages, self.longitudinal_subjects_ind)
            accelerations_longitudinal = np.take(heuristic_accelerations, self.longitudinal_subjects_ind)
            
            # Accelerations of single point subjects are unchanged
            heuristic_accelerations[[self.longitudinal_subjects_ind]] *= rescaling_factor(accelerations_longitudinal)
            self.global_initial_momenta /= rescaling_factor(accelerations_longitudinal)

            self.global_acceleration_std = get_acceleration_std(accelerations_longitudinal)
            self.global_time_shift_std = np.std(onset_ages_longitudinal)
                
        logger.info('>> Estimated fixed effects:')
        logger.info('\t\t time_shift_std    =\t%.3f' % self.global_time_shift_std)
        logger.info('\t\t acceleration_std  =\t%.3f' % self.global_acceleration_std)
        logger.info('>> Estimated random effect statistics:')
        logger.info('\t\t onset_ages    =\t%.3f\t[ mean ]\t+/-\t%.4f\t' %
                    (np.mean(heuristic_onset_ages), self.global_time_shift_std))
        logger.info('\t\t accelerations =\t%.4f\t[ mean ]\t+/-\t%.4f\t' %
                    (np.mean(heuristic_accelerations), np.std(heuristic_accelerations)))

        # Export the results -----------------------------------------------------------------------------------------------
        np.savetxt(self.global_initial_momenta_path, self.global_initial_momenta)
        np.savetxt(self.global_initial_onset_ages_path, heuristic_onset_ages)    
        np.savetxt(self.global_initial_accelerations_path, heuristic_accelerations)

        # Modify the original model.xml file accordingly.
        self.insert_xml()


    def compute_accelerations_and_onset_ages(self):
        """
        3]. Initializing heuristics for accelerations and onset ages.
        -----------------------------------------------------------------
            The individual accelerations are taken as the ratio of the regression momenta norm to the global one.
            The individual onset ages are computed as if all baseline ages were in correspondence.
        """

        logger.info('\n[ initializing heuristics for individual accelerations and onset ages ]\n')

        self.define_heuristics_outputs()

        global_initial_cp_torch = torch.from_numpy(self.global_initial_cp).type(default.tensor_scalar_type)
        global_initial_momenta_torch = torch.from_numpy(self.global_initial_momenta).type(default.tensor_scalar_type)
        global_initial_momenta_norm_squared = torch.dot(global_initial_momenta_torch.view(-1), 
                                                        self.kernel.convolve(
                                                        global_initial_cp_torch, global_initial_cp_torch,
                                                        global_initial_momenta_torch).view(-1)).detach().cpu().numpy()
        
        heuristic_onset_ages = []
        heuristic_accelerations = []
        
        for i in range(self.global_subjects_nb):
            # Heuristic for the initial onset age.
            heuristic_onset_ages.append(np.mean(self.global_visit_ages[i]))

            # Heuristic for the initial acceleration.
            if i not in self.longitudinal_subjects_ind:
                heuristic_accelerations.append(1.0)
            else:
                regression_momenta = read_3D_array(self.transported_regression_momenta_path[i])
                regression_momenta_torch = torch.from_numpy(regression_momenta).type(default.tensor_scalar_type)
                sc_product_with_pop_momenta = torch.dot(global_initial_momenta_torch.view(-1), self.kernel.convolve(
                                                    global_initial_cp_torch, global_initial_cp_torch, 
                                                    regression_momenta_torch).view(-1)).detach().cpu().numpy()

                if sc_product_with_pop_momenta <= 0.0:
                    heuristic_accelerations.append(1.0)
                else:
                    heuristic_accelerations.append(float(np.sqrt(sc_product_with_pop_momenta / global_initial_momenta_norm_squared)))            

        heuristic_onset_ages = np.array(heuristic_onset_ages)
        heuristic_accelerations = np.array(heuristic_accelerations)

        # Standard deviations.
        if self.dataset_type != "single_points":                        
            # Computations only on longitudinal subjects
            onset_ages_longitudinal = np.take(heuristic_onset_ages, self.longitudinal_subjects_ind)
            accelerations_longitudinal = np.take(heuristic_accelerations, self.longitudinal_subjects_ind)
            
            # Accelerations of single point subjects are unchanged
            heuristic_accelerations[[self.longitudinal_subjects_ind]] *= rescaling_factor(accelerations_longitudinal)
            self.global_initial_momenta /= rescaling_factor(accelerations_longitudinal)

            self.global_acceleration_std = get_acceleration_std(accelerations_longitudinal)
            self.global_time_shift_std = np.std(onset_ages_longitudinal)
                
        logger.info('>> Estimated fixed effects:')
        logger.info('\t\t time_shift_std    =\t%.3f' % self.global_time_shift_std)
        logger.info('\t\t acceleration_std  =\t%.3f' % self.global_acceleration_std)
        logger.info('>> Estimated random effect statistics:')
        logger.info('\t\t onset_ages    =\t%.3f\t[ mean ]\t+/-\t%.4f\t' %
                    (np.mean(heuristic_onset_ages), self.global_time_shift_std))
        logger.info('\t\t accelerations =\t%.4f\t[ mean ]\t+/-\t%.4f\t' %
                    (np.mean(heuristic_accelerations), np.std(heuristic_accelerations)))

        # Export the results -----------------------------------------------------------------------------------------------
        np.savetxt(self.global_initial_momenta_path, self.global_initial_momenta)
        np.savetxt(self.global_initial_onset_ages_path, heuristic_onset_ages)    
        np.savetxt(self.global_initial_accelerations_path, heuristic_accelerations)

        # Modify the original model.xml file accordingly.
        self.insert_xml()
    
    def shoot_average_baseline_to_global_baseline(self):
        """
        4]. Shoot from the average baseline age to the global average.
        --------------------------------------------------------------
            New values are obtained for the template, control points, and (longitudinal) momenta.
            Skipped if initial control points and momenta were given.
        """

        self.define_shooting_outputs()
        
        if self.dataset_type != "single_point":
            logger.info('\n[ shoot from the average baseline age to the global average ]')

            # Instantiate a geodesic.
            geodesic = Geodesic(self.kernel, t0=self.global_tmin, 
                                concentration_of_time_points=self.concentration_of_tp)
            geodesic.set_tmin(self.global_tmin) #average 1st observation age
            geodesic.set_tmax(self.global_t0) #average observation ages

            # Set the template, control points and momenta and update.
            template_points_t0 = {key: Variable(torch.from_numpy(value).type(default.tensor_scalar_type), requires_grad=False)
                                for key, value in self.global_initial_template.get_points().items()}
            control_points_t0 = Variable(torch.from_numpy(self.global_initial_cp).type(default.tensor_scalar_type))
            momenta_t0 = Variable(torch.from_numpy(self.global_initial_momenta).type(default.tensor_scalar_type), requires_grad=False)
            
            geodesic.set_template_points_t0(template_points_t0)
            geodesic.set_control_points_t0(control_points_t0)
            geodesic.set_momenta_t0(momenta_t0)
            geodesic.update()

            geodesic.write('Shooting', self.global_objects_name, self.global_objects_ext, self.global_initial_template,
                        template_data = {key: Variable(torch.from_numpy(value).type(default.tensor_scalar_type), requires_grad=False)
                        for key, value in self.global_initial_template_data.items()}, 
                        output_dir = self.shooting_output_path, write_adjoint_parameters=True)

            # Export results -----------------------------------------------------------------------------------------------        
            for k in range(len(self.global_objects_name)):
                shutil.copyfile(self.shooted_template_path[k], self.global_initial_template_path[k])

            shutil.copyfile(self.shooted_cp_path, self.global_initial_cp_path)
            shutil.copyfile(self.shooted_momenta_path, self.global_initial_momenta_path)
        
        self.global_initial_cp = read_2D_array(self.global_initial_cp_path)
        self.global_initial_momenta = read_3D_array(self.global_initial_momenta_path)
        
        # Modify and write the model.xml file accordingly.
        self.insert_xml()

    def tangent_space_ica(self):
        """
        5]. Tangent-space ICA on the individual momenta outputted by the atlas estimation.
        ----------------------------------------------------------------------------------
            Those momenta are first projected on the space orthogonal to the initial (longitudinal) momenta.
            Skipped if initial control points and modulation matrix were specified.
        """

        logger.info('\n[ Tangent-space ICA on the projected individual momenta ]\n')

        self.define_ica_outputs()

        # Compute RKHS matrix.
        K = np.zeros((self.global_cp_nb * self.dimension, self.global_cp_nb * self.dimension))
        for i in range(self.global_cp_nb):
            for j in range(self.global_cp_nb):
                cp_i = self.global_initial_cp[i, :]
                cp_j = self.global_initial_cp[j, :]
                kernel_distance = math.exp(- np.sum((cp_j - cp_i) ** 2) / (self.global_kernel_width ** 2))
                for d in range(self.dimension):
                    K[self.dimension * i + d, self.dimension * j + d] = kernel_distance
                    K[self.dimension * j + d, self.dimension * i + d] = kernel_distance

        Km = np.dot(K, self.global_initial_momenta.ravel())
        mKm = np.dot(self.global_initial_momenta.ravel().transpose(), Km)
        w = np.array([self.global_initial_momenta[i].ravel() - np.dot(self.global_initial_momenta[i].ravel(), Km) / mKm * self.global_initial_momenta.ravel()\
                    for i in range(self.global_initial_momenta.shape[0])])

        # Dimensionality reduction.
        ica = FastICA(n_components=self.number_of_sources, max_iter=50000)
        global_initial_sources = ica.fit_transform(w)
        global_initial_mod_matrix = ica.mixing_

        # Rescale.
        for s in range(self.number_of_sources):
            std = np.std(global_initial_sources[:, s])
            global_initial_sources[:, s] /= std
            global_initial_mod_matrix[:, s] *= std

        # Print.
        residuals = []
        for i in range(self.global_subjects_nb):
            residuals.append(w[i] - np.dot(global_initial_mod_matrix, global_initial_sources[i]))
        mean_relative_residual = np.mean(np.absolute(np.array(residuals))) / np.mean(np.absolute(w))
        logger.info('>> Mean relative residual: %.3f %%.' % (100 * mean_relative_residual))

        # Save.
        np.savetxt(self.global_initial_mod_matrix_path, global_initial_mod_matrix)
        np.savetxt(self.global_initial_sources_path, global_initial_sources)

        # Modify the original model.xml file accordingly.
        self.insert_xml()

        logger.info('>> Estimated random effect statistics:')
        logger.info('\t\t sources =\t%.3f\t[ mean ]\t+/-\t%.4f\t' % (np.mean(global_initial_sources), np.std(global_initial_sources)))

    def longitudinal_registration(self):
        """
        6]. Longitudinal registration of all target subjects.
        ~ estimate a longitudinal atlas for each subject... 
        -> update accelerations, time shift, sources and their stds...
        -----------------------------------------------------
            The reference is the average of the ages at all visits.
            The template, control points and modulation matrix are from the atlas estimation.
            The momenta is from the individual regressions.
        """
        self.define_longitudinal_registration_outputs()

        logger.info('\n [ longitudinal registration of all subjects ]\n')
                
        self.global_deformetrica.output_dir = self.registration_output_path
        estimate_longitudinal_registration(self.global_deformetrica, copy.deepcopy(self.xml_parameters))

        # Load results.
        global_onset_ages = read_2D_array(self.estimated_onset_ages_path)
        global_accelerations = read_2D_array(self.estimated_accelerations_path)
        global_sources = read_2D_array(self.estimated_sources_path)
        
        # Rescaling the initial momenta according to the mean of the acceleration factors.
        global_accelerations *= rescaling_factor(global_accelerations)
        self.global_initial_momenta /= rescaling_factor(global_accelerations)

        self.global_acceleration_std = get_acceleration_std(global_accelerations)
        self.global_time_shift_std = np.std(global_onset_ages)

        # Copy the output individual effects into the data folder.
        np.savetxt(self.global_initial_momenta_path, self.global_initial_momenta)
        np.savetxt(self.global_initial_accelerations_path, global_accelerations)
        np.savetxt(self.global_initial_onset_ages_path, global_onset_ages)
        np.savetxt(self.global_initial_sources_path, global_sources)

        # Modify the original model.xml file accordingly.
        self.insert_xml()

        logger.info('\n>> Estimated fixed effects:')
        logger.info('\t\t time_shift_std    =\t%.3f' % self.global_time_shift_std)
        logger.info('\t\t acceleration_std  =\t%.3f' % self.global_acceleration_std)
        logger.info('>> Estimated random effect statistics:')
        logger.info('\t\t onset_ages    =\t%.3f\t[ mean ]\t+/-\t%.4f\t' % (np.mean(global_onset_ages), self.global_time_shift_std))
        logger.info('\t\t accelerations =\t%.4f\t[ mean ]\t+/-\t%.4f\t' % (np.mean(global_accelerations), np.std(global_accelerations)))
        logger.info('\t\t sources       =\t%.4f\t[ mean ]\t+/-\t%.4f\t' % (np.mean(global_sources), np.std(global_sources)))

    def gradient_based_longitudinal_atlas(self):
        """
        7]. Gradient-based optimization on population parameters.
        ---------------------------------------------------------
            Ignored if the user-specified optimization method is not the MCMC-SAEM.
        """
        self.define_longitudinal_atlas_outputs()

        logger.info('\n[ longitudinal atlas estimation with the GradientAscent optimizer ]\n')    

        self.global_deformetrica.output_dir = self.longitudinal_atlas_output
        model = estimate_longitudinal_atlas(self.global_deformetrica, self.xml_parameters)

        # Export the results -------------------------------------------------------------------------------------------
        shutil.copyfile(self.estimated_cp_path, self.global_initial_cp_path)
        shutil.copyfile(self.estimated_momenta_path, self.global_initial_momenta_path)
        shutil.copyfile(self.estimated_onset_ages_path, self.global_initial_onset_ages_path)
        shutil.copyfile(self.estimated_accelerations_path, self.global_initial_accelerations_path)
        shutil.copyfile(self.estimated_sources_path, self.global_initial_sources_path)

        # Update the XML file
        self.global_initial_reference_time = np.loadtxt(self.estimated_reference_time_path)
        self.global_time_shift_std = np.loadtxt(self.estimated_time_shift_std_path)
        self.global_acceleration_std = np.loadtxt(self.estimated_acceleration_std_path)

        global_initial_noise_variance = model.get_noise_variance()
        self.global_initial_noise_std_string = ['{:.4f}'.format(math.sqrt(elt)) for elt in global_initial_noise_variance]
        
        self.insert_xml()

def initialize_longitudinal_atlas_simplified(model_xml_path, dataset_xml_path, optimization_parameters_xml_path):
    initializer = LongitudinalAtlasInitializer(model_xml_path, dataset_xml_path, optimization_parameters_xml_path)
    for v in vars(initializer).keys():
        print("\n", v)
        print(vars(initializer)[v])
    print("\n ORIGINAL XML PARAM -----------------------------------------------------------")
    for v in vars(initializer.xml_parameters).keys():
        print("\n", v)
        print(vars(initializer.xml_parameters)[v])

    initializer.initialize_template_image_with_atlas()
    initializer.compute_individual_geodesic_regressions()
    initializer.compute_accelerations_and_onset_ages()
    initializer.shoot_average_baseline_to_global_baseline()
    initializer.tangent_space_ica()
    initializer.longitudinal_registration()
    initializer.gradient_based_longitudinal_atlas()


