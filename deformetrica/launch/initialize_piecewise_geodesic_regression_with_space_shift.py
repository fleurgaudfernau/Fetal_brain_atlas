import os
import copy
import torch
import os.path as op
from os.path import join
from ..launch.compute_shooting import compute_shooting
from torch.autograd import Variable
import warnings
import math
import torch
from sklearn.decomposition import PCA, FastICA
import xml.etree.ElementTree as et
from xml.dom.minidom import parseString
from ..support import utilities

from ..core import default
from ..in_out.xml_parameters import XmlParameters, get_dataset_specifications, get_estimator_options, get_model_options
from ..in_out.dataset_functions import create_template_metadata, create_dataset
from ..in_out.array_readers_and_writers import *
from ..support import kernels as kernel_factory
from ..api.deformetrica import Deformetrica
from .deformetrica_functions import *
from .compute_parallel_transport import compute_parallel_transport, compute_piecewise_parallel_transport

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
    K = torch.zeros((global_cp_nb * dimension, global_cp_nb * dimension))
    global_initial_cp = torch.from_numpy(global_initial_cp)
    #K = np.zeros((global_cp_nb * dimension, global_cp_nb * dimension))
    for i in range(global_cp_nb):
        for j in range(global_cp_nb):
            cp_i = global_initial_cp[i, :]
            cp_j = global_initial_cp[j, :]
            kernel_distance = torch.exp(- torch.sum((cp_j - cp_i) ** 2) / (kernel_width ** 2))
            for d in range(dimension):
                K[dimension * i + d, dimension * j + d] = kernel_distance
                K[dimension * j + d, dimension * i + d] = kernel_distance
    return K

#############################################################################""

class SubjectFiles():
    def __init__(self, objects_name, global_visit_ages, global_dataset_filenames, global_subject_ids,
               registration_output, shooting_output, global_times,
               concentration_of_tp, global_t0_for_pt):
        
        self.objects_name = objects_name
        self.global_dataset_filenames = global_dataset_filenames
        self.global_visit_ages = global_visit_ages
        self.global_subject_ids = global_subject_ids
        self.registration_output = registration_output
        self.shooting_output = shooting_output
        self.global_times = global_times
        self.concentration_of_tp = concentration_of_tp
        self.global_t0_for_pt = global_t0_for_pt

        self.accepted_difference = (1/self.concentration_of_tp)/2 + 0.01
    
    def filename(self, i):
        return self.global_dataset_filenames[i][0][self.objects_name[0]]
    
    def filename_(self, i):
        return self.global_dataset_filenames[i]
    def age(self, i):
        return round(self.global_visit_ages[i][0], 2)

    def id(self, i):
        return self.global_subject_ids[i]
    
    def registration_path(self, i):
        return join(self.registration_output, 'Registration__subject_'+ self.id(i))

    def registration_momenta_path(self, i):
        return join(self.registration_path(i), "DeterministicAtlas__EstimatedParameters__Momenta.txt")

    def shooting_path(self, i):
        return join(self.shooting_output, 'Shooting__subject_'+ self.id(i))

    def transported_momenta_path(self, i):
        return join(self.shooting_path(i), 
                    "Transported_Momenta_tp_{}__age_{:.2f}.txt".format(1, self.global_t0_for_pt))
    
    def set_regression_output(self, regression_templates, regression_momenta):
        self.regression_templates = regression_templates
        self.regression_momenta = regression_momenta
    
    def target_age(self, i):
        return [a for a in self.global_times if np.abs(a - self.age(i)) <= self.accepted_difference][0]

    def same_age_template(self, i):
        try:
            template = [f for f in self.regression_templates if str(self.target_age(i)) in f][0]
        except:
            template = [f for f in self.regression_templates if str(self.age(i)) in f][0]
        
        return template
    
    def same_age_momenta(self, i):
        return [m for m in self.regression_momenta if str(self.target_age(i)) in m][0]

    def shot_momenta_path(self, i):
        ctp = 1
        accepted_difference = (1/ctp)/2 + 0.01
        
        if np.abs(self.age(i) - self.global_t0_for_pt) < accepted_difference:
            tp = 0
            t0 = self.age(i)
        elif self.age(i) > self.global_t0_for_pt:
            tp = 0
            t0 = self.global_t0_for_pt
        else:
            tp = int((self.global_t0_for_pt - self.age(i)) * ctp + 0.5)
            t0 = self.global_t0_for_pt

        return join(self.shooting_path(i), 
                "Transported_Momenta_tp_{}__age_{:.2f}.txt".format(tp, t0)) 
    
    def shot_subject(self, i):
        return join(self.shooting_path(i), 
                "Transported_Momenta_tp_{}__age_{:.2f}.txt".format(1, self.global_t0_for_pt)) 

    def tmin(self, i):
        return min(self.global_t0_for_pt, self.age(i))
    
    def tmax(self, i):
        return max(self.global_t0_for_pt, self.age(i))

class BayesianRegressionInitializer():
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
        
        for _, object in self.xml_parameters.template_specifications.items():
            self.global_initial_template_path = object['filename']

        self.dataset = create_dataset(self.xml_parameters.template_specifications, 
                                    copy.deepcopy(self.global_visit_ages), #to avoid modification
                                    self.global_dataset_filenames, self.global_subject_ids,
                                    self.dimension)
        self.deformable_objects_dataset = self.dataset.deformable_objects #List of DeformableMultiObjects
        self.global_subjects_nb = len(self.global_dataset_filenames)
        self.global_observations_nb = sum([len(elt) for elt in self.global_visit_ages])

        # Deformation parameters
        self.global_kernel_width = self.xml_parameters.deformation_kernel_width
        self.kernel = kernel_factory.factory(kernel_width=self.global_kernel_width)
        
        # Times
        self.concentration_of_tp = self.xml_parameters.concentration_of_time_points
        self.global_t0 = self.xml_parameters.t0
        self.global_t0_for_pt = self.global_t0
        self.global_t0_for_pt = 30
        self.global_tR = self.xml_parameters.tR
        self.global_tmin = np.min([e[0] for e in self.global_visit_ages])
        self.global_tmax = np.max([e[-1] for e in self.global_visit_ages])
        # times 23.00, 23.2, 23.4... 
        self.global_times = [np.round(i, 2) for i in np.arange(np.floor(self.global_tmin), np.ceil(self.global_tmax) + 1, 1/self.concentration_of_tp)]
        # Number of time points (entiers eg 11)
        self.global_nb_of_tp = self.xml_parameters.number_of_time_points
        self.geodesic_nb_of_tp = int(1 + (self.global_t0 - self.global_tmin) * self.concentration_of_tp)
        
        self.num_component = self.xml_parameters.num_component
        self.piecewise = True if self.global_tR else False
        self.piecewise = False

        self.number_of_sources = 4
        if self.xml_parameters.number_of_sources:
            self.number_of_sources = self.xml_parameters.number_of_sources            
    
    def to_torch_tensor(self, array):
        return Variable(utilities.move_data(array), requires_grad=False)

    def create_folders(self):
        self.regression_output = join(self.output_dir, '2_subjects_geodesic_regression')
        self.registration_output = join(self.output_dir, '3_atlas_registration_to_subjects')
        self.shooting_output = join(self.output_dir, '4_subjects_shootings_to_t0')
        self.ICA_output = join(self.output_dir, '5_ICA')

        self.subjects = SubjectFiles(self.objects_name, self.global_visit_ages, 
                                     self.global_dataset_filenames, self.global_subject_ids, 
                                     self.registration_output, self.shooting_output, 
                                     self.global_times, self.concentration_of_tp, 
                                     self.global_t0_for_pt)
        
        self.registration_subjects_paths = [self.subjects.registration_path(i)\
                                        for i in range(self.global_subjects_nb)]

        self.shooted_subjects_paths = [self.subjects.shooting_path(i)\
                                        for i in range(self.global_subjects_nb)]
                
        for path in [self.regression_output, self.registration_output, \
                    self.shooting_output, self.ICA_output]\
                    + self.registration_subjects_paths + self.shooted_subjects_paths:
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
    
    def set_template_xml(self, xml_parameters, file = None):
        if file is None: 
            xml_parameters.template_specifications[self.objects_name[0]]['filename'] = self.global_initial_template_path

        xml_parameters.template_specifications[self.objects_name[0]]['filename'] = file
        
        return xml_parameters

    def insert_xml(self):
        model_xml_0 = et.parse(self.model_xml_path).getroot()
        self.model_xml_path = join(os.path.dirname(self.output_dir), 'initialized_model.xml')

        if self.global_initial_template_path:
            model_xml_0 = insert_model_xml_template(model_xml_0, 'filename', self.global_initial_template_path)
        if self.global_initial_noise_std_string:
            model_xml_0 = insert_model_xml_template(model_xml_0, 'noise-std', self.global_initial_noise_std_string)

        names = ['initial-control-points', 'initial-momenta', 'initial-onset-ages', 
                 'initial-accelerations', 'initial-modulation-matrix', 'initial-sources']
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
        if self.piecewise:
            for tR in self.global_tR:
                model_xml_0 = insert_model_xml_deformation_parameters(model_xml_0, 'tR', '%.4f' % tR)     
        # save the xml file
        doc = parseString((et.tostring(model_xml_0).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
        np.savetxt(self.model_xml_path, [doc], fmt='%s')

    def define_regression_outputs(self):
        mod = "Regression_EstimatedParameters"
        flow = "GeodesicRegression__GeodesicFlow__"
        fi = 'ForInitialization__'

        # Momenta flow (for parallel transport)/ template flow for registration

        self.global_initial_cp_path = join(self.regression_output, '{}__ControlPoints.txt'.format(mod))
        self.regression_templates = []
        self.regression_momenta_path = []

        self.global_initial_momenta_path = join(self.path_to_initialization, '{}Momenta_from_regression.txt'.format(fi))

        for t, age in enumerate(self.global_times):
            self.regression_momenta_path.append(join(self.regression_output, 
                                                '{}Momenta__tp_{}__age_{:.2f}.txt'.format(flow, t, age)))
    
        if not self.piecewise:    
            global_visit_ages = sorted([a[0] for a in self.global_visit_ages])
            for age in self.global_visit_ages:
                t = global_visit_ages.index(age[0])
                self.regression_templates.append(join(self.regression_output, "GeodesicRegression__Reconstruction__{}__tp_{}__age_{:.2f}{}"
                                                .format(self.objects_name[0], t, age[0], self.objects_ext[0])))
        else:            
            for t, age in enumerate(self.global_times):
                tr = [r for r in self.global_tR if age < r]
                c = self.global_tR.index(tr[0]) if len(tr) > 0 else self.num_component - 1
                self.regression_templates.append(join(self.regression_output, '{}{}__component_{}__tp_{}__age_{:.2f}{}'\
                                                .format(flow, self.objects_name[0], c, t, age, self.objects_ext[0])))

        self.subjects.set_regression_output(self.regression_templates, self.regression_momenta_path)

    def set_regression_xml(self, xml_parameters):
        xml_parameters = self.set_template_xml(xml_parameters)

        # Adapt the specific xml parameters and update
        #join filenames as if from a single subject = [[{obs 1}] [{obs2}]] -> [[{obs 1} {obs2}]]
        xml_parameters.dataset_filenames = [sum(self.global_dataset_filenames, [])]
        xml_parameters.visit_ages = [sum(self.global_visit_ages, [])]
        xml_parameters.subject_ids = [self.global_subject_ids[0]]
        xml_parameters.t0 = self.global_t0   
        xml_parameters.tR = self.global_tR
        xml_parameters.num_component = self.num_component
        xml_parameters.multiscale_meshes = False

        if self.piecewise: xml_parameters.model_type = 'PiecewiseRegression'.lower()

        if self.global_subjects_nb > 30 and self.objects_ext == ".nii":
            xml_parameters.optimization_method_type = "StochasticGradientAscent".lower()     

        return xml_parameters

    def compute_geodesic_regression(self):
        """
        2]. Compute individual geodesic regression
        """
        self.define_regression_outputs()

        logger.info('\n [ Geodesic regression on all subjects ]')

        if not op.exists(self.global_initial_momenta_path):
            xml_parameters = copy.deepcopy(self.xml_parameters)

            # Read the current model xml parameters.            
            xml_parameters = set_xml_for_regression(xml_parameters)
            xml_parameters = self.set_regression_xml(xml_parameters)

            self.global_deformetrica.output_dir = self.regression_output
            model = estimate_regression(self.global_deformetrica, xml_parameters)

            # Save results    
            np.savetxt(self.global_initial_cp_path, model.get_control_points()) 
            np.savetxt(self.global_initial_momenta_path, model.get_momenta())
            self.global_initial_momenta = read_3D_array(self.global_initial_momenta_path)
                    
    def set_registration_xml(self, i, xml_parameters):
        xml_parameters.dataset_filenames = [self.subjects.filename_(i)]
        xml_parameters.visit_ages = [[self.subjects.age(i)]]
        xml_parameters.subject_ids = [self.subjects.id(i)]
        xml_parameters = self.set_template_xml(xml_parameters, self.subjects.same_age_template(i))
        xml_parameters.print_every_n_iters = 100
        xml_parameters.multiscale_meshes = False

        # to ensure we keep the same bounding box
        xml_parameters.initial_control_points = self.global_initial_cp_path
        self.global_deformetrica.output_dir = self.subjects.registration_path(i)
        
        return xml_parameters

    def parallel_transport(self, xml_parameters, i):        
            compute_piecewise_parallel_transport(xml_parameters.template_specifications, self.dimension, 
                                   self.global_kernel_width,
                                    None, self.global_initial_cp_path, self.global_initial_momenta_path, # initial mom
                                    self.subjects.registration_momenta_path(i), # mom to transport
                                    tmin, tmax, self.concentration_of_tp,
                                    t0 = self.global_t0,
                                    t1 = self.subjects.age(i),
                                    tR=self.global_tR, nb_components=self.num_component,
                                    number_of_time_points=self.global_nb_of_tp,
                                    output_dir=self.subjects.shooting_path(i), 
                                    perform_shooting = True)


    def compute_shootings_to_t0(self):
        """
        3]. Register subjects to age matched template then transport momenta to t0
        """
        xml_parameters = copy.deepcopy(self.xml_parameters)

        for i in range(self.global_subjects_nb):
            if not op.exists(self.subjects.registration_momenta_path(i)):
                print(self.subjects.registration_momenta_path(i))
                logger.info('\n[ Register shooted template to subject {} of age {}]'.format(self.subjects.id(i), self.subjects.age(i)))

                # Set the target (subject) and the source (template at subject age)
                xml_parameters = self.set_registration_xml(i, xml_parameters)
                
                estimate_registration(self.global_deformetrica, xml_parameters)

            if not op.exists(self.subjects.shot_momenta_path(i)):
                print("!!!", self.subjects.shot_momenta_path(i))
                logger.info('\n[ Parallel transport registration momenta of subject {} of age {} to t0 {}]'.format(self.subjects.id(i), self.subjects.age(i), self.global_t0_for_pt))
                xml_parameters = self.set_template_xml(xml_parameters, self.subjects.same_age_template(i))
                compute_parallel_transport(xml_parameters.template_specifications, self.dimension, 
                                 self.global_kernel_width, None, self.global_initial_cp_path, 
                                self.subjects.same_age_momenta(i), # initial mom
                                self.subjects.registration_momenta_path(i), # mom to transport
                                self.subjects.tmin(i), self.subjects.tmax(i), 
                                concentration_of_time_points=1,
                                t0 = self.subjects.age(i),
                                number_of_time_points=self.global_nb_of_tp,
                                output_dir=self.subjects.shooting_path(i))

        self.insert_xml()
    
    def compute_shootings_to_t0_(self):
        """
        3]. Register subjects to age matched template then transport momenta to t0
        """
        xml_parameters = copy.deepcopy(self.xml_parameters)

        for i in range(range(self.global_subjects_nb)):
            if self.subjects.age(i) != self.global_t0_for_pt:
                if not op.exists(self.subjects.registration_momenta_path(i)):
                    logger.info('\n[ Register shooted template to subject {} of age {}]'.format(self.subjects.id(i), self.subjects.age(i)))

                    # Set the target (subject) and the source (template at subject age)
                    xml_parameters = self.set_registration_xml(i, xml_parameters)
                    
                    estimate_registration(self.global_deformetrica, xml_parameters)

                if not op.exists(self.subjects.shot_momenta_path(i)):
                    logger.info('\n[ Parallel transport regression momenta along registration momenta of subject {} of age]'.format(self.subjects.age(i)))
                    
                    compute_parallel_transport(xml_parameters.template_specifications, self.dimension, 
                                        self.global_kernel_width,None, self.global_initial_cp_path, 
                                        self.subjects.registration_momenta_path(i),
                                        self.subjects.same_age_momenta(i),
                                        tmin=0, tmax=1, t0 = 0,
                                        concentration_of_time_points=1,
                                        number_of_time_points=self.global_nb_of_tp,
                                        output_dir=self.subjects.shooting_path(i), 
                                        perform_shooting = False)

                if not op.exists(self.subjects.shot_subject(i)):
                    logger.info("\n [ Shoot subject {} to t0 ]".format(self.subjects.id(i)))
                    xml_parameters = self.set_template_xml(xml_parameters, self.subjects.filename(i))
                    compute_shooting(xml_parameters.template_specifications, dimension=self.dimension,
                                    deformation_kernel_width=self.global_kernel_width,
                                    initial_control_points=self.global_initial_cp_path, 
                                    initial_momenta=self.subjects.shot_momenta_path(i), 
                                    concentration_of_time_points=1, 
                                    t0=self.subjects.age(i), 
                                    tmin=self.subjects.tmin(i), tmax=self.subjects.tmax(i), 
                                    output_dir=self.subjects.shooting_path(i), 
                                    write_adjoint_parameters = False)   

        self.insert_xml()

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

        self.insert_xml()

        if not op.exists(self.global_initial_sources_path):

            # Compute RKHS matrix.
            control_points = read_3D_array(self.global_initial_cp_path) 
            K = compute_RKHS_matrix(control_points.shape[0], self.dimension, 
                                    self.global_kernel_width, control_points)
            print("RKHS matrix computed")
            # Project regression momenta
            if self.piecewise:
                momenta = self.global_initial_momenta[0]
            else:
                momenta = [m for m in self.regression_momenta_path if str(self.global_t0_for_pt) in m][0]
                momenta = read_3D_array(momenta) 
                momenta = torch.from_numpy(momenta.ravel())
            #Km = np.dot(K, momenta) # dot product/ produit scalaire
            Km = torch.matmul(K, momenta)
            print("KM multiplied")
            mKm = torch.matmul(torch.transpose(momenta), Km)
            #mKm = np.dot(momenta.transpose(), Km)
            print("mKm multiplied")
            print("mKm.shape", mKm.shape)

            w = []
            for i in range(self.global_subjects_nb):
                print("i", i)
                subject_momenta = read_3D_array(self.subjects.shot_momenta_path(i))
                subject_momenta = torch.from_numpy(subject_momenta.ravel()) #flattened array
                #w.append(subject_momenta - np.dot(subject_momenta, Km) / mKm * momenta)
                w.append(subject_momenta - torch.matmul(subject_momenta, Km) / mKm * momenta)
            w = np.array(w)
            print("w.shape", w.shape)

            ica = FastICA(n_components = self.number_of_sources, max_iter=50000)
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

            np.savetxt(self.global_initial_mod_matrix_path, global_initial_mod_matrix)
            np.savetxt(self.global_initial_sources_path, global_initial_sources)

            self.insert_xml()

            logger.info('>> Estimated random effect statistics:')
            logger.info('\t\t sources =\t%.3f\t[ mean ]\t+/-\t%.4f\t[std]' %
                        (np.mean(global_initial_sources), np.std(global_initial_sources)))
        
            # Visualization
            for s in range(self.number_of_sources):
                space_shift = global_initial_mod_matrix[:, s].contiguous().view(self.geodesic.momenta[0].size())
                np.savetxt(self.global_initial_sources_path, global_initial_sources)

            compute_shooting(xml_parameters.template_specifications, dimension=self.dimension,
                            deformation_kernel_width=self.global_kernel_width,
                            initial_control_points=self.global_initial_cp_path, 
                            initial_momenta=self.subjects.shot_momenta_path(i), 
                            concentration_of_time_points=1, 
                            output_dir=self.ICA_output, 
                            write_adjoint_parameters = False) 
            
def initialize_piecewise_geodesic_regression_with_space_shift(model_xml_path, dataset_xml_path, optimization_parameters_xml_path):
    initializer = BayesianRegressionInitializer(model_xml_path, dataset_xml_path, 
                                                optimization_parameters_xml_path)
    initializer.create_folders()
    initializer.initialize_outputs()
    
    initializer.compute_geodesic_regression()
    initializer.compute_shootings_to_t0()
    initializer.tangent_space_ica()
