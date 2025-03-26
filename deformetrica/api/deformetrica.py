import gc
import copy
import logging
import math
import os
import os.path as op
import resource
import sys
import nibabel as nib
from copy import deepcopy
import torch
import numpy as np
import time 
from time import strftime, gmtime
from ..core import default
from ..core.estimators.gradient_ascent import GradientAscent
from ..core.models import DeformableTemplate, GeodesicRegression, PiecewiseGeodesicRegression, \
                        BayesianPiecewiseGeodesicRegression
from ..in_out.dataset_functions import create_dataset, filter_dataset, make_dataset_timeseries,\
                                        age, id, dataset_for_registration, maxi, mini, ages_histogram
from ..launch.compute_parallel_transport import launch_parallel_transport, launch_piecewise_parallel_transport, \
                                                compute_distance_to_flow
from ..in_out.array_readers_and_writers import read_3D_array, write_3D_array
from ..launch.compute_shooting import compute_shooting
from ..launch.compute_ica import perform_ICA, plot_ica
from ..support import utilities
from ..support.utilities.vtk_tools import screenshot_vtk
from ..support.utilities.tools import gaussian_kernel
from ..support.probability_distributions.multi_scalar_normal_distribution import MultiScalarNormalDistribution
from ..support.probability_distributions.uniform_distribution import UniformDistribution
from ..core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject


from ..core.estimators.stochastic_gradient_ascent import StochasticGradientAscent
from .piecewise_rg_tools import make_dir, complete, PlotResiduals, options_for_registration

global logger
logger = logging.getLogger()


class Deformetrica:
    """ Analysis of 2D and 3D shape data.
    Compute deformations of the 2D or 3D ambient space, which, in turn, warp any object embedded in this space, whether this object is a curve, a surface,
    a structured or unstructured set of points, an image, or any combination of them.
    2 main applications are contained within Deformetrica: `compute` and `estimate`.
    """

    ####################################################################################################################
    # Constructor & destructor.
    ####################################################################################################################

    def __init__(self, output_dir=default.output_dir, verbosity='INFO'):
        """
        Constructor
        :param str output_dir: Path to the output directory
        :param str verbosity: Defines the output log verbosity level. By default the verbosity level is set to 'INFO'.
                          Possible values are: CRITICAL, ERROR, WARNING, INFO or DEBUG

        :raises toto: :py:class:`BaseException`.
        """
        self.output_dir = output_dir

        # create output dir if it does not already exist
        if not op.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if logger.hasHandlers():
            logger.handlers.clear()

        # file logger
        logger_file_handler = logging.FileHandler(
            op.join(self.output_dir, strftime("%Y-%m-%d-%H%M%S", gmtime()) + '_info.log'), mode='w')
        logger_file_handler.setFormatter(logging.Formatter(default.logger_format))
        logger_file_handler.setLevel(logging.INFO)
        logger.addHandler(logger_file_handler)

        # console logger
        logger_stream_handler = logging.StreamHandler(stream=sys.stdout)
        try:
            logger_stream_handler.setLevel(verbosity)
            logger.setLevel(verbosity)
        except ValueError:
            logger.warning('Logging level was not recognized. Using INFO.')
            logger_stream_handler.setLevel(logging.INFO)

        logger.addHandler(logger_stream_handler)

        logger.error("Logger has been set to: " + logging.getLevelName(logger_stream_handler.level))

    def __del__(self):
        logger.debug('Deformetrica.__del__()')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # remove previously set env variable
        if 'OMP_NUM_THREADS' in os.environ:
            del os.environ['OMP_NUM_THREADS']

        logging.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug('Deformetrica.__exit__()')

    ####################################################################################################################
    # Main methods.
    ####################################################################################################################

    def set_bounding_box(self, dataset):
        bounding_boxes = np.array([object.bounding_box for obj in dataset.objects for object in obj])
        new_bounding_box = np.array([np.min(bounding_boxes[:, :, 0], axis=0), 
                                    np.max(bounding_boxes[:, :, 1], axis=0)]).T

        # new_bounding_box = np.zeros((dataset.dimension, 2))
        # new_bounding_box[:, 0] = np.min(bounding_boxes, axis = 0)[:, 0]
        # new_bounding_box[:, 1] = np.max(bounding_boxes, axis = 0)[:, 1]

        return new_bounding_box

    def registration(self, template_spec, dataset_spec, model_options={}, estimator_options={}, 
                            write_output=True):
        """ Estimates the best possible deformation between two sets of objects.
        Note: Particular case of the deformable template application with a fixed template object.

        :dict template_spec: Task description  
                              + some hyper-parameters for the objects and the deformations used.
        :dict dataset_spec: Paths to input objects from which a model will be estimated.
        :dict model_options: Details about the model .
        :dict estimator_options: Details about the optimization method. 
        :bool write_output: Boolean that defines is output files will be written to disk.
        """
        # Check and completes the input parameters.
        model_options, estimator_options = self.further_initialization('Registration', 
                                            model_options, dataset_spec, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(**dataset_spec)
        assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."
        
        # Instantiate model.
        model = DeformableTemplate(template_spec, dataset.n_subjects, **model_options)
        model.initialize_noise_variance(dataset)

        # Instantiate estimator.
        estimator = self.__instantiate_estimator(model, dataset, estimator_options)

        try:
            self.__launch_estimator(estimator, write_output)
        finally:
            model.cleanup()

        return model

    def deformable_template(self, template_spec, dataset_spec, model_options={}, 
                        estimator_options={}, write_output=True):
        """ Estimate deformable template model.
        Given a family of objects, the atlas model learns a template shape (mean of the objects),
        + a low number of coordinates for each object from this template shape.

        :dict template_spec: Task description + some hyper-parameters for the objects and the deformations used.
        :dict dataset_spec: Paths to input objects from which a model will be estimated.
        :dict model_options: Details about the model .
        :dict estimator_options: Details about the optimization method. 
        :bool write_output: Boolean that defines is output files will be written to disk.
        """
        # Check and completes the input parameters.
        model_options, estimator_options = self.further_initialization('DeformableTemplate', 
                                                                model_options, dataset_spec, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(**dataset_spec)
        assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

        # Instantiate model.
        model = DeformableTemplate(template_spec, dataset.n_subjects, **model_options)
        model.initialize_noise_variance(dataset)

        # Instantiate estimator.
        estimator = self.__instantiate_estimator(model, dataset, estimator_options)

        try:
            self.__launch_estimator(estimator, write_output)
        finally:
            model.cleanup()

        return model

    def bayesian_atlas(self, template_spec, dataset_spec,
                                model_options={}, estimator_options={}, write_output=True):
        """ Estimate bayesian atlas.
        Bayesian version of the deformable template. 
        In addition to the template and the registrations, the variability of the geometry and the data noise are learned.

        :dict template_spec: Task description  
                              + some hyper-parameters for the objects and the deformations used.
        :dict dataset_spec: Paths to input objects from which a model will be estimated.
        :dict model_options: Details about the model .
        :dict estimator_options: Details about the optimization method. 
        :bool write_output: Boolean that defines is output files will be written to disk.
        """
        # Check and completes the input parameters.
        model_options, estimator_options = self.further_initialization('BayesianAtlas', 
                                                    model_options, dataset_spec, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset( **dataset_spec)
        assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

        # Instantiate model.
        model = BayesianAtlas(template_spec, **model_options)
        individual_RER = model.initialize_random_effects_realization(dataset.n_subjects,
                                                                                **model_options)
        model.initialize_noise_variance(dataset, individual_RER)

        # Instantiate estimator.
        estimator_options['individual_RER'] = individual_RER
        estimator = self.__instantiate_estimator(model, dataset, estimator_options)

        try:
            self.__launch_estimator(estimator, write_output)
        finally:
            model.cleanup()

        return model, estimator.individual_RER

    def geodesic_regression(self, template_spec, dataset_spec,
                                     model_options={}, estimator_options={}, write_output=True):
        """ Construct a shape trajectory that is as close as possible to the given targets at the given times.

        :dict template_spec: Task description+ some hyper-parameters for the objects 
                            and the deformations used.
        :dict dataset_spec: Paths to input objects from which a model will be estimated.
        :dict model_options: Details about the model
        :dict estimator_options: Details about the optimization method. 
        :bool write_output: Boolean that defines is output files will be written to disk.
        """
        # Check and completes the input parameters.
        model_options, estimator_options = self.further_initialization('Regression', 
                                            model_options, dataset_spec, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(**dataset_spec)

        assert (dataset.is_time_series()), "Cannot estimate a geodesic regression from a non-time-series dataset."
        
        # Handle the case where template has smaller bounding box than the subjects
        new_bounding_box = self.set_bounding_box(dataset)

        # Instantiate model.
        model = GeodesicRegression(template_spec, **model_options, new_bounding_box=new_bounding_box)
        model.initialize_noise_variance(dataset)

        # Instantiate estimator.
        estimator = self.__instantiate_estimator(model, dataset, estimator_options)

        try:
            self.__launch_estimator(estimator, write_output)
        finally:
            model.cleanup()

        return model
    
    def piecewise_geodesic_regression(self, template_spec, dataset_spec,
                                     model_options={}, estimator_options={}, write_output=True):
        """ Construct a shape trajectory that is as close as possible to the given targets 
            at the given times.

        :dict template_spec: Task description  
                              + some hyper-parameters for the objects and the deformations used.
        :dict dataset_spec: Paths to input objects from which a model will be estimated.
        :dict model_options: Details about the model .
        :dict estimator_options: Details about the optimization method. 
        :bool write_output: Boolean that defines is output files will be written to disk.
        """
        # Check and completes the input parameters.
        model_options, estimator_options = self.further_initialization(
                                    'PiecewiseRegression', model_options, dataset_spec, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(**dataset_spec)
        
        assert (dataset.is_time_series()), "Cannot estimate a geodesic regression from a non-time-series dataset."

        # Handle the case where template has smaller bounding box than the subjects
        new_bounding_box = self.set_bounding_box(dataset)

        # Instantiate model.
        model = PiecewiseGeodesicRegression(template_spec, **model_options, new_bounding_box = new_bounding_box)
        model.initialize_noise_variance(dataset)

        # Instantiate estimator.
        estimator = self.__instantiate_estimator(model, dataset, estimator_options)

        try:
            self.__launch_estimator(estimator, write_output)
        finally:
            model.cleanup()

        return model
    
    def piecewise_bayesian_geodesic_regression(self, template_spec, dataset_spec,
                                     model_options={}, estimator_options={}, write_output=True):
        """ Construct a shape trajectory that is as close as possible to the given targets at the given times.

        :dict template_spec: Task description  
                              + some hyper-parameters for the objects and the deformations used.
        :dict dataset_spec: Paths to input objects from which a model will be estimated.
        :dict model_options: Details about the model .
        :dict estimator_options: Details about the optimization method. 
        :bool write_output: Boolean that defines is output files will be written to disk.
        """
        # Check and completes the input parameters.
        model_options, estimator_options = self.further_initialization('BayesianGeodesicRegression', 
                                                        model_options, dataset_spec, estimator_options)
        # Instantiate dataset.
        dataset = create_dataset(**dataset_spec)

        # Check there are several subjects, with only 1 visit each. 
        assert (dataset.is_cross_sectional() and not dataset.is_time_series()), \
            "Cannot estimate a bayesian regression from a cross-sectional or time-series dataset."

        # Handle the case where template has smaller bounding box than the subjects
        new_bounding_box = self.set_bounding_box(dataset)

        # Instantiate model.
        model = BayesianPiecewiseGeodesicRegression(template_spec, **model_options, new_bounding_box = new_bounding_box)
        individual_RER = model.initialize_random_effects_realization(dataset.n_subjects,
                                                                                 **model_options)
        model.initialize_noise_variance(dataset, individual_RER)

        # Instantiate estimator.
        estimator_options['individual_RER'] = individual_RER
        estimator = self.__instantiate_estimator(model, dataset, estimator_options)

        try:
            self.__launch_estimator(estimator, write_output)
        finally:
            model.cleanup()

        return model
    
    def atlas_at_t0_for_initialisation(self, template_spec, dataset_spec,
                                     model_options, estimator_options, write_output):
        estimator_options_ = copy.deepcopy(estimator_options)

        t0 = model_options["t0"]

        # Select subjects around t0
        new_dataset_spec = { 
            'subject_ids': [dataset_spec['subject_ids'][i]\
                    for i in range(dataset_spec['n_subjects']) if t0 - 3 < age(dataset_spec, i) < t0 + 3],
            'filenames': [[dataset_spec['filenames'][i][0]]\
                    for i in range(dataset_spec['n_subjects']) if t0 - 3 < age(dataset_spec, i) < t0 + 3],
            'visit_ages': [[age(dataset_spec, i)]\
                    for i in range(dataset_spec['n_subjects']) if t0 - 3 < age(dataset_spec, i) < t0 + 3],
                            }
        
        logger.info("\n >>>> 0_Atlas estimation >>>> \n")
        
        # Check and completes the input parameters.
        model_options, estimator_options_ = self.further_initialization('KernelRegression', 
                                            model_options, dataset_spec, estimator_options_)
        
        # Kernel weigting of the subjects
        model_options['visit_ages'] = new_dataset_spec['visit_ages']
        model_options['time'] = t0
        model_options["freeze_template"] = False
        #model_options["freeze_momenta"] = True
        estimator_options_["multiscale_objects"] = True

        # Instantiate dataset.
        dataset = create_dataset(**new_dataset_spec)
        assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

        model = DeformableTemplate(template_spec, dataset.n_subjects, **model_options)
        model.initialize_noise_variance(dataset)

        estimator = self.__instantiate_estimator(model, dataset, estimator_options_)

        try:
            self.__launch_estimator(estimator, write_output)
        finally:
            template = model.template_path
            object_name = model.objects_name[0]
            model.cleanup()

        return template, object_name

    def piecewise_regression_for_initialisation(self, template_spec, dataset_spec,
                                     model_options, estimator_options, write_output):
        estimator_options_ = copy.deepcopy(estimator_options)

        model_options, estimator_options_ = self.further_initialization(
                                        'PiecewiseRegression', model_options, dataset_spec, estimator_options_)
        
        logger.info("\n >>>> 1_piecewise_regression >>>> \n")

        new_dataset_spec = make_dataset_timeseries(dataset_spec)

        dataset = create_dataset(**new_dataset_spec)
        
        assert (dataset.is_time_series()), "Cannot estimate a geodesic regression from a non-time-series dataset."

        new_bounding_box = self.set_bounding_box(dataset)

        model_options["freeze_template"] = True # important

        model = PiecewiseGeodesicRegression(template_spec, **model_options, new_bounding_box = new_bounding_box)
        model.initialize_noise_variance(dataset)

        estimator = self.__instantiate_estimator(model, dataset, estimator_options_)

        try:
            self.__launch_estimator(estimator, write_output)
        finally:
            initial_cp_path = model.cp_path
            initial_momenta_path = model.momenta_path
            flow_path = model.geodesic.flow_path
            model.cleanup()
        
        return initial_cp_path, initial_momenta_path, flow_path
    
    def parallel_transport_subject(self, i, template_spec, dataset_spec,
                                    model_options, estimator_options, main_output_dir3, 
                                    flow_path, object_name, initial_cp_path, initial_momenta_path, 
                                    registration_model, start_time, target_time):
        
        # Parallel transport from t_i to t0 -we keep original tmin, tmax, t0 and tR
        logger.info("\n >>>> Parallel transport for subject {} - id {} to {}".format(i, id(dataset_spec, i), target_time))

        model_options_ = {"tmin" : mini(dataset_spec), 
                            "tmax" : maxi(dataset_spec), 
                            "start_time" : int(start_time), "target_time" : target_time,
                            'initial_momenta_to_transport' : registration_model.momenta_path,
                            "initial_cp" : initial_cp_path,
                            "initial_momenta_tR" : initial_momenta_path,
                            "perform_shooting" : False}
        
        model_options_ = complete(model_options, model_options_)

        print("start_time", start_time)
        print("target_time", target_time)
        
        template_spec[object_name]['filename'] = flow_path[model_options_["t0"]]

        if target_time != model_options["t0"]:
            self.output_dir = op.join(main_output_dir3, "Subject_{}_age_{}_to_{}".format(i, age(dataset_spec, i), target_time))
        else:
            self.output_dir = op.join(main_output_dir3, "Subject_{}_age_{}".format(i, age(dataset_spec, i)))
        
        make_dir(self.output_dir)


        model_options_, _ = self.further_initialization('ParallelTransport', model_options_)

        trajectory = launch_piecewise_parallel_transport(template_spec, output_dir=self.output_dir, 
                                                        overwrite = estimator_options["overwrite"],
                                                        **model_options_)        
        # residuals = compute_distance_to_flow(template_spec, output_dir=self.output_dir, 
        #                               flow_path = flow_path, **model_options_)
        # #print("\n residuals", residuals)
        # r.plot(residuals, i, time, age(dataset_spec, i))

        return trajectory

    def registration_and_transport_for_initialization(self, template_spec, dataset_spec,
                                                    model_options, estimator_options, main_output_dir2,
                                                    main_output_dir3, flow_path, object_name, initial_cp_path, 
                                                    initial_momenta_path, target_times = []):
        
        accepted_difference = (1/model_options["time_concentration"]) / 2 + 0.01

        transported_momenta_path = {t: [] for t in target_times}

        r = PlotResiduals(main_output_dir3)

        for i, subject in enumerate(dataset_spec['filenames']):

            self.output_dir = op.join(main_output_dir2, "Subject_{}_age_{}".format(i, age(dataset_spec, i)))
            make_dir(self.output_dir)

            new_dataset_spec = dataset_for_registration(subject, age(dataset_spec, i), 
                                                        id(dataset_spec, i))
            
            for start_time, template in flow_path.items():
                if np.abs(start_time - age(dataset_spec, i)) <= accepted_difference:
                    template_spec[object_name]['filename'] = template
                    break
                
            logger.info("\n >>>> Registration for subject {}".format(id(dataset_spec, i)))

            model_options_ = copy.deepcopy(model_options)
            model_options_["initial_cp"] = initial_cp_path
            model_options_["kernel_regression"] = False
            
            estimator_options_ = options_for_registration(estimator_options)

            model = self.registration(template_spec, new_dataset_spec,
                                            model_options_, estimator_options_, write_output=True)
            
            trajectory = dict()
            for target_time in target_times:
                if target_time not in trajectory.keys():
                    trajectory = self.parallel_transport_subject(i, template_spec, 
                                                            dataset_spec,
                                                            model_options, estimator_options, 
                                                            main_output_dir3, flow_path, object_name, initial_cp_path, 
                                                            initial_momenta_path, model, start_time,
                                                            target_time = target_time)
                
                transported_momenta_path[target_time].append(trajectory[target_time])
        
        #r.end(residuals, i, age(dataset_spec, i))

        return transported_momenta_path, model_options_

    def initialize_piecewise_bayesian_geodesic_regression(self, template_spec, dataset_spec,
                                     model_options={}, estimator_options={}, write_output=True):
        """ Construct a shape trajectory that is as close as possible to the given targets at the given times.

        :dict template_spec: Task description  
                              + some hyper-parameters for the objects and the deformations used.
        :dict dataset_spec: Paths to input objects from which a model will be estimated.
        :dict model_options: Details about the model .
        :dict estimator_options: Details about the optimization method. 
        :bool write_output: Boolean that defines is output files will be written to disk.
        """
        main_output_dir = self.output_dir  
        self.output_dir = op.join(main_output_dir, "0_template_estimation")
        make_dir(self.output_dir)          
        main_output_dir1 = op.join(main_output_dir, "1_piecewise_regression")
        make_dir(main_output_dir1)
        main_output_dir2 = op.join(main_output_dir, "2_registration")
        make_dir(main_output_dir2)
        main_output_dir3 = op.join(main_output_dir, "3_parallel_transport")
        make_dir(main_output_dir3)

        ages_histogram(dataset_spec, main_output_dir)

        ## 0 - estimate initial template shape
        estimator_options["overwrite"] = False

        # template, object_name = self.atlas_at_t0_for_initialisation(template_spec, dataset_spec,
        #                                             model_options, estimator_options, write_output)

        object_name = "img"
        #template_spec[object_name]['filename'] = template

        ## 1 - estimate piecewise regression
        self.output_dir = main_output_dir1

        initial_cp_path, initial_momenta_path, flow_path = \
        self.piecewise_regression_for_initialisation(template_spec, dataset_spec,
                                                    model_options, estimator_options, write_output)

        ## 2 - Register the mean trajectory to each observation at t_i
        transported_momenta_path, model_options_ = self.registration_and_transport_for_initialization(template_spec, 
                                                                                                dataset_spec,
                                     model_options, estimator_options, main_output_dir2,
                                     main_output_dir3, flow_path, object_name, initial_cp_path, 
                                     initial_momenta_path, target_times = [model_options["t0"], 32])
        
        # ICA
        for target_time in [model_options["t0"], 32]:
            logger.info("\n >>>> Tangent space ICA at time {}".format(target_time))
            if target_time == model_options["t0"]:
                self.output_dir = op.join(main_output_dir, "4_ICA")
            else:
                self.output_dir = op.join(main_output_dir, "4_ICA_{}".format(target_time))
            make_dir(self.output_dir)

            model_options_["perform_shooting"] = True
            model_options_["tmin"] = mini(dataset_spec)
            model_options_["tmax"] = maxi(dataset_spec)

            path_to_sources, path_to_mm = perform_ICA(self.output_dir, initial_cp_path, 
                                                    model_options["deformation_kernel_width"], 
                                                    initial_momenta_path, transported_momenta_path[target_time], 
                                                    model_options["number_of_sources"], target_time,
                                                    model_options["tR"], overwrite = estimator_options["overwrite"])

            model_options_ = complete(model_options, model_options_)
            model_options_["initial_momenta"] = initial_momenta_path
            model_options_["initial_momenta_tR"] = initial_momenta_path
            plot_ica(path_to_sources, path_to_mm, template_spec, dataset_spec,
                    output_dir=self.output_dir, overwrite = estimator_options["overwrite"],
                    nb_components = model_options_["num_component"], target_time = target_time,
                    **model_options_)
        
    def initialized_piecewise_bayesian_geodesic_regression(self, template_spec, dataset_spec,
                                     model_options={}, estimator_options={}, write_output=True):
        
        main_output_dir = self.output_dir      
        
        logger.info("\n >>>> Piecewise regression with space shift")
        
        estimator_options["overwrite"] = False

        ############## Get initialized parameters ##############

        # Get average trajectory initialization
        self.output_dir = op.join(main_output_dir, "1_piecewise_regression")
        initial_cp_path, initial_momenta_path, _, _ = \
        self.piecewise_regression_for_initialisation(template_spec, dataset_spec,
                                                    model_options, estimator_options, write_output)
        
        # Get ICA output
        self.output_dir = op.join(main_output_dir, "4_ICA")
        path_to_sources, path_to_mm = perform_ICA(self.output_dir, overwrite = False)

        self.output_dir = op.join(main_output_dir, "5_piecewise_regression_with_space_shift")
        make_dir(self.output_dir)

        model_options["initial_sources"] = path_to_sources
        model_options["initial_modulation_matrix"] = path_to_mm
        model_options["initial_momenta"] = initial_momenta_path
        model_options["initial_cp"] = initial_cp_path

        ############## Final Model ##############

        logger.info("\n >>>> Final model")

        estimator_options["overwrite"] = True

        # Check and completes the input parameters.
        model_options, estimator_options = self.further_initialization(
                                'BayesianGeodesicRegression', model_options, dataset_spec, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(**dataset_spec)

        # Check there are several subjects, with only 1 visit each. 
        assert (dataset.is_cross_sectional() and not dataset.is_time_series()), \
            "Cannot estimate a bayesian regression from a cross-sectional or time-series dataset."

        # Handle the case where template has smaller bounding box than the subjects
        new_bounding_box = self.set_bounding_box(dataset)

        # Instantiate model.
        model = BayesianPiecewiseGeodesicRegression(template_spec, **model_options, new_bounding_box = new_bounding_box)
        individual_RER = model.initialize_random_effects_realization(dataset.n_subjects,
                                                                                 **model_options)
        model.initialize_noise_variance(dataset, individual_RER)

        # Instantiate estimator.
        estimator_options['individual_RER'] = individual_RER
        estimator = self.__instantiate_estimator(model, dataset, estimator_options)

        try:
            self.__launch_estimator(estimator, write_output)
        finally:
            model.cleanup()

        return model
    
    def kernel_regression(self, time, template_spec, dataset_spec, model_options={}, 
                            estimator_options = {}, write_output = True):
        # Check and completes the input parameters.
        model_options, estimator_options = self.further_initialization('KernelRegression', 
                                            model_options, dataset_spec, estimator_options)
        visit_ages = [d[0] for d in dataset_spec['visit_ages']]
        original_template_spec = template_spec["Object_1"]
        
        time = int(time)
        model_options['time'] = time
        self.output_dir = self.output_dir + '/time_' + str(time)
        if not op.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Select the subjects that contribute to the atlas
        visit_ages = dataset_spec['visit_ages']
        total_weights = np.sum([ gaussian_kernel(time, age[0]) for age in dataset_spec['visit_ages'] ])
        weights = [ gaussian_kernel(time, age[0]) for age in visit_ages ]
        selection = [ i for i, w in enumerate(weights) if w > 0.01 ]
        
        # Adaptative kernel
        
        # Update the dataset accordingly
        new_dataset_spec = copy.deepcopy(dataset_spec)        
        new_dataset_spec['visit_ages'] = [[visit_ages[i][0]] for i in selection]
        new_dataset_spec['subject_ids'] = [dataset_spec['subject_ids'][i] for i in selection]
        new_dataset_spec['filenames'] = [dataset_spec['filenames'][i] for i in selection]
        model_options['visit_ages'] = new_dataset_spec['visit_ages']

        dataset = create_dataset(**new_dataset_spec)
        
        # Compute a mean image
        # if ".nii" in new_dataset_spec['filenames'][0][0]["img"]:
        #     data_list = [nib.load(f[0]["img"]) for f in new_dataset_spec['filenames']]
        #     mean = np.zeros((data_list[0].get_fdata().shape))
        #     for i, f in enumerate(data_list):
        #         mean += f.get_fdata() * (weights[i]/total_weights)
        #     image_new = nib.nifti1.Nifti1Image(mean, data_list[0].affine, data_list[0].header)
        #     output_image = original_template_spec.replace("mean", "mean_age_{}".format(time))
        #     nib.save(image_new, output_image)
        # else:            
        #     i = weights.index(max(weights))
        #     name = new_dataset_spec['filenames'][i][0]["img"]
        #     name = "/home/fleur.gaudfernau/Necker_atlas_SPT/Inner_cortical_surface_bayesian_piecewise_regression_60_subjects_with_init_/Kernel_regression/template_smooth_2000.vtk"
        #template_spec['img']["filename"] = name
        
        # Instantiate model.
        print("Number of subjects:", len(new_dataset_spec['filenames']), "at age", time)
        model = DeformableTemplate(template_spec, dataset.n_subjects, **model_options)
        model.initialize_noise_variance(dataset)

        # Instantiate estimator.
        estimator = self.__instantiate_estimator(model, dataset, estimator_options)

        try:
            self.__launch_estimator(estimator, write_output)
        finally:
            model.cleanup()

        return 

    def kernel_regression_2(self, template_spec, dataset_spec,
                                     model_options={}, estimator_options={}, write_output=True):
        """ Estimate a deterministic atlas
        """
        # Check and completes the input parameters.
        model_options, estimator_options = self.further_initialization('KernelRegression', 
                                            model_options, dataset_spec, estimator_options)
        
        visit_ages = [d[0] for d in dataset_spec['visit_ages']]
        tmin = int(min(visit_ages))
        tmax = int(max(visit_ages))
        original_template_spec = template_spec['img']["filename"]
        root_output_dir = self.output_dir
        
        for time in range(tmin, tmax):
            model_options['time'] = time

            self.output_dir = root_output_dir + '/time_' + str(time)
            if not op.exists(self.output_dir):
                os.makedirs(self.output_dir)

            # Select the subjects that contribute to the atlas
            selection = []
            weights = []
            total_weights = np.sum([gaussian_kernel(time, age[0]) for age in dataset_spec['visit_ages']])
            for i, age in enumerate(dataset_spec['visit_ages']):
                weight = gaussian_kernel(time, age[0])
                if weight > 0.01:
                    selection.append(i)
                    weights.append(weight)

            # Update the dataset accordingly
            new_dataset_spec = copy.deepcopy(dataset_spec)
            new_dataset_spec['visit_ages'] = [[age] for i, age in enumerate(visit_ages) if i in selection]
            new_dataset_spec['subject_ids'] = [id for i, id in enumerate(dataset_spec['subject_ids'])\
                                                        if i in selection]
            new_dataset_spec['filenames'] = [name for i, name in enumerate(dataset_spec['filenames'])\
                                                            if i in selection]
            model_options['visit_ages'] = new_dataset_spec['visit_ages']

            dataset = create_dataset(**new_dataset_spec)
            
            # Compute a mean image
            data_list = [nib.load(f[0]["img"]) for f in new_dataset_spec['filenames']]
            mean = np.zeros((data_list[0].get_fdata().shape))
            for i, f in enumerate(data_list):
                mean += f.get_fdata() * (weights[i]/total_weights)
            image_new = nib.nifti1.Nifti1Image(mean, data_list[0].affine, data_list[0].header)
            output_image = original_template_spec.replace("mean", "mean_age_{}".format(time))
            nib.save(image_new, output_image)
            template_spec['img']["filename"] = output_image
            
            # Instantiate model.
            model = DeformableTemplate(template_spec, dataset.n_subjects, **model_options)
            model.initialize_noise_variance(dataset)

            # Instantiate estimator.
            estimator = self.__instantiate_estimator(model, dataset, estimator_options)

            try:
                self.__launch_estimator(estimator, write_output)
            finally:
                model.cleanup()

        return 

    def compute_parallel_transport(self, template_spec, model_options={}):
        """ Given a known progression of shapes, to transport this progression onto a new shape.

        :dict template_spec: Task description  
        :dict model_options: Details about the model .
        """
        # Check and completes the input parameters.
        model_options, _ = self.further_initialization('ParallelTransport', model_options)

        launch_parallel_transport(template_spec, output_dir=self.output_dir, **model_options)
    
    def compute_piecewise_parallel_transport(self, template_spec, model_options={}):
        """ Given a known progression of shapes, to transport this progression onto a new shape.

        :dict template_spec: Task description  
        :dict model_options: Details about the model .
        """
        # Check and completes the input parameters.
        model_options, _ = self.further_initialization('ParallelTransport', model_options)

        launch_piecewise_parallel_transport(template_spec, output_dir=self.output_dir, **model_options)

    def compute_shooting(self, template_spec, model_options={}):
        """ If control points and momenta corresponding to a deformation have been obtained, 
        it is possible to shoot the corresponding deformation of obtain the flow of a shape under this deformation.

        :dict template_spec: Task description  
                              + some hyper-parameters for the objects and the deformations used.
        :dict model_options: Details about the model .
        """

        # Check and completes the input parameters.
        model_options, _ = self.further_initialization('ParallelTransport', model_options)

        # Launch.
        compute_shooting(template_spec, output_dir=self.output_dir, **model_options)

    ####################################################################################################################
    # Auxiliary methods.
    ####################################################################################################################

    @staticmethod
    def __launch_estimator(estimator = GradientAscent, write_output=True):
        """
        Launch the estimator. This will iterate until a stop condition is reached.

        :param estimator:   Estimator that is to be used.
                            eg: :class:`GradientAscent <core.estimators.gradient_ascent.GradientAscent>`
        """        
        if estimator.stop:
            return 
        
        start_time = time.time()
        logger.info('>> Started estimator: {}\n'.format(estimator.name))
        write = estimator.update() 
        duration = time.time() - start_time

        if write_output and write:
            print("Final writing")
            estimator.write()

        if duration > 86400: 
            fmt = "%d days, %H hours, %M minutes and %S seconds"
        elif duration > 3600:  
            fmt = "%H hours, %M minutes and %S seconds"
        elif duration > 60: 
            fmt = "%M minutes and %S seconds"
        else:
            fmt = "%S seconds"
    
        logger.info(f">> Estimation took: {strftime(fmt, gmtime(duration))}")

        
    def __instantiate_estimator(self, model, dataset, estimator_options):
        optimization_method = estimator_options['optimization_method'].lower()
        
        if optimization_method == 'GradientAscent'.lower():
            estimator = GradientAscent

        elif optimization_method == 'StochasticGradientAscent'.lower():
            estimator = StochasticGradientAscent
            # set batch number
            if dataset.total_number_of_observations < 6 and dataset.n_subjects < 6:
                print("\nDefaulting to GradientAscent optimizer")
                estimator_options['optimization_method'] = 'GradientAscent'
                estimator = GradientAscent
            else:
                batch_size = 4
                if dataset.total_number_of_observations > 0:
                    estimator_options["number_of_batches"] = math.ceil(dataset.total_number_of_observations/batch_size)
                else:
                    estimator_options["number_of_batches"] = math.ceil(dataset.n_subjects/batch_size)

                print("\nSetting number of batches to", estimator_options["number_of_batches"])

        else:
            estimator = default

        logger.debug(estimator_options)
        return estimator(model, dataset, output_dir=self.output_dir, **estimator_options)

    def further_initialization(self, model_type, model_options, dataset_spec = None, 
                                estimator_options = None, time = None):
        model_type = model_type.lower()

        if dataset_spec is None or estimator_options is None:
            assert model_type in ['Shooting'.lower(), 'ParallelTransport'.lower()], \
            'Only shooting and parallel transport can run without dataset and estimator.'

        if estimator_options is not None:
            if 'state_file' not in estimator_options:
                estimator_options['state_file'] = default.state_file
            if 'load_state_file' not in estimator_options:
                estimator_options['load_state_file'] = default.load_state_file
            if 'multiscale_momenta' not in estimator_options: 
                estimator_options['multiscale_momenta'] = default.multiscale_momenta
            if 'multiscale_objects' not in estimator_options: 
                estimator_options['multiscale_objects'] = default.multiscale_objects
            if 'multiscale_strategy' not in estimator_options: 
                estimator_options['multiscale_strategy'] = default.multiscale_strategy

        if 'perform_shooting' not in model_options: 
            model_options['perform_shooting'] = default.perform_shooting
        if 'freeze_template' not in model_options:
            model_options['freeze_template'] = default.freeze_template
        if 'initial_cp' not in model_options:
            model_options['initial_cp'] = default.initial_cp
        if 'deformation_kernel_width' not in model_options:
            model_options['deformation_kernel_width'] = default.deformation_kernel_width
        if 't0' not in model_options:
            model_options['t0'] = default.t0
        if 'tR' not in model_options:
            model_options['tR'] = []
        if 't1' not in model_options:
            model_options['t1'] = default.t0 
        if 'initial_modulation_matrix' not in model_options:
            model_options['initial_modulation_matrix'] = default.initial_modulation_matrix
        if 'number_of_sources' not in model_options:
            model_options['number_of_sources'] = default.number_of_sources
        if 'downsampling_factor' not in model_options:
            model_options['downsampling_factor'] = default.downsampling_factor
        if 'interpolation' not in model_options: 
            model_options['interpolation'] = default.interpolation
        if "kernel_regression" not in model_options:
            model_options['kernel_regression'] = False

        # try and automatically set best number of thread per spawned process if not overridden by uer
        if 'OMP_NUM_THREADS' not in os.environ:
            hyperthreading = utilities.has_hyperthreading()
            omp_num_threads = math.floor(os.cpu_count() / 1)

            if hyperthreading:
                omp_num_threads = math.ceil(omp_num_threads / 2)

            omp_num_threads = max(1, int(omp_num_threads))

            os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

        # If longitudinal model and t0 is not initialized, initializes it.
        if model_type in ['Regression'.lower(), 'PiecewiseRegression'.lower(), 'BayesianGeodesicRegression'.lower()]:
            
            assert 'visit_ages' in dataset_spec, 'Visit ages are needed to estimate a Regression'

            if model_options['t0'] is None:
                ages = [a[0] for a in dataset_spec['visit_ages']]   
                logger.info('>> Initial t0 set to the minimal visit age: %.2f' % min(ages))
                model_options['t0'] = min(ages)
            else:
                logger.info('>> Initial t0 set by the user to %.2f' % (model_options['t0']))

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        try:
            torch.multiprocessing.set_start_method("spawn")
            torch.multiprocessing.set_sharing_strategy('file_descriptor')
            logger.debug("nofile (soft): " + str(rlimit[0]) + ", nofile (hard): " + str(rlimit[1]))
            resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))
        except RuntimeError as e:
            logger.warning(str(e))
        except AssertionError:
            logger.warning('Could not set torch settings.')
        except ValueError:
            logger.warning('Could not set max open file. Currently using: ' + str(rlimit))

        if estimator_options is not None:
            if estimator_options['state_file'] is None:
                path_to_state_file = op.join(self.output_dir, "deformetrica-state.p")
                logger.info('>> Deformetrica state is saved in: %s.' % path_to_state_file)
                if op.isfile(path_to_state_file):
                    os.remove(path_to_state_file)
                estimator_options['state_file'] = path_to_state_file
            else:
                if op.exists(estimator_options['state_file']):
                    estimator_options['load_state_file'] = True
                    logger.info(
                        '>> Deformetrica will resume computation from state file %s.' % estimator_options['state_file'])

        # Freeze the fixed effects in case of a registration.
        if model_type == 'Registration'.lower():
            model_options['freeze_template'] = True

        elif model_type == 'KernelRegression'.lower():
            model_options['kernel_regression'] = True
            model_options['time'] = time
            model_options['visit_ages'] = []

        # Initialize the number of sources if needed.
        if model_type in ['BayesianGeodesicRegression'.lower()]:

            if model_options['initial_modulation_matrix'] is None and model_options['number_of_sources'] is None:
                model_options['number_of_sources'] = 4
                logger.info('>> No initial modulation matrix neither number of sources. '
                            'The latter will be ARBITRARILY defaulted to %d.' % model_options['number_of_sources'])
        
            if 'sources_proposal_std' not in estimator_options:
                estimator_options['sources_proposal_std'] = 1

                estimator_options['individual_proposal_distributions'] = {
                    'sources': MultiScalarNormalDistribution(std=estimator_options['sources_proposal_std'])}

        logger.info("Downsampling factor set to: {}".format(model_options['downsampling_factor']))
        logger.info("Deformation kernel width set to: {}".format(model_options['deformation_kernel_width']))

        return model_options, estimator_options
