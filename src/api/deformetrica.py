import logging
import os
import time

import torch
import math
from sys import platform

from core import default
from core.default import logger_format
from in_out.array_readers_and_writers import read_2D_array, read_3D_array
from launch.compute_parallel_transport import compute_parallel_transport
from launch.compute_shooting import run_shooting
from launch.estimate_affine_atlas import instantiate_affine_atlas_model
from launch.estimate_bayesian_atlas import instantiate_bayesian_atlas_model
from launch.estimate_deterministic_atlas import instantiate_deterministic_atlas_model
from launch.estimate_geodesic_regression import instantiate_geodesic_regression_model
from launch.estimate_longitudinal_atlas import instantiate_longitudinal_atlas_model
from launch.estimate_principal_geodesic_analysis import instantiate_principal_geodesic_model

from in_out.dataset_functions import create_dataset

from core.models.deterministic_atlas import DeterministicAtlas

from core.estimators.scipy_optimize import ScipyOptimize
from core.estimators.gradient_ascent import GradientAscent
from core.estimators.mcmc_saem import McmcSaem

logger = logging.getLogger(__name__)


class Deformetrica:
    def __init__(self, output_dir=default.output_dir, verbosity='DEBUG'):
        self.output_dir = output_dir

        # create output dir if it does not already exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # set logging level
        try:
            log_level = logging.getLevelName(verbosity)
            logging.basicConfig(level=log_level, format=logger_format)
        except ValueError:
            logger.warning('Logging level was not recognized. Using INFO.')
            log_level = logging.INFO

        logger.debug('Using verbosity level: ' + verbosity)
        logging.basicConfig(level=log_level, format=logger_format)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug('Deformetrica.__exit__()')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def estimate_deterministic_atlas(self, template_specifications, dataset_specifications,
                                     estimator_options={}, model_options={}, write_output=True):
        """
        Estimate deterministic atlas
        """
        # Check and completes the input parameters.
        template_specifications, estimator_options, model_options, tensor_types = self.__further_initialization(
            'DeterministicAtlas', template_specifications, dataset_specifications, estimator_options, model_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications, tensor_types, **dataset_specifications)
        assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

        # Instantiate model.
        statistical_model = instantiate_deterministic_atlas_model(dataset, template_specifications, **model_options)

        # Instantiate estimator.
        estimator = self.__instantiate_estimator(
            statistical_model, dataset, self.output_dir, estimator_options, default=ScipyOptimize)

        # Launch.
        self.__launch_estimator(estimator, write_output)

        return statistical_model

    def estimate_bayesian_atlas(self, template_specifications, dataset_specifications,
                                estimator_options={}, model_options={}, write_output=True):
        """
        Estimate bayesian atlas
        """
        # Check and completes the input parameters.
        (template_specifications, estimator_options, model_options, tensor_types) = self.__further_initialization(
            'BayesianAtlas', template_specifications, dataset_specifications, estimator_options, model_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications, tensor_types, **dataset_specifications)
        assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

        # Instantiate model.
        statistical_model, individual_RER = instantiate_bayesian_atlas_model(
            dataset, template_specifications, **model_options)

        # Instantiate estimator.
        estimator_options['individual_RER'] = individual_RER
        estimator = self.__instantiate_estimator(
            statistical_model, dataset, self.output_dir, estimator_options, default=ScipyOptimize)

        # Launch.
        self.__launch_estimator(estimator, write_output)

        return statistical_model

    def estimate_longitudinal_atlas(self, template_specifications, dataset_specifications,
                                    estimator_options={}, model_options={}, write_output=True):
        """
        Estimate longitudinal atlas
        """

        # Check and completes the input parameters.
        (template_specifications, estimator_options, model_options, tensor_types) = self.__further_initialization(
            'LongitudinalAtlas', template_specifications, dataset_specifications, estimator_options, model_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications, tensor_types, **dataset_specifications)
        assert (not dataset.is_cross_sectional() and not dataset.is_time_series()), \
            "Cannot estimate an atlas from a cross-sectional or time-series dataset."

        # Instantiate model.
        statistical_model, individual_RER = instantiate_longitudinal_atlas_model(dataset, template_specifications,
                                                                                 **model_options)

        # Instantiate estimator.
        estimator_options['individual_RER'] = individual_RER
        estimator = self.__instantiate_estimator(
            statistical_model, dataset, self.output_dir, estimator_options, default=McmcSaem)

        # Launch.
        self.__launch_estimator(estimator, write_output)

        return statistical_model

    def estimate_affine_atlas(self, template_specifications, dataset, estimator=ScipyOptimize,
                              estimator_options={}, model_options={}, write_output=True):
        """
        Estimate affine atlas
        :return:
        """
        # sanitize estimator_options
        if 'output_dir' in estimator_options:
            raise RuntimeError('estimator_options cannot contain output_dir key')

        statistical_model = instantiate_affine_atlas_model(dataset, template_specifications, **model_options)

        # instantiate estimator
        estimator = estimator(statistical_model, dataset, output_dir=self.output_dir, **estimator_options)

        self.__launch_estimator(estimator, write_output)

        return statistical_model

    def estimate_longitudinal_metric_model(self):
        """
        TODO
        :return:
        """
        raise NotImplementedError

    def estimate_longitudinal_metric_registration(self):
        """
        TODO
        :return:
        """
        raise NotImplementedError

    def estimate_longitudinal_registration(self):
        """
        TODO
        :return:
        """
        raise NotImplementedError

    def estimate_geodesic_regression(self, template_specifications, dataset_specifications,
                                     estimator_options={}, model_options={}, write_output=True):
        """
        Estimate geodesic regression.
        """
        # Check and completes the input parameters.
        template_specifications, estimator_options, model_options, tensor_types = self.__further_initialization(
            'Regression', template_specifications, dataset_specifications, estimator_options, model_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications, tensor_types, **dataset_specifications)
        assert (dataset.is_time_series()), "Cannot estimate a geodesic regression from a non-time-series dataset."

        # Instantiate model.
        statistical_model = instantiate_geodesic_regression_model(dataset, template_specifications, **model_options)

        # Instantiate estimator.
        estimator = self.__instantiate_estimator(
            statistical_model, dataset, self.output_dir, estimator_options, default=ScipyOptimize)

        # Launch.
        self.__launch_estimator(estimator, write_output)

        return statistical_model

    def estimate_deep_pga(self):
        """
        TODO
        :return:
        """
        raise NotImplementedError

    def estimate_principal_geodesic_analysis(self, template_specifications, dataset, estimator=ScipyOptimize,
                                             estimator_options={}, model_options={}, write_output=True):
        """
        Estimate principal geodesic analysis
        """
        statistical_model, individual_RER = instantiate_principal_geodesic_model(self, dataset, template_specifications,
                                                                                 **model_options)

        # sanitize estimator_options
        if 'output_dir' in estimator_options:
            raise RuntimeError('estimator_options cannot contain output_dir key')

        # instantiate estimator
        estimator = estimator(statistical_model, dataset, output_dir=self.output_dir, individual_RER=individual_RER,
                              **estimator_options)

        """
        Launch
        """
        self.__launch_estimator(estimator, write_output)

        return statistical_model

    def compute_parallel_transport(self, template_specifications, dataset, model_options={}, write_output=True):

        """
        Get needed parameters.
        """
        initial_control_points = model_options['initial_control_points']
        initial_momenta = model_options['initial_momenta']
        initial_momenta_to_transport = model_options['initial_momenta_to_transport']
        initial_control_points_to_transport = model_options['initial_control_points_to_transport']
        deformation_kernel = model_options['deformation_kernel']

        """
        Compute parallel transport
        """
        if initial_control_points is None:
            raise RuntimeError("Please provide initial control points")
        if initial_momenta is None:
            raise RuntimeError("Please provide initial momenta")
        if initial_momenta_to_transport is None:
            raise RuntimeError("Please provide initial momenta to transport")

        control_points = read_2D_array(initial_control_points)
        initial_momenta = read_3D_array(initial_momenta)
        initial_momenta_to_transport = read_3D_array(initial_momenta_to_transport)

        if initial_control_points_to_transport is None:
            logger.warning(
                "initial-control-points-to-transport was not specified, "
                "I am assuming they are the same as initial-control-points")
            control_points_to_transport = control_points
            need_to_project_initial_momenta = False
        else:
            control_points_to_transport = read_2D_array(initial_control_points_to_transport)
            need_to_project_initial_momenta = True

        control_points_torch = torch.from_numpy(control_points).type(dataset.tensor_scalar_type)
        initial_momenta_torch = torch.from_numpy(initial_momenta).type(dataset.tensor_scalar_type)
        initial_momenta_to_transport_torch = torch.from_numpy(initial_momenta_to_transport).type(
            dataset.tensor_scalar_type)

        # We start by projecting the initial momenta if they are not carried at the reference progression control points.
        if need_to_project_initial_momenta:
            control_points_to_transport_torch = torch.from_numpy(control_points_to_transport).type(
                dataset.tensor_scalar_type)
            velocity = deformation_kernel.convolve(control_points_torch, control_points_to_transport_torch,
                                                   initial_momenta_to_transport_torch)
            kernel_matrix = deformation_kernel.get_kernel_matrix(control_points_torch)
            cholesky_kernel_matrix = torch.potrf(kernel_matrix)
            # cholesky_kernel_matrix = torch.Tensor(np.linalg.cholesky(kernel_matrix.data.numpy()).type_as(kernel_matrix))#Dirty fix if pytorch fails.
            projected_momenta = torch.potrs(velocity, cholesky_kernel_matrix).squeeze().contiguous()

        else:
            projected_momenta = initial_momenta_to_transport_torch

        compute_parallel_transport(control_points_torch, initial_momenta_torch, projected_momenta,
                                   template_specifications,
                                   output_dir=self.output_dir, **model_options)

    def compute_shooting(self, template_specifications, dataset, model_options={}, write_output=True):
        """
        Compute shooting
        """
        run_shooting(template_specifications, dataset, output_dir=self.output_dir, **model_options)

    @staticmethod
    def __launch_estimator(estimator, write_output=True):
        """
        Launch the estimator. This will iterate until a stop condition is reached.

        :param estimator:   Estimator that is to be used.
                            eg: :class:`GradientAscent <core.estimators.gradient_ascent.GradientAscent>`, :class:`ScipyOptimize <core.estimators.scipy_optimize.ScipyOptimize>`
        """
        start_time = time.time()
        logger.info('Started estimator: ' + estimator.name)
        estimator.update()
        end_time = time.time()

        if write_output:
            estimator.write()

        logger.info('Estimation took: ' + str(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))

    def __instantiate_estimator(self, statistical_model, dataset, output_dir, estimator_options, default=ScipyOptimize):
        if estimator_options['optimization_method_type'].lower() == 'GradientAscent'.lower():
            estimator = GradientAscent
        elif estimator_options['optimization_method_type'].lower() == 'ScipyLBFGS'.lower():
            estimator = ScipyOptimize
        elif estimator_options['optimization_method_type'].lower() == 'McmcSaem'.lower():
            estimator = McmcSaem
        else:
            estimator = default
        return estimator(statistical_model, dataset, output_dir=self.output_dir, **estimator_options)

    def __further_initialization(self, model_type, template_specifications, dataset_specifications,
                                 estimator_options, model_options):

        #
        # Initializes variables that will be checked.
        #
        if 'dense_mode' not in model_options:
            model_options['dense_mode'] = default.dense_mode
        if 'freeze_control_points' not in model_options:
            model_options['freeze_control_points'] = default.freeze_control_points
        if 'freeze_template' not in model_options:
            model_options['freeze_template'] = default.freeze_template
        if 'initial_control_points' not in model_options:
            model_options['initial_control_points'] = default.initial_control_points
        if 'initial_cp_spacing' not in model_options:
            model_options['initial_cp_spacing'] = default.initial_cp_spacing
        if 'deformation_kernel_width' not in model_options:
            model_options['deformation_kernel_width'] = default.deformation_kernel_width
        if 'deformation_kernel_type' not in model_options:
            model_options['deformation_kernel_type'] = default.deformation_kernel_type
        if 'number_of_threads' not in model_options:
            model_options['number_of_threads'] = default.number_of_threads
        if 't0' not in model_options:
            model_options['t0'] = default.t0
        if 'initial_time_shift_variance' not in model_options:
            model_options['initial_time_shift_variance'] = default.initial_time_shift_variance
        if 'initial_modulation_matrix' not in model_options:
            model_options['initial_modulation_matrix'] = default.initial_modulation_matrix
        if 'number_of_sources' not in model_options:
            model_options['number_of_sources'] = default.number_of_sources
        if 'initial_log_acceleration_variance' not in model_options:
            model_options['initial_log_acceleration_variance'] = default.initial_log_acceleration_variance
        if 'downsampling_factor' not in model_options:
            model_options['downsampling_factor'] = default.downsampling_factor
        if 'use_sobolev_gradient' not in model_options:
            model_options['use_sobolev_gradient'] = default.use_sobolev_gradient
        if 'sobolev_kernel_width_ratio' not in model_options:
            model_options['sobolev_kernel_width_ratio'] = default.sobolev_kernel_width_ratio

        if 'use_cuda' not in estimator_options:
            estimator_options['use_cuda'] = default.use_cuda
        if 'state_file' not in estimator_options:
            estimator_options['state_file'] = default.state_file
        if 'load_state_file' not in estimator_options:
            estimator_options['load_state_file'] = default.load_state_file

        #
        # Global variables for this method.
        #
        tensor_scalar_type = default.tensor_scalar_type
        tensor_integer_type = default.tensor_integer_type
        cuda_is_used = estimator_options['use_cuda']

        #
        # Check and completes the user-given parameters.
        #

        # Smoothing kernel width.
        if model_options['use_sobolev_gradient']:
            model_options['smoothing_kernel_width'] = \
                model_options['deformation_kernel_width'] * model_options['sobolev_kernel_width_ratio']

        # Dense mode.
        if model_options['dense_mode']:
            print('>> Dense mode activated. No distinction will be made between template and control points.')
            assert len(template_specifications) == 1, \
                'Only a single object can be considered when using the dense mode.'
            if not model_options['freeze_control_points']:
                model_options['freeze_control_points'] = True
                msg = 'With active dense mode, the freeze_template (currently %s) and freeze_control_points ' \
                      '(currently %s) flags are redundant. Defaulting to freeze_control_poi∂nts = True.' \
                      % (str(model_options['freeze_template']), str(model_options['freeze_control_points']))
                print('>> ' + msg)
            if model_options['initial_control_points'] is not None:
                model_options['initial_control_points'] = None
                msg = 'With active dense mode, specifying initial_control_points is useless. Ignoring this xml entry.'
                print('>> ' + msg)

        if model_options['initial_cp_spacing'] is None and model_options['initial_control_points'] is None \
                and not model_options['dense_mode']:
            print('>> No initial CP spacing given: using diffeo kernel width of '
                  + str(model_options['deformation_kernel_width']))
            model_options['initial_cp_spacing'] = model_options['deformation_kernel_width']

        # We also set the type to FloatTensor if keops is used.
        def keops_is_used():
            if model_options['deformation_kernel_type'].lower() == 'keops':
                return True
            for elt in template_specifications.values():
                if 'kernel_type' in elt and elt['kernel_type'].lower() == 'keops':
                    return True
            return False

        if keops_is_used():
            assert platform not in ['darwin'], 'The "keops" kernel is not available with the Mac OS X platform.'

            print(">> KEOPS is used at least in one operation, all operations will be done with FLOAT precision.")
            tensor_scalar_type = torch.FloatTensor

            if torch.cuda.is_available():
                print('>> CUDA is available: the KEOPS backend will automatically be set to "gpu".')
                cuda_is_used = True
            else:
                print('>> CUDA seems to be unavailable: the KEOPS backend will automatically be set to "cpu".')

        # Setting tensor types according to CUDA availability and user choices.
        if cuda_is_used:

            if not torch.cuda.is_available():
                msg = 'CUDA seems to be unavailable. All computations will be carried out on CPU.'
                print('>> ' + msg)

            else:
                print(">> CUDA is used at least in one operation, all operations will be done with FLOAT precision.")
                if estimator_options['use_cuda']:
                    print(">> All tensors will be CUDA tensors.")
                    tensor_scalar_type = torch.cuda.FloatTensor
                    tensor_integer_type = torch.cuda.LongTensor
                else:
                    print(">> Setting tensor type to float.")
                    tensor_scalar_type = torch.FloatTensor

        # Multi-threading/processing only available for the deterministic atlas for the moment.
        if model_options['number_of_threads'] > 1:

            if model_type.lower() in ['Shooting'.lower(), 'ParallelTransport'.lower(), 'Registration'.lower()]:
                model_options['number_of_threads'] = 1
                msg = 'It is not possible to estimate a "%s" model with multithreading. ' \
                      'Overriding the "number-of-threads" option, now set to 1.' % model_type
                print('>> ' + msg)

            elif model_type.lower() in ['BayesianAtlas'.lower(), 'Regression'.lower(),
                                        'LongitudinalAtlas'.lower(), 'LongitudinalRegistration'.lower()]:
                model_options['number_of_threads'] = 1
                msg = 'It is not possible at the moment to estimate a "%s" model with multithreading. ' \
                      'Overriding the "number-of-threads" option, now set to 1.' % model_type
                print('>> ' + msg)

        # Setting the number of threads in general settings
        if model_options['number_of_threads'] > 1:
            print(">> I will use", ['number_of_threads'],
                  "threads, and I set OMP_NUM_THREADS and torch_num_threads to 1.")
            os.environ['OMP_NUM_THREADS'] = "1"
            torch.set_num_threads(1)
        else:
            print('>> Setting OMP_NUM_THREADS and torch_num_threads to 4.')
            os.environ['OMP_NUM_THREADS'] = "4"
            torch.set_num_threads(4)

        # If longitudinal model and t0 is not initialized, initializes it.
        if (model_type.lower() == 'regression' or model_type.lower() == 'LongitudinalAtlas'.lower()
            or model_type.lower() == 'LongitudinalRegistration'.lower()) \
                and (model_options['t0'] is None or ['initial_time_shift_variance'] is None):
            total_number_of_visits = 0
            mean_visit_age = 0.0
            var_visit_age = 0.0
            assert 'visit_ages' in dataset_specifications, 'Visit ages are needed to estimate a Regression, ' \
                                                           'Longitudinal Atlas or Longitudinal Registration model.'
            for i in range(len(dataset_specifications['visit_ages'])):
                for j in range(len(dataset_specifications['visit_ages'][i])):
                    total_number_of_visits += 1
                    mean_visit_age += dataset_specifications['visit_ages'][i][j]
                    var_visit_age += dataset_specifications['visit_ages'][i][j] ** 2

            if total_number_of_visits > 0:
                mean_visit_age /= float(total_number_of_visits)
                var_visit_age = (var_visit_age / float(total_number_of_visits) - mean_visit_age ** 2)

                if model_options['t0'] is None:
                    print('>> Initial t0 set to the mean visit age: %.2f' % mean_visit_age)
                    model_options['t0'] = mean_visit_age
                else:
                    print('>> Initial t0 set by the user to %.2f ; note that the mean visit age is %.2f'
                          % (model_options['t0'], mean_visit_age))

                if not model_type.lower() == 'regression':
                    if model_options['initial_time_shift_variance'] is None:
                        print('>> Initial time-shift std set to the empirical std of the visit ages: %.2f'
                              % math.sqrt(var_visit_age))
                        model_options['initial_time_shift_variance'] = var_visit_age
                    else:
                        print(('>> Initial time-shift std set by the user to %.2f ; note that the empirical std of '
                               'the visit ages is %.2f') % (model_options['initial_time_shift_variance'],
                                                            math.sqrt(var_visit_age)))

        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError as error:
            print('>> Warning: ' + str(error) + ' [ in xml_parameters ]. Ignoring.')

        # Initializes the state file.
        if estimator_options['state_file'] is None:
            path_to_state_file = os.path.join(self.output_dir, "deformetrica-state.p")
            print('>> No specified state-file. By default, Deformetrica state will by saved in file: %s.' %
                  path_to_state_file)
            if os.path.isfile(path_to_state_file):
                os.remove(path_to_state_file)
                print('>> Removing the pre-existing state file with same path.')
            estimator_options['state_file'] = path_to_state_file
        else:
            if os.path.exists(estimator_options['state_file']):
                estimator_options['load_state_file'] = True
                print('>> Deformetrica will attempt to resume computation from the user-specified state file: %s.'
                      % estimator_options['state_file'])
            else:
                msg = 'The user-specified state-file does not exist: %s. State cannot be reloaded. ' \
                      'Future Deformetrica state will be saved at the given path.' % estimator_options['state_file']
                print('>> ' + msg)

        # Freeze the fixed effects in case of a registration.
        if model_type.lower() == 'Registration'.lower():
            model_options['freeze_template'] = True

        elif model_type.lower() == 'LongitudinalRegistration'.lower():
            model_options['freeze_template'] = True
            model_options['freeze_control_points'] = True
            model_options['freeze_momenta'] = True
            model_options['freeze_modulation_matrix'] = True
            model_options['freeze_reference_time'] = True
            model_options['freeze_time_shift_variance'] = True
            model_options['freeze_log_acceleration_variance'] = True
            model_options['freeze_noise_variance'] = True

        # Initialize the number of sources if needed.
        if model_type.lower() == 'LongitudinalAtlas'.lower() \
                and model_options['initial_modulation_matrix'] is None and model_options['number_of_sources'] is None:
            model_options['number_of_sources'] = 4
            print('>> No initial modulation matrix given, neither a number of sources. '
                  'The latter will be ARBITRARILY defaulted to %d.' % model_options['number_of_sources'])

        # Initialize the initial_log_acceleration_variance if needed.
        if (model_type == 'LongitudinalAtlas'.lower() or model_type == 'LongitudinalRegistration'.lower()) \
                and model_options['initial_log_acceleration_variance'] is None:
            log_acceleration_std = 0.5
            print('>> The initial log-acceleration std fixed effect is ARBITRARILY set to %.2f.' % log_acceleration_std)
            model_options['initial_log_acceleration_variance'] = (log_acceleration_std ** 2)

        # Checking the number of image objects, and moving as desired the downsampling_factor parameter.
        count = 0
        for elt in template_specifications.values():
            if elt['deformable_object_type'].lower() == 'image':
                count += 1
                if not model_options['downsampling_factor'] == 1:
                    if 'downsampling_factor' in elt.keys():
                        print('>> Warning: the downsampling_factor option is specified twice. '
                              'Taking the value: %d.' % elt['downsampling_factor'])
                    else:
                        elt['downsampling_factor'] = model_options['downsampling_factor']
                        print('>> Setting the image grid downsampling factor to: %d.' %
                              model_options['downsampling_factor'])
        if count > 1:
            raise RuntimeError('Only a single image object can be used.')
        if count == 0 and not model_options['downsampling_factor'] == 1:
            msg = 'The "downsampling_factor" parameter is useful only for image data, ' \
                  'but none is considered here. Ignoring.'
            print('>> ' + msg)

        return template_specifications, estimator_options, model_options, (tensor_scalar_type, tensor_integer_type)
