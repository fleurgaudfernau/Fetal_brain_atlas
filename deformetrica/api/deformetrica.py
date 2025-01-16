import gc
import copy
import logging
import math
import os
import os.path as op
import resource
import sys
import time
import nibabel as nib
from copy import deepcopy
import torch
import numpy as np

from ..core import default, GpuMode
from ..core.estimators.gradient_ascent import GradientAscent
from ..core.estimators.mcmc_saem import McmcSaem
from ..core.estimators.scipy_optimize import ScipyOptimize
from ..core.models import PrincipalGeodesicAnalysis, AffineAtlas, BayesianAtlas, \
                        DeterministicAtlas, GeodesicRegression, PiecewiseGeodesicRegression, \
                        BayesianPiecewiseGeodesicRegression, LongitudinalAtlas, LongitudinalAtlasSimplified
from ..in_out.dataset_functions import create_dataset, filter_dataset, make_dataset_timeseries, create_template_metadata,\
                                        age, id, dataset_for_registration, maxi, mini, ages_histogram
from ..in_out.deformable_object_reader import DeformableObjectReader
from ..launch.compute_parallel_transport import compute_parallel_transport, compute_piecewise_parallel_transport, compute_distance_to_flow
from ..in_out.array_readers_and_writers import read_3D_array, write_3D_array
from ..launch.compute_shooting import compute_shooting
from ..launch.compute_ica import perform_ICA, plot_ica
from ..launch.estimate_longitudinal_registration import estimate_longitudinal_registration
from ..support import utilities
from ..support.utilities.vtk_tools import screenshot_vtk
from ..support.probability_distributions.multi_scalar_normal_distribution import MultiScalarNormalDistribution

#ajouts vd
from ..core.estimators.proximal_gradient_ascent import ProximalGradientAscent
from ..core.estimators.hardthreshold_gradient_ascent import HardthresholdGradientAscent
from ..core.models import BayesianAtlasSparse, ClusteredBayesianAtlas, DeterministicAtlasSparse, DeterministicAtlasHypertemplate, DeterministicAtlasWithModule, ClusteredLongitudinalAtlas
from ..support.probability_distributions.multi_scalar_truncated_normal_distribution import MultiScalarTruncatedNormalDistribution
from ..support.probability_distributions.truncated_normal_distribution import TruncatedNormalDistribution
from ..support.probability_distributions.uniform_distribution import UniformDistribution
from ..core.models.model_functions import initialize_control_points
from ..core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject

#ajout fg
from ..core.estimators.stochastic_gradient_ascent import StochasticGradientAscent
from .piecewise_rg_tools import make_dir, complete, PlotResiduals, options_for_registration

######

global logger
logger = logging.getLogger()

def _gaussian_kernel(x, y, sigma = 1):
    return np.exp(-0.5 * ((x-y)/sigma)**2)/ sigma * np.sqrt(2 * np.pi)

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
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if logger.hasHandlers():
            logger.handlers.clear()

        # file logger
        logger_file_handler = logging.FileHandler(
            os.path.join(self.output_dir, time.strftime("%Y-%m-%d-%H%M%S", time.gmtime()) + '_info.log'), mode='w')
        logger_file_handler.setFormatter(logging.Formatter(default.logger_format))
        logger_file_handler.setLevel(logging.INFO)
        logger.addHandler(logger_file_handler)

        # console logger
        logger_stream_handler = logging.StreamHandler(stream=sys.stdout)
        # logger_stream_handler.setFormatter(logging.Formatter(default.logger_format))
        # logger_stream_handler.setLevel(verbosity)
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

    @staticmethod
    def set_seed(seed=None):
        """
        Set the random number generator's seed.
        :param seed: Can be set to None to reset to the original seed
        """
        if seed is None:
            torch.manual_seed(torch.initial_seed())
            np.random.seed(seed)
        else:
            assert isinstance(seed, int)
            torch.manual_seed(seed)
            np.random.seed(seed)

    ####################################################################################################################
    # Main methods.
    ####################################################################################################################

    def set_bounding_box(self, dataset, model_options):
        bounding_boxes = np.zeros((dataset.total_number_of_observations, model_options['dimension'], 2))
        k=0
        for j, obj in enumerate(dataset.deformable_objects):
            for i, object in enumerate(obj):
                bounding_boxes[k] = object.bounding_box
                k += 1
        new_bounding_box = np.zeros((model_options['dimension'], 2))
        new_bounding_box[:, 0] = np.min(bounding_boxes, axis = 0)[:, 0]
        new_bounding_box[:, 1] = np.max(bounding_boxes, axis = 0)[:, 1]

        return new_bounding_box

    def estimate_barycenter(self, template_specifications, dataset_specifications,
                                     model_options={}, estimator_options={}, write_output=True):
        estimator_options["overwrite"] = False
        main_output_dir = self.output_dir  
        momenta= []

        for target in range(len(dataset_specifications['subject_ids'])):

            self.output_dir = op.join(main_output_dir, "Target_{}__{}_age_{}".format(target, 
                                            dataset_specifications['subject_ids'][target],
                                            dataset_specifications['visit_ages'][target][0]))
            make_dir(self.output_dir)
            new_dataset_spec = dict()
            new_dataset_spec['subject_ids'] = [dataset_specifications['subject_ids'][target]]
            new_dataset_spec['dataset_filenames']= [[dataset_specifications['dataset_filenames'][target][0]]]
            new_dataset_spec['visit_ages'] = [[age(dataset_specifications, target)]]

            # Check and completes the input parameters.
            template_specifications_, model_options, estimator_options = self.further_initialization(
                'Registration', template_specifications, model_options, new_dataset_spec, 
                estimator_options)
            
            # Instantiate dataset.
            dataset = create_dataset(template_specifications_,
                                    dimension=model_options['dimension'], interpolation =model_options['interpolation'],
                                    **new_dataset_spec)
            assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

            statistical_model = DeterministicAtlas(template_specifications_, dataset.number_of_subjects, **model_options)
            statistical_model.initialize_noise_variance(dataset)
            statistical_model.setup_multiprocess_pool(dataset)

            estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=ScipyOptimize)

            try:
                self.__launch_estimator(estimator, write_output)
            finally:
                momenta.append(statistical_model.momenta_path)
                cp = statistical_model.fixed_effects['control_points']
                statistical_model.cleanup()
        from ..in_out.array_readers_and_writers import read_3D_array
        list = [read_3D_array(mom) for mom in momenta]
        final_momenta = np.mean(np.array(list), axis = 0)

        np.savetxt(op.join(main_output_dir, "Mean_momenta.txt"), final_momenta)
        np.savetxt(op.join(main_output_dir, "CP.txt"), cp)
        make_dir(op.join(main_output_dir, "shooting"))

        compute_shooting(template_specifications,
                     dimension=model_options["dimension"],
                     deformation_kernel_type=model_options["deformation_kernel_type"],
                     deformation_kernel_width=model_options["deformation_kernel_width"],
                     initial_control_points=op.join(main_output_dir, "CP.txt"),
                     initial_momenta=op.join(main_output_dir, "Mean_momenta.txt"),
                     output_dir=op.join(main_output_dir, "shooting"))

    def estimate_registration(self, template_specifications, dataset_specifications,
                              model_options={}, estimator_options={}, write_output=True):
        """ Estimates the best possible deformation between two sets of objects.
        Note: A registration is a particular case of the deterministic atlas application, with a fixed template object.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict dataset_specifications: Dictionary containing the paths to the input objects from which a statistical model will be estimated.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        :param dict estimator_options: Dictionary containing details about the optimization method. This will be passed to the optimizer's constructor.
        :param bool write_output: Boolean that defines is output files will be written to disk.
        :return:
        """
        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'Registration', template_specifications, model_options, dataset_specifications, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications,
                                 dimension=model_options['dimension'], interpolation =model_options['interpolation'],
                                 **dataset_specifications)
        assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

        # Instantiate model.
        statistical_model = DeterministicAtlas(template_specifications, dataset.number_of_subjects, **model_options)
        statistical_model.initialize_noise_variance(dataset)

        # Instantiate estimator.
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=ScipyOptimize)

        try:
            # Launch.
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

        return statistical_model

    def estimate_deterministic_atlas(self, template_specifications, dataset_specifications,
                                     model_options={}, estimator_options={}, write_output=True):
        """ Estimate deterministic atlas.
        Given a family of objects, the atlas model proposes to learn a template shape which corresponds to a mean of the objects,
        as well as to compute a low number of coordinates for each object from this template shape.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict dataset_specifications: Dictionary containing the paths to the input objects from which a statistical model will be estimated.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        :param dict estimator_options: Dictionary containing details about the optimization method. This will be passed to the optimizer's constructor.
        :param bool write_output: Boolean that defines is output files will be written to disk.
        """

        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'DeterministicAtlas', template_specifications, model_options, dataset_specifications, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications,
                                 dimension=model_options['dimension'], interpolation =model_options['interpolation'],
                                 **dataset_specifications)
        assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

        # Instantiate model.
        statistical_model = DeterministicAtlas(template_specifications, dataset.number_of_subjects, **model_options)
        statistical_model.initialize_noise_variance(dataset)
        statistical_model.setup_multiprocess_pool(dataset)

        # Instantiate estimator.
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=ScipyOptimize)

        try:
            # Launch.
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

        return statistical_model

    #ajouts vd
    def estimate_deterministic_hypertemplate_atlas(self, template_specifications, dataset_specifications,
                                     model_options={}, estimator_options={}, write_output=True):
        """ Estimate deterministic atlas.
        Given a family of objects, the atlas model proposes to learn a template shape which corresponds to a mean of the objects,
        as well as to compute a low number of coordinates for each object from this template shape.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict dataset_specifications: Dictionary containing the paths to the input objects from which a statistical model will be estimated.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        :param dict estimator_options: Dictionary containing details about the optimization method. This will be passed to the optimizer's constructor.
        :param bool write_output: Boolean that defines is output files will be written to disk.
        """

        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'DeterministicAtlasSparse', template_specifications, model_options, dataset_specifications, estimator_options)

        logger.info(estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications,
                                 dimension=model_options['dimension'], **dataset_specifications)
        assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

        # Instantiate model.
        statistical_model = DeterministicAtlasHypertemplate(template_specifications, dataset.number_of_subjects, **model_options)
        statistical_model.initialize_noise_variance(dataset)
        statistical_model.setup_multiprocess_pool(dataset)

        # Instantiate estimator.
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=ScipyOptimize)

        try:
            # Launch.
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

        return statistical_model

    def estimate_deterministic_atlas_sparse(self, template_specifications, dataset_specifications,
                                     model_options={}, estimator_options={}, write_output=True):
        """ Estimate deterministic atlas.
        Given a family of objects, the atlas model proposes to learn a template shape which corresponds to a mean of the objects,
        as well as to compute a low number of coordinates for each object from this template shape.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict dataset_specifications: Dictionary containing the paths to the input objects from which a statistical model will be estimated.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        :param dict estimator_options: Dictionary containing details about the optimization method. This will be passed to the optimizer's constructor.
        :param bool write_output: Boolean that defines is output files will be written to disk.
        """

        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'DeterministicAtlas', template_specifications, model_options, dataset_specifications, estimator_options)

        logger.info(estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications,
                                 dimension=model_options['dimension'], **dataset_specifications)
        assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

        # Instantiate model.
        statistical_model = DeterministicAtlasSparse(template_specifications, dataset.number_of_subjects, **model_options)
        statistical_model.initialize_noise_variance(dataset)
        statistical_model.setup_multiprocess_pool(dataset)

        # Instantiate estimator.
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=ScipyOptimize)

        try:
            # Launch.
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

        return statistical_model

    def estimate_deterministic_atlas_withmodule(self, template_specifications, dataset_specifications,
                                     model_options={}, estimator_options={}, write_output=True):
        """ Estimate deterministic atlas.
        Given a family of objects, the atlas model proposes to learn a template shape which corresponds to a mean of the objects,
        as well as to compute a low number of coordinates for each object from this template shape.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict dataset_specifications: Dictionary containing the paths to the input objects from which a statistical model will be estimated.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        :param dict estimator_options: Dictionary containing details about the optimization method. This will be passed to the optimizer's constructor.
        :param bool write_output: Boolean that defines is output files will be written to disk.
        """

        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'DeterministicAtlas', template_specifications, model_options, dataset_specifications, estimator_options)

        logger.info(estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications,
                                 dimension=model_options['dimension'], **dataset_specifications)
        assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

        # Instantiate model.
        statistical_model = DeterministicAtlasWithModule(template_specifications, dataset.number_of_subjects, **model_options)
        statistical_model.initialize_noise_variance(dataset)
        statistical_model.setup_multiprocess_pool(dataset)

        # Instantiate estimator.
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=ScipyOptimize)

        try:
            # Launch.
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

        return statistical_model

    ###########

    def estimate_bayesian_atlas(self, template_specifications, dataset_specifications,
                                model_options={}, estimator_options={}, write_output=True):
        """ Estimate bayesian atlas.
        Bayesian version of the deterministic atlas. In addition to the template and the registrations, the variability of the geometry and the data noise are learned.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict dataset_specifications: Dictionary containing the paths to the input objects from which a statistical model will be estimated.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        :param dict estimator_options: Dictionary containing details about the optimization method. This will be passed to the optimizer's constructor.
        :param bool write_output: Boolean that defines is output files will be written to disk.
        """
        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'BayesianAtlas', template_specifications, model_options, dataset_specifications, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications,
                                 dimension=model_options['dimension'], interpolation =model_options['interpolation'],
                                 **dataset_specifications)
        assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

        # Instantiate model.
        statistical_model = BayesianAtlas(template_specifications, **model_options)
        individual_RER = statistical_model.initialize_random_effects_realization(dataset.number_of_subjects,
                                                                                 **model_options)
        statistical_model.initialize_noise_variance(dataset, individual_RER)

        # Instantiate estimator.
        estimator_options['individual_RER'] = individual_RER
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=ScipyOptimize)

        try:
            # Launch.
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

        return statistical_model, estimator.individual_RER

    def estimate_sparse_bayesian_atlas(self, template_specifications, dataset_specifications,
                                model_options={}, estimator_options={}, write_output=True):
        """ Estimate bayesian atlas.
        Bayesian version of the deterministic atlas. In addition to the template and the registrations, the variability of the geometry and the data noise are learned.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict dataset_specifications: Dictionary containing the paths to the input objects from which a statistical model will be estimated.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        :param dict estimator_options: Dictionary containing details about the optimization method. This will be passed to the optimizer's constructor.
        :param bool write_output: Boolean that defines is output files will be written to disk.
        """
        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'BayesianAtlasSparse', template_specifications, model_options, dataset_specifications, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications,
                                 dimension=model_options['dimension'], **dataset_specifications)
        assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

        # Instantiate model.
        statistical_model = BayesianAtlasSparse(template_specifications, dataset.number_of_subjects, **model_options)
        individual_RER = statistical_model.initialize_random_effects_realization(dataset.number_of_subjects,
                                                                                 **model_options)
        statistical_model.initialize_noise_variance(dataset, individual_RER)

        # Instantiate estimator.
        estimator_options['individual_RER'] = individual_RER
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=ScipyOptimize)

        try:
            # Launch.
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

        return statistical_model, estimator.individual_RER

    def estimate_clustered_bayesian_atlas(self, template_specifications, dataset_specifications,
                                model_options={}, estimator_options={}, write_output=True):
        """ Estimate clustered bayesian atlas.
        Clustered Bayesian version of the deterministic atlas. In addition to the template and the registrations, the variability of the geometry and the data noise are learned.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict dataset_specifications: Dictionary containing the paths to the input objects from which a statistical model will be estimated.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        :param dict estimator_options: Dictionary containing details about the optimization method. This will be passed to the optimizer's constructor.
        :param bool write_output: Boolean that defines is output files will be written to disk.
        """
        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'ClusteredBayesianAtlas', template_specifications, model_options, dataset_specifications, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications,
                                 dimension=model_options['dimension'], **dataset_specifications)
        assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

        # Instantiate model.
        statistical_model = ClusteredBayesianAtlas(template_specifications, **model_options)
        individual_RER = statistical_model.initialize_random_effects_realization(dataset.number_of_subjects,
                                                                                 **model_options)
        statistical_model.initialize_noise_variance(dataset, individual_RER)

        # Instantiate estimator.
        estimator_options['individual_RER'] = individual_RER
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=ScipyOptimize)

        try:
            # Launch.
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

        return statistical_model, estimator.individual_RER

##########

    def estimate_longitudinal_atlas(self, template_specifications, dataset_specifications,
                                    model_options={}, estimator_options={}, write_output=True):
        """ Estimate longitudinal atlas.
        TODO

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict dataset_specifications: Dictionary containing the paths to the input objects from which a statistical model will be estimated.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        :param dict estimator_options: Dictionary containing details about the optimization method. This will be passed to the optimizer's constructor.
        :param bool write_output: Boolean that defines is output files will be written to disk.
        """

        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'LongitudinalAtlas', template_specifications, model_options, dataset_specifications, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications,
                                 dimension=model_options['dimension'], **dataset_specifications)
        assert (not dataset.is_cross_sectional() and not dataset.is_time_series()), \
            "Cannot estimate an atlas from a cross-sectional or time-series dataset."

        # Instantiate model.
        statistical_model = LongitudinalAtlas(template_specifications, **model_options)
        individual_RER = statistical_model.initialize_random_effects_realization(dataset.number_of_subjects,
                                                                                 **model_options)
        statistical_model.initialize_noise_variance(dataset, individual_RER)
        statistical_model.setup_multiprocess_pool(dataset)

        # Instantiate estimator.
        estimator_options['individual_RER'] = individual_RER
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=McmcSaem)

        try:
            # Launch.
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

        return statistical_model

    #ajouts vd

    def estimate_clustered_longitudinal_atlas(self, template_specifications, dataset_specifications,
                                    model_options={}, estimator_options={}, write_output=True):
        """ Estimate clustered longitudinal atlas.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict dataset_specifications: Dictionary containing the paths to the input objects from which a statistical model will be estimated.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        :param dict estimator_options: Dictionary containing details about the optimization method. This will be passed to the optimizer's constructor.
        :param bool write_output: Boolean that defines is output files will be written to disk.
        """

        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'ClusteredLongitudinalAtlas', template_specifications, model_options, dataset_specifications, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications,
                                 dimension=model_options['dimension'], **dataset_specifications)
        assert (not dataset.is_cross_sectional() and not dataset.is_time_series()), \
            "Cannot estimate an atlas from a cross-sectional or time-series dataset."

        # Instantiate model.
        mini = min(dataset.times[0])
        maxi = max(dataset.times[0])
        for k in range(1, dataset.times.__len__()):
            if min(dataset.times[k]) < mini: mini = min(dataset.times[k])
            if max(dataset.times[k]) > maxi: maxi = max(dataset.times[k])
        statistical_model = ClusteredLongitudinalAtlas(template_specifications, min_times=mini, max_times=maxi, **model_options)
        individual_RER = statistical_model.initialize_random_effects_realization(dataset.number_of_subjects,
                                                                                 **model_options)
        statistical_model.setup_multiprocess_pool(dataset)
        statistical_model.initialize_noise_variance(dataset, individual_RER)

        # Instantiate estimator.
        estimator_options['individual_RER'] = individual_RER
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=McmcSaem)

        try:
            # Launch.
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

        return statistical_model

    def estimate_longitudinal_atlas_simplified(self, template_specifications, dataset_specifications,
                                    model_options={}, estimator_options={}, write_output=True):
        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'LongitudinalAtlasSimplified', template_specifications, model_options, dataset_specifications, 
            estimator_options)
        
        # Instantiate dataset.
        dataset = create_dataset(template_specifications, dimension=model_options['dimension'], **dataset_specifications)        
        #the dataset can now be cross sectional

        # Instantiate model.
        statistical_model = LongitudinalAtlasSimplified(template_specifications, **model_options)
        individual_RER = statistical_model.initialize_random_effects_realization(dataset.number_of_subjects,
                                                                                 **model_options)
        statistical_model.initialize_noise_variance(dataset, individual_RER)
        statistical_model.setup_multiprocess_pool(dataset)

        # Instantiate estimator.
        estimator_options['individual_RER'] = individual_RER
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=McmcSaem)

        try:
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

        return statistical_model


    def estimate_longitudinal_registration(self, template_specifications, dataset_specifications,
                                           model_options={}, estimator_options={}, overwrite=True):
        """ Estimate longitudinal registration.
        This function does not simply estimate a statistical model, but will successively instantiate and estimate
        several, before gathering all the results in a common folder: that is why it calls a dedicated script.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict dataset_specifications: Dictionary containing the paths to the input objects from which a statistical model will be estimated.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        :param dict estimator_options: Dictionary containing details about the optimization method. This will be passed to the optimizer's constructor.
        """

        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'LongitudinalRegistration', template_specifications, model_options,
            dataset_specifications, estimator_options)

        # Launch the dedicated script.
        estimate_longitudinal_registration(template_specifications, dataset_specifications,
                                           model_options, estimator_options,
                                           output_dir=self.output_dir, overwrite=overwrite)


    def estimate_affine_atlas(self, template_specifications, dataset_specifications,
                              model_options={}, estimator_options={}, write_output=True):
        """ Estimate affine atlas
        TODO

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict dataset_specifications: Dictionary containing the paths to the input objects from which a statistical model will be estimated.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        :param dict estimator_options: Dictionary containing details about the optimization method. This will be passed to the optimizer's constructor.
        :param bool write_output: Boolean that defines is output files will be written to disk.
        """
        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'AffineAtlas', template_specifications, model_options, dataset_specifications, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications,
                                 dimension=model_options['dimension'], **dataset_specifications)

        # Instantiate model.
        statistical_model = AffineAtlas(dataset, template_specifications, **model_options)

        # instantiate estimator
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=ScipyOptimize)

        try:
            # Launch.
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

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

    def estimate_geodesic_regression(self, template_specifications, dataset_specifications,
                                     model_options={}, estimator_options={}, write_output=True):
        """ Construct a shape trajectory that is as close as possible to the given targets at the given times.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict dataset_specifications: Dictionary containing the paths to the input objects from which a statistical model will be estimated.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        :param dict estimator_options: Dictionary containing details about the optimization method. This will be passed to the optimizer's constructor.
        :param bool write_output: Boolean that defines is output files will be written to disk.
        """
        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'Regression', template_specifications, model_options, dataset_specifications, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications,
                                 dimension=model_options['dimension'], **dataset_specifications)

        assert (dataset.is_time_series()), "Cannot estimate a geodesic regression from a non-time-series dataset."
        # Handle the case where template has smaller bounding box than the subjects
        new_bounding_box = self.set_bounding_box(dataset, model_options)

        # Instantiate model.
        statistical_model = GeodesicRegression(template_specifications, **model_options, new_bounding_box=new_bounding_box)
        statistical_model.initialize_noise_variance(dataset)

        # Instantiate estimator.
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=ScipyOptimize)

        try:
            # Launch.
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

        return statistical_model
    
    def estimate_piecewise_geodesic_regression(self, template_specifications, dataset_specifications,
                                     model_options={}, estimator_options={}, write_output=True):
        """ Construct a shape trajectory that is as close as possible to the given targets at the given times.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict dataset_specifications: Dictionary containing the paths to the input objects from which a statistical model will be estimated.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        :param dict estimator_options: Dictionary containing details about the optimization method. This will be passed to the optimizer's constructor.
        :param bool write_output: Boolean that defines is output files will be written to disk.
        """
        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'PiecewiseRegression', template_specifications, model_options, dataset_specifications, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications,
                                 dimension=model_options['dimension'], **dataset_specifications)
        
        
        assert (dataset.is_time_series()), "Cannot estimate a piecewise geodesic regression from a non-time-series dataset."

        # Handle the case where template has smaller bounding box than the subjects
        new_bounding_box = self.set_bounding_box(dataset, model_options)

        # Instantiate model.
        statistical_model = PiecewiseGeodesicRegression(template_specifications, **model_options, new_bounding_box = new_bounding_box)
        statistical_model.initialize_noise_variance(dataset)

        # Instantiate estimator.
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=ScipyOptimize)

        try:
            # Launch.
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

        return statistical_model
    
    def estimate_piecewise_bayesian_geodesic_regression(self, template_specifications, dataset_specifications,
                                     model_options={}, estimator_options={}, write_output=True):
        """ Construct a shape trajectory that is as close as possible to the given targets at the given times.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict dataset_specifications: Dictionary containing the paths to the input objects from which a statistical model will be estimated.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        :param dict estimator_options: Dictionary containing details about the optimization method. This will be passed to the optimizer's constructor.
        :param bool write_output: Boolean that defines is output files will be written to disk.
        """
        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'BayesianPiecewiseRegression', template_specifications, model_options, dataset_specifications, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications,
                                 dimension=model_options['dimension'], **dataset_specifications)

        # Check there are several subjects, with only 1 visit each. 
        assert (dataset.is_cross_sectional() and not dataset.is_time_series()), \
            "Cannot estimate a bayesian regression from a cross-sectional or time-series dataset."

        # Handle the case where template has smaller bounding box than the subjects
        new_bounding_box = self.set_bounding_box(dataset, model_options)

        # Instantiate model.
        statistical_model = BayesianPiecewiseGeodesicRegression(template_specifications, **model_options, new_bounding_box = new_bounding_box)
        individual_RER = statistical_model.initialize_random_effects_realization(dataset.number_of_subjects,
                                                                                 **model_options)
        statistical_model.initialize_noise_variance(dataset, individual_RER)

        # Instantiate estimator.
        estimator_options['individual_RER'] = individual_RER
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=ScipyOptimize)

        try:
            # Launch.
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

        return statistical_model
    
    def atlas_at_t0_for_initialisation(self, template_specifications, dataset_specifications,
                                     model_options, estimator_options, write_output):
        estimator_options_ = copy.deepcopy(estimator_options)

        t0 = model_options["t0"]

        new_dataset_spec = {k : [] for k in dataset_specifications.keys()}

        # Select subjects around t0
        for i in range(len(dataset_specifications['subject_ids'])):
            if age(dataset_specifications, i) > t0 - 3 and age(dataset_specifications, i) < t0 + 3:
                new_dataset_spec['subject_ids'].append(dataset_specifications['subject_ids'][i])
                new_dataset_spec['dataset_filenames'].append([dataset_specifications['dataset_filenames'][i][0]])
                new_dataset_spec['visit_ages'].append([age(dataset_specifications, i)])

        logger.info("\n >>>> 0_Atlas estimation >>>> \n")
        
        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options_ = self.further_initialization(
            'KernelRegression', template_specifications, model_options, dataset_specifications, 
            estimator_options_)
        
        # Kernel weigting of the subjects
        model_options['visit_ages'] = new_dataset_spec['visit_ages']
        model_options['time'] = t0
        model_options["freeze_template"] = False
        #model_options["freeze_momenta"] = True
        estimator_options_["multiscale_images"] = True

        # Instantiate dataset.
        dataset = create_dataset(template_specifications,
                                 dimension=model_options['dimension'], interpolation =model_options['interpolation'],
                                 **new_dataset_spec)
        assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

        statistical_model = DeterministicAtlas(template_specifications, dataset.number_of_subjects, **model_options)
        statistical_model.initialize_noise_variance(dataset)
        statistical_model.setup_multiprocess_pool(dataset)

        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options_, default=ScipyOptimize)

        try:
            self.__launch_estimator(estimator, write_output)
        finally:
            template = statistical_model.template_path
            object_name = statistical_model.objects_name[0]
            statistical_model.cleanup()

        return template, object_name

    def longitudinal_weighting(self, dataset_specifications):
        ages = [a[0] for a in dataset_specifications['visit_ages']]   
        print(ages)     

        effectifs_ages = { a : 0 for a in range(int(min(ages)), int(max(ages)) + 1)}
        weights = []

        for age in effectifs_ages.keys():
            n = len([a for a in ages if a >= age and a < age + 1])
            effectifs_ages[age] = n
        
        for age in ages:
            a = int(np.floor(age))
            weights.append(4 / effectifs_ages[a])

        return weights
    
    def longitudinal_weighting_(self, dataset_specifications):
        ages = [a[0] for a in dataset_specifications['visit_ages']]    
        effectifs_ages = { a : 0 for a in range(int(min(ages)), int(max(ages)) + 1)}  
        for age in effectifs_ages.keys():
            n = len([a for a in ages if a >= age and a < age + 1])
            effectifs_ages[age] = n

        poids_ages = { a : 0 for a in range(int(min(ages)), int(max(ages)) + 1)} 
        sigma_ages = { a : 1 for a in range(int(min(ages)), int(max(ages)) + 1)} 
        for age in poids_ages.keys():
            sigma = 1
            poids = [_gaussian_kernel(age, a, sigma) if _gaussian_kernel(age, a, sigma) > 0.01 else 0 for a in ages]
            poids_tot =  np.sum(poids)

            if poids_tot < 15:
                while poids_tot < 15:
                    sigma += 0.1
                    poids = [_gaussian_kernel(age, a, sigma) if _gaussian_kernel(age, a, sigma) > 0.01 else 0 for a in ages]
                    poids_tot =  np.sum(poids)
            else:
                while poids_tot > 15:
                    sigma -= 0.1
                    poids = [_gaussian_kernel(age, a, sigma) if _gaussian_kernel(age, a, sigma) > 0.01 else 0 for a in ages]
                    poids_tot =  np.sum(poids)
            sigma_ages[age] = sigma

        print(poids_ages)

    def piecewise_regression_for_initialisation(self, template_specifications, dataset_specifications,
                                     model_options, estimator_options, write_output):
        estimator_options_ = copy.deepcopy(estimator_options)

        template_specifications, model_options, estimator_options_ = self.further_initialization(
            'PiecewiseRegression', template_specifications, model_options, dataset_specifications, estimator_options_)
        
        logger.info("\n >>>> 1_piecewise_regression >>>> \n")

        new_dataset_spec = make_dataset_timeseries(dataset_specifications)

        dataset = create_dataset(template_specifications, dimension=model_options['dimension'], **new_dataset_spec)
        
        assert (dataset.is_time_series()), "Cannot estimate a geodesic regression from a non-time-series dataset."

        new_bounding_box = self.set_bounding_box(dataset, model_options)

        model_options["freeze_template"] = True # important
        #estimator_options_["multiscale_images"] = True

        #weights = self.longitudinal_weighting(dataset_specifications)
        #model_options["weights"] = weights

        statistical_model = PiecewiseGeodesicRegression(template_specifications, **model_options, new_bounding_box = new_bounding_box)
        statistical_model.initialize_noise_variance(dataset)

        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options_, default=ScipyOptimize)

        try:
            self.__launch_estimator(estimator, write_output)
        finally:
            initial_cp_path = statistical_model.cp_path
            initial_momenta_path = statistical_model.momenta_path
            flow_path = statistical_model.geodesic.flow_path
            statistical_model.cleanup()
        
        return initial_cp_path, initial_momenta_path, flow_path
    
    def piecewise_regression_for_initialisation_(self, template_specifications, dataset_specifications,
                                     model_options, estimator_options, write_output):
        # Résout la piecewise régression par morceaux... 
        estimator_options_ = copy.deepcopy(estimator_options)

        template_specifications, model_options, estimator_options_ = self.further_initialization(
            'PiecewiseRegression', template_specifications, model_options, dataset_specifications, estimator_options_)
        
        new_dataset_spec = make_dataset_timeseries(dataset_specifications)
        model_options["freeze_template"] = True # important
        estimator_options_["multiscale_momenta"] = False

        dataset = create_dataset(template_specifications, dimension=model_options['dimension'], **new_dataset_spec)

        new_bounding_box = self.set_bounding_box(dataset, model_options)

        main_output_dir = self.output_dir

        for t, age_limit in enumerate(model_options['tR'][1:]):

            logger.info("\n >>>> 1_piecewise_regression {}>>>> \n".format(t+1))

            model_options_ = deepcopy(model_options)
            model_options_["tR"] = model_options['tR'][:t+1]
            model_options_["t1"] = age_limit
            model_options_["num_component"] = len(model_options_["tR"]) + 1

            self.output_dir = main_output_dir + '/regression_' + str(t+1)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            if not op.exists(op.join(self.output_dir, "GeodesicRegression__EstimatedParameters__Momenta.txt")):
                if t > 0:
                    momenta = read_3D_array(initial_momenta_path)
                    if momenta.shape[0] != model_options_["num_component"]:
                        new_momenta = np.zeros((momenta.shape[0] + 1, momenta.shape[1], momenta.shape[2]))
                        new_momenta[:momenta.shape[0]] = momenta
                        name = initial_momenta_path.split("/")[-1]
                        write_3D_array(new_momenta, self.output_dir, name)
                    model_options_["initial_momenta"] = initial_momenta_path
                
                new_dataset_spec_ = filter_dataset(new_dataset_spec, float(age_limit))

                dataset = create_dataset(template_specifications, dimension=model_options_['dimension'], **new_dataset_spec_)
                
                assert (dataset.is_time_series()), "Cannot estimate a geodesic regression from a non-time-series dataset."

                statistical_model = PiecewiseGeodesicRegression(template_specifications, **model_options_, new_bounding_box = new_bounding_box)
                statistical_model.initialize_noise_variance(dataset)

                estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options_, default=ScipyOptimize)

                try:
                    self.__launch_estimator(estimator, write_output)
                finally:
                    initial_momenta_path = statistical_model.momenta_path
                
                statistical_model.cleanup()
            
            initial_momenta_path = op.join(self.output_dir, "GeodesicRegression__EstimatedParameters__Momenta.txt")

    def parallel_transport_subject(self, i, template_specifications, dataset_specifications,
                                    model_options, estimator_options, main_output_dir3, 
                                    flow_path, object_name, initial_cp_path, initial_momenta_path, 
                                    registration_model, start_time, target_time):
        # Parallel transport from t_i to t0 -we keep original tmin, tmax, t0 and tR
        logger.info("\n >>>> Parallel transport for subject {} - id {} to {}".format(i, id(dataset_specifications, i), target_time))

        model_options_ = {"tmin" : mini(dataset_specifications), 
                            "tmax" : maxi(dataset_specifications), 
                            "start_time" : int(start_time), "target_time" : target_time,
                            'initial_momenta_to_transport' : registration_model.momenta_path,
                            "initial_control_points" : initial_cp_path,
                            "initial_momenta_tR" : initial_momenta_path,
                            "perform_shooting" : False}
        
        model_options_ = complete(model_options, model_options_)

        print("start_time", start_time)
        print("target_time", target_time)
        
        template_specifications[object_name]['filename'] = flow_path[model_options_["t0"]]

        if target_time != model_options["t0"]:
            self.output_dir = op.join(main_output_dir3, "Subject_{}_age_{}_to_{}".format(i, age(dataset_specifications, i), target_time))
        else:
            self.output_dir = op.join(main_output_dir3, "Subject_{}_age_{}".format(i, age(dataset_specifications, i)))
        
        make_dir(self.output_dir)


        template_specifications, model_options_, _ = self.further_initialization(
        'ParallelTransport', template_specifications, model_options_)

        trajectory = compute_piecewise_parallel_transport(template_specifications, 
                                                            output_dir=self.output_dir, 
                                                            overwrite = estimator_options["overwrite"],
                                                            **model_options_)        
        # residuals = compute_distance_to_flow(template_specifications, output_dir=self.output_dir, 
        #                               flow_path = flow_path, **model_options_)
        # #print("\n residuals", residuals)
        # r.plot(residuals, i, time, age(dataset_specifications, i))

        return trajectory

    def registration_and_transport_for_initialization(self, template_specifications, dataset_specifications,
                                                    model_options, estimator_options, main_output_dir2,
                                                    main_output_dir3, flow_path, object_name, initial_cp_path, 
                                                    initial_momenta_path, target_times = []):
        
        accepted_difference = (1/model_options["concentration_of_time_points"]) / 2 + 0.01

        transported_momenta_path = {t: [] for t in target_times}

        r = PlotResiduals(main_output_dir3)

        for i, subject in enumerate(dataset_specifications['dataset_filenames']):

            self.output_dir = op.join(main_output_dir2, "Subject_{}_age_{}".format(i, age(dataset_specifications, i)))
            make_dir(self.output_dir)

            new_dataset_spec = dataset_for_registration(subject, age(dataset_specifications, i), 
                                                        id(dataset_specifications, i))
            
            for start_time, template in flow_path.items():
                if np.abs(start_time - age(dataset_specifications, i)) <= accepted_difference:
                    template_specifications[object_name]['filename'] = template
                    break
                
            logger.info("\n >>>> Registration for subject {}".format(id(dataset_specifications, i)))

            model_options_ = copy.deepcopy(model_options)
            model_options_["initial_control_points"] = initial_cp_path
            model_options_["kernel_regression"] = False
            
            estimator_options_ = options_for_registration(estimator_options)

            model = self.estimate_registration(template_specifications, new_dataset_spec,
                                            model_options_, estimator_options_, write_output=True)
            
            trajectory = dict()
            for target_time in target_times:
                if target_time not in trajectory.keys():
                    trajectory = self.parallel_transport_subject(i, template_specifications, 
                                                            dataset_specifications,
                                                            model_options, estimator_options, 
                                                            main_output_dir3, flow_path, object_name, initial_cp_path, 
                                                            initial_momenta_path, model, start_time,
                                                            target_time = target_time)
                
                transported_momenta_path[target_time].append(trajectory[target_time])
        
        #r.end(residuals, i, age(dataset_specifications, i))

        return transported_momenta_path, model_options_

    def initialize_piecewise_bayesian_geodesic_regression(self, template_specifications, dataset_specifications,
                                     model_options={}, estimator_options={}, write_output=True):
        """ Construct a shape trajectory that is as close as possible to the given targets at the given times.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict dataset_specifications: Dictionary containing the paths to the input objects from which a statistical model will be estimated.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        :param dict estimator_options: Dictionary containing details about the optimization method. This will be passed to the optimizer's constructor.
        :param bool write_output: Boolean that defines is output files will be written to disk.
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

        ages_histogram(dataset_specifications, main_output_dir)

        ## 0 - estimate initial template shape
        estimator_options["overwrite"] = False

        # template, object_name = self.atlas_at_t0_for_initialisation(template_specifications, dataset_specifications,
        #                                             model_options, estimator_options, write_output)

        object_name = "img"
        #template_specifications[object_name]['filename'] = template

        ## 1 - estimate piecewise regression
        self.output_dir = main_output_dir1

        initial_cp_path, initial_momenta_path, flow_path = \
        self.piecewise_regression_for_initialisation(template_specifications, dataset_specifications,
                                                    model_options, estimator_options, write_output)

        ## 2 - Register the mean trajectory to each observation at t_i
        transported_momenta_path, model_options_ = self.registration_and_transport_for_initialization(template_specifications, 
                                                                                                dataset_specifications,
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
            model_options_["tmin"] = mini(dataset_specifications)
            model_options_["tmax"] = maxi(dataset_specifications)
            

            path_to_sources, path_to_mm = perform_ICA(self.output_dir, initial_cp_path, 
                                                    model_options["deformation_kernel_width"], 
                                                    initial_momenta_path, transported_momenta_path[target_time], 
                                                    model_options["number_of_sources"], target_time,
                                                    model_options["tR"], overwrite = estimator_options["overwrite"])

            model_options_ = complete(model_options, model_options_)
            model_options_["initial_momenta"] = initial_momenta_path
            model_options_["initial_momenta_tR"] = initial_momenta_path
            plot_ica(path_to_sources, path_to_mm, template_specifications, dataset_specifications,
                    output_dir=self.output_dir, overwrite = estimator_options["overwrite"],
                    nb_components = model_options_["num_component"], target_time = target_time,
                    **model_options_)
        
    def test_deterministic_atlas(self, template_specifications, dataset_specifications,
                                     model_options={}, estimator_options={}, write_output=True):
        ages_histogram(dataset_specifications, self.output_dir)

        estimator_options["overwrite"] = False

        main_output_dir = self.output_dir  

        for template in range(len(dataset_specifications['subject_ids'])):
            new_dataset_spec = {"subject_ids" : [], 'dataset_filenames' : [], "visit_ages" : []}
            self.output_dir = op.join(main_output_dir, "Template_{}__{}_age_{}".format(template, 
                                            dataset_specifications['subject_ids'][template],
                                            dataset_specifications['visit_ages'][template][0]))
            make_dir(self.output_dir)

            template_specifications["img"]['filename'] = dataset_specifications['dataset_filenames'][template][0]['img']

            logger.info("Setting template to {}".format(template_specifications["img"]['filename'] ))

            for i in range(len(dataset_specifications['subject_ids'])):
                if i != template:
                    new_dataset_spec['subject_ids'].append(dataset_specifications['subject_ids'][i])
                    new_dataset_spec['dataset_filenames'].append([dataset_specifications['dataset_filenames'][i][0]])
                    new_dataset_spec['visit_ages'].append([age(dataset_specifications, i)])

            # Check and completes the input parameters.
            template_specifications_, model_options, estimator_options = self.further_initialization(
                'DeterministicAtlas', template_specifications, model_options, new_dataset_spec, 
                estimator_options)
            
            # Instantiate dataset.
            dataset = create_dataset(template_specifications_,
                                    dimension=model_options['dimension'], interpolation =model_options['interpolation'],
                                    **new_dataset_spec)
            assert (dataset.is_cross_sectional()), "Cannot estimate an atlas from a non-cross-sectional dataset."

            statistical_model = DeterministicAtlas(template_specifications_, dataset.number_of_subjects, **model_options)
            statistical_model.initialize_noise_variance(dataset)
            statistical_model.setup_multiprocess_pool(dataset)

            estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=ScipyOptimize)

            try:
                self.__launch_estimator(estimator, write_output)
            finally:
                statistical_model.cleanup()

        
    def initialized_piecewise_bayesian_geodesic_regression(self, template_specifications, dataset_specifications,
                                     model_options={}, estimator_options={}, write_output=True):
        
        main_output_dir = self.output_dir      
        
        logger.info("\n >>>> Piecewise regression with space shift")
        
        estimator_options["overwrite"] = False

        ############## Get initialized parameters ##############

        # Get average trajectory initialization
        self.output_dir = op.join(main_output_dir, "1_piecewise_regression")
        initial_cp_path, initial_momenta_path, _, _ = \
        self.piecewise_regression_for_initialisation(template_specifications, dataset_specifications,
                                                    model_options, estimator_options, write_output)
        
        # Get ICA output
        self.output_dir = op.join(main_output_dir, "4_ICA")
        path_to_sources, path_to_mm = perform_ICA(self.output_dir, overwrite = False)

        self.output_dir = op.join(main_output_dir, "5_piecewise_regression_with_space_shift")
        make_dir(self.output_dir)

        weights = self.longitudinal_weighting(dataset_specifications)
        model_options["weights"] = weights

        model_options["initial_sources"] = path_to_sources
        model_options["initial_modulation_matrix"] = path_to_mm
        model_options["initial_momenta"] = initial_momenta_path
        model_options["initial_control_points"] = initial_cp_path

        ############## Final Model ##############

        logger.info("\n >>>> Final model")

        estimator_options["overwrite"] = True

        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'BayesianPiecewiseRegression', template_specifications, model_options, dataset_specifications, estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications,
                                 dimension=model_options['dimension'], **dataset_specifications)

        # Check there are several subjects, with only 1 visit each. 
        assert (dataset.is_cross_sectional() and not dataset.is_time_series()), \
            "Cannot estimate a bayesian regression from a cross-sectional or time-series dataset."

        # Handle the case where template has smaller bounding box than the subjects
        new_bounding_box = self.set_bounding_box(dataset, model_options)

        # Instantiate model.
        statistical_model = BayesianPiecewiseGeodesicRegression(template_specifications, **model_options, new_bounding_box = new_bounding_box)
        individual_RER = statistical_model.initialize_random_effects_realization(dataset.number_of_subjects,
                                                                                 **model_options)
        statistical_model.initialize_noise_variance(dataset, individual_RER)

        # Instantiate estimator.
        estimator_options['individual_RER'] = individual_RER
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=ScipyOptimize)

        try:
            # Launch.
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

        return statistical_model
    
    def estimate_kernel_regression(self, time, template_specifications, dataset_specifications,
                                     model_options={}, estimator_options={}, write_output=True):
        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'KernelRegression', template_specifications, model_options, dataset_specifications, estimator_options)
        
        visit_ages = [d[0] for d in dataset_specifications['visit_ages']]
        original_template_spec = template_specifications['img']["filename"]
        
        time = int(time)
        model_options['time'] = time
        self.output_dir = self.output_dir + '/time_' + str(time)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Select the subjects that contribute to the atlas
        selection, weights = [], []

        # Regular kernel
        total_weights = np.sum([_gaussian_kernel(time, age[0]) for age in dataset_specifications['visit_ages']])
        for i, age in enumerate(dataset_specifications['visit_ages']):
            weight = _gaussian_kernel(time, age[0])
            if weight > 0.01:
                selection.append(i)
                weights.append(weight)
        
        # Adaptative kernel
        

        # Update the dataset accordingly
        new_dataset_spec = copy.deepcopy(dataset_specifications)
        new_dataset_spec['visit_ages'] = [[age] for i, age in enumerate(visit_ages) if i in selection]
        new_dataset_spec['subject_ids'] = [id for i, id in enumerate(dataset_specifications['subject_ids'])\
                                                    if i in selection]
        new_dataset_spec['dataset_filenames'] = [name for i, name in enumerate(dataset_specifications['dataset_filenames'])\
                                                        if i in selection]
        model_options['visit_ages'] = new_dataset_spec['visit_ages']

        dataset = create_dataset(template_specifications, dimension=model_options['dimension'], 
                                **new_dataset_spec)
        
        # Compute a mean image
        # if ".nii" in new_dataset_spec['dataset_filenames'][0][0]["img"]:
        #     data_list = [nib.load(f[0]["img"]) for f in new_dataset_spec['dataset_filenames']]
        #     mean = np.zeros((data_list[0].get_fdata().shape))
        #     for i, f in enumerate(data_list):
        #         mean += f.get_fdata() * (weights[i]/total_weights)
        #     image_new = nib.nifti1.Nifti1Image(mean, data_list[0].affine, data_list[0].header)
        #     output_image = original_template_spec.replace("mean", "mean_age_{}".format(time))
        #     nib.save(image_new, output_image)
        # else:            
        #     i = weights.index(max(weights))
        #     name = new_dataset_spec['dataset_filenames'][i][0]["img"]
        #     name = "/home/fleur.gaudfernau/Necker_atlas_SPT/Inner_cortical_surface_bayesian_piecewise_regression_60_subjects_with_init_/Kernel_regression/template_smooth_2000.vtk"
        #template_specifications['img']["filename"] = name
        
        # Instantiate model.
        print("Number of subjects", len(new_dataset_spec['dataset_filenames']), "at age", time)
        statistical_model = DeterministicAtlas(template_specifications, dataset.number_of_subjects, **model_options)
        statistical_model.initialize_noise_variance(dataset)
        statistical_model.setup_multiprocess_pool(dataset)

        # Instantiate estimator.
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=ScipyOptimize)

        try:
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

        return 


    def kernel_regression_2(self, template_specifications, dataset_specifications,
                                     model_options={}, estimator_options={}, write_output=True):
        """ Estimate a deterministic atlas
        """
        
        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'KernelRegression', template_specifications, model_options, dataset_specifications, estimator_options)
        
        visit_ages = [d[0] for d in dataset_specifications['visit_ages']]
        tmin = int(min(visit_ages))
        tmax = int(max(visit_ages))
        original_template_spec = template_specifications['img']["filename"]
        root_output_dir = self.output_dir
        
        for time in range(tmin, tmax):
            model_options['time'] = time

            self.output_dir = root_output_dir + '/time_' + str(time)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            # Select the subjects that contribute to the atlas
            selection = []
            weights = []
            total_weights = np.sum([_gaussian_kernel(time, age[0]) for age in dataset_specifications['visit_ages']])
            for i, age in enumerate(dataset_specifications['visit_ages']):
                weight = _gaussian_kernel(time, age[0])
                if weight > 0.01:
                    selection.append(i)
                    weights.append(weight)

            # Update the dataset accordingly
            new_dataset_spec = copy.deepcopy(dataset_specifications)
            new_dataset_spec['visit_ages'] = [[age] for i, age in enumerate(visit_ages) if i in selection]
            new_dataset_spec['subject_ids'] = [id for i, id in enumerate(dataset_specifications['subject_ids'])\
                                                        if i in selection]
            new_dataset_spec['dataset_filenames'] = [name for i, name in enumerate(dataset_specifications['dataset_filenames'])\
                                                            if i in selection]
            model_options['visit_ages'] = new_dataset_spec['visit_ages']

            dataset = create_dataset(template_specifications, dimension=model_options['dimension'], 
                                    **new_dataset_spec)
            
            # Compute a mean image
            data_list = [nib.load(f[0]["img"]) for f in new_dataset_spec['dataset_filenames']]
            mean = np.zeros((data_list[0].get_fdata().shape))
            for i, f in enumerate(data_list):
                mean += f.get_fdata() * (weights[i]/total_weights)
            image_new = nib.nifti1.Nifti1Image(mean, data_list[0].affine, data_list[0].header)
            output_image = original_template_spec.replace("mean", "mean_age_{}".format(time))
            nib.save(image_new, output_image)
            template_specifications['img']["filename"] = output_image
            
            # Instantiate model.
            statistical_model = DeterministicAtlas(template_specifications, dataset.number_of_subjects, **model_options)
            statistical_model.initialize_noise_variance(dataset)
            statistical_model.setup_multiprocess_pool(dataset)

            # Instantiate estimator.
            estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=ScipyOptimize)

            try:
                self.__launch_estimator(estimator, write_output)
            finally:
                statistical_model.cleanup()

        return 

    def estimate_deep_pga(self):
        """
        TODO
        :return:
        """
        raise NotImplementedError

    def estimate_principal_geodesic_analysis(self, template_specifications, dataset_specifications,
                                             model_options={}, estimator_options={}, write_output=True):
        """ Estimate principal geodesic analysis

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict dataset_specifications: Dictionary containing the paths to the input objects from which a statistical model will be estimated.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        :param dict estimator_options: Dictionary containing details about the optimization method. This will be passed to the optimizer's constructor.
        :param bool write_output: Boolean that defines is output files will be written to disk.
        """
        # Check and completes the input parameters.
        template_specifications, model_options, estimator_options = self.further_initialization(
            'PrincipalGeodesicAnalysis', template_specifications, model_options, dataset_specifications,
            estimator_options)

        # Instantiate dataset.
        dataset = create_dataset(template_specifications,
                                 dimension=model_options['dimension'], **dataset_specifications)
        assert (dataset.is_cross_sectional()), \
            "Cannot estimate a PGA from a non cross-sectional dataset."

        # Instantiate model.
        statistical_model = PrincipalGeodesicAnalysis(template_specifications, **model_options)

        # Runs a tangent pca on a deterministic atlas to initialize
        individual_RER = statistical_model.initialize(dataset, template_specifications, dataset_specifications,
                                                      model_options, estimator_options, self.output_dir)

        statistical_model.initialize_noise_variance(dataset, individual_RER)

        # Instantiate estimator.
        estimator_options['individual_RER'] = individual_RER
        estimator = self.__instantiate_estimator(statistical_model, dataset, estimator_options, default=McmcSaem)

        try:
            # Launch.
            self.__launch_estimator(estimator, write_output)
        finally:
            statistical_model.cleanup()

        return statistical_model

    def compute_parallel_transport(self, template_specifications, model_options={}):
        """ Given a known progression of shapes, to transport this progression onto a new shape.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        """

        # Check and completes the input parameters.
        template_specifications, model_options, _ = self.further_initialization(
            'ParallelTransport', template_specifications, model_options)

        logger.debug("dtype=" + default.dtype)

        # Launch.
        compute_parallel_transport(template_specifications, output_dir=self.output_dir, **model_options)

    def compute_shooting(self, template_specifications, model_options={}):
        """ If control points and momenta corresponding to a deformation have been obtained, 
        it is possible to shoot the corresponding deformation of obtain the flow of a shape under this deformation.

        :param dict template_specifications: Dictionary containing the description of the task that is to be performed (such as estimating a registration, an atlas, ...)
                as well as some hyper-parameters for the objects and the deformations used.
        :param dict model_options: Dictionary containing details about the model that is to be run.
        """

        # Check and completes the input parameters.
        template_specifications, model_options, _ = self.further_initialization(
            'ParallelTransport', template_specifications, model_options)

        logger.debug("dtype=" + default.dtype)

        # Launch.
        compute_shooting(template_specifications, output_dir=self.output_dir, **model_options)

    ####################################################################################################################
    # Auxiliary methods.
    ####################################################################################################################

    @staticmethod
    def __launch_estimator(estimator, write_output=True):
        """
        Launch the estimator. This will iterate until a stop condition is reached.

        :param estimator:   Estimator that is to be used.
                            eg: :class:`GradientAscent <core.estimators.gradient_ascent.GradientAscent>`, :class:`ScipyOptimize <core.estimators.scipy_optimize.ScipyOptimize>`
        """        
        if estimator.stop:
            return 
        
        logger.debug("dtype=" + default.dtype)
        start_time = time.time()
        logger.info('>> Started estimator: ' + estimator.name)
        write = estimator.update() 
        end_time = time.time()

        if write_output and write:
            print("Final writing")
            estimator.write()

        if end_time - start_time > 60 * 60 * 24:
            logger.info('>> Estimation took: %s' %
                        time.strftime("%d days, %H hours, %M minutes and %S seconds",
                                      time.gmtime(end_time - start_time)))
        elif end_time - start_time > 60 * 60:
            logger.info('>> Estimation took: %s' %
                        time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(end_time - start_time)))
        elif end_time - start_time > 60:
            logger.info('>> Estimation took: %s' %
                        time.strftime("%M minutes and %S seconds", time.gmtime(end_time - start_time)))
        else:
            logger.info('>> Estimation took: %s' % time.strftime("%S seconds", time.gmtime(end_time - start_time)))
        

    def __instantiate_estimator(self, statistical_model, dataset, estimator_options, default=ScipyOptimize):
        if estimator_options['optimization_method_type'].lower() == 'GradientAscent'.lower():
            estimator = GradientAscent
        #ajouts vd
        elif estimator_options['optimization_method_type'].lower() == 'ProximalGradientAscent'.lower():
            estimator = ProximalGradientAscent
        elif estimator_options['optimization_method_type'].lower() == 'HardthresholdGradientAscent'.lower():
            estimator = HardthresholdGradientAscent
        elif estimator_options['optimization_method_type'].lower() == 'StochasticGradientAscent'.lower():
            estimator = StochasticGradientAscent
            # set batch number
            if dataset.total_number_of_observations < 6 and dataset.number_of_subjects < 6:
                print("\nDefaulting to GradientAscent optimizer")
                estimator_options['optimization_method_type'] = 'GradientAscent'
                estimator = GradientAscent
            else:
                batch_size = 4
                if dataset.total_number_of_observations > 0:
                    estimator_options["number_of_batches"] = math.ceil(dataset.total_number_of_observations/batch_size)
                else:
                    estimator_options["number_of_batches"] = math.ceil(dataset.number_of_subjects/batch_size)

                print("\nSetting number of batches to", estimator_options["number_of_batches"])


        elif estimator_options['optimization_method_type'].lower() == 'ScipyLBFGS'.lower():
            estimator = ScipyOptimize
        elif estimator_options['optimization_method_type'].lower() == 'McmcSaem'.lower():
            estimator = McmcSaem
        else:
            estimator = default

        logger.debug(estimator_options)
        return estimator(statistical_model, dataset, output_dir=self.output_dir, **estimator_options)

    def further_initialization(self, model_type, template_specifications, model_options,
                               dataset_specifications=None, estimator_options=None, time = None):

        #
        # Consistency checks.
        #
        if dataset_specifications is None or estimator_options is None:
            assert model_type.lower() in ['Shooting'.lower(), 'ParallelTransport'.lower()], \
                'Only the "shooting" and "parallel transport" can run without a dataset and an estimator.'

        #
        # Initializes variables that will be checked.
        #
        if estimator_options is not None:
            if 'gpu_mode' not in estimator_options:
                estimator_options['gpu_mode'] = default.gpu_mode
            if estimator_options['gpu_mode'] is GpuMode.FULL and not torch.cuda.is_available():
                logger.warning("GPU computation is not available, falling-back to CPU.")
                estimator_options['gpu_mode'] = GpuMode.NONE

            if 'state_file' not in estimator_options:
                estimator_options['state_file'] = default.state_file
            if 'load_state_file' not in estimator_options:
                estimator_options['load_state_file'] = default.load_state_file
            if 'memory_length' not in estimator_options:
                estimator_options['memory_length'] = default.memory_length
            
            if 'multiscale_momenta' not in estimator_options: #ajout fg
                estimator_options['multiscale_momenta'] = default.multiscale_momenta
            if 'multiscale_images' not in estimator_options: #ajout fg
                estimator_options['multiscale_images'] = default.multiscale_images
            if 'multiscale_meshes' not in estimator_options: #ajout fg
                estimator_options['multiscale_meshes'] = default.multiscale_meshes
            if "gamma" not in estimator_options:
                estimator_options['gamma'] = default.gamma
            if "start_scale" not in estimator_options:
                estimator_options['start_scale'] = None
            if 'multiscale_strategy' not in estimator_options: #ajout fg
                estimator_options['multiscale_strategy'] = default.multiscale_strategy

        if 'dimension' not in model_options:
            model_options['dimension'] = default.dimension
        if 'dtype' not in model_options:
            model_options['dtype'] = default.dtype
        else:
            default.update_dtype(new_dtype=model_options['dtype'])

        model_options['tensor_scalar_type'] = default.tensor_scalar_type
        model_options['tensor_integer_type'] = default.tensor_integer_type

        if 'dense_mode' not in model_options:
            model_options['dense_mode'] = default.dense_mode
        if 'freeze_control_points' not in model_options:
            model_options['freeze_control_points'] = default.freeze_control_points
        if 'perform_shooting' not in model_options: #ajout fg
            model_options['perform_shooting'] = default.perform_shooting
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
        if 'number_of_processes' not in model_options:
            model_options['number_of_processes'] = default.number_of_processes
        if 't0' not in model_options:
            model_options['t0'] = default.t0
        if 'tR' not in model_options:
            model_options['tR'] = []
        if 't1' not in model_options:
            model_options['t1'] = default.t0 #ajout fg
        if 'initial_time_shift_variance' not in model_options:
            model_options['initial_time_shift_variance'] = default.initial_time_shift_variance
        if 'initial_modulation_matrix' not in model_options:
            model_options['initial_modulation_matrix'] = default.initial_modulation_matrix
        if 'number_of_sources' not in model_options:
            model_options['number_of_sources'] = default.number_of_sources
        if 'initial_acceleration_variance' not in model_options:
            model_options['initial_acceleration_variance'] = default.initial_acceleration_variance
        if 'downsampling_factor' not in model_options:
            model_options['downsampling_factor'] = default.downsampling_factor
        if 'interpolation' not in model_options: #ajout fg
            model_options['interpolation'] = default.interpolation
        if 'use_sobolev_gradient' not in model_options:
            model_options['use_sobolev_gradient'] = default.use_sobolev_gradient
        if 'sobolev_kernel_width_ratio' not in model_options:
            model_options['sobolev_kernel_width_ratio'] = default.sobolev_kernel_width_ratio
        if "kernel_regression" not in model_options:
            model_options['kernel_regression'] = default.kernel_regression

        #
        # Check and completes the user-given parameters.
        #

        # Optional random seed.
        if 'random_seed' in model_options and model_options['random_seed'] is not None:
            self.set_seed(model_options['random_seed'])

        # If needed, infer the dimension from the template specifications.
        if model_options['dimension'] is None:
            model_options['dimension'] = self.__infer_dimension(template_specifications)

        # Smoothing kernel width.
        if model_options['use_sobolev_gradient']:
            model_options['smoothing_kernel_width'] = \
                model_options['deformation_kernel_width'] * model_options['sobolev_kernel_width_ratio']

        # Dense mode.
        if model_options['dense_mode']:
            logger.info('>> Dense mode activated. No distinction will be made between template and control points.')
            assert len(template_specifications) == 1, \
                'Only a single object can be considered when using the dense mode.'
            if not model_options['freeze_control_points']:
                model_options['freeze_control_points'] = True
                msg = 'With active dense mode, the freeze_template (currently %s) and freeze_control_points ' \
                      '(currently %s) flags are redundant. Defaulting to freeze_control_points = True.' \
                      % (str(model_options['freeze_template']), str(model_options['freeze_control_points']))
                logger.info('>> ' + msg)
            if model_options['initial_control_points'] is not None:
                # model_options['initial_control_points'] = None
                msg = 'With active dense mode, specifying initial_control_points is useless. Ignoring this xml entry.'
                logger.info('>> ' + msg)

        if model_options['initial_cp_spacing'] is None and model_options['initial_control_points'] is None \
                and not model_options['dense_mode']:
            # logger.info('>> No initial CP spacing given: using diffeo kernel width of '
            #             + str(model_options['deformation_kernel_width']))
            model_options['initial_cp_spacing'] = model_options['deformation_kernel_width']

        # Multi-threading/processing only available for the deterministic atlas for the moment.
        if model_options['number_of_processes'] > 1:

            if model_type.lower() in ['Shooting'.lower(), 'ParallelTransport'.lower(), 'Registration'.lower()]:
                model_options['number_of_processes'] = 1
                msg = 'It is not possible to estimate a "%s" model with multithreading. ' \
                      'Overriding the "number-of-processes" option, now set to 1.' % model_type
                logger.info('>> ' + msg)

            elif model_type.lower() in ['BayesianAtlas'.lower(), 'Regression'.lower(),
                                        'LongitudinalRegistration'.lower(), 'ClusteredBayesianAtlas'.lower()]:
                model_options['number_of_processes'] = 1
                msg = 'It is not possible at the moment to estimate a "%s" model with multithreading. ' \
                      'Overriding the "number-of-processes" option, now set to 1.' % model_type
                logger.info('>> ' + msg)

        # try and automatically set best number of thread per spawned process if not overridden by uer
        if 'OMP_NUM_THREADS' not in os.environ:
            #logger.info('OMP_NUM_THREADS was not found in environment variables. An automatic value will be set.')
            hyperthreading = utilities.has_hyperthreading()
            #omp_num_threads = math.floor(os.cpu_count() / model_options['number_of_processes'])
            #modif vd
            omp_num_threads = max(1,math.floor(os.cpu_count() / model_options['number_of_processes'] - 5))

            if hyperthreading:
                omp_num_threads = math.ceil(omp_num_threads / 2)

            omp_num_threads = max(1, int(omp_num_threads))

            #logger.info('OMP_NUM_THREADS will be set to ' + str(omp_num_threads))
            os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
        # else:
        #     logger.info('OMP_NUM_THREADS found in environment variables. Using value OMP_NUM_THREADS=' + str(
        #         os.environ['OMP_NUM_THREADS']))

        # If longitudinal model and t0 is not initialized, initializes it.
        if model_type.lower() in ['Regression'.lower(), 'PiecewiseRegression'.lower(), 'BayesianPiecewiseRegression'.lower(),
                                    'ClusteredLongitudinalAtlas'.lower(),
                                  'LongitudinalAtlas'.lower(), 'LongitudinalAtlasSimplified'.lower(), 'LongitudinalRegistration'.lower()]:
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
                    logger.info('>> Initial t0 set to the mean visit age: %.2f' % mean_visit_age)
                    model_options['t0'] = mean_visit_age
                else:
                    logger.info('>> Initial t0 set by the user to %.2f ; note that the mean visit age is %.2f'
                                % (model_options['t0'], mean_visit_age))

                if not model_type.lower() == 'regression':
                    if model_options['initial_time_shift_variance'] is None:
                        logger.info('>> Initial time-shift std set to the empirical std of the visit ages: %.2f'
                                    % math.sqrt(var_visit_age))
                        model_options['initial_time_shift_variance'] = var_visit_age
                    else:
                        logger.info(
                            ('>> Initial time-shift std set by the user to %.2f ; note that the empirical std of '
                             'the visit ages is %.2f') % (math.sqrt(model_options['initial_time_shift_variance']),
                                                          math.sqrt(var_visit_age)))

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
            # Initializes the state file.
            if estimator_options['state_file'] is None:
                path_to_state_file = os.path.join(self.output_dir, "deformetrica-state.p")
                logger.info('>> No specified state-file. By default, Deformetrica state will by saved in file: %s.' %
                            path_to_state_file)
                if os.path.isfile(path_to_state_file):
                    os.remove(path_to_state_file)
                    logger.info('>> Removing the pre-existing state file with same path.')
                estimator_options['state_file'] = path_to_state_file
            else:
                if os.path.exists(estimator_options['state_file']):
                    estimator_options['load_state_file'] = True
                    logger.info(
                        '>> Deformetrica will attempt to resume computation from the user-specified state file: %s.'
                        % estimator_options['state_file'])
                # else:
                #     msg = 'The user-specified state-file does not exist: %s. State cannot be reloaded. ' \
                #           'Future Deformetrica state will be saved at the given path.' % estimator_options['state_file']
                #     logger.info('>> ' + msg)

            # Warning if scipy-LBFGS with memory length > 1 and sobolev gradient.
            if estimator_options['optimization_method_type'].lower() == 'ScipyLBFGS'.lower() \
                    and estimator_options['memory_length'] > 1 \
                    and not model_options['freeze_template'] and model_options['use_sobolev_gradient']:
                logger.info(
                    '>> Using a Sobolev gradient for the template data with the ScipyLBFGS estimator memory length '
                    'being larger than 1. Beware: that can be tricky.')

        # Freeze the fixed effects in case of a registration.
        if model_type.lower() == 'Registration'.lower():
            model_options['freeze_template'] = True
        elif model_type.lower() == 'KernelRegression'.lower():
            model_options['kernel_regression'] = True
            model_options['time'] = time
            model_options['visit_ages'] = []
        elif model_type.lower() == 'LongitudinalRegistration'.lower():
            model_options['freeze_template'] = True
            model_options['freeze_control_points'] = True
            model_options['freeze_momenta'] = True
            model_options['freeze_modulation_matrix'] = True
            model_options['freeze_reference_time'] = True
            model_options['freeze_time_shift_variance'] = True
            model_options['freeze_acceleration_variance'] = True
            model_options['freeze_noise_variance'] = True

        # Initialize the number of sources if needed.
        if model_type.lower() in ['BayesianPiecewiseRegression'.lower(), 'LongitudinalAtlas'.lower(), 'LongitudinalAtlasSimplified'.lower()] \
                and model_options['initial_modulation_matrix'] is None and model_options['number_of_sources'] is None:
            model_options['number_of_sources'] = 4
            logger.info('>> No initial modulation matrix given, neither a number of sources. '
                        'The latter will be ARBITRARILY defaulted to %d.' % model_options['number_of_sources'])

        # Initialize the initial_acceleration_variance if needed.
        if (model_type in ['LongitudinalAtlas'.lower(), 'LongitudinalAtlasSimplified'.lower(), 'LongitudinalRegistration'.lower()]) \
                and model_options['initial_acceleration_variance'] is None:
            acceleration_std = 0.5
            logger.info('>> The initial acceleration std fixed effect is ARBITRARILY set to %.2f.' % acceleration_std)
            model_options['initial_acceleration_variance'] = (acceleration_std ** 2)
        
        #ajout vd
        if model_type == 'ClusteredBayesianAtlas'.lower() and model_options['nb_classes'] is None:
            model_options['nb_classes'] = 1
            print('≥> The number of classes is set to 1')
        ###########

        # Checking the number of image objects, and moving as desired the downsampling_factor parameter.
        count = 0
        for elt in template_specifications.values():
            if elt['deformable_object_type'].lower() == 'image':
                count += 1
                if not model_options['downsampling_factor'] == 1:
                    if 'downsampling_factor' in elt.keys():
                        logger.info('>> Warning: the downsampling_factor option is specified twice. ')
                    else:
                        elt['downsampling_factor'] = model_options['downsampling_factor']
                        logger.info('>> Setting the image grid downsampling factor to: %d.' % model_options['downsampling_factor'])
                elt["interpolation"] = model_options["interpolation"]
                
        if count > 1:
            raise RuntimeError('Only a single image object can be used.')
        if count == 0 and not model_options['downsampling_factor'] == 1:
            msg = 'The "downsampling_factor" parameter is useful only for image data, ' \
                  'but none is considered here. Ignoring.'
            logger.info('>> ' + msg)

        # Initializes the proposal distributions.
        if estimator_options is not None and \
                estimator_options['optimization_method_type'].lower() == 'McmcSaem'.lower():

            assert model_type.lower() in ['LongitudinalAtlas'.lower(), 'LongitudinalAtlasSimplified'.lower(), 
                                            'BayesianAtlas'.lower(), 'ClusteredLongitudinalAtlas'.lower(), 
                                            'ClusteredBayesianAtlas'.lower(), 'BayesianAtlasSparse'.lower(),
                                            'BayesianPiecewiseRegression'.lower()], \
                'Only the "BayesianAtlas" and "LongitudinalAtlas" models can be estimated with the "McmcSaem" ' \
                'algorithm, when here was specified a "%s" model.' % model_type

            if model_type.lower() in ['BayesianPiecewiseRegression'.lower(), 'LongitudinalAtlas'.lower(), 'LongitudinalAtlasSimplified'.lower()]:

                if 'onset_age_proposal_std' not in estimator_options:
                    estimator_options['onset_age_proposal_std'] = default.onset_age_proposal_std
                if 'acceleration_proposal_std' not in estimator_options:
                    estimator_options['acceleration_proposal_std'] = default.acceleration_proposal_std
                if 'sources_proposal_std' not in estimator_options:
                    estimator_options['sources_proposal_std'] = default.sources_proposal_std

                estimator_options['individual_proposal_distributions'] = {
                    'onset_age': MultiScalarNormalDistribution(std=estimator_options['onset_age_proposal_std']),
                    'acceleration': MultiScalarNormalDistribution(std=estimator_options['acceleration_proposal_std']),
                    'sources': MultiScalarNormalDistribution(std=estimator_options['sources_proposal_std'])}

            #ajouts vd
            elif model_type.lower() == 'ClusteredLongitudinalAtlas'.lower():

                if 'onset_age_proposal_std' not in estimator_options:
                    estimator_options['onset_age_proposal_std'] = default.onset_age_proposal_std
                if 'acceleration_proposal_std' not in estimator_options:
                    estimator_options['acceleration_proposal_std'] = default.acceleration_proposal_std/50
                if 'sources_proposal_std' not in estimator_options:
                    estimator_options['sources_proposal_std'] = default.sources_proposal_std*50

                estimator_options['individual_proposal_distributions'] = {
                    'classes': UniformDistribution(max=model_options['nb_classes']),
                    'onset_ages': MultiScalarNormalDistribution(std=estimator_options['onset_age_proposal_std']),
                    'accelerations': MultiScalarNormalDistribution(std=estimator_options['acceleration_proposal_std']),
                    'sources': MultiScalarNormalDistribution(std=estimator_options['sources_proposal_std'])}
            ###############
            elif model_type.lower() == 'BayesianAtlas'.lower():
                if 'momenta_proposal_std' not in estimator_options:
                    estimator_options['momenta_proposal_std'] = default.momenta_proposal_std

                estimator_options['individual_proposal_distributions'] = {
                    'momenta': MultiScalarNormalDistribution(std=estimator_options['momenta_proposal_std'])}

            #ajouts vd
            elif model_type.lower() == 'BayesianAtlasSparse'.lower():
                if 'momenta_proposal_std' not in estimator_options:
                    estimator_options['momenta_proposal_std'] = 100*default.momenta_proposal_std
                    estimator_options['module_positions_proposal_std'] = 200*default.momenta_proposal_std
                    estimator_options['module_intensities_proposal_std'] = 30*default.momenta_proposal_std
                    estimator_options['module_variances_proposal_std'] = 100*default.momenta_proposal_std

                (object_list, objects_name, objects_name_extension,
                 objects_noise_variance, multi_object_attachment) = create_template_metadata(
                    template_specifications, model_options['dimension'], gpu_mode='cpu')

                template = DeformableMultiObject(object_list)
                initial_cp = initialize_control_points(None, template, model_options['space_between_modules'], None,
                                                       model_options['dimension'], False)
                nb_modules = initial_cp.shape[0]
                nb_subjects = dataset_specifications['subject_ids'].__len__()
                estimator_options['individual_proposal_distributions'] = {
                    'momenta': MultiScalarNormalDistribution(std=estimator_options['momenta_proposal_std']),
                    #'module_intensities': MultiScalarTruncatedNormalDistribution(std=estimator_options['module_intensities_proposal_std']),
                    #'module_variances': MultiScalarNormalDistribution(std=estimator_options['module_variances_proposal_std']),
                    #'module_directions': MultiScalarNormalDistribution(std=estimator_options['module_variances_proposal_std'])
                }

                #for i in range(nb_modules):
                #    estimator_options['individual_proposal_distributions']['module_' + str(i)] = MultiScalarNormalDistribution(std= np.array([10.,10.,50.,2.,2.,1.,1.]))
                estimator_options['individual_proposal_distributions'][
                    'sparse_matrix'] = MultiScalarTruncatedNormalDistribution(std=0.05)
                #for i in range(nb_subjects):
                #    estimator_options['individual_proposal_distributions']['module_positions_subj' + str(i)] = MultiScalarNormalDistribution(std=estimator_options['module_positions_proposal_std'])
                        #estimator_options['individual_proposal_distributions'][
                        #    'module_positions_subj' + str(i)].set_boundary(np.min(object_list[0].bounding_box,1), np.max(object_list[0].bounding_box,1))
            
            elif model_type.lower() == 'ClusteredBayesianAtlas'.lower():
                if 'momenta_proposal_std' not in estimator_options:
                    estimator_options['momenta_proposal_std'] = default.momenta_proposal_std*10

                estimator_options['individual_proposal_distributions'] = {
                    'momenta': MultiScalarNormalDistribution(std=estimator_options['momenta_proposal_std']),
                    'classes': UniformDistribution(max=model_options['nb_classes'])
                }
        ###############
        
        return template_specifications, model_options, estimator_options

    @staticmethod
    def __infer_dimension(template_specifications):
        reader = DeformableObjectReader()
        max_dimension = 0
        for elt in template_specifications.values():
            object_filename = elt['filename']
            object_type = elt['deformable_object_type']
            o = reader.create_object(object_filename, object_type, dimension=None)
            d = o.dimension
            max_dimension = max(d, max_dimension)
        return max_dimension
