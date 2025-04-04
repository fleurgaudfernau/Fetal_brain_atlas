#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import logging
import os
import ast
import sys
import deformetrica as dfca

import torch
logger = logging.getLogger(__name__)

def main():
    # common options
    common_parser = argparse.ArgumentParser()
    common_parser.add_argument('--output', '-o', type=str, help='output folder')
    common_parser.add_argument('--verbosity', '-v', type=str, default='INFO',
                               choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                               help='set output verbosity')
    # Model parameters
    common_parser.add_argument('--model-type', "-m", type = str,  help = "Model type")
    common_parser.add_argument('--num-component', "-components", type = int,  help = "Number of components")
    common_parser.add_argument('--deformation-kernel-width', "-k", type = float,  help = "Deformation kernel width")

    # Template parameters
    common_parser.add_argument('--template', "-t", type = str,  help = "Template file")
    common_parser.add_argument('--attachment-type', "-attachment", type = str,  help = "Attachment type")
    common_parser.add_argument('--attachment-kernel', "-ak", type = float,  help = "Attachment kernel")
    common_parser.add_argument('--noise-std', "-n", type = float,  help = "Template file")

    # Time
    common_parser.add_argument('--time-concentration', "-tc", type = int,  help = "Concentration of time points")
    common_parser.add_argument('--number_of_sources', "-sources", type = int,  help = "Number of sources")
    common_parser.add_argument('--t0', "-t0", type = float,  help = "Initial time t0")
    common_parser.add_argument('--t1', "-t1", type = float,  help = "time")
    common_parser.add_argument('--tmin', "-tmin", type = float,  help = "Minimum time")
    common_parser.add_argument('--tmax', "-tmax", type = float,  help = "Maxmimal time")
    common_parser.add_argument('--tR', "-tR", type = ast.literal_eval, help = 'List of values for tR')

    # Initial files
    common_parser.add_argument('--initial-control-points', "-icp", type = str, help = "Initial control points file")
    common_parser.add_argument('--initial-momenta', "-im", type = str,  help = "Initial momenta file")
    common_parser.add_argument('--initial-modulation-matrix', "-imm", type = str,  help = "Initial modulation matrix file")
    common_parser.add_argument('--initial-sources', "-is", type = str,  help = "Initial sources file")
    common_parser.add_argument('--initial-momenta-to-transport', "-imt", type = str,  help = "Initial momenta to transport file")
    common_parser.add_argument('--initial-control-points-to-transport', "-icpt", type = int,  help = "Initial control points to transport file")

    # Optimization parameters
    common_parser.add_argument('--multiscale-momenta', "-ctf-momenta", action='store_true', 
                                help = "Multiscale on momentum vectors")
    common_parser.add_argument('--multiscale-objects', "-ctf-objects", action='store_true', 
                                help = "Multiscale on objects")
    common_parser.add_argument('--freeze-template', "-ft", action='store_true', 
                                help = "Freeze template optimization")
    common_parser.add_argument('--freeze-momenta', "-fm", action='store_true', 
                                help = "Freeze momenta optimization")

    common_parser.add_argument('--initial-step-size', "-step", type = float,  help = "Initial step size")
    common_parser.add_argument('--save-every-n-iters', "-save", type = int,  help = "Save every N iterations")
    common_parser.add_argument('--max-iterations', "-max-iter", type = int,  help = "Maximum number of iterations")
    common_parser.add_argument('--convergence-tolerance', "-convergence", type = int,  help = "Convergence threshold")
    
    common_parser.add_argument('--downsampling-factor', "-df", type = int,  help = "Image downsampling factor")
    common_parser.add_argument('--interpolation', "-i", type = str,  help = "Image interpolation type")

    common_parser.add_argument('--age', '-a', type=int, nargs='+', help="Time(s) at which to estimate Kernel regression (int or space-separated list of ints)")
    
    # main parser
    description = 'Statistical analysis of 2D and 3D shape data. {}{} version {}'.format(os.linesep, os.linesep, dfca.__version__)
    parser = argparse.ArgumentParser(prog='deformetrica', description=description, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(title='command', dest='command')
    subparsers.required = True

    # estimate command
    parser_estimate = subparsers.add_parser('estimate', add_help=False, parents=[common_parser])
    parser_estimate.add_argument('dataset', type=str, help='dataset xml file')

    # compute command
    parser_compute = subparsers.add_parser('compute', add_help=False, parents=[common_parser])
    parser_compute.add_argument('model', type=str, help='model xml file')

    # initialize command
    parser_initialize = subparsers.add_parser('initialize', add_help=False, parents=[common_parser])
    parser_initialize.add_argument('dataset', type=str, help='dataset xml file')

    # initialize command
    parser_initialize = subparsers.add_parser('finalize', add_help=False, parents=[common_parser])

    args = parser.parse_args()

    try:
        logger.setLevel(args.verbosity)
    except ValueError:
        logger.setLevel(logging.INFO)

    """
    Read xml files, set general settings, and call the adapted function.
    """

    xml = dfca.io.XmlParameters()
    xml.read_all_xmls(args)

    dataset_spec = dfca.io.get_dataset_specifications(xml)
    model_options = dfca.io.get_model_options(xml)
    estimator_options = dfca.io.get_estimator_options(xml)

    deformetrica = dfca.Deformetrica(output_dir=xml.output_dir, verbosity=logger.level)

    logger.info("\n*******************************************************************")
    logger.info(">>> Launching Model {} ".format(xml.model_type))
    logger.info("       Output directory: {}".format(xml.output_dir))
    logger.info("\n*******************************************************************")
    logger.info(">>> Dataset information: ")
    logger.info("       Number of subjects: {}".format(dataset_spec["n_subjects"]))
    logger.info("       Total number of observations: {}".format(dataset_spec["n_observations"]))
    logger.info("       Number of objects: {}".format(dataset_spec["n_objects"]))
    
    if model_options['downsampling_factor'] > 1:
        logger.info("       Downsampling factor: {}".format(model_options['downsampling_factor']))

    logger.info("\n*******************************************************************")
    logger.info(">>> Model information: ")
    logger.info("       Deformation kernel width: {}".format(model_options['deformation_kernel_width']))
    logger.info("       Noise std: {}".format(dataset_spec['noise_std']))
    
    if model_options["freeze_template"] is True:
        logger.info("       Template object is frozen during estimation")

    if model_options["num_component"] is not None:
        logger.info("       Number of component: {}".format(model_options['num_component']))
        logger.info(f"       Rupture times:{str(model_options['tR'])}")
    
    if args.age is not None:
        logger.info("Age(s): {}".format(args.age))

    if model_options["t0"] is not None:
        logger.info("       Initial time t0: {}".format(model_options['t0']))
    
    if "bayesian" in xml.model_type:
        logger.info("       Number of sources: {}".format(model_options['number_of_sources']))

    logger.info("\n*******************************************************************")
    logger.info(">>> Estimator information: ")
    logger.info("       Initial step size: {}".format(estimator_options['initial_step_size']))
    logger.info("       Convergence threshold: {}".format(estimator_options['convergence_tolerance']))
    if estimator_options["multiscale_momenta"] == True:
        logger.info("       Multiscale strategy on momentum vectors")
    if estimator_options["multiscale_objects"] == True:
        logger.info("       Multiscale strategy on the objects")

    logger.info("*******************************************************************")

    if xml.model_type in ['Registration'.lower(), 'DeformableTemplate'.lower(), 'Regression'.lower(),
                        'PiecewiseRegression'.lower(), 'KernelRegression'.lower(),
                        'InitializedBayesianGeodesicRegression'.lower()]:

        assert args.command == 'estimate', \
            'The estimation should be launched with the command deformetrica estimate '

    if xml.model_type == 'Registration'.lower():
        deformetrica.registration(xml.template_specifications, dataset_spec,
                                    model_options, estimator_options)
        
    elif xml.model_type == 'DeformableTemplate'.lower():
        deformetrica.deformable_template(xml.template_specifications, dataset_spec,
                                            model_options, estimator_options)        	

    elif xml.model_type == 'Regression'.lower():
        deformetrica.geodesic_regression(xml.template_specifications, dataset_spec,
                                                    model_options, estimator_options)
    
    elif xml.model_type == 'KernelRegression'.lower(): # ajout fg
        assert args.age is not None, "Age(s) must be submitted with --age or -a"
        for age in args.age:
            deformetrica.kernel_regression(age, xml.template_specifications, dataset_spec,
                                            model_options, estimator_options)
    
    elif xml.model_type == 'PiecewiseRegression'.lower(): #ajout fg
        deformetrica.piecewise_geodesic_regression(xml.template_specifications, dataset_spec,
                                                    model_options, estimator_options)
    
    elif xml.model_type == 'BayesianGeodesicRegression'.lower(): #ajout fg
        assert args.command in ['estimate', 'initialize'],\
            'The estimation of a regression model should be launched with the command: ' \
            '"deformetrica estimate" (and not {}).'.format(args.command)
        if args.command == 'estimate':
            deformetrica.piecewise_bayesian_geodesic_regression(
                xml.template_specifications, dataset_spec,  model_options, estimator_options)
        elif args.command == 'initialize':
            dfca.initialize_piecewise_geodesic_regression_with_space_shift(xml, args.dataset)
    
    elif xml.model_type == 'InitializeBayesianGeodesicRegression'.lower(): #ajout fg
        assert args.command in ['estimate', 'initialize'],\
            'The estimation of a regression model should be launched with the command: ' \
            '"deformetrica estimate" (and not {}).'.format(args.command)
        deformetrica.initialize_piecewise_bayesian_geodesic_regression(
                            xml.template_specifications, dataset_spec,  model_options, estimator_options)
    
    elif xml.model_type == 'InitializedBayesianGeodesicRegression'.lower(): #ajout fg
        assert args.command in ['estimate'],\
            'The estimation of a regression model should be launched with the command: ' \
            '"deformetrica estimate" (and not {}).'.format(args.command)
        deformetrica.initialized_piecewise_bayesian_geodesic_regression(
                            xml.template_specifications, dataset_spec, model_options, estimator_options)
                        
    elif xml.model_type == 'Shooting'.lower():
        assert args.command == 'compute', \
            'The computation of a shooting task should be launched with the command: ' \
            '"deformetrica compute" (and not {}).'.format(args.command)
        deformetrica.compute_shooting(xml.template_specifications, model_options)

    elif xml.model_type == 'ParallelTransport'.lower():
        assert args.command == 'compute', \
            'The computation of a parallel transport task should be launched with the command: ' \
            '"deformetrica compute" (and not {}).'.format(args.command)
        deformetrica.compute_parallel_transport(xml.template_specifications, model_options)
    
    elif xml.model_type == 'PiecewiseParallelTransport'.lower():
        assert args.command == 'compute', \
            'The computation of a parallel transport task should be launched with the command: ' \
            '"deformetrica compute" (and not {}).'.format(args.command)
        deformetrica.compute_piecewise_parallel_transport(xml.template_specifications, model_options)

    else:
        raise RuntimeError('Unrecognized model-type: {}'.format(xml.model_type))
        
if __name__ == "__main__":
    main()
    sys.exit(0)
