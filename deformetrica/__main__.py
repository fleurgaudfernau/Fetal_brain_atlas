#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import logging
import os
import sys
import deformetrica as dfca

import torch
logger = logging.getLogger(__name__)

def main():
    # common options
    print("torch.cuda.is_available", torch.cuda.is_available())
    common_parser = argparse.ArgumentParser()
    common_parser.add_argument('--parameters', '-p', type=str, help='parameters xml file')
    common_parser.add_argument('--output', '-o', type=str, help='output folder')
    common_parser.add_argument('--age', "-a", type=int)
    common_parser.add_argument('--verbosity', '-v', type=str, default='INFO',
                               choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                               help='set output verbosity')
    
    # main parser
    description = 'Statistical analysis of 2D and 3D shape data. ' + os.linesep + os.linesep + 'version ' + dfca.__version__
    parser = argparse.ArgumentParser(prog='deformetrica', description=description, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(title='command', dest='command')
    subparsers.required = True  # make 'command' mandatory

    # estimate command
    parser_estimate = subparsers.add_parser('estimate', add_help=False, parents=[common_parser])
    parser_estimate.add_argument('model', type=str, help='model xml file')
    parser_estimate.add_argument('dataset', type=str, help='dataset xml file')

    # compute command
    parser_compute = subparsers.add_parser('compute', add_help=False, parents=[common_parser])
    parser_compute.add_argument('model', type=str, help='model xml file')

    # initialize command
    parser_initialize = subparsers.add_parser('initialize', add_help=False, parents=[common_parser])
    parser_initialize.add_argument('model', type=str, help='model xml file')
    parser_initialize.add_argument('dataset', type=str, help='dataset xml file')

    # initialize command
    parser_initialize = subparsers.add_parser('finalize', add_help=False, parents=[common_parser])
    parser_initialize.add_argument('model', type=str, help='model xml file')

    args = parser.parse_args()

    try:
        logger.setLevel(args.verbosity)
    except ValueError:
        logger.setLevel(logging.INFO)

    """
    Read xml files, set general settings, and call the adapted function.
    """
    output_dir = None
    try:
        if args.output is None:
            if not args.command == 'initialize':
                output_dir = dfca.default.output_dir
            else:
                output_dir = dfca.default.preprocessing_dir
            logger.info('Setting output directory to: ' + output_dir)
            os.makedirs(output_dir)
        else:
            logger.info('Setting output directory to: ' + args.output)
            output_dir = args.output
    except FileExistsError:
        pass
    
    age = args.age if args.age else None

    deformetrica = dfca.Deformetrica(output_dir=output_dir, verbosity=logger.level)

    xml = dfca.io.XmlParameters()
    xml.read_all_xmls(args.model, args.dataset if args.command == 'estimate' else None,
                    args.parameters)

    dataset_spec = dfca.io.get_dataset_specifications(xml)
    estimator_options = dfca.io.get_estimator_options(xml)
    model_options = dfca.io.get_model_options(xml)

    logger.info("*******************************************************************")
    logger.info(">>> Launching Model {} ".format(xml.model_type))
    logger.info("*******************************************************************")
    logger.info(">>> Dataset information: ")
    logger.info("       Number of subjects: {}".format(dataset_spec["n_subjects"]))
    logger.info("       Total number of observations: {}".format(dataset_spec["n_observations"]))
    logger.info("       Number of objects: {}".format(dataset_spec["n_objects"]))
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
        assert args.age is not None, "An age must be submitted with --age or -a"
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
            dfca.initialize_piecewise_geodesic_regression_with_space_shift(
                args.model, args.dataset, args.parameters)#, #output_dir=output_dir, overwrite=True)
    
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
