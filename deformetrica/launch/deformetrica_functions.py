from ..in_out.xml_parameters import XmlParameters, get_dataset_specifications, get_estimator_options, get_model_options
import xml.etree.ElementTree as et
from ..core import default
from ..core.model_tools.deformations.exponential import Exponential
from ..in_out.array_readers_and_writers import *
from ..support import kernels as kernel_factory
import torch
import shutil
from os.path import join

def estimate_registration(deformetrica, xml_parameters):
    xml_parameters.model_type = 'Registration'.lower()
    xml_parameters.optimization_method_type = 'GradientAscent'.lower()
    xml_parameters.freeze_template = True
    xml_parameters.save_every_n_iters = 100

    model = deformetrica.estimate_registration(
        xml_parameters.template_specifications,
        get_dataset_specifications(xml_parameters),
        estimator_options=get_estimator_options(xml_parameters),
        model_options=get_model_options(xml_parameters))

    return model


def estimate_bayesian_atlas(deformetrica, xml_parameters):
    xml_parameters.model_type = 'BayesianAtlas'.lower()
    xml_parameters.freeze_template = False
    xml_parameters.save_every_n_iters = 10

    model, individual_RER = deformetrica.estimate_bayesian_atlas(
        xml_parameters.template_specifications,
        get_dataset_specifications(xml_parameters),
        estimator_options=get_estimator_options(xml_parameters),
        model_options=get_model_options(xml_parameters))

    return model, individual_RER['momenta']


def estimate_deterministic_atlas(deformetrica, xml_parameters):
    xml_parameters.freeze_template = False
    xml_parameters.save_every_n_iters = 10
    xml_parameters.model_type = "DeterministicAtlas".lower()

    return deformetrica.estimate_deterministic_atlas(
        xml_parameters.template_specifications,
        get_dataset_specifications(xml_parameters),
        estimator_options=get_estimator_options(xml_parameters),
        model_options=get_model_options(xml_parameters)), None

def estimate_regression(deformetrica, xml_parameters):
    if "piecewise" in xml_parameters.model_type.lower():
        return estimate_piecewise_geodesic_regression(deformetrica, xml_parameters)
    else:
        return estimate_geodesic_regression(deformetrica, xml_parameters)


def estimate_geodesic_regression(deformetrica, xml_parameters):
    model_options=get_model_options(xml_parameters)
    model_options["write_adjoint_parameters"] = True
    return deformetrica.estimate_geodesic_regression(
        xml_parameters.template_specifications,
        get_dataset_specifications(xml_parameters),
        estimator_options=get_estimator_options(xml_parameters),
        model_options=model_options)

def estimate_piecewise_geodesic_regression(deformetrica, xml_parameters):
    model_options=get_model_options(xml_parameters)
    model_options["write_adjoint_parameters"] = True
    
    return deformetrica.estimate_piecewise_geodesic_regression(
        xml_parameters.template_specifications,
        get_dataset_specifications(xml_parameters),
        estimator_options=get_estimator_options(xml_parameters),
        model_options=model_options)

def set_xml_for_regression(xml_parameters):
    xml_parameters.model_type = 'Regression'.lower()
    xml_parameters.optimization_method_type = 'GradientAscent'.lower()
    xml_parameters.freeze_template = True
    xml_parameters.freeze_control_points = True
    xml_parameters.state_file = None
    xml_parameters.print_every_n_iters = 50

    return xml_parameters


def estimate_geodesic_regression_single_point(deformetrica, xml_parameters, output_path,
                                 global_dataset_filenames, global_visit_ages, global_subject_ids,
                                 global_tmin, global_initial_objects_template_path):
    # Create folder.
    regression_output_path = join(output_path, 'Global_GeodesicRegression')
    if os.path.isdir(regression_output_path): 
        shutil.rmtree(regression_output_path)
    os.mkdir(regression_output_path)

    # set the initial template as the Bayesian template
    for k, (_, object_specs) in enumerate(xml_parameters.template_specifications.items()):
        object_specs['filename'] = global_initial_objects_template_path[k]
    
    # Adapt the specific xml parameters and update.Logger has been set to
    #join filenames as if from a single subject = [[{obs 1}] [{obs2}]] -> [[{obs 1} {obs2}]]
    xml_parameters = set_xml_for_regression(xml_parameters)
    xml_parameters.dataset_filenames = [sum(global_dataset_filenames, [])]
    xml_parameters.visit_ages = [sum(global_visit_ages, [])]
    xml_parameters.subject_ids = [global_subject_ids[0]]
    xml_parameters.t0 = global_tmin

    deformetrica.output_dir = regression_output_path

    model = estimate_geodesic_regression(deformetrica, xml_parameters)

    return model.get_control_points(), model.get_momenta()

def estimate_subject_geodesic_regression(i, deformetrica, xml_parameters, output_path,
                                        global_dataset_filenames, global_visit_ages, global_subject_ids,
                                        t0 = None):

    # Create folder.
    subject_regression_output = join(output_path, 'GeodesicRegression__subject_' + global_subject_ids[i])
    if os.path.isdir(subject_regression_output): 
        shutil.rmtree(subject_regression_output)
    os.mkdir(subject_regression_output)

    # Adapt the specific xml parameters and update.
    xml_parameters = set_xml_for_regression(xml_parameters)
    xml_parameters.dataset_filenames = [global_dataset_filenames[i]]
    xml_parameters.visit_ages = [global_visit_ages[i]]
    xml_parameters.subject_ids = [global_subject_ids[i]]
    xml_parameters.t0 = t0
    xml_parameters.tmin = min(global_visit_ages[i])
    xml_parameters.tmax = max(global_visit_ages[i])

    deformetrica.output_dir = subject_regression_output

    model = estimate_geodesic_regression(deformetrica, xml_parameters)

    # Add the estimated momenta.
    return model.get_control_points(), model.get_momenta()


def estimate_longitudinal_registration(deformetrica, xml_parameters, overwrite=True):
    xml_parameters.model_type = 'LongitudinalRegistration'.lower()
    xml_parameters.optimization_method_type = 'ScipyLBFGS'.lower()

    return deformetrica.estimate_longitudinal_registration(
        xml_parameters.template_specifications,
        get_dataset_specifications(xml_parameters),
        estimator_options=get_estimator_options(xml_parameters),
        model_options=get_model_options(xml_parameters), overwrite=overwrite)


def estimate_longitudinal_atlas(deformetrica, xml_parameters):
    xml_parameters.model_type = 'LongitudinalAtlasSimplified'.lower()
    xml_parameters.optimization_method_type = 'GradientAscent'.lower()
    # Means we only optimize the parameters of class 2 ... y0, A0, m0. 
    

    xml_parameters.max_line_search_iterations = 20

    return deformetrica.estimate_longitudinal_atlas_simplified(
                        xml_parameters.template_specifications,
                        get_dataset_specifications(xml_parameters),
                        estimator_options=get_estimator_options(xml_parameters),
                        model_options=get_model_options(xml_parameters))


def insert_model_xml_level1_entry(model_xml_0, key, value):
    found_tag = False
    for model_xml_level1 in model_xml_0:
        if model_xml_level1.tag.lower() == key:
            model_xml_level1.text = value
            found_tag = True
    if not found_tag:
        new_element_xml = et.SubElement(model_xml_0, key)
        new_element_xml.text = value
    return model_xml_0


def insert_model_xml_template(model_xml_0, key, values):
    for model_xml_level1 in model_xml_0:
        if model_xml_level1.tag.lower() == 'template':
            k = -1
            for model_xml_level2 in model_xml_level1:
                if model_xml_level2.tag.lower() == 'object':
                    k += 1
                    found_tag = False
                    for model_xml_level3 in model_xml_level2:
                        if model_xml_level3.tag.lower() == key.lower():
                            model_xml_level3.text = values[k]
                            found_tag = True
                    if not found_tag:
                        new_element_xml = et.SubElement(model_xml_level2, key)
                        new_element_xml.text = values[k]
    return model_xml_0


def insert_model_xml_deformation_parameters(model_xml_0, key, value):
    for model_xml_level1 in model_xml_0:
        if model_xml_level1.tag.lower() == 'deformation-parameters':
            found_tag = False
            for model_xml_level2 in model_xml_level1:
                if model_xml_level2.tag.lower() == key:
                    model_xml_level2.text = value
                    found_tag = True
            if not found_tag:
                new_element_xml = et.SubElement(model_xml_level1, key)
                new_element_xml.text = value
    return model_xml_0


def shoot(control_points, momenta, kernel_width, kernel_type,
          number_of_time_points=default.number_of_time_points,
          dense_mode=default.dense_mode,
          tensor_scalar_type=default.tensor_scalar_type):
    control_points_torch = torch.from_numpy(control_points).type(tensor_scalar_type)
    momenta_torch = torch.from_numpy(momenta).type(tensor_scalar_type)
    exponential = Exponential(
                            dense_mode=dense_mode, number_of_time_points=number_of_time_points,
                            kernel=kernel_factory.factory(kernel_type, kernel_width=kernel_width),
                            initial_control_points=control_points_torch, initial_momenta=momenta_torch)
    exponential.shoot()
    return exponential.control_points_t[-1].detach().cpu().numpy(), exponential.momenta_t[-1].detach().cpu().numpy()


def reproject_momenta(source_control_points, source_momenta, target_control_points,
                      kernel_width, kernel_type='torch', kernel_device='cpu',
                      tensor_scalar_type=default.tensor_scalar_type):
    kernel = kernel_factory.factory(kernel_type, kernel_width=kernel_width)
    source_cp_torch = tensor_scalar_type(source_control_points)
    source_momenta_torch = tensor_scalar_type(source_momenta)
    target_control_points_torch = tensor_scalar_type(target_control_points)
    target_momenta_torch = torch.cholesky_solve(
                            kernel.convolve(target_control_points_torch, source_cp_torch, source_momenta_torch),
                            torch.cholesky(kernel.get_kernel_matrix(target_control_points_torch), upper=True), upper=True)

    return target_momenta_torch.detach().cpu().numpy()


def parallel_transport(source_control_points, source_momenta, driving_momenta,
                       kernel_width, kernel_type='torch', kernel_device='cpu',
                       number_of_time_points=default.number_of_time_points,
                       dense_mode=default.dense_mode,
                       tensor_scalar_type=default.tensor_scalar_type):
    
    source_cp_torch = tensor_scalar_type(source_control_points)
    source_momenta_torch = tensor_scalar_type(source_momenta)
    driving_momenta_torch = tensor_scalar_type(driving_momenta)

     
    #
    print("1")
    exponential = Exponential(
        dense_mode=dense_mode, number_of_time_points=number_of_time_points, use_rk2_for_shoot=True,
        kernel = kernel_factory.factory(kernel_type, kernel_width=kernel_width),
        initial_control_points=source_cp_torch, initial_momenta=driving_momenta_torch)
    print("2")
    # Usually there is an error shooting the exponential
    # this is not a PT issue 
    exponential.shoot()
    transported_cp_torch = exponential.control_points_t[-1]
    print("1")
    transported_momenta_torch = exponential.parallel_transport(source_momenta_torch)[-1]
    
    return transported_cp_torch.detach().cpu().numpy(), transported_momenta_torch.detach().cpu().numpy()
