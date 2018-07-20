import torch

from core import default
from core.model_tools.deformations.exponential import Exponential
from core.model_tools.deformations.geodesic import Geodesic
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from in_out.array_readers_and_writers import *
from in_out.dataset_functions import create_template_metadata


def compute_parallel_transport(control_points, initial_momenta, projected_momenta, template_specifications, tmin, tmax,
                               dense_mode=default.dense_mode,
                               concentration_of_time_points=default.concentration_of_time_points,
                               t0=None,
                               deformation_kernel=default.deformation_kernel, shoot_kernel=None, flow_kernel=None,
                               number_of_time_points=default.number_of_time_points,
                               use_rk2_for_shoot=default.use_rk2_for_shoot, use_rk2_for_flow=default.use_rk2_for_flow,
                               dimension=default.dimension,
                               tensor_scalar_type=default.tensor_scalar_type,
                               output_dir=default.output_dir):

    objects_list, objects_name, objects_name_extension, _, _ = create_template_metadata(template_specifications,
                                                                                        dimension,
                                                                                        tensor_scalar_type)
    template = DeformableMultiObject(objects_list, dimension)

    template_points = template.get_points()
    template_points = {key: torch.from_numpy(value).type(tensor_scalar_type) for key, value in template_points.items()}

    template_data = template.get_data()
    template_data = {key: torch.from_numpy(value).type(tensor_scalar_type) for key, value in template_data.items()}

    geodesic = Geodesic(dimension=dimension, dense_mode=dense_mode, tensor_scalar_type=tensor_scalar_type,
                        concentration_of_time_points=concentration_of_time_points, t0=t0,
                        deformation_kernel=deformation_kernel, shoot_kernel=shoot_kernel, flow_kernel=flow_kernel,
                        number_of_time_points=number_of_time_points,
                        use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow)

    # geodesic.concentration_of_time_points = concentration_of_time_points
    geodesic.set_kernel(deformation_kernel)
    geodesic.set_use_rk2_for_shoot(True)
    geodesic.set_use_rk2_for_flow(use_rk2_for_flow)

    # Those are mandatory parameters.
    assert tmin != -float("inf"), "Please specify a minimum time for the geodesic trajectory"
    assert tmax != float("inf"), "Please specify a maximum time for the geodesic trajectory"

    geodesic.set_tmin(tmin)
    geodesic.set_tmax(tmax)
    if t0 is None:
        geodesic.set_t0(geodesic.tmin)
    else:
        geodesic.set_t0(t0)

    geodesic.set_momenta_t0(initial_momenta)
    geodesic.set_control_points_t0(control_points)
    geodesic.set_template_points_t0(template_points)
    geodesic.update()

    # We write the flow of the geodesic
    geodesic.write("Regression", objects_name, objects_name_extension, template, template_data, output_dir=output_dir)

    # Now we transport!
    parallel_transport_trajectory = geodesic.parallel_transport(projected_momenta)

    # Getting trajectory caracteristics:
    times = geodesic._get_times()
    control_points_traj = geodesic._get_control_points_trajectory()
    momenta_traj = geodesic._get_momenta_trajectory()

    exponential = Exponential(dimension=dimension, dense_mode=dense_mode, tensor_scalar_type=tensor_scalar_type,
                              kernel=deformation_kernel, shoot_kernel=shoot_kernel, flow_kernel=flow_kernel,
                              number_of_time_points=number_of_time_points,
                              use_rk2_for_shoot=True, use_rk2_for_flow=use_rk2_for_flow)

    # exponential.number_of_time_points = xml_parameters.number_of_time_points
    # exponential.set_kernel(deformation_kernel)
    exponential.set_use_rk2_for_shoot(True)
    exponential.set_use_rk2_for_flow(use_rk2_for_flow)

    # We save the parallel trajectory
    for i, (time, cp, mom, transported_mom) in enumerate(zip(times, control_points_traj, momenta_traj, parallel_transport_trajectory)):
        # Writing the momenta/cps
        write_2D_array(cp.detach().cpu().numpy(), output_dir, "ControlPoints_tp_{0:d}__age_{1:.2f}.txt".format(i, time))
        write_3D_array(mom.detach().cpu().numpy(), output_dir, "Momenta_tp_{0:d}__age_{1:.2f}.txt".format(i, time))
        write_3D_array(transported_mom.detach().cpu().numpy(), output_dir, "Transported_Momenta_tp_{0:d}__age_{1:.2f}.txt".format(i, time))

        deformed_points = geodesic.get_template_points(time)

        # Shooting from the geodesic:
        exponential.set_initial_template_points(deformed_points)
        exponential.set_initial_control_points(cp)
        exponential.set_initial_momenta(transported_mom)
        exponential.update()

        parallel_points = exponential.get_template_points()
        parallel_data = template.get_deformed_data(parallel_points, template_data)

        names = [
            objects_name[k] + "_parallel_curve_tp_{0:d}__age_{1:.2f}".format(i, time) + objects_name_extension[k]
            for k in range(len(objects_name))]

        template.write(output_dir, names, {key: value.detach().cpu().numpy() for key, value in parallel_data.items()})
