
import numpy as np
from core.models.longitudinal_atlas import LongitudinalAtlas


from ..core import default
from .support import kernels as kernel_factory
from .in_out.array_readers_and_writers import *
from .in_out.deformable_object_reader import DeformableObjectReader
from observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from model_tools.deformations.spatiotemporal_reference_frame import SpatiotemporalReferenceFrame


n_subjects = 100
longitudinal_subjects_ratio = 0.1
n_sources = 4

# Choose fixed effects
output_dir = "home/fleur.gaudfernau/longitudinal_atlas/simulated_data/output"
template_shape = "home/fleur.gaudfernau/longitudinal_atlas/simulated_data/template.png"
global_cp =  "home/fleur.gaudfernau/longitudinal_atlas/simulated_data/ControlPoints.txt"
global_cp_array = read_3D_array(global_cp)
global_momenta =  "home/fleur.gaudfernau/longitudinal_atlas/simulated_data/Momenta.txt"
global_momenta_array = read_3D_array(global_momenta)
mod_matrix =  "home/fleur.gaudfernau/longitudinal_atlas/simulated_data/ModulationMatrix.txt"
mod_matrix_array = read_2D_array(mod_matrix)
sigma_e = 0.01

print("mod_matrix_array", mod_matrix_array)

#sigma_tau = 
#No acceleration

# Generate the trajectories
avg_observations_nb = 1
tmin = 1
t_0 = 5
tmax = 10

observation_times = []

for i in range(n_subjects):

    # Draw a number of observations
    n_obs = np.random.poisson(avg_observations_nb)
    while n_obs == 0:
        n_obs = np.random.poisson(avg_observations_nb)
    
    # Draw the observation times
    times = []
    for k in range(n_obs):
        time = np.random.uniform(tmin, tmax)
    observation_times.append(times)

    # Simulate the sources
    sources = np.random.standard_normal(n_sources) #draw n_sources samples from N(0,1)

    # Simulate tau_i


# Create a dataset
reader = DeformableObjectReader()
object_list = []
for object_id in ['main_object']:
    object_list.append(reader.create_object(template_shape, object_type = 'Image', dimension = 2))

template = DeformableMultiObject(object_list)
template_data = template.get_data()
template_points = template.get_points()

# Compute the average geodesic
spatiotemporal_reference_frame = SpatiotemporalReferenceFrame(
    dense_mode=default.dense_mode, kernel=kernel_factory.factory(default.deformation_kernel_type,
                                    gpu_mode=default.gpu_mode, kernel_width=5),
    shoot_kernel_type=default.shoot_kernel_type, concentration_of_time_points=10, number_of_time_points=11)

spatiotemporal_reference_frame.set_template_points_t0(template_points)
spatiotemporal_reference_frame.set_control_points_t0(global_cp_array)
spatiotemporal_reference_frame.set_momenta_t0(global_momenta_array)
spatiotemporal_reference_frame.set_modulation_matrix_t0(mod_matrix_array)
spatiotemporal_reference_frame.set_t0(t_0)
spatiotemporal_reference_frame.set_tmin(tmin)
spatiotemporal_reference_frame.set_tmax(tmax)
spatiotemporal_reference_frame.update()

for time in spatiotemporal_reference_frame.times:
    template_points_t = spatiotemporal_reference_frame.template_points_t[time]
    deformed_data = template.get_deformed_data(template_points_t, template_data)
    template.write(output_dir, ["A_shape"], {key: value.detach().cpu().numpy() for key, value in deformed_data.items()})


# Write the geodesic and the orthogonal flow
#spatiotemporal_reference_frame.geodesic.write("Test", "a_shape", ".png", template, template_data, output_dir,
#                                        write_adjoint_parameters = False)

#spatiotemporal_reference_frame.write(root_name, "a_shape", ".png", template, template_data, output_dir,
#                                    write_adjoint_parameters=False, write_exponential_flow=False)

#deformed_control_points = self.spatiotemporal_reference_frame.exponential.control_points_t[-1]
#deformed_points = self.spatiotemporal_reference_frame.exponential.get_template_points()
#deformed_data = self.template.get_deformed_data(deformed_points, template_data)
# self.set_template_data({key: value.detach().cpu().numpy() for key, value in deformed_data.items()})








# for each source l
# project column l of A on mo_|_
# compute the inivital velocity field w_l
# transport it along y

# for each subject i
# for each visit j
# compute the initial velocity field vij : a CL of the the wl (SUM w_l . si_l)
# shoot y(tij) with vij

        
deformed_points = self.spatiotemporal_reference_frame.get_template_points(time[i], sources[i], device=device)
deformed_data = self.template.get_deformed_data(deformed_points, template_data)
    
