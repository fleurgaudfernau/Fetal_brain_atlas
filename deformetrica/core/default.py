import os
from ..core import GpuMode
from ..support import utilities

logger_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

tensor_scalar_type = utilities.get_torch_scalar_type('float32')

output_dir = os.path.join(os.getcwd(), 'Output')
preprocessing_dir = os.path.join(os.getcwd(), 'preprocessing')
state_file = None
load_state_file = False

process_per_gpu = 1

model_type = 'undefined'
deformation_kernel_device = 'auto'

n_time_points = 11
time_concentration = 10
number_of_sources = None
use_rk2_for_shoot = False
use_rk2_for_flow = False
t0 = None
tmin = float('inf')
tmax = - float('inf')
covariance_momenta_prior_norm_dof = 0.001

optimization_method = 'GradientAscent'
optimized_log_likelihood = 'complete'
max_iterations = 1000
save_every_n_iters = 20
print_every_n_iters = 1
sample_every_n_mcmc_iters = 10
smoothing_kernel_width = None
initial_step_size = 1e-2
convergence_tolerance = 1e-4
noise_variance_prior_norm_dof = 0.01
memory_length = 10
downsampling_factor = 1
interpolation = "linear"

gpu_mode = GpuMode.FULL

freeze_template = False
multiscale_momenta = False
multiscale_objects = False
multiscale_strategy = "stairs"
freeze_momenta = False
freeze_modulation_matrix = False
freeze_reference_time = False
freeze_noise_variance = False
freeze_rupture_time = True

# For metric learning atlas
freeze_p0 = False
freeze_v0 = False
initial_cp = None
initial_momenta = None
initial_cp_to_transport = None
initial_momenta_to_transport = None
initial_modulation_matrix = None
initial_sources = None
initial_sources_mean = None
initial_sources_std = None

momenta_proposal_std = 0.01
sources_proposal_std = 0.01

verbose = 1

perform_shooting = True

def update_dtype(new_dtype):
    global tensor_scalar_type
    tensor_scalar_type = utilities.get_torch_scalar_type(dtype)

