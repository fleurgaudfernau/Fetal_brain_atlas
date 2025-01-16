__version__ = '2025.01.15'

# api
from .api import Deformetrica

# core
from .core import default, GpuMode

# models
from .core import models as models

# model_tools
from .core.model_tools import attachments as attachments
from .core.model_tools import deformations as deformations
from .launch.initialize_longitudinal_atlas import initialize_longitudinal_atlas
from .launch.finalize_longitudinal_atlas import finalize_longitudinal_atlas
from .launch.estimate_longitudinal_metric_registration import estimate_longitudinal_metric_registration
#from .launch.initialize_longitudinal_atlas_simplified import initialize_longitudinal_atlas_simplified
from .launch.initialize_longitudinal_atlas_development import initialize_longitudinal_atlas_development
from .launch.initialize_piecewise_geodesic_regression_with_space_shift import initialize_piecewise_geodesic_regression_with_space_shift

# estimators
from .core import estimators as estimators

# io
from . import in_out as io

# kernels
from .support import kernels as kernels

# utils
from .support import utilities as utils

# gui
from . import gui as gui
