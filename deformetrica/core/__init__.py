from enum import Enum, auto


class GpuMode(Enum):
    """ Defines members of the GpuMode enumeration.
    # auto() assigns a unique, automatically generated value to AUTO, FULL, NONE and KERNEL
    """
    AUTO = auto()
    FULL = auto() # all operations on GPU
    NONE = auto()
    KERNEL = auto() # only kernel-intensive operations on GPU

def get_best_gpu_mode(model):
    """ Returns the best gpu mode with respect to the model that is to be run, gpu resources, ...
    TODO
    :return: GpuMode
    """
    from ..core.models.abstract_statistical_model import AbstractStatisticalModel
    from ..core.models.longitudinal_atlas import LongitudinalAtlas
    assert isinstance(model, AbstractStatisticalModel)

    gpu_mode = GpuMode.FULL
    if model.name == "LongitudinalAtlas":
        gpu_mode = GpuMode.KERNEL

    return gpu_mode
