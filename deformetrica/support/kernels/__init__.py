from enum import Enum
from ...core import default
from ...support.kernels.abstract_kernel import AbstractKernel


class Type(Enum):
    from ...support.kernels.keops_kernel import KeopsKernel

    UNDEFINED = None
    NO_KERNEL = None
    KEOPS = KeopsKernel

instance_map = dict()

def factory(kernel_type = "KEOPS", cuda_type=None, gpu_mode=None, *args, **kwargs):
    """Return an instance of a kernel corresponding to the requested kernel_type"""
    
    if cuda_type is None:
        cuda_type = "float32"
        
    if gpu_mode is None:
        gpu_mode = default.gpu_mode

        if not hasattr(factory, "gpu_mode_printed"):
            print("\nSetting GPU mode by default to {}".format(gpu_mode))
            factory.gpu_mode_printed = True

    # turn enum string to enum object
    if isinstance(kernel_type, str):
        try:
            for c in [' ', '-']:    # chars to be replaced for normalization
                kernel_type = kernel_type.replace(c, '_')
            kernel_type = Type[kernel_type.upper()]
        except:
            raise TypeError('kernel_type ' + kernel_type + ' could not be found')

    if not isinstance(kernel_type, Type):
        raise TypeError('kernel_type must be an instance of KernelType Enum')

    if kernel_type in [Type.UNDEFINED, Type.NO_KERNEL]:
        return None
    
    res = None
    hash = AbstractKernel.hash(kernel_type, cuda_type, gpu_mode, *args, **kwargs) #special code to id kernel

    #if hash not in instance_map: #store the new kernel object in a dict instance_map
    #     res = kernel_type.value(gpu_mode=gpu_mode, cuda_type=cuda_type, *args, **kwargs)    # instantiate : long!
    #     instance_map[hash] = res
    #else:
    #     res = instance_map[hash]
    #assert res is not None

    #ajout fg: pour retourner le bon kernel...
    res = kernel_type.value(gpu_mode=gpu_mode, cuda_type=cuda_type, *args, **kwargs)
    
    return res #KeopsKernelObject
