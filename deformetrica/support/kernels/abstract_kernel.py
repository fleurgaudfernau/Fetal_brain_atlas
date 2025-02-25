from abc import ABC, abstractmethod
import torch
import numpy as np
from ...core import default
import logging
logger = logging.getLogger(__name__)


class AbstractKernel(ABC):
    def __init__(self, kernel_type='undefined', gpu_mode=default.gpu_mode, kernel_width=None):
        self.kernel_type = kernel_type
        self.kernel_width = kernel_width
        self.gpu_mode = gpu_mode
        logger.debug('instantiating kernel %s with kernel_width %s and gpu_mode %s. addr: %s',
                     self.kernel_type, self.kernel_width, self.gpu_mode, hex(id(self)))

    def __eq__(self, other):
        return self.kernel_type == other.kernel_type \
               and self.gpu_mode == other.gpu_mode \
               and self.kernel_width == other.kernel_width

    @staticmethod
    def hash(kernel_type, cuda_type, gpu_mode, *args, **kwargs): #kernel_width in kwargs
        #ajout fg to avoid np.ndarray unashable -> give frozenset a dict instead of a np.ndarray
        #need different hash codes for different kernels!
        if "kernel_width" in kwargs.keys() and isinstance(kwargs["kernel_width"], np.ndarray):
            return hash((kernel_type, cuda_type, gpu_mode, frozenset(args), frozenset({'kernel_width': kwargs["kernel_width"][-1][0]})))
        return hash((kernel_type, cuda_type, gpu_mode, frozenset(args), frozenset(kwargs.items())))

    def __hash__(self, **kwargs):
        return AbstractKernel.hash(self.kernel_type, None, self.gpu_mode, **kwargs)

    @abstractmethod
    def convolve(self, x, y, p, mode=None):
        raise NotImplementedError

    @abstractmethod
    def convolve_gradient(self, px, x, y=None, py=None):
        raise NotImplementedError

    def get_kernel_matrix(self, x, y=None):
        """
        returns the kernel matrix, A_{ij} = exp(-|x_i-x_j|^2/sigma^2)
        """
        if y is None:
            y = x
        assert (x.size(0) == y.size(0))
        sq = self._squared_distances(x, y)
        return torch.exp(-sq / (self.kernel_width ** 2))
    
    def get_normalized_kernel_matrix(self, x, y=None):
        """
        returns the kernel matrix, A_{ij} = exp(-|x_i-x_j|^2/sigma^2)
        """
        K = self.get_kernel_matrix(x, y)
        sum = torch.sum(K, dim = 1)

        return K/sum.unsqueeze(1)

    @staticmethod
    def _squared_distances(x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1) # n_centers x 1
        y_norm = (y ** 2).sum(1).view(1, -1) # 1 x n_centers

        #mm=matrix multiplication
        # x: n_centers x d, yt : d x n_centers
        # torch.mm output: n_centersxn_centers
        # torch.mm memory error!
        # x_norm + y_norm memory error

        if x.size()[0] > 30000:
            #out of memory error...
            dist = torch.zeros((x.size()[0], y.size()[0]), device = x_norm.device)
            n_centers = int(x.size()[0]/4)

            # norm
            repeat = x_norm.repeat([1, round(n_centers)]) #N1x1 -> N1 x n_centers
            dist[:,:round(n_centers)] += repeat
            dist[:,round(n_centers):round(n_centers)*2] += repeat
            dist[:,round(n_centers)*2:round(n_centers)*3] += repeat
            dist[:,round(n_centers)*3:] += x_norm.repeat([1, x.size()[0] - round(n_centers)*3])

            repeat = y_norm.repeat([round(n_centers), 1]) #1xN2 -> n_centersxN2
            dist[:round(n_centers), :] += repeat
            dist[round(n_centers):round(n_centers)*2, :] += repeat
            dist[round(n_centers)*2:round(n_centers)*3, :] += repeat
            dist[round(n_centers)*3:, :] += y_norm.repeat([y.size()[0] - round(n_centers)*3, 1])

            # scalar product
            cut_x, cut_x_2 = x[:round(n_centers)], x[round(n_centers):round(n_centers)*2]
            cut_x_3, cut_x_4 = x[round(n_centers)*2:round(n_centers)*3], x[round(n_centers)*3:]
            dist[:round(n_centers)] = -2.0*torch.mm(cut_x, torch.transpose(y, 0, 1))
            dist[round(n_centers):round(n_centers)*2] = -2.0*torch.mm(cut_x_2, torch.transpose(y, 0, 1))
            dist[round(n_centers)*2:round(n_centers)*3] = -2.0*torch.mm(cut_x_3, torch.transpose(y, 0, 1))
            dist[round(n_centers)*3:] = -2.0*torch.mm(cut_x_4, torch.transpose(y, 0, 1))

        else:
            dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        
        return dist

    @staticmethod
    def _move_to_device(t, gpu_mode):
        """
        AUTO, FULL, NONE, KERNEL

        gpu_mode:
        - auto: use get_best_gpu_mode(model)
        - full: tensors should already be on gpu, if not, move them and do not return to cpu
        - none: tensors should be on cpu, if not, move them
        - kernel: tensors should be on cpu. Move them to cpu and return to cpu
        """

        from ...core import GpuMode
        from ...support import utilities

        if gpu_mode in [GpuMode.FULL, GpuMode.KERNEL]:
            # tensors should already be on the target device, if not, they will be moved
            dev, _ = utilities.get_best_device(gpu_mode=gpu_mode)
            t = utilities.move_data(t, device=dev)

        elif gpu_mode is GpuMode.NONE:
            # tensors should be on cpu. If not they should be moved to cpu.
            t = utilities.move_data(t, device='cpu')

        return t
