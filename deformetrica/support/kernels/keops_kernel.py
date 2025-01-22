import torch
import numpy as np

from ...support.kernels import AbstractKernel
from ...core import default, GpuMode
from pykeops.torch import Genred
import numpy as np

import logging
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")


class KeopsKernel(AbstractKernel):
    def __init__(self, gpu_mode=default.gpu_mode, kernel_width=None, cuda_type=None, **kwargs):
        super().__init__('keops', gpu_mode, kernel_width)
        
        if cuda_type is None:
            cuda_type = "float32"

        self.cuda_type = cuda_type

        #ajout fg
        if not isinstance(self.kernel_width, np.ndarray):
            self.gamma = 1. / default.tensor_scalar_type([self.kernel_width ** 2])

            self.gaussian_convolve = []
            self.gaussian_convolve_weighted = []
            self.point_cloud_convolve = []
            self.varifold_convolve = []
            self.gaussian_convolve_gradient_x = []
            
            #Genred: creates a new generic operation
            for dimension in [2, 3]:
                
                self.gaussian_convolve.append(Genred(
                    "Exp(-G*SqDist(X,Y)) * P",
                    ["G = Pm(1)", #1st input: no indexation, the input tensor is a vector of dim 1 and not a 2d array.
                    "X = Vi(" + str(dimension) + ")",  # 2nd input: dimension (2/3) vector per line (i means axis 0)
                    "Y = Vj(" + str(dimension) + ")", # 3rd input: dimension (2/3) vector per column (j means axis 1)
                    "P = Vj(" + str(dimension) + ")"],
                    reduction_op='Sum', axis=1))

                    # output size = i Vi x d
                    # We sum of the j (=axis 1)
                
                # square = ² 
                # (a|b) : scalar product between vectors
                
                #softmax = EXP(d(x,y)) / SUM (EXP(d(x,y)))
                self.gaussian_convolve_weighted.append(Genred(
                    "-G * SqDist(X, Y)", 
                    ["G = Pm(1)", "X = Vi("+str(dimension)+")",  
                    "Y = Vj("+str(dimension)+")", "P = Vj("+str(dimension)+")"],
                    reduction_op='SumSoftMaxWeight', axis=1, formula2 = "P"))
                
                self.point_cloud_convolve.append(Genred(
                    "Exp(-G*SqDist(X,Y)) * P",
                    ["G = Pm(1)",
                    "X = Vi(" + str(dimension) + ")",
                    "Y = Vj(" + str(dimension) + ")",
                    "P = Vj(1)"],
                    reduction_op='Sum', axis=1))

                 # Error in this formula!
                # WeightedSqDist takes 2 arguments not 3...

                # self.varifold_convolve.append(Genred(
                #     "Exp(-(WeightedSqDist(G, X, Y))) * Square((Nx|Ny)) * P",
                #     ["G = Pm(1)",
                #     "X = Vi(" + str(dimension) + ")",
                #     "Y = Vj(" + str(dimension) + ")",
                #     "Nx = Vi(" + str(dimension) + ")",
                #     "Ny = Vj(" + str(dimension) + ")",
                #     "P = Vj(1)"],
                #     reduction_op='Sum', axis=1))

                self.varifold_convolve.append(Genred(
                    "Exp(-G*SqDist(X, Y)) * Square((Nx|Ny)) * P",
                    ["G = Pm(1)",
                    "X = Vi(" + str(dimension) + ")",
                    "Y = Vj(" + str(dimension) + ")",
                    "Nx = Vi(" + str(dimension) + ")",
                    "Ny = Vj(" + str(dimension) + ")",
                    "P = Vj(1)"],
                    reduction_op='Sum', axis=1))

                self.gaussian_convolve_gradient_x.append(Genred(
                    "(Px|Py) * Exp(-G*SqDist(X,Y)) * (X-Y)",
                    ["G = Pm(1)",
                    "X = Vi(" + str(dimension) + ")",
                    "Y = Vj(" + str(dimension) + ")",
                    "Px = Vi(" + str(dimension) + ")",
                    "Py = Vj(" + str(dimension) + ")"],
                    reduction_op='Sum', axis=1))

    def __eq__(self, other):
        return AbstractKernel.__eq__(self, other) and self.cuda_type == other.cuda_type

    def convolve(self, x, y, p, mode='gaussian'):
        if mode == 'gaussian':
            assert isinstance(x, torch.Tensor), 'x variable must be a torch Tensor'
            assert isinstance(y, torch.Tensor), 'y variable must be a torch Tensor'
            assert isinstance(p, torch.Tensor), 'p variable must be a torch Tensor'

            # move tensors with respect to gpu_mode
            x, y, p = (self._move_to_device(t, gpu_mode=self.gpu_mode) for t in [x, y, p])
            assert x.device == y.device == p.device, 'tensors must be on the same device. x.device=' + str(x.device) \
                                                     + ', y.device=' + str(y.device) + ', p.device=' + str(p.device)

            d = x.size(1)
            gamma = self.gamma.to(x.device, dtype=x.dtype)

            device_id = x.device.index if x.device.index is not None else -1
            res = self.gaussian_convolve[d - 2](gamma, x.contiguous(), y.contiguous(), p.contiguous(), device_id=device_id)

            return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res
            
            #x: nb points controle x dim, y : nb points ctrl x dim -> les positions des pt ctl
            #ou bien x 156 000*3 et y tjrs les pts controles (x position des pixels)
        elif mode == 'gaussian_weighted':
            assert isinstance(x, torch.Tensor), 'x variable must be a torch Tensor'
            assert isinstance(y, torch.Tensor), 'y variable must be a torch Tensor'
            assert isinstance(p, torch.Tensor), 'p variable must be a torch Tensor'

            x, y, p = (self._move_to_device(t, gpu_mode=self.gpu_mode) for t in [x, y, p])
            assert x.device == y.device == p.device, 'tensors must be on the same device. x.device=' + str(x.device) \
                                                     + ', y.device=' + str(y.device) + ', p.device=' + str(p.device)

            d = x.size(1)
            gamma = self.gamma.to(x.device, dtype=x.dtype)

            device_id = x.device.index if x.device.index is not None else -1
            res = self.gaussian_convolve_weighted[d - 2](gamma, x.contiguous(), y.contiguous(), p.contiguous(), device_id=device_id)
            return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res

        elif mode == 'pointcloud':
            assert isinstance(x, torch.Tensor), 'x variable must be a torch Tensor'
            assert isinstance(y, torch.Tensor), 'y variable must be a torch Tensor'
            assert isinstance(p, torch.Tensor), 'p variable must be a torch Tensor'

            # move tensors with respect to gpu_mode
            x, y, p = (self._move_to_device(t, gpu_mode=self.gpu_mode) for t in [x, y, p])
            assert x.device == y.device == p.device, 'tensors must be on the same device. x.device=' + str(x.device) \
                                                     + ', y.device=' + str(y.device) + ', p.device=' + str(p.device)

            d = x.size(1)
            gamma = self.gamma.to(x.device, dtype=x.dtype)

            device_id = x.device.index if x.device.index is not None else -1
            res = self.point_cloud_convolve[d - 2](gamma, x.contiguous(), y.contiguous(), p.contiguous(), device_id=device_id)
            return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res

        elif mode == 'varifold':
            assert isinstance(x, tuple), 'x must be a tuple'
            assert len(x) == 2, 'tuple length must be 2'
            assert isinstance(y, tuple), 'y must be a tuple'
            assert len(y) == 2, 'tuple length must be 2'

            # tuples are immutable, mutability is needed to mode to device
            x = list(x)
            y = list(y)

            # move tensors with respect to gpu_mode
            x[0], x[1], y[0], y[1], p = (self._move_to_device(t, gpu_mode=self.gpu_mode) for t in [x[0], x[1], y[0], y[1], p])
            assert x[0].device == y[0].device == p.device, 'x, y and p must be on the same device'
            assert x[1].device == y[1].device == p.device, 'x, y and p must be on the same device'

            x, nx = x
            y, ny = y
            d = x.size(1)
            gamma = self.gamma.to(x.device, dtype=x.dtype)
            # print(d)
            # print(gamma.size())
            # print(x.size())
            # print(y.size())
            # print(nx.size())
            # print(ny.size())
            # print(p.size())

            # contiguous makes a copy of the tensors
            device_id = x.device.index if x.device.index is not None else -1
            res = self.varifold_convolve[d - 2](gamma, x.contiguous(), y.contiguous(), 
                                              nx.contiguous(), ny.contiguous(), p.contiguous(), device_id=device_id)            
            #print("res", res)
            return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res

        else:
            raise RuntimeError('Unknown kernel mode.')

    def convolve_gradient(self, px, x, y=None, py=None, mode='gaussian'):
        
        
        if y is None:
            y = x
        if py is None:
            py = px

        assert isinstance(px, torch.Tensor), 'px variable must be a torch Tensor'
        assert isinstance(x, torch.Tensor), 'x variable must be a torch Tensor'
        assert isinstance(y, torch.Tensor), 'y variable must be a torch Tensor'
        assert isinstance(py, torch.Tensor), 'py variable must be a torch Tensor'

        # move tensors with respect to gpu_mode
        x, px, y, py = (self._move_to_device(t, gpu_mode=self.gpu_mode) for t in [x, px, y, py])
        assert px.device == x.device == y.device == py.device, 'tensors must be on the same device'

        d = x.size(1)
        gamma = self.gamma.to(x.device, dtype=x.dtype)

        device_id = x.device.index if x.device.index is not None else -1
        
        res = (-2 * gamma * self.gaussian_convolve_gradient_x[d - 2](gamma, x, y, px, py, device_id=device_id))
        
        return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res
    
    def update_width(self, new_width):
        """
        Ajout fg
        """
        self.kernel_width = new_width
        self.gamma = 1. / default.tensor_scalar_type([new_width ** 2])

