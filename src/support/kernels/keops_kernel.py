import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../')

import numpy as np
import torch
from torch.autograd import Variable, grad

from pydeformetrica.libs.libkp.python.bindings.torch.kernels import Kernel, kernel_product
from pydeformetrica.src.support.utilities.general_settings import Settings


class KeopsKernel:
    def __init__(self):
        self.kernel_type = 'keops'
        self.kernel_width = None

    def convolve(self, x, y, p, mode='gaussian(x,y)'):

        assert self.kernel_width != None, "pykp kernel width not initialized"

        kw = Variable(torch.from_numpy(np.array([self.kernel_width])).type(Settings().tensor_scalar_type),
                      requires_grad=False)

        params = {
            'id': Kernel(mode),
            'gamma': 1. / kw ** 2 if mode == 'gaussian(x,y)' else (1. / kw ** 2, 1. / kw ** 2),
            'backend': 'auto'
        }

        return kernel_product(x, y, p, params).type(Settings().tensor_scalar_type)

    def convolve_gradient(self, px, x, y=None, py=None, mode='gaussian(x,y)'):

        factor = 1.0
        if y is None:
            y = x
            factor = 0.5
        if py is None:
            py = px

        kw = Variable(torch.from_numpy(np.array([self.kernel_width])).type(Settings().tensor_scalar_type),
                      requires_grad=True)

        params = {
            'id': Kernel(mode),
            'gamma': 1. / kw ** 2,
            'backend': 'auto'
        }

        px_xKy_py = torch.dot(px.view(-1),
                              kernel_product(x, y, py, params).type(Settings().tensor_scalar_type).view(-1))

        return factor * grad(px_xKy_py, [x], create_graph=True)[0]

    def get_kernel_matrix(self, x, y=None):
        """
        returns the kernel matrix, A_{ij} = exp(-|x_i-x_j|^2/sigma^2)
        """
        if y is None: y = x
        assert (x.size()[0] == y.size()[0])
        sq = self._squared_distances(x, y)
        return torch.exp(-sq / (self.kernel_width ** 2))

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _squared_distances(self, x, y):
        """
        Returns the matrix of $(x_i - y_j)^2$.
        Output is of size (1, M, N).
        """
        return torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, 2)
