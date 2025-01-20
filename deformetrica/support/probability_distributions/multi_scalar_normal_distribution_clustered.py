from math import sqrt

import numpy as np

import torch
from torch.autograd import Variable

from ...support import utilities
from ...support.utilities.general_settings import Settings


class MultiScalarNormalDistributionClustered:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, variance=None, std=None):
        self.mean = None
        self.num_iter = -1
        self.classes = None
        # self.variance_sqrt = None
        # self.variance_inverse = None

        if variance is not None:
            self.set_variance(variance)
        elif std is not None:
            self.set_variance_sqrt(std)


    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def get_mean(self):
        return self.mean

    def set_mean(self, m):
        self.mean = m

    def get_variance_sqrt(self):
        return self.variance_sqrt

    def set_variance_sqrt(self, std):
        self.variance_sqrt = std
        self.variance_inverse = 1.0 / std ** 2

    def set_variance(self, var):
        self.variance_sqrt = sqrt(var)
        self.variance_inverse = 1.0 / var

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def compute_log_likelihood(self, observation):
        """
        Fully numpy method.
        Returns only the part that includes the observation argument.
        """
        self.num_iter += 1
        subject = int(self.num_iter/2)
        mean = np.array([self.mean[int(c)] for c in self.classes[subject]])
        assert mean.size == 1 or mean.shape == observation.shape
        delta = observation.ravel() - mean.ravel()
        return - 0.5 * self.variance_inverse * np.sum(delta ** 2)

