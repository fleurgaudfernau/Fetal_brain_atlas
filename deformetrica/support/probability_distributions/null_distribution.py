from math import sqrt

import numpy as np

import torch

from ...support import utilities


class NullDistribution:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, variance=None, std=None):
        self.mean = 0
        self.variance = 0

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
        self.variance_sqrt = 0
        self.variance_inverse = 0

    def set_variance(self, var):
        self.variance_sqrt = 0
        self.variance_inverse = 0

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def sample(self):
        return None 
        #self.mean + self.variance_sqrt * np.random.standard_normal(self.mean.shape)

    def compute_log_likelihood(self, observation):
        """
        Fully numpy method.
        Returns only the part that includes the observation argument.
        """
        assert self.mean.size == 1 or self.mean.shape == observation.shape
        delta = observation.ravel() - self.mean.ravel()
        return - 0.5 * self.variance_inverse * np.sum(delta ** 2)

    def compute_log_likelihood_torch(self, observation, device='cpu'):
        """
        Fully torch method.
        Returns only the part that includes the observation argument.
        """
        mean = utilities.move_data(self.mean, requires_grad=False, device=device)
        observation = utilities.move_data(observation, device=device)
        assert mean.detach().cpu().numpy().size == observation.detach().cpu().numpy().size, \
            'mean.detach().cpu().numpy().size = %d, \t observation.detach().cpu().numpy().size = %d' \
            % (mean.detach().cpu().numpy().size, observation.detach().cpu().numpy().size)

        delta = observation.contiguous().view(-1, 1) - mean.contiguous().view(-1, 1)
        return -0.5 * torch.sum(delta ** 2) * self.variance_inverse
