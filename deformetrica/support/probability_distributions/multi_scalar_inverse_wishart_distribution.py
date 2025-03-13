import math

import numpy as np


class MultiScalarInverseWishartDistribution:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.dof = []
        self.scale_scalars = []

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def sample(self):
        raise RuntimeError(
            'The "sample" method is not implemented yet for the multi scalar inverse Wishart distribution.')
        pass

    def compute_log_likelihood(self, observations):
        """
        The input is a 1D array containing scalar variances, or a simple scalar.
        """
        assert len(self.dof) == len(self.scale_scalars)

        if not isinstance(observations, np.ndarray):
            return - 0.5 * self.dof[0] * (math.log(observations) + self.scale_scalars[0] / observations)

        else:
            assert len(self.scale_scalars) == observations.shape[0]
            out = 0.0
            for k in range(observations.shape[0]):
                out -= 0.5 * self.dof[k] * (
                    math.log(observations[k]) + self.scale_scalars[k] / observations[k])
            return out
