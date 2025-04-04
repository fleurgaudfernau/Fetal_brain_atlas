import numpy as np
import math
import logging
logger = logging.getLogger(__name__)


class LongitudinalDataset:
    """
    A longitudinal data set is a collection of sets of deformable objects
    for a series of subjects at multiple time-points.

    """

    ################################################################################
    ### Constructor:
    ################################################################################

    def __init__(self, ids, times=None, objects=None):

        self.ids = ids
        self.times = times
        self.tmin, self.tmax = None, None            
        self.objects = objects # list of lists of deformable multi objects
        self.dimension = objects[0][0].dimension
        self.type = objects[0][0].object_list[0].type.lower()

        self.n_subjects = len(ids)
        self.n_obs = len(sum(self.objects, []))

        # Total number of observations.
        if times and times != [[]]:
            self.total_number_of_observations = sum(len(times) for times in self.times)
            self.tmin = math.trunc(min(min(t) for t in times)) # arrondi à entier inf
            self.tmax = math.ceil(max(max(t) for t in times)) # entier sup

        elif objects is not None:
            self.total_number_of_observations = sum(len(objets) for objets in self.objects)

        # Order the observations.
        if times is not None and len(times) > 0 and len(times[0]) > 0 and objects is not None:
            self.order_observations()

    ################################################################################
    ### Public methods:
    ################################################################################

    def is_cross_sectional(self):
        """
        Checks whether there is a single visit per subject
        """
        b = True
        for elt in self.objects: b = (b and len(elt) == 1)
        return b

    def is_time_series(self):
        """
        Checks whether there is only one subject, with several visits
        """
        return len(self.objects) == 1 and len(self.objects[0]) > 1 and \
               len(self.times) == 1 and len(self.objects[0]) == len(self.times[0])

    def check_image_shapes(self):
        """
        In the case of non deformable objects, checks the dimension of the images are the same.
        """
        shape = self.objects[0][0].get_points().shape
        for subj in self.objects:
            for img in subj:
                assert img.get_points().shape == shape, "Different images dimensions detected."

    def order_observations(self):
        """ sort the visits for each individual, by time"""
        for i in range(len(self.times)):
            
            self.times[i] = np.sort(self.times[i])
            self.objects[i] = [self.objects[i][j] \
                                            for j in np.argsort(self.times[i])]
