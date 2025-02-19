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

    def __init__(self, subject_ids, times=None, deformable_objects=None):

        self.subject_ids = subject_ids
        self.times = times
        self.tmin = math.trunc(min(min(t) for t in times)) # arrondi à entier inf
        self.tmax = math.ceil(max(max(t) for t in times)) # entier sup
        self.deformable_objects = deformable_objects
        self.dimension = deformable_objects[0][0].dimension

        self.number_of_subjects = len(subject_ids)

        # Total number of observations.
        if times is not None:
            self.total_number_of_observations = sum(len(times) for times in self.times)

        elif deformable_objects is not None:
            self.total_number_of_observations = sum(len(objets) for objets in self.deformable_objects)

        # Order the observations.
        if times is not None and len(times) > 0 and len(times[0]) > 0 and deformable_objects is not None:
            self.order_observations()

    ################################################################################
    ### Public methods:
    ################################################################################

    def is_cross_sectional(self):
        """
        Checks whether there is a single visit per subject
        """
        b = True
        for elt in self.deformable_objects: b = (b and len(elt) == 1)
        return b

    def is_time_series(self):
        """
        Checks whether there is only one subject, with several visits
        """
        return len(self.deformable_objects) == 1 and len(self.deformable_objects[0]) > 1 and \
               len(self.times) == 1 and len(self.deformable_objects[0]) == len(self.times[0])

    def check_image_shapes(self):
        """
        In the case of non deformable objects, checks the dimension of the images are the same.
        """
        shape = self.deformable_objects[0][0].get_points().shape
        for subj in self.deformable_objects:
            for img in subj:
                assert img.get_points().shape == shape, "Different images dimensions detected."

    def order_observations(self):
        """ sort the visits for each individual, by time"""
        for i in range(len(self.times)):
            
            self.times[i] = np.sort(self.times[i])
            self.deformable_objects[i] = [self.deformable_objects[i][j] \
                                            for j in np.argsort(self.times[i])]
