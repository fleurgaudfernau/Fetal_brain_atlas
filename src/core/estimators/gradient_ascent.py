import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np

from pydeformetrica.src.core.estimators.abstract_estimator import AbstractEstimator

class GradientAscent(AbstractEstimator):

    """
    GradientAscent object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ################################################################################
    ### Constructor:
    ################################################################################

    def __init__(self):
        self.InitialStepSize = None
        self.MaxLineSearchIterations = None

        self.LineSearchShrink = None
        self.LineSearchExpand = None
        self.ConvergenceTolerance = None

        self.LogLikelihoodHistory = []


    ################################################################################
    ### Public methods:
    ################################################################################

    # Runs the gradient ascent algorithm and updates the statistical model.
    def Update(self):

        # Declare variables --------------------------------------------------------
        newFixedEffects = None
        newPopRER = None
        newIndRER = None

        # Initialisation -----------------------------------------------------------
        logLikelihoodTerms = None
        self.StatisticalModel.UpdateFixedEffectsAndComputeCompleteLogLikelihood(
            self.Dataset, self.PopulationRER, self.IndividualRER, logLikelihoodTerms)
        self.LogLikelihoodHistory.append(logLikelihoodTerms)

        lsqRef = self.LogLikelihoodHistory[0].sum()
        fixedEffects = self.StatisticalModel.FixedEffects
        self.Print()

        popGrad, indGrad = self.StatisticalModel.ComputeCompleteLogLikelihoodGradient(
            self.Dataset, self.PopulationRER, self.IndividualRER)

        nbParams = len(popGrad) + len(indGrad)
        step = np.ones((nbParams, 1)) * self.InitialStepSize

        # Main loop ----------------------------------------------------------------
        for iter in range(1, self.MaxIterations + 1):
            self.CurrentIteration = iter

            foundMin = False
            for li in range(self.MaxLineSearchIterations):

                # Print step size --------------------------------------------------
                if not(iter % self.PrintEveryNIters):
                    k = 0
                    print('Step size = ')
                    for dict in [popGrad, indGrad]:
                        for key in dict.keys():
                            print('\t ' + str(step[k]) + ' [' + key + ']')
                            k += 1

                # Try a simple gradient ascent step --------------------------------
                self.GradientAscentStep(fixedEffects, popGrad, indGrad, )


    def Print(self):
        # TODO.
        print('GradientAscent::Print')

    ################################################################################
    ### Private methods:
    ################################################################################