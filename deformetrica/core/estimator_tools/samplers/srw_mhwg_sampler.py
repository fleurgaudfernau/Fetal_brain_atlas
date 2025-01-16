import logging
logger = logging.getLogger(__name__)

import math

import numpy as np

from ....core import default


class SrwMhwgSampler:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self,
                 individual_proposal_distributions=default.individual_proposal_distributions,
                 acceptance_rates_target=30.0):

        # Dictionary of probability distributions.
        self.population_proposal_distributions = {}
        self.individual_proposal_distributions = individual_proposal_distributions

        self.acceptance_rates_target = acceptance_rates_target  # Percentage.

        #ajout fg
        self.accepted_realisations = {}
        for random_effect_name, _ in self.individual_proposal_distributions.items():
            self.accepted_realisations[random_effect_name] = []

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def sample(self, statistical_model, dataset, population_RER, individual_RER, current_model_terms=None):
        print("\nSampling...")
        # Initialization -----------------------------------------------------------------------------------------------

        # Initialization of the memory of the current model terms.
        # The contribution of each subject is stored independently.
        if current_model_terms is None:
            current_model_terms = self._compute_model_log_likelihood(statistical_model, dataset,
                                                                     population_RER, individual_RER)

        # Acceptance rate metrics initialization.
        acceptance_rates = {key: 0.0 for key in self.individual_proposal_distributions.keys()}

        # Main loop ----------------------------------------------------------------------------------------------------
        for random_effect_name, proposal_RED in self.individual_proposal_distributions.items():
            
            if not statistical_model.is_frozen[random_effect_name]: #ajout fg

                print("\n Random_effect:", random_effect_name)

                # RED: random effect distribution. eg multiscalarnormaldistribution
                model_RED = statistical_model.individual_random_effects[random_effect_name]

                # Initialize subject lists.
                current_regularity_terms = []
                candidate_regularity_terms = []
                current_RER = []
                candidate_RER = []

                # Shape parameters of the current random effect realization.
                shape_parameters = individual_RER[random_effect_name][0].shape

                # Simulate the random variable for each subject
                for i in range(dataset.number_of_subjects):
                    # Evaluate the current part.
                    # LL of the previous random effect realization
                    current_regularity_terms.append(model_RED.compute_log_likelihood(individual_RER[random_effect_name][i]))
                    current_RER.append(individual_RER[random_effect_name][i].flatten())

                    # Draw the candidate.
                    proposal_RED.mean = current_RER[i] #proposal random effect distribution of the RE
                    candidate_RER.append(proposal_RED.sample()) #sample a candidate

                    # Evaluate the candidate part (LL of this RE realization)
                    individual_RER[random_effect_name][i] = candidate_RER[i].reshape(shape_parameters)
                    candidate_regularity_terms.append(model_RED.compute_log_likelihood(candidate_RER[i]))

                # Evaluate the candidate terms for all subjects at once, since all contributions are independent.
                candidate_model_terms = self._compute_model_log_likelihood(
                    statistical_model, dataset, population_RER, individual_RER, modified_individual_RER=random_effect_name)
                
                print("candidate_RER for sub 1", candidate_RER[0])
                print("candidate_RER for sub 2", candidate_RER[1])

                for i in range(dataset.number_of_subjects):

                    # Acceptance rate of the candidate (Log -> / )
                    tau = candidate_model_terms[i] + candidate_regularity_terms[i] \
                        - current_model_terms[i] - current_regularity_terms[i]

                    # Reject.
                    if math.log(np.random.uniform()) > tau or math.isnan(tau):
                        individual_RER[random_effect_name][i] = current_RER[i].reshape(shape_parameters)
                        if i == 0:
                            self.accepted_realisations[random_effect_name].append(current_RER[i])

                    # Accept.
                    else:  #already added to individual_RER - see above
                        current_model_terms[i] = candidate_model_terms[i] #update new model term of LL
                        current_regularity_terms[i] = candidate_regularity_terms[i] #regularity term of LL
                        acceptance_rates[random_effect_name] += 1.0

                        if i in [0,1]:
                            print("Candidate accepted for sub", i+1)

                        if i == 0:
                            self.accepted_realisations[random_effect_name].append(candidate_RER[i])

                # Acceptance rate final scaling for the considered random effect.
                acceptance_rates[random_effect_name] *= 100.0 / float(dataset.number_of_subjects)

        return acceptance_rates, current_model_terms

    ####################################################################################################################
    ### Auxiliary methods:
    ####################################################################################################################

    def _compute_model_log_likelihood(self, statistical_model, dataset, population_RER, individual_RER,
                                      modified_individual_RER='all'):
        try:
            return statistical_model.compute_log_likelihood(
                dataset, population_RER, individual_RER, mode='model', modified_individual_RER=modified_individual_RER)

        except ValueError as error:
            logger.info('>> ' + str(error) + ' \t[ in srw_mhwg_sampler ]')
            statistical_model.clear_memory()
            return np.zeros((dataset.number_of_subjects,)) - float('inf')

    def adapt_proposal_distributions(self, statistical_model, current_acceptance_rates_in_window, iteration_number, verbose):
        goal = self.acceptance_rates_target #30%
        msg = '>> Proposal std re-evaluated from:\n'

        for random_effect_name, proposal_distribution in self.individual_proposal_distributions.items():
            if not statistical_model.is_frozen[random_effect_name]: #ajout fg
                ar = current_acceptance_rates_in_window[random_effect_name]
                std = proposal_distribution.get_variance_sqrt()
                msg += '\t\t %.3f ' % std

                if ar > self.acceptance_rates_target:
                    #we raise std
                    std *= 1 + (ar - goal) / ((100 - goal) * math.sqrt(iteration_number + 1))
                else:
                    #we diminish std
                    std *= 1 - (goal - ar) / (goal * math.sqrt(iteration_number + 1))

                msg += '\tto\t%.3f \t[ %s ]\n' % (std, random_effect_name)
                proposal_distribution.set_variance_sqrt(std)

        if verbose > 0: logger.info(msg[:-1])

    ####################################################################################################################
    ### Pickle dump methods.
    ####################################################################################################################

    def get_proposal_standard_deviations(self):
        out = {}
        for random_effect_name, proposal_distribution in self.individual_proposal_distributions.items():
            out[random_effect_name] = proposal_distribution.get_variance_sqrt()
        return out

    def set_proposal_standard_deviations(self, stds):
        for random_effect_name, std in stds.items():
            self.individual_proposal_distributions[random_effect_name].set_variance_sqrt(std)