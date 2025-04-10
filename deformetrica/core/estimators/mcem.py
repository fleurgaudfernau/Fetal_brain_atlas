import logging
import os.path as op
import _pickle as pickle

from ...core import default
from ...core.estimator_tools.samplers.srw_mhwg_sampler import SrwMhwgSampler
from ...core.estimators.abstract_estimator import AbstractEstimator
from ...core.estimators.gradient_ascent import GradientAscent
from ...in_out.array_readers_and_writers import *
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class MCEM(AbstractEstimator):
    """
    GradientAscent object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, statistical_model, dataset, optimization_method='undefined', 
                individual_RER={}, individual_proposal_distributions = {},
                 max_iterations=default.max_iterations,
                 print_every_n_iters=default.print_every_n_iters, 
                 save_every_n_iters=default.save_every_n_iters,
                 sample_every_n_mcmc_iters=default.sample_every_n_mcmc_iters,
                 convergence_tolerance=default.convergence_tolerance,
                 output_dir=default.output_dir,
                initial_step_size=default.initial_step_size,
                 load_state_file=default.load_state_file, state_file=default.state_file,

                 multiscale_momenta = default.multiscale_momenta, #ajout fg
                 naive = default.naive, multiscale_images = default.multiscale_images, #ajout fg
                 multiscale_strategy = default.multiscale_strategy, gamma = default.gamma,
                 **kwargs):

        super().__init__(statistical_model=statistical_model, dataset=dataset, name='MCEM',
                         max_iterations=max_iterations,
                         convergence_tolerance=convergence_tolerance,
                         print_every_n_iters=print_every_n_iters, save_every_n_iters=save_every_n_iters,
                         individual_RER=individual_RER,
                         state_file=state_file, output_dir=output_dir)

        assert optimization_method.lower() == self.name.lower()

        self.sampler = SrwMhwgSampler(individual_proposal_distributions)
        self.current_mcmc_iteration = 0
        self.sample_every_n_mcmc_iters = sample_every_n_mcmc_iters #10 fois
        self.number_of_burn_in_iterations = None  # Number of iterations without memory.
        self.memory_window_size = 1  # Size of the averaging window for the acceptance rates.

        self.number_of_trajectory_points = min(self.max_iterations, 500)
        self.save_model_parameters_every_n_iters = max(1, int(self.max_iterations / float(self.number_of_trajectory_points)))

        #ajouts fg
        self.errors_rates = []

        # Initialization of the gradient-based optimizer.
        self.gradient_based_estimator = GradientAscent(
            statistical_model, dataset, optimized_log_likelihood='class2', max_iterations=5, 
            convergence_tolerance=convergence_tolerance, print_every_n_iters=1, save_every_n_iters=100000, 
            initial_step_size=initial_step_size, output_dir=output_dir, 
            individual_RER=individual_RER, optimization_method='GradientAscent',
            multiscale_momenta = multiscale_momenta, #ajout fg
            multiscale_images = multiscale_images)

        self._initialize_number_of_burn_in_iterations()

        if load_state_file:
            (self.current_iteration, parameters, proposal_stds,
             self.current_acceptance_rates, self.average_acceptance_rates,
             self.current_acceptance_rates_in_window, self.average_acceptance_rates_in_window,
             self.model_parameters_trajectory, self.individual_random_effects_samples_stack) = self._load_state_file()
            self._set_parameters(parameters)
            self.sampler.set_proposal_standard_deviations(proposal_stds)
            logger.info("State file loaded, it was at iteration %d." % self.current_iteration)

        else:
            self.current_iteration = 0
            self.current_acceptance_rates = {}  # Acceptance rates of the current iteration.
            self.average_acceptance_rates = {}  # Mean acceptance rates, computed over all past iterations.
            self.current_acceptance_rates_in_window = None  # Memory of the last memory_window_size acceptance rates.
            self.average_acceptance_rates_in_window = None  # Moving average of current_acceptance_rates_in_window.
            self.model_parameters_trajectory = None  # Memory of the model parameters along the estimation.
            self.individual_random_effects_samples_stack = None  # Stack of the last individual random effect samples.

            self._initialize_acceptance_rate_information()
            self._initialize_model_parameters_trajectory()
            self._initialize_individual_random_effects_samples_stack()

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Runs the MCMC-SAEM algorithm and updates the statistical model.
        """

        # Print initial console information.
        logger.info('-------------------------------- Iteration: {}--------------------------------'.format(self.iteration))
        logger.info('>> MCEM algorithm launched for ' + str(self.max_iterations) + ' iterations (' + str(
            self.number_of_burn_in_iterations) + ' iterations of burn-in).')

        # Initialization of the average random effects realizations.
        averaged_individual_RER = {key: np.zeros(value.shape) for key, value in self.individual_RER.items()}

        # Main loop ------------------------------------------------------------------------------------------
        while self.current_iteration < self.max_iterations:

            self.current_iteration += 1

            if self.current_iteration <= self.number_of_burn_in_iterations - 1
                step = 1.0
            else:
                step = 1.0 / (self.current_iteration - self.number_of_burn_in_iterations + 1)

            # Simulation.
            current_model_terms = None

            #print("\nSimulate the random effects")
            for n in range(self.sample_every_n_mcmc_iters):
                
                self.current_mcmc_iteration += 1

                #print("self.current_mcmc_iteration", self.current_mcmc_iteration)

                # Single iteration of the MCMC (Metropolis Hastings with Gibbs sampler)
                #return acceptance rates for all latent variables
                #the sampling modifies self.individual_RER : 1 realization/subject
                
                self.current_acceptance_rates, current_model_terms = self.sampler.sample(
                    self.statistical_model, self.dataset, self.individual_RER,
                    current_model_terms)

                ###  Adapt proposal variances.

                #update self.average_acceptance_rates and self.current_acceptance_rate_in_window
                self._update_acceptance_rate_information() 
                
                #print("self.average_acceptance_rates", self.average_acceptance_rates)
                #print("self.average_acceptance_rates_in_window", self.average_acceptance_rates_in_window)
                
                #if we are not at the end of the average window for acceptance rate
                if not (self.current_mcmc_iteration % self.memory_window_size):
                    self.average_acceptance_rates_in_window = {
                        key: np.mean(self.current_acceptance_rates_in_window[key])
                        for key in self.sampler.individual_proposal_distributions.keys()}
                    
                    #print("self.average_acceptance_rates_in_window", self.average_acceptance_rates_in_window)    
                    
                    #if average_acceptance_rates_in_window[z] > 30%: raise std_z
                    #if average_acceptance_rates_in_window[z] < 30%: diminish std_z
                    self.sampler.adapt_proposal_distributions(
                        self.statistical_model, self.average_acceptance_rates_in_window, self.current_mcmc_iteration,
                        not self.current_iteration % self.print_every_n_iters and n == self.sample_every_n_mcmc_iters - 1)
                
            current_fixed_effects = self.statistical_model.fixed_effects.copy()
            
            #self.individual_random_effects_samples_stack: sample_every_n_mcmc_iters x n_subjects for each z
            #contains all the realizations of z

            # Maximization for the class 2 fixed effects - GradientAscent: call self.gradient_based_estimator
            print("\nOptimize class 2 fixed effetcs")
            fixed_effects_before_maximization = self.statistical_model.get_fixed_effects()
            self._maximize_over_fixed_effects()

            fixed_effects_after_maximization = self.statistical_model.get_fixed_effects()
            fixed_effects = {key: value + step * (fixed_effects_after_maximization[key] - value) for key, value in
                             fixed_effects_before_maximization.items()}
            self.statistical_model.set_fixed_effects(fixed_effects)

            #print("AFTER fixed_effects['modulation_matrix']", fixed_effects['modulation_matrix'])
            #print("AFTER fixed_effects['modulation_matrix']", fixed_effects['momenta'][:10])

            # Averages the random effect realizations in the concentration phase.
            #print("Average the realization of the random effects")
            if step < 1.0:
                print("step")

                coefficient_1 = float(self.current_iteration + 1 - self.number_of_burn_in_iterations)
                coefficient_2 = (coefficient_1 - 1.0) / coefficient_1
                averaged_individual_RER = {key: value * coefficient_2 + self.individual_RER[key] / coefficient_1 for
                                           key, value in averaged_individual_RER.items()}
                
                #update self.individual_random_effects_samples_stack with self.individual_RER 
                self._update_individual_random_effects_samples_stack()
                
                print("self.individual_random_effects_samples_stack", self.individual_random_effects_samples_stack)

            else:
                averaged_individual_RER = self.individual_RER

            #print("self.individual_RER", self.individual_RER)

            # Saving, printing, writing.
            self.save_errors(current_fixed_effects)
            if not (self.current_iteration % self.save_model_parameters_every_n_iters):
                self._update_model_parameters_trajectory()
            if not (self.current_iteration % self.print_every_n_iters):
                self.print()
            if not (self.current_iteration % self.save_every_n_iters):
                self.write()
                self.write_errors()
            

        # Finalization ---------------------------------------------------------------------------------------
        self.individual_RER = averaged_individual_RER

    def print(self):
        """
        Prints information.
        """
        # Iteration number.
        logger.info('\n -------------------------------- Iteration: {}--------------------------------'.format(self.iteration))

        # Averaged acceptance rates over all the past iterations.
        logger.info('>> Average acceptance rates (all past iterations):')
        for random_effect_name, average_acceptance_rate in self.average_acceptance_rates.items():
            logger.info('\t\t %.2f \t[ %s ]' % (average_acceptance_rate, random_effect_name))

        # Let the model under optimization print information about itself.
        self.statistical_model.print(self.individual_RER)
    
    def save_errors(self, previous_fixed_effects):
        current_error_rate = self.statistical_model.compute_errors(previous_fixed_effects)
        self.errors_rates.append(current_error_rate)
    
    def write_errors(self):
        iterations = [k for k in range(self.current_iteration)]
        plt.plot(iterations, self.errors_rates)
        plt.xlabel('Iterations')
        plt.ylabel('Error rate')
        plt.ylim([0, max(self.errors_rates)])
        plt.xlim([0, max(iterations)])
        plt.savefig(self.output_dir + '/Error_rate.png')
        plt.close()

    def write(self, individual_RER=None):
        """
        Save the current results.
        """
        if individual_RER is None:
            individual_RER = self.individual_RER
        self.statistical_model.write(self.dataset, individual_RER, self.output_dir, update_fixed_effects=False)

        # Save the recorded model parameters trajectory.
        # self.model_parameters_trajectory is a list of dictionaries
        #modif fg : avoid memory error
        np.save(op.join(self.output_dir, self.statistical_model.name + '__EstimatedParameters__Trajectory.npy'),
                np.array(
                    {key: value[:(1 + int(self.current_iteration / float(self.save_model_parameters_every_n_iters)))]
                     for key, value in self.model_parameters_trajectory.items()}))

        # Save the memorized individual random effects samples.
        if self.current_iteration > self.number_of_burn_in_iterations:
            np.save(op.join(self.output_dir,
                                 self.statistical_model.name + '__EstimatedParameters__IndividualRandomEffectsSamples.npy'),
                    {key: value[:(self.current_iteration - self.number_of_burn_in_iterations)] for key, value in
                     self.individual_random_effects_samples_stack.items()})
        
         
        # Dump state file.
        #modif fg : fichier très lourd !
        self._dump_state_file()

    ####################################################################################################################
    ### Private_maximize_over_remaining_fixed_effects() method and associated utilities:
    ####################################################################################################################

    def _maximize_over_fixed_effects(self):
        """
        Update the model fixed effects for which no closed-form update is available
        """
        # Default optimizer, if not initialized in the launcher.
        # Should better be done in a dedicated initializing method. TODO.
        if self.statistical_model.has_maximization_procedure is not None \
                and self.statistical_model.has_maximization_procedure:
            self.statistical_model.maximize(self.individual_RER, self.dataset)

        else:
            self.gradient_based_estimator.initialize()

            if self.gradient_based_estimator.verbose > 0:
                logger.info('\n[ maximizing over the fixed effects with the GradientAscent optimizer ]')

            success = False
            while not success:
                try:
                    self.gradient_based_estimator.update()
                    success = True
                except RuntimeError as error:
                    logger.info('>> ' + str(error.args[0]) + ' [ in mcmc_saem ]')
                    self.statistical_model.adapt_to_error(error.args[1])

            if self.gradient_based_estimator.verbose > 0:
                logger.info('\n [ end of the gradient-based maximization ]')

    ####################################################################################################################
    ### Other private methods:
    ####################################################################################################################        

    def _initialize_number_of_burn_in_iterations(self):
        if self.number_of_burn_in_iterations is None:
            # Because some models will set it manually (e.g. deep Riemannian models)
            if self.max_iterations > 4000:
                self.number_of_burn_in_iterations = self.max_iterations - 2000
            else:
                self.number_of_burn_in_iterations = int(self.max_iterations / 2)

    def _initialize_acceptance_rate_information(self):
        # Initialize average_acceptance_rates.
        self.average_acceptance_rates = {key: 0.0 for key in self.sampler.individual_proposal_distributions.keys()}

        # Initialize current_acceptance_rates_in_window.
        self.current_acceptance_rates_in_window = {key: np.zeros((self.memory_window_size,))
                                                   for key in self.sampler.individual_proposal_distributions.keys()}
        self.average_acceptance_rates_in_window = {key: 0.0
                                                   for key in self.sampler.individual_proposal_distributions.keys()}

    def _update_acceptance_rate_information(self):
        # Update average_acceptance_rates.
        coefficient_1 = float(self.current_mcmc_iteration)
        coefficient_2 = (coefficient_1 - 1.0) / coefficient_1
        #coefficient_2 higher for highervalue of current_mcmc_iteration

        #self.average_acceptance_rates[z] = moyenne des taux d'acceptations pour VL z
        self.average_acceptance_rates = {key: value * coefficient_2 + self.current_acceptance_rates[key] / coefficient_1
                                         for key, value in self.average_acceptance_rates.items()}

        # Update current_acceptance_rates_in_window.
        for key in self.current_acceptance_rates_in_window.keys():
            self.current_acceptance_rates_in_window[key][(self.current_mcmc_iteration - 1) % self.memory_window_size] = \
                self.current_acceptance_rates[key]

    ####################################################################################################################
    ### Model parameters trajectory saving methods:
    ####################################################################################################################

    def _initialize_model_parameters_trajectory(self):
        self.model_parameters_trajectory = {}
        for (key, value) in self.statistical_model.get_fixed_effects(mode='all').items():
            if not self.statistical_model.is_frozen[key]:
                self.model_parameters_trajectory[key] = np.zeros((self.number_of_trajectory_points + 1, value.size))
                self.model_parameters_trajectory[key][0, :] = value.flatten()

    def _update_model_parameters_trajectory(self):
        for (key, value) in self.statistical_model.get_fixed_effects(mode='all').items():
            if not key in self.statistical_model.is_frozen.keys() \
                or not self.statistical_model.is_frozen[key]:
                self.model_parameters_trajectory[key][
                int(self.current_iteration / float(self.save_model_parameters_every_n_iters)), :] = value.flatten()

    def _get_vectorized_individual_RER(self):
        return np.concatenate([value.flatten() for value in self.individual_RER.values()])

    def _initialize_individual_random_effects_samples_stack(self):
        number_of_concentration_iterations = self.max_iterations - self.number_of_burn_in_iterations
        self.individual_random_effects_samples_stack = {}
        for (key, value) in self.individual_RER.items():
            if number_of_concentration_iterations > 0:
                self.individual_random_effects_samples_stack[key] = np.zeros(
                    (number_of_concentration_iterations, value.size))
                self.individual_random_effects_samples_stack[key][0, :] = value.flatten()

    def _update_individual_random_effects_samples_stack(self):
        for (key, value) in self.individual_RER.items():
            self.individual_random_effects_samples_stack[key][
            self.current_iteration - self.number_of_burn_in_iterations - 1, :] = value.flatten()

    ####################################################################################################################
    ### Pickle dump methods.
    ####################################################################################################################

    def _get_parameters(self):
        out = self.statistical_model.get_fixed_effects()
        out.update(self.individual_RER)
        assert len(out) == len(self.statistical_model.get_fixed_effects()) \
                + len(self.individual_RER)
        return out

    def _set_parameters(self, parameters):
        fixed_effects = {key: parameters[key] for key in self.statistical_model.get_fixed_effects().keys()}
        self.statistical_model.set_fixed_effects(fixed_effects)
        self.individual_RER = {key: parameters[key] for key in self.individual_RER.keys()}

    def _load_state_file(self):
        with open(self.state_file, 'rb') as f:
            d = pickle.load(f)
            return (d['current_iteration'],
                    d['current_parameters'],
                    d['current_proposal_stds'],
                    d['current_acceptance_rates'],
                    d['average_acceptance_rates'],
                    d['current_acceptance_rates_in_window'],
                    d['average_acceptance_rates_in_window'],
                    d['trajectory'],
                    d['samples'])

    def _dump_state_file(self):
        d = {
            'current_iteration': self.current_iteration,
            'current_parameters': self._get_parameters(),
            'current_proposal_stds': self.sampler.get_proposal_standard_deviations(),
            'current_acceptance_rates': self.current_acceptance_rates,
            'average_acceptance_rates': self.average_acceptance_rates,
            'current_acceptance_rates_in_window': self.current_acceptance_rates_in_window,
            'average_acceptance_rates_in_window': self.average_acceptance_rates_in_window,
            'trajectory': self.model_parameters_trajectory,
            'samples': self.individual_random_effects_samples_stack
        }
        with open(self.state_file, 'wb') as f:
            pickle.dump(d, f, protocol=4)