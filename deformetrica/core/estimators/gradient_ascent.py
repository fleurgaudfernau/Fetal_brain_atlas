import _pickle as pickle
import copy
from time import perf_counter
import logging
import math
import warnings
from decimal import Decimal
from ...core.estimator_tools.multiscale_functions import Multiscale
from ...core.estimator_tools.residuals_functions import Residuals
import numpy as np
import os

from ...core import default
from ...core.estimators.abstract_estimator import AbstractEstimator

logger = logging.getLogger(__name__)


class GradientAscent(AbstractEstimator):
    """
    GradientAscent object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, statistical_model, dataset, optimization_method='undefined', individual_RER={},
                 optimized_log_likelihood=default.optimized_log_likelihood,
                 max_iterations=default.max_iterations, 
                 convergence_tolerance=default.convergence_tolerance,
                 print_every_n_iters=default.print_every_n_iters, 
                 save_every_n_iters=default.save_every_n_iters,
                 initial_step_size=default.initial_step_size,
                 max_line_search_iterations=default.max_line_search_iterations,
                 line_search_shrink=default.line_search_shrink,
                 line_search_expand=default.line_search_expand,
                 output_dir=default.output_dir, 
                 load_state_file=default.load_state_file, 
                 state_file=default.state_file,

                 multiscale_momenta = default.multiscale_momenta, #ajout fg
                 multiscale_objects = default.multiscale_objects, #ajout fg
                 multiscale_strategy = default.multiscale_strategy,
                 overwrite = True,

                 **kwargs):
        
        self.overwrite = overwrite 
        
        super().__init__(statistical_model=statistical_model, dataset=dataset, name='GradientAscent',
                         optimized_log_likelihood=optimized_log_likelihood,
                         max_iterations=max_iterations, convergence_tolerance=convergence_tolerance,
                         print_every_n_iters=print_every_n_iters, save_every_n_iters=save_every_n_iters,
                         individual_RER=individual_RER,
                        state_file=state_file, output_dir=output_dir)
        
        assert optimization_method.lower() == self.name.lower()
        
        self.current_attachment = None
        self.current_regularity = None
        self.current_log_likelihood = None

        self.step = None
        self.line_search_shrink = line_search_shrink
        self.line_search_expand = line_search_expand

        self.initial_step_size = initial_step_size
        self.max_line_search_iterations = max_line_search_iterations
        self.current_iteration = 0

        self.multiscale = Multiscale(multiscale_momenta, multiscale_objects, 
                                    multiscale_strategy, self.statistical_model, self.initial_step_size, 
                                    self.output_dir, self.dataset)
        
        if load_state_file:
            self.current_parameters, self.current_iteration, object_scale, momenta_scale, \
            iter_objects, iter_momenta, order = self._load_state_file()
            
            if multiscale_objects:
                self.multiscale.object_scale = object_scale
                self.multiscale.iter_objects = iter_objects
            if multiscale_momenta:
                self.multiscale.momenta_scale = momenta_scale
                self.multiscale.iter_momenta = iter_momenta
            if multiscale_momenta and multiscale_objects:
                self.multiscale.order = order

            self._set_parameters(self.current_parameters)
            logger.info("State file loaded, it was at iteration {}".format(self.current_iteration))

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def initialize(self):
        self.current_iteration = 0
        self.current_attachment = None
        self.current_regularity = None
        self.current_log_likelihood = None

        self.current_parameters = self._get_parameters()

    def update(self):

        """
        Runs the gradient ascent algorithm and updates the statistical model.
        """
        super().update()
        
        self.statistical_model.output_path(self.output_dir, self.dataset)

        # ajout fg: before stopping, need output path
        if not self.overwrite and len(os.listdir(self.output_dir)) > 5:
            logger.info("\nOutput directory not empty - Stopping.\n _____________________________________________\n")  
            self.stop = True
            return 
        
        logger.info("Initial step size set to {}".format(self.initial_step_size))
        logger.info("Convergence tolerance set to {}".format(self.convergence_tolerance))

        self.multiscale.initialize()

        # Initialize residuals (before filtering)
        self.residuals = Residuals(self.statistical_model, self.dataset, self.print_every_n_iters, self.output_dir)
        self.residuals.compute_residuals(self.dataset, self.current_iteration, self.individual_RER, self.multiscale)

        # Initialize filter of subjects images/meshes and template (before getting parameters)
        self.dataset, new_parameters = self.multiscale.filter(self._get_parameters(), self.current_iteration)
        self.current_parameters = new_parameters
        self._set_parameters(self.current_parameters)

        # Initialize LL
        self.current_attachment, self.current_regularity, gradient = self._evaluate_model_fit(self.current_parameters,
                                                                                              with_grad=True)
        gradient = self.multiscale.compute_gradients(gradient)

        self.current_log_likelihood = self.current_attachment + self.current_regularity
        self.print()

        initial_log_likelihood = self.current_log_likelihood
        last_log_likelihood = initial_log_likelihood

        # Initialize steps
        self.step = self._initialize_step_size(gradient)
        self.step = self.multiscale.initialize_momenta_step(self.step, gradient, self, self.current_iteration)

        # Main loop ----------------------------------------------------------------------------------------------------
        while self.callback_ret and self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            t1 = perf_counter()

            # Line search ----------------------------------------------------------------------------------------------
            found_min = False
            end = False
            for li in range(self.max_line_search_iterations):

                # Print step size --------------------------------------------------------------------------------------
                if not (self.current_iteration % self.print_every_n_iters):
                    self.multiscale.info(self.step, gradient)

                # Try a simple gradient ascent step --------------------------------------------------------------------
                new_parameters = self.multiscale.gradient_ascent(self.current_parameters, gradient, self.step)

                new_attachment, new_regularity = self._evaluate_model_fit(new_parameters)
                                                
                q = new_attachment + new_regularity - last_log_likelihood

                if q > 0 or self.multiscale.no_convergence_after_ctf(self.current_iteration):
                    found_min = True
                    self.step = {key: value * self.line_search_expand for key, value in self.step.items()}
                    break
                
                # Adapting the step sizes ------------------------------------------------------------------------------
                # Step sizes reduction when the min is not found in order to go slower
                self.step = {key: value * self.line_search_shrink for key, value in self.step.items()}                

                if len(gradient) > 1:
                    new_parameters_prop = {}
                    new_attachment_prop = {}
                    new_regularity_prop = {}
                    q_prop = {}

                    # We try step shrinking for each parameter to update
                    # We keep the step shrinking for the parameter which most change the LL
                    for key in self.step.keys():
                        local_step = self.step.copy()
                        local_step[key] /= self.line_search_shrink
                        new_parameters_prop[key] = self.multiscale.gradient_ascent(self.current_parameters, gradient, local_step)
                        new_attachment_prop[key], new_regularity_prop[key] = self._evaluate_model_fit(new_parameters_prop[key])
                        q_prop[key] = new_attachment_prop[key] + new_regularity_prop[key] - last_log_likelihood
                    
                    key_max = max(q_prop.keys(), key=(lambda key: q_prop[key]))

                    if q_prop[key_max] > 0:
                        new_attachment = new_attachment_prop[key_max]
                        new_regularity = new_regularity_prop[key_max]
                        new_parameters = new_parameters_prop[key_max]
                        self.step[key_max] /= self.line_search_shrink
                        found_min = True
                        break

            # End of line search ---------------------------------------------------------------------------------------
            if not found_min:
                self._set_parameters(self.current_parameters)
                logger.info('Number of line search loops exceeded. Stopping.')
                end = True # to allow coarse to fine
                if self.multiscale.check_convergence_condition(self.current_iteration):
                    break
            
            # Update parameters
            self.current_attachment = new_attachment
            self.current_regularity = new_regularity
            self.current_log_likelihood = new_attachment + new_regularity
            self.current_parameters = new_parameters
            self._set_parameters(self.current_parameters)

            # Coarse to fine------------------------------------------------------------------------------------------
            self.residuals.compute_residuals(self.dataset, self.current_iteration, self.individual_RER, self.multiscale)
            new_parameters = self.coarse_to_fine(new_parameters, end)

            # Test the stopping criterion ------------------------------------------------------------------------------
            delta_current = last_log_likelihood - self.current_log_likelihood
            delta_initial = initial_log_likelihood - self.current_log_likelihood

            if math.fabs(delta_current) < self.convergence_tolerance * math.fabs(delta_initial):
                if self.multiscale.check_convergence_condition(self.current_iteration): 
                    logger.info('Tolerance threshold met. Stopping the optimization process.')
                    break
            
            # Printing and writing -------------------------------------------------------------------------------------
            t2 = perf_counter()
            logger.info("Time taken for iteration: {}".format(round(t2-t1, 1)))
            if not self.current_iteration % self.print_every_n_iters: self.print()
            if not self.current_iteration % self.save_every_n_iters: self.write()

            # Prepare next iteration -----------------------------------------------------------------------------------
            last_log_likelihood = self.current_log_likelihood
            if not self.current_iteration == self.max_iterations:
                gradient = self._evaluate_model_fit(self.current_parameters, with_grad=True)[2]
                gradient = self.multiscale.compute_gradients(gradient)

            # Save the state.
            if not self.current_iteration % self.save_every_n_iters: self._dump_state_file()

            # Reinitialize step sizes after coarse to fine and after gradients recomputation
            self.multiscale.reinitialize_step(self, gradient, self.current_iteration, self.step)
            
        # end of estimator loop
        self.write()

        return True

    def print(self):
        """
        Prints information.
        """
        logger.info('------------------------------------- Iteration: ' + str(self.current_iteration)
              + ' -------------------------------------')
        logger.info('>> Log-likelihood = %.3E \t [ attachment = %.3E ; regularity = %.3E ]' %
              (Decimal(str(self.current_log_likelihood)),
               Decimal(str(self.current_attachment)), Decimal(str(self.current_regularity))))

    def write(self):
        """
        Save the current results.
        """
        self.statistical_model.write(self.dataset, self.individual_RER, 
                                    self.output_dir, self.current_iteration, write_all = True)
        self.residuals.write(self.output_dir, self.dataset, self.individual_RER, self.current_iteration)
        self.residuals.plot_residuals_evolution(self.output_dir, self.multiscale, self.individual_RER)

        logger.info("\nTotal residuals diminution: {} %".format(
                    self.residuals.percentage_residuals_diminution()))
        
        self._dump_state_file()
            
        
    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################
    def coarse_to_fine(self, new_parameters, end = False):
        """
         If possible, we go down one scale
         Updated parameters may contain filtered template
        """
        self.dataset, new_parameters = self.multiscale.coarse_to_fine(new_parameters, self.dataset, self.current_iteration, 
                                                                    self.residuals.get_values("Residuals_average"), 
                                                                    end)                                        
        self.current_parameters = new_parameters
        self._set_parameters(self.current_parameters)

        return new_parameters
        
    def _initialize_step_size(self, gradient):
        """
        Initialization of the step sizes for the descent for the different variables.
        We rescale the initial sizes by the gradient squared norms.
        """
        if self.step is None or max(list(self.step.values())) < 1e-12:
            step = {}
            remaining_keys = []
            
            for key, value in gradient.items(): 
                if key != "haar_coef_momenta":
                    gradient_norm = math.sqrt(np.sum(value ** 2))
                    if gradient_norm < 1e-8:
                        remaining_keys.append(key)

                    elif math.isinf(gradient_norm):
                        step[key] = 1e-10 if math.isinf(gradient_norm) else 1.0 / gradient_norm

            if len(remaining_keys) > 0:
                for key in remaining_keys:
                    step[key] = min(list(step.values())) if len(list(step.values())) > 0 else 1e-5

            initial_step_size = 1 if self.initial_step_size is None else self.initial_step_size
            
            return {key: value * initial_step_size for key, value in step.items()}

        else:
            return self.step

    def _evaluate_model_fit(self, parameters, with_grad=False):
        # Propagates the parameter value to all necessary attributes.
        self._set_parameters(parameters)

        return self.statistical_model.compute_log_likelihood(self.dataset, self.individual_RER,
                                                                 mode=self.optimized_log_likelihood,
                                                                 with_grad=with_grad)

    def _get_parameters(self):
        out = self.statistical_model.get_fixed_effects()
        out.update(self.individual_RER)
        assert len(out) == len(self.statistical_model.get_fixed_effects()) + len(self.individual_RER)
        return out

    def _set_parameters(self, parameters):
        fixed_effects = {key: parameters[key] for key in self.statistical_model.get_fixed_effects().keys()}
        self.statistical_model.set_fixed_effects(fixed_effects)
        self.individual_RER = {key: parameters[key] for key in self.individual_RER.keys()}

    def _load_state_file(self):
        with open(self.state_file, 'rb') as f:
            d = pickle.load(f)
            return d['current_parameters'], d['current_iteration'], \
                    d["object_scale"], d["momenta_scale"],\
                    d["iter_multiscale_objects"], d["iter_multiscale_momenta"], d["order"]

    def _dump_state_file(self):
        d = {'current_parameters': self.current_parameters, 
             'current_iteration': self.current_iteration}
        d = self.multiscale.dump_state_file(d)     
        
        if self.state_file:
            with open(self.state_file, 'wb') as f:
                pickle.dump(d, f)
