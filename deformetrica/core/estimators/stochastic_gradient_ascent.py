import _pickle as pickle
import copy
import logging
import torch
import math
import warnings
from decimal import Decimal
from ...core.estimator_tools.multiscale_functions import Multiscale
from ...core.estimator_tools.residuals_functions import Residuals
import numpy as np

from ...core import default
from ...core.estimators.abstract_estimator import AbstractEstimator

logger = logging.getLogger(__name__)


class StochasticGradientAscent(AbstractEstimator):
    """
    GradientAscent object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, statistical_model, dataset, optimization_method='undefined', individual_RER={},
                 optimized_log_likelihood=default.optimized_log_likelihood,
                 max_iterations=default.max_iterations, convergence_tolerance=default.convergence_tolerance,
                 print_every_n_iters=default.print_every_n_iters, save_every_n_iters=default.save_every_n_iters,
                 initial_step_size=default.initial_step_size,
                 output_dir=default.output_dir, callback=None,
                 load_state_file=default.load_state_file, state_file=default.state_file,
                 last_residuals = None, initial_residuals = None, #ajouts fg

                 multiscale_momenta = default.multiscale_momenta, #ajout fg
                 multiscale_objects = default.multiscale_objects, #ajout fg
                 multiscale_strategy = default.multiscale_strategy,
                 number_of_batches = 9,
                 overwrite = True,

                 **kwargs):

        super().__init__(statistical_model=statistical_model, dataset=dataset, name='StochasticGradientAscent',
                         optimized_log_likelihood=optimized_log_likelihood,
                         max_iterations=max_iterations, convergence_tolerance=convergence_tolerance,
                         print_every_n_iters=print_every_n_iters, save_every_n_iters=save_every_n_iters,
                         individual_RER=individual_RER,
                         callback=callback, state_file=state_file, output_dir=output_dir)

        assert optimization_method.lower() == self.name.lower()
        
        self.current_attachment = None
        self.current_regularity = None
        self.current_log_likelihood = None

        self.step = None
        self.line_search_shrink = 0.5
        self.line_search_expand = 1.5

        self.initial_step_size = initial_step_size
        self.max_line_search_iterations = 10
        self.current_iteration = 0

        # Stochastic gradient
        self.line_search_expand = 1.5
        self.number_of_batches = number_of_batches

        self.overwrite = overwrite
        
        # Multiscale
        self.multiscale = Multiscale(multiscale_momenta, multiscale_objects, multiscale_strategy,
                                    self.statistical_model, self.initial_step_size, self.output_dir, self.dataset)
        self.multiscale.initialize()

        # Initialize residuals (before filtering)
        self.residuals = Residuals(self.statistical_model, self.dataset, self.print_every_n_iters)
        self.residuals.compute_residuals(self.dataset, self.current_iteration, self.individual_RER, self.multiscale)

        # Initialize filter of subjects images and template (before getting parameters)
        self.dataset, self.current_parameters = self.multiscale.filter(self._get_parameters(), self.current_iteration)
        self._set_parameters(self.current_parameters)

        if self.statistical_model.name == "GeodesicRegression" \
            and not multiscale_momenta and not multiscale_objects:
            self.line_search_expand = 1.01
        elif self.statistical_model.name == "GeodesicRegression":
            self.line_search_expand = 1.3

        # If the load_state_file flag is active, restore context.
        if load_state_file:
            self.current_parameters, self.current_iteration, \
            object_scale, momenta_scale, iter_multiscale_objects, iter_multiscale_momenta, order \
            = self._load_state_file()
            if multiscale_objects:
                self.multiscale.object_scale = object_scale
                self.multiscale.iter_multiscale_objects = iter_multiscale_objects
            if multiscale_momenta:
                self.multiscale.momenta_scale = momenta_scale
                self.multiscale.iter_multiscale_momenta = iter_multiscale_momenta
            if multiscale_momenta and multiscale_objects:
                self.multiscale.order = order

            self._set_parameters(self.current_parameters)
            logger.info("State file loaded, it was at iteration", self.current_iteration)

        
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

        self.statistical_model.output_path(self.output_dir)

        # ajout fg
        if not self.overwrite and len(os.listdir(self.output_dir)) > 5:
            logger.info("\nOutput directory not empty - Stopping.")
            return False

        # Initialize LL
        self.current_attachment, self.current_regularity = self.compute_log_likelihood(self.current_parameters)
        self.current_log_likelihood = self.current_attachment + self.current_regularity

        # Use gradient of 1 mii batch to initialize step sizes
        batch = self.statistical_model.mini_batches(self.dataset, self.number_of_batches)[0]
        _, _, batch_gradient = self.statistical_model.compute_mini_batch_gradient(batch, self.dataset, 
                                                                                self.individual_RER, with_grad=True)
        batch_gradient = self.multiscale.compute_gradients(batch_gradient)

        # Initialize steps
        self.step = self._initialize_step_size(batch_gradient)
        self.step = self.multiscale.initialize_momenta_step(self.step, batch_gradient, self, self.current_iteration) #step[haar_coef_momenta]

        self.print()
        initial_log_likelihood = self.current_log_likelihood
        last_log_likelihood_batch = initial_log_likelihood
        last_log_likelihood = initial_log_likelihood
        nb_params = len(batch_gradient)

        # Main loop ----------------------------------------------------------------------------------------------------
        while self.callback_ret and self.current_iteration < self.max_iterations:
            self.current_iteration += 1

            # Divide dataset into mini natches
            mini_batches = self.statistical_model.mini_batches(self.dataset,  self.number_of_batches)
            
            for b, batch in enumerate(mini_batches):
                logger.info("batch {}".format(b))

                _, _, batch_gradient = self.statistical_model.compute_mini_batch_gradient(batch, self.dataset, self.individual_RER, with_grad=True)

                # the gradient of momenta[i] == 0 if subject i not in batch
                batch_gradient = self.multiscale.compute_gradients(batch_gradient)
                self.multiscale.reinitialize_step(self, batch_gradient, self.current_iteration-1, self.step)
                
                # Line search ----------------------------------------------------------------------------------------------
                found_min = False
                for li in range(self.max_line_search_iterations):

                    if not (self.current_iteration % self.print_every_n_iters):
                        self.multiscale.info(self.step, batch_gradient)

                    # Try a simple gradient ascent step --------------------------------------------------------------------                    
                    new_parameters = self.multiscale.gradient_ascent(self.current_parameters, batch_gradient, self.step)

                    new_attachment, new_regularity = self._evaluate_model_fit(new_parameters)
                                                        
                    # Adapting the step sizes ------------------------------------------------------------------------------
                    q = new_attachment + new_regularity - last_log_likelihood_batch
                    if q > 0 or self.multiscale.no_convergence_after_ctf(self.current_iteration):
                        found_min = True
                        break
                    
                    # Adapting the step sizes when the min is not found------------------------------------------------------------------------------
                    self.step = {key: value * self.line_search_shrink for key, value in self.step.items()}                

                    if nb_params > 1:
                        new_parameters_prop, new_attachment_prop, new_regularity_prop, q_prop = {}, {}, {}, {}
                        
                        for key in self.step.keys():
                            local_step = self.step.copy()
                            local_step[key] /= self.line_search_shrink
                            new_parameters_prop[key] = self.multiscale.gradient_ascent(self.current_parameters, batch_gradient, local_step)
                            new_attachment_prop[key], new_regularity_prop[key] = self._evaluate_model_fit(new_parameters_prop[key])
                            q_prop[key] = new_attachment_prop[key] + new_regularity_prop[key] - last_log_likelihood_batch
                        key_max = max(q_prop.keys(), key=(lambda key: q_prop[key]))

                        if q_prop[key_max] > 0:
                            new_attachment = new_attachment_prop[key_max]
                            new_regularity = new_regularity_prop[key_max]
                            new_parameters = new_parameters_prop[key_max]
                            self.step[key_max] /= self.line_search_shrink
                            found_min = True
                            break

                # End of line search for the batch---------------------------------------------------------------------------------------
                end = False
                if not found_min:
                    self._set_parameters(self.current_parameters)
                    logger.info('Number of line search loops exceeded. Stopping.')
                    end = True
                    break # break only the batches loop

                self.current_attachment = new_attachment
                self.current_regularity = new_regularity
                self.current_log_likelihood = new_attachment + new_regularity
                self.current_parameters = new_parameters
                self._set_parameters(self.current_parameters)
                last_log_likelihood_batch = self.current_log_likelihood
                
            # Back to main loop------------------------------------------------------------------------------------------
            if end and self.multiscale.check_convergence_condition(self.current_iteration):
                break
                
            # Test the stopping criterion ------------------------------------------------------------------------------
            delta_f_current = last_log_likelihood - self.current_log_likelihood
            delta_f_initial = initial_log_likelihood - self.current_log_likelihood

            if math.fabs(delta_f_current) < self.convergence_tolerance * math.fabs(delta_f_initial):
                if self.multiscale.check_convergence_condition(self.current_iteration): 
                    logger.info('Tolerance threshold met. Stopping the optimization process.')
                    break  

            # Prepare next iteration
            self.step = {key: value * self.line_search_expand for key, value in self.step.items()}
            last_log_likelihood = self.current_log_likelihood
            
            # Coarse to fine------------------------------------------------------------------------------------------
            self.residuals.compute_residuals(self.dataset, self.current_iteration, self.individual_RER, self.multiscale)
            new_parameters = self.coarse_to_fine(new_parameters, end)
            
            # Printing and writing -------------------------------------------------------------------------------------
            if not self.current_iteration % self.print_every_n_iters: self.print()
            if not self.current_iteration % self.save_every_n_iters: self.write()

            # Call user callback function ------------------------------------------------------------------------------
            if self.callback is not None:
                self._call_user_callback(float(self.current_log_likelihood), float(self.current_attachment),
                                         float(self.current_regularity), batch_gradient)

            # Save the state.
            if not self.current_iteration % self.save_every_n_iters: self._dump_state_file()

        # end of estimator loop
        self.write()

    def print(self):
        """
        Prints information.
        """
        logger.info('------------------------------------- Iteration: ' + str(self.current_iteration)
              + ' -------------------------------------')
        logger.info('>> Log-likelihood = %.3E \t [ attachment = %.3E ; regularity = %.3E ]' %
              (Decimal(str(self.current_log_likelihood)),
               Decimal(str(self.current_attachment)),
               Decimal(str(self.current_regularity))))

    def write(self):
        """
        Save the current results.
        """
        self.statistical_model.write(self.dataset, self.individual_RER, self.output_dir, self.current_iteration)
        self.residuals.write(self.output_dir, self.dataset)
        self.residuals.plot_residuals_evolution(self.output_dir, self.multiscale)
        self._dump_state_file()
    
    def save_model_state_after_ctf(self):
        output_dir = self.multiscale.save_model_after_ctf(self.current_iteration)
        if output_dir:
            self.statistical_model.write(self.dataset, self.individual_RER, output_dir, self.current_iteration)

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def coarse_to_fine(self, new_parameters, end = False):
        self.dataset, new_parameters = self.multiscale.coarse_to_fine(new_parameters, self.dataset, self.current_iteration, 
                                                        self.residuals.get_values("Residuals_average"), end)                    
        self.current_parameters = new_parameters
        self._set_parameters(self.current_parameters)
        self.save_model_state_after_ctf()

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
                    else:
                        step[key] = 1.0 / gradient_norm
                                        
            if len(remaining_keys) > 0:
                if len(list(step.values())) > 0:
                    default_step = min(list(step.values()))
                else:
                    default_step = 1e-5
                    msg = 'Warning: no initial non-zero gradient to guide to choice of the initial step size. ' \
                            'Defaulting to the ARBITRARY initial value of %.2E.' % default_step
                    warnings.warn(msg)
                for key in remaining_keys:
                    step[key] = default_step
            
            if self.initial_step_size is None:
                return step
            else:
                return {key: value * self.initial_step_size for key, value in step.items()}

        else:
            return self.step

    def _evaluate_model_fit(self, parameters, with_grad=False):
        # Propagates the parameter value to all necessary attributes.
        self._set_parameters(parameters)

        # Call the model method.
        try:
            return self.statistical_model.compute_log_likelihood(self.dataset, self.individual_RER,
                                                                 mode=self.optimized_log_likelihood,
                                                                 with_grad=with_grad)

        except ValueError as error:
            logger.info('>> ' + str(error) + ' [ in gradient_ascent ]')
            self.statistical_model.clear_memory()
            if with_grad:
                raise RuntimeError('Failure of the gradient_ascent algorithm: the gradient of the model log-likelihood '
                                   'fails to be computed.', str(error))
            else:
                return - float('inf'), - float('inf')

    def compute_log_likelihood(self, parameters):
        # Propagates the parameter value to all necessary attributes.
        self._set_parameters(parameters)

        # Call the model method.
        return self.statistical_model.compute_log_likelihood(self.dataset, self.individual_RER,
                                                                 mode=self.optimized_log_likelihood,
                                                                 with_grad=False)
            


    def _get_parameters(self):
        out = self.statistical_model.get_fixed_effects()
        out.update(self.individual_RER)
        assert len(out) == len(self.statistical_model.get_fixed_effects())  + len(self.individual_RER)
        return out

    def _set_parameters(self, parameters):
        fixed_effects = {key: parameters[key] for key in self.statistical_model.get_fixed_effects().keys()}
        self.statistical_model.set_fixed_effects(fixed_effects)
        self.individual_RER = {key: parameters[key] for key in self.individual_RER.keys()}

    def _load_state_file(self):
        with open(self.state_file, 'rb') as f:
            d = pickle.load(f)
            return d['current_parameters'], d['current_iteration'], d["object_scale"], d["momenta_scale"],\
                    d["iter_multiscale_objects"], d["iter_multiscale_momenta"], d["order"]

    def _dump_state_file(self):
        d = {'current_parameters': self.current_parameters, 'current_iteration': self.current_iteration}
        d = self.multiscale.dump_state_file(d)        
        
        with open(self.state_file, 'wb') as f:
            pickle.dump(d, f)

    def _check_model_gradient(self):
        attachment, regularity, gradient = self._evaluate_model_fit(self.current_parameters, with_grad=True)
        parameters = copy.deepcopy(self.current_parameters)

        epsilon = 1e-3

        for key in gradient.keys():
            if key in ['image_intensities', 'landmark_points', 'modulation_matrix', 'sources']: continue

            logger.info('Checking gradient of ' + key + ' variable')
            parameter_shape = gradient[key].shape

            # To limit the cost if too many parameters of the same kind.
            nb_to_check = 100
            for index, _ in np.ndenumerate(gradient[key]):
                if nb_to_check > 0:
                    nb_to_check -= 1
                    perturbation = np.zeros(parameter_shape)
                    perturbation[index] = epsilon

                    # Perturb in +epsilon direction
                    new_parameters_plus = copy.deepcopy(parameters)
                    new_parameters_plus[key] += perturbation
                    new_attachment_plus, new_regularity_plus = self._evaluate_model_fit(new_parameters_plus)
                    total_plus = new_attachment_plus + new_regularity_plus

                    # Perturb in -epsilon direction
                    new_parameters_minus = copy.deepcopy(parameters)
                    new_parameters_minus[key] -= perturbation
                    new_attachment_minus, new_regularity_minus = self._evaluate_model_fit(new_parameters_minus)
                    total_minus = new_attachment_minus + new_regularity_minus

                    # Numerical gradient:
                    numerical_gradient = (total_plus - total_minus) / (2 * epsilon)
                    if gradient[key][index] ** 2 < 1e-5:
                        relative_error = 0
                    else:
                        relative_error = abs((numerical_gradient - gradient[key][index]) / gradient[key][index])
                    # assert relative_error < 1e-6 or np.isnan(relative_error), \
                    #     "Incorrect gradient for variable {} {}".format(key, relative_error)
                    # Extra printing
                    logger.info("Relative error for index " + str(index) + ': ' + str(relative_error)
                          + '\t[ numerical gradient: ' + str(numerical_gradient)
                          + '\tvs. torch gradient: ' + str(gradient[key][index]) + ' ].')
