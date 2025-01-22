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

    def __init__(self, statistical_model, dataset, optimization_method_type='undefined', individual_RER={},
                 optimized_log_likelihood=default.optimized_log_likelihood,
                 max_iterations=default.max_iterations, convergence_tolerance=default.convergence_tolerance,
                 print_every_n_iters=default.print_every_n_iters, save_every_n_iters=default.save_every_n_iters,
                 scale_initial_step_size=default.scale_initial_step_size, initial_step_size=default.initial_step_size,
                 max_line_search_iterations=default.max_line_search_iterations,
                 line_search_shrink=default.line_search_shrink,
                 line_search_expand=default.line_search_expand,
                 output_dir=default.output_dir, callback=None,
                 load_state_file=default.load_state_file, state_file=default.state_file,

                 multiscale_momenta = default.multiscale_momenta, #ajout fg
                 naive = default.naive, #ajout fg
                 multiscale_images = default.multiscale_images, #ajout fg
                 multiscale_meshes = default.multiscale_meshes,
                 multiscale_strategy = default.multiscale_strategy,
                 start_scale = None,
                 overwrite = True,

                 **kwargs):
        
        self.overwrite = overwrite

           
        
        super().__init__(statistical_model=statistical_model, dataset=dataset, name='GradientAscent',
                         optimized_log_likelihood=optimized_log_likelihood,
                         max_iterations=max_iterations, convergence_tolerance=convergence_tolerance,
                         print_every_n_iters=print_every_n_iters, save_every_n_iters=save_every_n_iters,
                         individual_RER=individual_RER,
                         callback=callback, state_file=state_file, output_dir=output_dir)
        
        
        
        assert optimization_method_type.lower() == self.name.lower()
        
        self.current_attachment = None
        self.current_regularity = None
        self.current_log_likelihood = None

        self.step = None
        self.line_search_shrink = line_search_shrink
        self.line_search_expand = line_search_expand

        self.scale_initial_step_size = scale_initial_step_size
        self.initial_step_size = initial_step_size
        self.max_line_search_iterations = max_line_search_iterations
        self.current_iteration = 0

        self.multiscale = Multiscale(multiscale_momenta, multiscale_images, multiscale_meshes, 
                                    multiscale_strategy, naive, self.statistical_model, self.initial_step_size, 
                                    self.scale_initial_step_size, self.output_dir, self.dataset, start_scale)
        
        if load_state_file:
            self.current_parameters, self.current_iteration, \
            image_scale, momenta_scale, iter_images, iter_momenta, order \
            = self._load_state_file()
            if multiscale_images:
                self.multiscale.image_scale = image_scale
                self.multiscale.iter_images = iter_images
            if multiscale_momenta:
                self.multiscale.momenta_scale = momenta_scale
                self.multiscale.iter_momenta = iter_momenta
            if multiscale_momenta and multiscale_images:
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
        
        self.multiscale.initialize()

        # Initialize residuals (before filtering)
        self.residuals = Residuals(self.statistical_model, self.dataset, self.print_every_n_iters, 
                                   self.output_dir)
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

        nb_params = len(gradient)

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

                if nb_params > 1:
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

            #self.residuals.add_new_component(self.dataset, self.current_iteration)
            # recover residuals in case a new component was added in piecewise regression
            #self.current_parameters = self._get_parameters()

            # Test the stopping criterion ------------------------------------------------------------------------------
            delta_f_current = last_log_likelihood - self.current_log_likelihood
            delta_f_initial = initial_log_likelihood - self.current_log_likelihood

            if math.fabs(delta_f_current) < self.convergence_tolerance * math.fabs(delta_f_initial):
                if self.multiscale.check_convergence_condition(self.current_iteration): 
                    logger.info('Tolerance threshold met. Stopping the optimization process.')
                    break
            
            

            # Printing and writing -------------------------------------------------------------------------------------
            t2 = perf_counter()
            logger.info("Time taken for iteration: {}".format(t2-t1))
            if not self.current_iteration % self.print_every_n_iters: self.print()
            if not self.current_iteration % self.save_every_n_iters: self.write()

            # Call user callback function ------------------------------------------------------------------------------
            if self.callback is not None:
                self._call_user_callback(float(self.current_log_likelihood), float(self.current_attachment),
                                         float(self.current_regularity), gradient)

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
        self.statistical_model.write(self.dataset, self.population_RER, self.individual_RER, 
                                    self.output_dir, self.current_iteration, write_all = True)
        self.residuals.write(self.output_dir, self.dataset, self.individual_RER, self.current_iteration)
        self.residuals.plot_residuals_evolution(self.output_dir, self.multiscale, self.individual_RER)
        self._dump_state_file()
            
    
    def save_model_state_after_ctf(self):
        output_dir = self.multiscale.save_model_after_ctf(self.current_iteration)
        if output_dir:
            self.statistical_model.write(self.dataset, self.population_RER, self.individual_RER, output_dir, 
                                        self.current_iteration, write_all = False)
    
    def save_model_after_adding_component(self):
        output_dir = self.residuals.save_model_after_new_component(self.current_iteration, self.output_dir)
        if output_dir:
            self.statistical_model.write(self.dataset, self.population_RER, self.individual_RER, output_dir, 
                                        self.current_iteration, write_all = True, 
                                        write_adjoint_parameters = False)

    
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
                                                                    self.residuals.get_values("Residuals_components"), 
                                                                    end)                                        
        self.current_parameters = new_parameters
        self._set_parameters(self.current_parameters)
        self.save_model_state_after_ctf()

        return new_parameters
        
    def _initialize_step_size(self, gradient):
        """
        Initialization of the step sizes for the descent for the different variables.
        If scale_initial_step_size is On, we rescale the initial sizes by the gradient squared norms.
        """
        if self.step is None or max(list(self.step.values())) < 1e-12:
            step = {}
            if self.scale_initial_step_size:
                remaining_keys = []
                
                for key, value in gradient.items(): 
                    if key != "haar_coef_momenta":
                        gradient_norm = math.sqrt(np.sum(value ** 2))
                        if gradient_norm < 1e-8:
                            remaining_keys.append(key)
                        elif math.isinf(gradient_norm):
                            step[key] = 1e-10
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

            if not self.scale_initial_step_size:
                if self.initial_step_size is None:
                    msg = 'Initializing all initial step sizes to the ARBITRARY default value: 1e-5.'
                    warnings.warn(msg)
                    return {key: 1e-5 for key in gradient.keys()}
                else:
                    return {key: self.initial_step_size for key in gradient.keys() if key != "haar_coef_momenta"}
        else:
            return self.step

    def _evaluate_model_fit(self, parameters, with_grad=False):
        # Propagates the parameter value to all necessary attributes.
        self._set_parameters(parameters)

        # Call the model method.
        try:
            return self.statistical_model.compute_log_likelihood(self.dataset, self.population_RER, self.individual_RER,
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

    def _get_parameters(self):
        out = self.statistical_model.get_fixed_effects()
        out.update(self.population_RER)
        out.update(self.individual_RER)
        assert len(out) == len(self.statistical_model.get_fixed_effects()) \
                           + len(self.population_RER) + len(self.individual_RER)
        return out

    def _set_parameters(self, parameters):
        fixed_effects = {key: parameters[key] for key in self.statistical_model.get_fixed_effects().keys()}
        self.statistical_model.set_fixed_effects(fixed_effects)
        self.population_RER = {key: parameters[key] for key in self.population_RER.keys()}
        self.individual_RER = {key: parameters[key] for key in self.individual_RER.keys()}

    def _load_state_file(self):
        with open(self.state_file, 'rb') as f:
            d = pickle.load(f)
            return d['current_parameters'], d['current_iteration'], \
                    d["image_scale"], d["momenta_scale"],\
                    d["iter_multiscale_images"], d["iter_multiscale_momenta"], d["order"]

    def _dump_state_file(self):
        d = {'current_parameters': self.current_parameters, 
             'current_iteration': self.current_iteration}
        d = self.multiscale.dump_state_file(d)     
        
        if self.state_file:
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
