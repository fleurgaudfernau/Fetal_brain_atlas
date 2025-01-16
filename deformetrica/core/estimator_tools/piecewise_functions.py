import logging
import os
import numpy as np
import os.path as op
logger = logging.getLogger(__name__)


class Piecewise():
    def __init__(self, model, ages, n_obs):
        
        # Model parameters
        self.model = model
        self.convergence_threshold = 1e-3 #0.001
        self.ages = ages
        self.n_obs = n_obs

        # Piecewise
        self.n_components = 1
        self.rupture_times = [0]
        self.add_component = [0]

        if "Regression" in model.name and len(model.get_momenta().shape) > 2:
            self.n_components = model.get_momenta().shape[0]
            self.rupture_times = model.get_rupture_time().tolist()   
        
        self.components = [str(c) for c in range(self.n_components)]  

        self.n_sources = 0 
        if model.name in ["BayesianGeodesicRegression", "BayesianPiecewiseRegression"]:
            self.n_sources = self.model.number_of_sources
    
    def find_component(self, j):
        c = 0
        candidates = [t for t in self.rupture_times if t < self.ages[j]]
        if candidates:
            c = self.rupture_times.index(candidates[-1]) + 1 

        return c
    
    def component_residuals(self, subjects_residuals, c):
        component_residuals = []
        if self.n_components > 1:
            for j in range(self.n_obs):
                if self.find_component(j) == c:
                    component_residuals.append(subjects_residuals[j])
        
        return np.mean(component_residuals)
    
    def set_weights(self):
        comp = self.components_weights()
        weights = [1] * self.n_obs
        for j in range(self.n_obs):
            c = self.find_component(j)
            weights[j] =  (c + 1) * 0.5
        logger.info("Setting weights to {}".format(weights))
        #self.n_obs / (comp * self.n_components)
        self.model.set_weights(weights)
    

    def components_weights(self):
        components = {c : 0 for c in range(self.n_components)}
        for j in range(self.n_obs):
            c = self.find_component(j)
            components[c] += 1
        return components
    
    ####################################################################################################################
    ### ADD COMPONENTS TO THE PIECEWISE MODEL
    ####################################################################################################################
    
    def add_component_happening(self):
        max_components = 7
        return self.n_components > 1 and not self.model.freeze_components \
                and self.n_components < max_components

    def add_component_condition(self, iteration):        
        if self.add_component_happening():
            if np.abs(iteration - self.add_component[-1]) > 10:
                if self.ratio(self.get_values("Residuals_average")[-1], self.get_values("Residuals_average")[-2]) \
                    < self.convergence_threshold:
                    return True
        return False

    # def change_weights(self, iteration):
    #     if self.add_component_condition(iteration):
        
    
    def add_new_component(self, dataset, iteration):
        # if np.abs(iteration - self.add_component[-1]) > 30:
        #     logger.info("Weighths back to original")
        #     weights = [1] * self.n_obs
        #     self.model.set_weights(weights)
        
        if self.add_component_condition(iteration):

            residuals = { c : self.get_values("Residuals_components", c)[-1]\
                        for c in range(self.n_components)}
            residuals = {k: v for k, v in sorted(residuals.items(), key=lambda item: item[1])}
            target = list(residuals.keys())[-1]

            # Add a component to the model
            while self.model.add_component(dataset, target) is False:
                residuals = {k: v for k, v in residuals.items() if k != target}
                target = list(residuals.keys())[-1]
            
            self.n_components += 1
            self.rupture_times = self.model.get_rupture_time().tolist()  
            self.add_component.append(iteration)

            # Adjust weights in the cost function
            
            logger.info("Setting high weights to subjects in component {}".format(target + 1))
            weights = self.model.weights
            for j in range(self.n_obs):
                if self.find_component(j) == target + 1:
                    weights[j] = 5.
            self.model.set_weights(weights)

            comp = self.components_weights()
            weights = self.model.weights
            for j in range(self.n_obs):
                c = self.find_component(j)
                weights[j] =  self.n_obs / (comp * self.n_components)
            self.model.set_weights(weights)

            return target

    def after_new_component(self, iteration):
        return iteration > 1 and (self.add_component[-1] == iteration)
    
    def after_new_component_(self, iteration):
        return iteration > 1 and (self.add_component[-1] == iteration -1)
    
    def no_convergence_after_new_component(self, iteration):
        if self.after_new_component_(iteration):
            logger.info("Do not allow convergence after new component")
            return True
                        
        return False
    
    def check_convergence_condition(self, iteration):
        if self.add_component_happening():
            return False
        
        if np.abs(self.add_component[-1] - iteration) < 6:
            return False

        return True
    
    def save_model_after_new_component(self, iteration, output_dir):
        if self.after_new_component(iteration):
            output_dir = op.join(output_dir, "{}_components_{}".format(self.n_components, iteration))
            
            if not op.exists(output_dir): os.mkdir(output_dir)
                
            return output_dir
        
        return

