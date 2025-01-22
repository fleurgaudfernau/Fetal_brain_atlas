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
    