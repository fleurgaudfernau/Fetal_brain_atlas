import torch
import logging
import os
import numpy as np
import os.path as op
from copy import deepcopy
from ...in_out.array_readers_and_writers import write_2D_list
from ...core.model_tools.deformations.exponential_tools import norm, scalar_product, current_distance
from ...support.utilities.plot_tools import PlotTracker, scatter_plot, plot_value_evolution, plot_sources
from ...support.utilities.tools import total_residuals_change, residuals_change, change
from  ...support.utilities import detach
from .curvature_functions import Curvature
logger = logging.getLogger(__name__)

class Residuals(PlotTracker):
    def __init__(self, model, dataset, print_every_n_iters = None, output_dir = None):
        super().__init__() 
        
        # Model parameters
        self.model = model
        self.name = model.name
        self.type = self.model.template.object_list[0].type 
        self.width = self.model.deformation_kernel_width   
        self.print_every_n_iters = print_every_n_iters
        self.output_dir = output_dir

        self.n_sources = self.model.n_sources

        # Dataset characteristics
        self.n_obs = dataset.n_obs
        self.time_series = dataset.is_time_series()

        self.obs_names = [ op.splitext(op.basename(obj.object_filename))[0]
                            for subject_obs in dataset.objects
                            for observation in subject_obs
                            for obj in observation.object_list ]        

        if dataset.is_cross_sectional(): # several subjects
            self.observations = [d[0] for d in dataset.objects]
            self.ages = [float(d[0]) for d in dataset.times] if dataset.times and dataset.times[0] else []

        elif dataset.is_time_series(): # several visit for 1 subject
            self.observations = [d for d in dataset.objects[0]]
            self.ages = [float(d) for d in dataset.times[0]]   

        self.id = dataset.ids if "Regression" not in self.name else [n for n in range(self.n_obs)]

        # Residuals computation
        self.compute_model_residuals = {"DeformableTemplate" : self.compute_atlas_residuals,
                                    "KernelRegression" : self.compute_atlas_residuals,
                                    "BayesianPiecewiseRegression" : self.compute_bayesian_atlas_residuals,
                                    "PiecewiseRegression" : self.compute_regression_residuals,
                                    "Regression" : self.compute_regression_residuals} 

        # Piecewise
        self.n_components = 1

        if "Piecewise" in self.name:
            self.n_components = model.get_momenta().shape[0]
        
        self.components = [str(c) for c in range(self.n_components)]  

        self.initialize_plots(dataset)


    def initialize_plots(self, dataset):
        d = {"v": [], "condition": True, "ylab": [""], "plots" : [], "iter" : []}

        to_plot = ["Residuals_average", "Residuals_subjects",
                   "Template_changes", "Template_distance", 
                   "Gradient_norm", "Momenta_norm", "Modulation_matrix_distance", 
                   "Modulation_matrix_changes", "Modulation_matrix_norm"]
        self.plot = {k: deepcopy(d) for k in to_plot}

        # Plots that have several series to plot
        self.initialize_values("Gradient_norm")
        self.initialize_values("Residuals_subjects", {i: [] for i in range(self.n_obs)})
        self.initialize_values("Momenta_norm", {i: [] for i in range(self.n_components)})
        self.initialize_values("Modulation_matrix_distance", {c : [] for c in range(self.n_sources)})
        self.initialize_values("Modulation_matrix_changes", {c : [] for c in range(self.n_sources)})
        self.initialize_values("Modulation_matrix_norm", {c : [] for c in range(self.n_sources)})

        # Conditions
        self.set_condition("Residuals_subjects")
        self.set_condition("Template_changes", not self.model.freeze_template)
        self.set_condition("Template_distance", ("Atlas" in self.name))
        self.set_condition("Modulation_matrix_distance", self.name == "BayesianGeodesicRegression")
        self.set_condition("Modulation_matrix_changes",self.name == "BayesianGeodesicRegression")
        self.set_condition("Modulation_matrix_norm", self.name == "BayesianGeodesicRegression")

        # Residuals computations
        self.templates = []
        self.ss = {c : [] for c in range(self.n_sources)}

        # Curvatures
        self.curvature = Curvature(self.output_dir, dataset, self.model, self.name, self.ages, self.n_obs)

        #self.first_write = [True, True]
        
        #self.compute_mesh_distance(dataset, 0)
                
    ####################################################################################################################
    ### Residuals tools
    ####################################################################################################################
    
    def percentage_residuals_diminution(self):
        return round(100 * total_residuals_change(self.get_values("Residuals_average")), 2)
    
    def last_residuals_diminution(self):
        return round(100 * residuals_change(self.get_values("Residuals_average")), 2)

    
    ####################################################################################################################
    ### PARAMETERS CHANGE
    ####################################################################################################################
    
    def compute_image_distances(self, dataset, current_iteration):
        """
        Provides more objective measures than residuals
        """
        if current_iteration not in self.iterations_dist:
            self.iterations_dist.append(current_iteration)

        if self.type == "Image" and len(self.iterations_dist) > len(self.get_values("Distances")[0][0]): 
            for j, obj in enumerate(self.observations):
                for k in self.distances:
                    distance = k.replace("Distance_", "")
                    self.model.compute_objects_distances(dataset, j, distance)
                        
                for i, object in enumerate(obj.object_list):
                    self.add_value(k, float(object.distance[distance]), j)

    def compute_mesh_distance(self, dataset, current_iteration, individual_RER = None):
        """
        Compute different distances between deformed template and subject
        Distances computed in the deformed template space
        /!\ Computing point wise current/varifold distance is WRONG

        here "hausdorff = average of point wise 
        """
        return
        if current_iteration not in self.iterations_dist:
            self.iterations_dist.append(current_iteration)

        if self.type == "SurfaceMesh" and len(self.iterations_dist) > len(list(self.plot["Distances"]["v"].values())[0][0]): 

            for j, obj in enumerate(self.get_objects(dataset)):
                
                # Store 3 average distance over cortex for each subject
                for d in self.distances:
                    distance = d.replace("Distance_", "")
                    self.model.compute_objects_distances(dataset, j, distance, individual_RER)

                    #if current_iteration != 0: obj = self.model.template.object_list
                    for i, object in enumerate(obj.object_list):
                        self.plot[d]["v"][j].append(float(object.distance[distance]))

                        if object.distance[list(self.plot[d]["v"].keys())[0]] > 1e-3:
                            current_iteration = current_iteration if self.first_write[0] else "_end"
                            self.first_write[0] = False
                            object.polydata.save(op.join(self.output_dir, 
                            "Distances_subject_t{}_i{}.vtk".format(j, current_iteration)), binary=False)
                            self.model.template.object_list[i].polydata.save(op.join(self.output_dir, 
                            "Distances_deformed_template_to_t{}_i{}.vtk".format(j, current_iteration)), binary=False)

    def template_change(self, iteration):
        if self.to_plot("Template_changes"):
            self.templates.append(self.model.get_template_data().values())
            if len(self.templates) > 2: #do not consider first template change because of CTF on images
                self.add_value("Template_changes", change(self.templates))
                self.add_iter("Template_changes", iteration)
                self.templates.pop(0)
    
    def distance_to_initial_template(self, iteration):
        if self.to_plot("Template_distance"):
            self.templates.append(self.model.get_template_data().values())
            if len(self.templates) > 2:
                self.add_value("Template_distance", change(self.templates))
                self.add_iter("Template_distance", iteration)
                self.templates.pop() #last element

    def momenta_change(self, iteration):
        if self.to_plot("Momenta_norm"):
                        
            for i in range(self.n_components):
                n = norm(self.model.cp, self.model.get_momenta()[i], self.width)
                self.add_value("Momenta_norm", n, i)
                self.add_iter("Momenta_norm", iteration)

    def gradient_norms(self, multiscale, iteration):
        if not self.plot["Gradient_norm"]["v"]:
            self.set_value("Gradient_norm", {k: [] for k in multiscale.gradient_norms.keys()})
            self.set_ylab("Gradient_norm", list(multiscale.gradient_norms.keys()))
        
        for k in multiscale.gradient_norms.keys():
            self.add_value("Gradient_norm", multiscale.gradient_norms[k][-1], k)
            self.add_iter("Gradient_norm", iteration)        
    
    def modulation_matrix_change(self, iteration):
        if self.to_plot("Modulation_matrix_norm"):

            # Compute MM norm and distance to original MM
            modulation_matrix = self.model.get_modulation_matrix()

            for s in range(modulation_matrix.shape[1]):
                self.ss[s].append(np.reshape(modulation_matrix[:, s], self.model.cp.shape))

                dist = current_distance(self.model.cp, self.ss[s][-1], self.ss[s][0], self.width)
                norm_ = norm(self.model.cp, self.ss[s][-1], self.width)
                changes = change(self.ss[s])

                self.add_value("Modulation_matrix_distance", dist, s)  
                self.add_value("Modulation_matrix_norm", norm_, s) 
                self.add_value("Modulation_matrix_changes", changes, s)

                if len(self.ss[s]) > 3: self.ss[s].pop(1)
            
            self.add_iter("Modulation_matrix_distance", iteration)
            self.add_iter("Modulation_matrix_norm", iteration)
            self.add_iter("Modulation_matrix_changes", iteration)    
    
    ####################################################################################################################
    ### Model-specific residuals tools
    ####################################################################################################################
    
    def compute_atlas_residuals(self, dataset, individual_RER = None):
        return [detach(r) for r in self.model.compute_residuals(dataset, individual_RER)]
    
    def compute_bayesian_atlas_residuals(self, dataset, individual_RER):
        return [detach(r[0]) for r in self.model._write_model_predictions(dataset, 
                                            individual_RER, output_dir="", write = False)]

    def compute_regression_residuals(self, dataset, individual_RER = None):
        # Avoid residuals recomputation: already computed in LL computation
        residuals_list = self.model.compute_residuals(dataset, individual_RER) \
                        if self.model.current_residuals is None else self.model.current_residuals
                
        return [detach(r) for r in residuals_list]         

    ####################################################################################################################
    ### Common Residuals tools
    ####################################################################################################################
    
    def compute_attachement_residuals(self, dataset, current_iteration, individual_RER):
        subjects_residuals = self.compute_model_residuals[self.name](dataset, individual_RER)

        # average residuals over subjects
        self.add_value("Residuals_average", np.sum(subjects_residuals))
        
        # subjects residuals
        for j in range(self.n_obs):
            self.add_value("Residuals_subjects", float(subjects_residuals[j]), j)
                
        self.add_iter("Residuals_average", current_iteration)
        self.add_iter("Residuals_subjects", current_iteration)        
                   
    def compute_residuals(self, dataset, current_iteration, individual_RER = None, multiscale = None):
        """
        Compute residuals at each pixel/voxel between objects and deformed template.
        """

        # Template and momenta changes
        self.template_change(current_iteration)
        self.momenta_change(current_iteration)
        self.gradient_norms(multiscale, current_iteration)
        self.modulation_matrix_change(current_iteration)

        # Residuals (return a list for each object)
        self.compute_attachement_residuals(dataset, current_iteration, individual_RER)
        
        if self.print_every_n_iters and not (current_iteration % self.print_every_n_iters):
            if len(self.get_values("Residuals_average")) > 1: 
                logger.info("\nResiduals diminution: {} (%)".format(self.last_residuals_diminution() ))
                        
    ####################################################################################################################
    ### Plot tools
    ####################################################################################################################
    
    def set_ylabels(self):
        self.set_ylab("Residuals_average", 'Average residuals (attachement term)')
        self.set_ylab("Residuals_subjects", self.obs_names)
        self.set_ylab("Template_changes", 'Template differences')
        self.set_ylab("Template_distance", 'Distance to initial template')
        self.set_ylab("Momenta_norm", self.components)
        self.set_ylab("Modulation_matrix_norm", ["Space_shift_{}".format(c) for c in range(self.n_sources)])
        self.set_ylab("Modulation_matrix_changes", ["Space_shift_{}".format(c) for c in range(self.n_sources)])
        self.set_ylab("Modulation_matrix_distance", ["Space_shift_{}".format(c) for c in range(self.n_sources)])

    def set_to_plot(self):
        for k in self.plot.keys():
            self.plot[k]["plots"] = [v for v in self.get_values(k)]

    def plot_residuals_evolution(self, output_dir, multiscale, individual_RER = None):
        # Scatter plots
        self.plot_sources(self.output_dir, individual_RER)
        self.plot_residuals_by_age(self.output_dir)

        # Curvatures
        self.curvature.plots()
        
        self.set_ylabels()
        self.set_to_plot()

        for t in self.plot.keys():
            if self.to_plot(t):
                plot_value_evolution(self.output_dir, t, self.plot[t]["plots"], 
                                    self.plot[t]["iter"], self.plot[t]["ylab"], 
                                    [multiscale.objects.iter, multiscale.momenta.iter],
                                    ["multiscale_images", "multiscale_momenta"])
                
    def plot_sources(self, output_dir, individual_RER = None):
        if individual_RER and "sources" in individual_RER:
            sources = individual_RER["sources"]
            plot_sources(self.output_dir, sources, self.ages)
    
    def plot_residuals_by_age(self, output_dir):    
        if self.n_obs > 1 and self.time_series:    
            ratios = [total_residuals_change(self.get_values('Residuals_subjects')[i]) \
                      for i in range(self.n_obs)]
            scatter_plot(self.output_dir, "Residuals_changes_by_age.png", self.ages, 
                         ratios, labels = self.id, xlab = "Age (GW)", ylab = "Residuals")
            
            values = [self.get_values('Residuals_subjects')[i][-1] for i in range(self.n_obs)]
            scatter_plot(self.output_dir, "Residual_error_by_age.png", self.ages, 
                         values, labels = self.id,  xlab = "Age (GW)", ylab = "Residual error")
    
    def write(self, output_dir, dataset, individual_RER =None, current_iteration = None):
        # Recompute residuals to display remaining residuals per subject

        # Curvature computed only every save_every_n_iters to save time
        #self.compute_image_distances(dataset, current_iteration)
        #self.compute_mesh_distance(dataset, current_iteration, individual_RER)

        self.curvature.update(dataset, current_iteration, individual_RER)

        to_write = [["\nTotal residuals left: {}".format(self.get_values("Residuals_average")[-1])],
                    ["Total initial residuals:", self.get_values("Residuals_average")[0]],
                    ["Residuals diminution: {} (%)".format(self.percentage_residuals_diminution())]]
                
        # save distances and curvatures
        to_write.append(["\n******** Template ********"])
        #to_write = self.curvature.write(to_write)

        to_write.append(["\n******** Observations ********"])
        for i in range(self.n_obs):
            if self.ages:
                to_write.append(["\n>> Observation: {} age {}---".format(i, self.ages[i])])
            else:
                to_write.append(["\n>> Observation: {}".format(i)])
            to_write.append(["Residuals diminution (%):{}"\
                        .format(total_residuals_change(self.get_values('Residuals_subjects')[i]))])
            
            #to_write = self.curvature.write(to_write, i)

        write_2D_list(to_write, self.output_dir, "{}___Residuals.txt".format(self.name))
