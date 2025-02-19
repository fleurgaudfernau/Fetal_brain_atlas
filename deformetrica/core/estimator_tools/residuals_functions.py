import torch
import logging
import numpy as np
import os
from ...in_out.array_readers_and_writers import write_2D_list
from ...support.utilities.plot_tools import scatter_plot, plot_value_evolution, plot_sources
from ...support.utilities.tools import ratio, change, norm, scalar_product, current_distance
#from ...core.models.model_functions import gaussian_kernel
from .curvature_functions import Curvature
import os.path as op
from copy import deepcopy
logger = logging.getLogger(__name__)

class Residuals():
    def __init__(self, model, dataset, print_every_n_iters = None, output_dir = None):
        
        # Model parameters
        self.model = model
        self.name = model.name
        self.ext = self.model.objects_extension[0]     
        self.width = self.model.deformation_kernel_width   
        self.print_every_n_iters = print_every_n_iters
        self.convergence_threshold = 1e-2 #0.001

        self.n_sources = self.model.n_sources

        # Dataset characteristics
        self.n_subjects = len(dataset.subject_ids)
        self.n_obs = len(sum(dataset.deformable_objects, [])) #useful in regression
        self.obs_names = []

        for i, _ in enumerate(dataset.subject_ids): 
            for o, observation in enumerate(dataset.deformable_objects[i]):
                for object in observation.object_list: # Surface mesh
                    self.obs_names.append(object.object_filename.split("/")[-1].replace(self.ext, ""))

        self.time_series = dataset.is_time_series()

        if dataset.is_cross_sectional(): # several subjects
            self.observations = [d[0] for d in dataset.deformable_objects]
            if len(dataset.times[0]) > 0:
                self.ages = [float(d[0]) for d in dataset.times]
            else:
                self.ages = []
        elif dataset.is_time_series(): # several visit for 1 subject
            self.observations = [d for d in dataset.deformable_objects[0]]
            self.ages = [float(d) for d in dataset.times[0]]   

        self.id = dataset.subject_ids if "Regression" not in self.name else [n for n in range(self.n_obs)]

        # Residuals computation
        if self.name == "DeformableTemplate":
            self.compute_model_residuals = self.compute_atlas_residuals
        elif self.name == "BayesianAtlas":
            self.compute_model_residuals = self.compute_bayesian_atlas_residuals
        elif "Regression" in self.name:
            self.compute_model_residuals = self.compute_regression_residuals

        # Piecewise
        self.n_components = 1
        self.rupture_times = [0]

        if "Regression" in self.name and len(model.get_momenta().shape) > 2:
            self.n_components = model.get_momenta().shape[0]
            self.rupture_times = model.get_rupture_time().tolist()   
        
        self.components = [str(c) for c in range(self.n_components)]  

        self.initialize_plots(dataset)


    def initialize_plots(self, dataset):
        d = {"v": [], "condition": True, "ylab": [""], "plots" : [], "iter" : []}

        to_plot = ["Residuals_average", "Residuals_subjects",
                   "Template_changes", "Template_distance", 
                   "Gradient_norm", "Momenta_norm", "Modulation_matrix_distance", 
                   "Modulation_matrix_changes", 
                    "Modulation_matrix_norm", "Rupture_time"]
        self.plot = {k: deepcopy(d) for k in to_plot}

        # Plots that have several series to plot
        self.initialize_values("Gradient_norm")
        self.initialize_values("Residuals_subjects", {i: [] for i in range(self.n_obs)})
        self.initialize_values("Momenta_norm", {i: [] for i in range(self.n_components)})
        self.initialize_values("Modulation_matrix_distance", {c : [] for c in range(self.n_sources)})
        self.initialize_values("Modulation_matrix_changes", {c : [] for c in range(self.n_sources)})
        self.initialize_values("Modulation_matrix_norm", {c : [] for c in range(self.n_sources)})
        self.initialize_values("Rupture_time", {c : [] for c in range(self.n_components)[:-1]})

        # Conditions
        self.set_condition("Residuals_subjects")
        self.set_condition("Rupture_time", self.n_components > 1 and not self.model.freeze_rupture_time)
        self.set_condition("Template_changes")
        self.set_condition("Template_distance", ("Atlas" in self.name))
        self.set_condition("Modulation_matrix_distance", (self.name == "BayesianGeodesicRegression"))
        self.set_condition("Modulation_matrix_changes",(self.name == "BayesianGeodesicRegression"))
        self.set_condition("Modulation_matrix_norm", (self.name == "BayesianGeodesicRegression"))

        # Residuals computations
        self.templates = []
        self.ss = {c : [] for c in range(self.n_sources)}

        # Curvatures
        self.curvature = Curvature(dataset, self.model, self.ext, self.name, self.ages, self.n_obs)

        #self.first_write = [True, True]
        
        #self.compute_mesh_distance(dataset, 0)
                
    ####################################################################################################################
    ### Residuals tools
    ####################################################################################################################
    
    def initialize_values(self, k, v = {}):
        self.plot[k]["v"] = v
    
    def set_condition(self, k, v = False):
        self.plot[k]["condition"] = v
    
    def set_ylab(self, k, v):
        if type(v) != list: v = list(v)
        
        self.plot[k]["ylab"] = v

    def add_iter(self, k, iteration):
        if iteration not in self.plot[k]["iter"]:
            self.plot[k]["iter"].append(iteration)
    
    def set_value(self, k, value, i = None):
        if i is not None:
            self.plot[k]["v"][i] = value
        else:
            self.plot[k]["v"] = value

    def add_value(self, k, value, i = None):
        if i is not None:
            self.plot[k]["v"][i].append(value)
        else:
            self.plot[k]["v"].append(value)
    
    def get_values(self, k, i = None):
        if i is not None:
            return deepcopy(self.plot[k]["v"][i])
        
        if type(self.plot[k]["v"]) == dict:

            return deepcopy(list(self.plot[k]["v"].values()))
        
        return deepcopy(self.plot[k]["v"])
    
    def percentage_residuals_diminution(self):
        return ratio(self.get_values("Residuals_average")[-1], 
                    self.get_values("Residuals_average")[0])
            
    def to_plot(self, k):
        return self.plot[k]["condition"]
    
    ####################################################################################################################
    ### PARAMETERS CHANGE
    ####################################################################################################################
    
    def compute_image_distances(self, dataset, current_iteration):
        """
        Provides more objective measures than residuals
        """
        if current_iteration not in self.iterations_dist:
            self.iterations_dist.append(current_iteration)

        if self.ext == ".nii" and len(self.iterations_dist) > len(self.get_values("Distances")[0][0]): 
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

        if self.ext == ".vtk" and len(self.iterations_dist) > len(list(self.plot["Distances"]["v"].values())[0][0]): 

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
                n = norm(self.model.control_points, self.model.get_momenta()[i], self.width)
                self.add_value("Momenta_norm", n, i)
                self.add_iter("Momenta_norm", iteration)

    def gradient_norms(self, multiscale, iteration):
        if not self.plot["Gradient_norm"]["v"]:
            self.set_value("Gradient_norm", {k: [] for k in multiscale.gradient_norms.keys()})
            self.set_ylab("Gradient_norm", list(multiscale.gradient_norms.keys()))
        
        for k in multiscale.gradient_norms.keys():
            self.add_value("Gradient_norm", multiscale.gradient_norms[k][-1], k)
            self.add_iter("Gradient_norm", iteration)        

    def rupture_time_change(self, iteration):
        if self.to_plot("Rupture_time"):    
            tR = self.model.get_rupture_time()
            
            for i, t in enumerate(tR):
                self.add_value("Rupture_time", t, i)
                self.add_iter("Rupture_time", iteration)
    
    def modulation_matrix_change(self, iteration):
        if self.to_plot("Modulation_matrix_norm"):

            # Compute MM norm and distance to original MM
            cp = self.model.control_points
            modulation_matrix = self.model.get_modulation_matrix()

            for s in range(modulation_matrix.shape[1]):
                self.ss[s].append(np.reshape(modulation_matrix[:, s], cp.shape))

                dist = current_distance(cp, self.ss[s][-1], self.ss[s][0], self.width)
                norm_ = norm(cp, self.ss[s][-1], self.width)
                changes = change(self.ss[s])

                self.add_value("Modulation_matrix_distance", dist, s)  
                self.add_value("Modulation_matrix_norm", norm_, s) 
                self.add_value("Modulation_matrix_changes", changes, s)

                if len(self.ss[s]) > 3: self.ss[s].pop(1)
            
            self.add_iter("Modulation_matrix_distance", iteration)
            self.add_iter("Modulation_matrix_norm", iteration)
            self.add_iter("Modulation_matrix_changes", iteration)    
    
    ####################################################################################################################
    ### Residuals tools for other models
    ####################################################################################################################
    def compute_atlas_residuals(self, dataset, individual_RER = None):
        residuals_list = self.model.compute_residuals(dataset, individual_RER)
        residuals_list = [r.cpu().numpy() for r in residuals_list]

        # irrelevant for meshes
        #residuals_by_points = self.model.compute_residuals_per_point(dataset, individual_RER)
        return residuals_list
    
    def compute_bayesian_atlas_residuals(self, dataset, individual_RER):
        residuals_list = self.model._write_model_predictions(dataset, individual_RER, output_dir="", write = False)
        residuals_list = [r[0].cpu().numpy() for r in residuals_list]

        return residuals_list

    def compute_regression_residuals(self, dataset, individual_RER = None):
        # Avoid residuals recomputation: already computed in LL computation
        if self.model.current_residuals is None:
            residuals_list = self.model.compute_residuals(dataset, individual_RER)
        else:
            residuals_list = self.model.current_residuals
        
        if torch.is_tensor(residuals_list[0]):
            residuals_list = [r.cpu().numpy() for r in residuals_list]
        
        return residuals_list         

    ####################################################################################################################
    ### Common Residuals tools
    ####################################################################################################################
    def compute_attachement_residuals(self, dataset, current_iteration, individual_RER):
        subjects_residuals = self.compute_model_residuals(dataset, individual_RER)

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
        self.rupture_time_change(current_iteration)
        self.modulation_matrix_change(current_iteration)

        # Residuals (return a list for each object)
        self.compute_attachement_residuals(dataset, current_iteration, individual_RER)
        
        if self.print_every_n_iters and not (current_iteration % self.print_every_n_iters):
            try:
                logger.info("Last average residuals {}".format(self.get_values("Residuals_average")[-5:]))
            except:
                logger.info("Avg_residuals {}".format(self.get_values("Residuals_average")))

            if len(self.get_values("Residuals_average")) > 1: 
                logger.info("Residuals diminution: {}".format(ratio(self.get_values("Residuals_average")[-1], 
                                                                    self.get_values("Residuals_average")[-2])))
                        
    ####################################################################################################################
    ### Plot tools
    ####################################################################################################################
    def set_ylabels(self):
        # Labels
        self.set_ylab("Residuals_average", 'Average residuals (attachement term)')
        self.set_ylab("Residuals_subjects", self.obs_names)
        self.set_ylab("Template_changes", 'Template differences')
        self.set_ylab("Template_distance", 'Distance to initial template')
        self.set_ylab("Momenta_norm", self.components)
        self.set_ylab("Modulation_matrix_norm", ["Space_shift_{}".format(c) for c in range(self.n_sources)])
        self.set_ylab("Modulation_matrix_changes", ["Space_shift_{}".format(c) for c in range(self.n_sources)])
        self.set_ylab("Modulation_matrix_distance", ["Space_shift_{}".format(c) for c in range(self.n_sources)])
        self.set_ylab("Rupture_time", ["tR_{}".format(c) for c in self.components[:-1]])

    def set_to_plot(self):
        for k in self.plot.keys():
            self.plot[k]["plots"] = [v for v in self.get_values(k)]

    def plot_residuals_evolution(self, output_dir, multiscale, individual_RER = None):
        # Scatter plots
        self.plot_sources(output_dir, individual_RER)
        self.plot_residuals_by_age(output_dir)

        # Curvatures
        # self.curvature.plot_reconstructions_curvature(output_dir)
        # self.curvature.plot_regression_curvature(output_dir)
        
        self.set_ylabels()
        self.set_to_plot()

        for t in self.plot.keys():
            if self.to_plot(t):
                try:
                    plot_value_evolution(output_dir, t, self.plot[t]["plots"], self.plot[t]["iter"],
                                         self.plot[t]["ylab"], [multiscale.images.iter, 
                                         multiscale.momenta.iter, multiscale.meshes.iter],
                                         ["multiscale_images", "multiscale_momenta", "multiscale_meshes"])
                except:
                      print("\nProblem plotting {}".format(t), "\n", self.plot[t]["plots"])
                
    def plot_sources(self, output_dir, individual_RER = None):
        if individual_RER and "sources" in individual_RER:
            sources = individual_RER["sources"]
            plot_sources(output_dir, sources, self.ages)
    
    def plot_residuals_by_age(self, output_dir):    
        if self.n_obs > 1 and self.time_series:    
            ratios = [ratio(self.get_values('Residuals_subjects')[i][-1], 
                            self.get_values('Residuals_subjects')[i][0]) \
                      for i in range(self.n_obs)]
            scatter_plot(output_dir, "Residuals_changes_by_age.png", self.ages, 
                         ratios, labels = self.id, xlab = "Age (GW)", ylab = "Residuals")
            
            values = [self.get_values('Residuals_subjects')[i][-1] for i in range(self.n_obs)]
            scatter_plot(output_dir, "Residual_error_by_age.png", self.ages, 
                         values, labels = self.id,  xlab = "Age (GW)", ylab = "Residual error")
    

    def write(self, output_dir, dataset, individual_RER =None, current_iteration = None):
        # Recompute residuals to display remaining residuals per subject

        # Curvature computed only every save_every_n_iters to save time
        #self.compute_image_distances(dataset, current_iteration)
        #self.compute_mesh_distance(dataset, current_iteration, individual_RER)
        # self.curvature.compute_mesh_curvatures(dataset, current_iteration, individual_RER)
        # self.curvature.compute_regression_curvature(dataset)

        to_write = [["\nTotal residuals left: {}".format(self.get_values("Residuals_average")[-1])],
                    ["Total initial residuals:", self.get_values("Residuals_average")[0]],
                    ["Percentage of residuals diminution:{}"\
                        .format(self.percentage_residuals_diminution())]]
                
        # save distances and curvatures
        to_write.append(["\n******** Template ********"])
        #to_write = self.curvature.write(to_write)

        to_write.append(["\n******** Observations ********"])
        for i in range(self.n_obs):
            if self.ages:
                to_write.append(["\n---Observation: {} age {}---".format(i, self.ages[i])])
            else:
                to_write.append(["\n---Observation: {}".format(i)])
            to_write.append(["Percentage of residuals diminution:{}"\
                        .format(ratio(self.get_values('Residuals_subjects')[i][-1], 
                                self.get_values('Residuals_subjects')[i][0]))])
            
            #to_write = self.curvature.write(to_write, i)

        write_2D_list(to_write, output_dir, "{}___Residuals.txt".format(self.name))
