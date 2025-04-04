from copy import deepcopy
from numpy import mean
import logging
from ...support.utilities.plot_tools import scatter_plot, plot_value_evolution
logger = logging.getLogger(__name__)

class Curvature(PlotTracker):
    def __init__(self, output_dir, dataset, model, name, ages, n_obs):
        super().__init__()  

        self.name = name
        self.ages = ages
        self.model = model
        self.n_obs = n_obs
        self.output_dir = output_dir
        
        self.curvatures = dict()

        if dataset.type != "image":
            self.curvatures = ["surface_area", "GI", "mean", "gaussian"]

            d = {"v": [], "condition": True, "ylab": [""], "plots" : [], "iter" : []}

            for c in self.curvatures:
                self.plot.update({"Template_" + c: deepcopy(d)})
                self.set_condition("Template_" + c, not self.model.freeze_template)

                self.plot.update({"Subjects_" + c: deepcopy(d)})
                self.plot.update({"Reconstructions_" + c: deepcopy(d)})

        self.subjects_curvature(dataset)
        self.template_curvature(0)
            
    ###########################################################################################
    #### UPDATE TOOLS
    ###########################################################################################

    def update(self, dataset, iteration, individual_RER):
        self.template_curvature(iteration)
        #self.deformed_subjects_curvature(dataset, iteration, individual_RER)
        #self.compute_regression_curvature(dataset)

    def compute_regression_curvature(self, dataset):
        if "Regression" in self.name and "Kernel" not in self.name:
            for c in self.curvatures:
                for time in (range(int(min(self.ages)), int(max(self.ages)))):
                    results = self.model.compute_flow_curvature(dataset, time, curvature)
                    for obj in results.object_list:
                        self.add_curvature(c, obj, "Regression")

    def template_curvature(self, iteration):
        """ Curvature of the template object """

        logger.info("\nComputing estimated template curvature...")
        for c in self.curvatures:
            if self.to_plot("Template_" + c):
                self.model.template.compute_curvature(c)

                for obj in self.model.template.object_list:
                    self.add_value("Template_" + c, obj.curv[c]["mean"])
                    self.add_iter("Template_" + c, iteration)

                    logger.info("Template {} = {}".format(c, round(obj.curv[c]["mean"],2)))

    def subjects_curvature(self, dataset):
        """ Curvature of the input subjects """
        logger.info("\nComputing input objects curvature...")

        for sujet_visites in dataset.objects:
            for observations in sujet_visites:
                for obj in observations.object_list:

                    for c in self.curvatures:
                        obj.curvature_metrics(c)
                        self.add_value("Subjects_" + c, obj.curv[c]["mean"])
        
    def deformed_subjects_curvature(self, dataset, iteration, individual_RER = None):
        """
            Deformed template to subject curvature. 
        """        
        # Compute deformed object curvatures
        for j in range(self.n_obs):  
            for c in self.curvatures:
                if self.to_plot("Template_" + c):
                    self.model.compute_curvature(dataset, j, individual_RER, c)   
                    for obj in self.model.template.object_list:
                        self.add_value("Reconstructions_" + c, obj.curv[c]["mean"])
                        self.add_iter("Reconstructions_" + c, iteration)              
    
    ###########################################################################################
    #### PLOTS
    ###########################################################################################
    def plots(self):
        # Curvatures
        self.plot_template_curvature()
        # self.plot_reconstructions_curvature()
        # self.plot_regression_curvature()
    
    def plot_template_curvature(self):
        self.set_to_plot()
        
        for c in self.curvatures:
            key = "Template_" + c
            self.set_ylab(key, c)

            if self.to_plot(key):

                plot_value_evolution(self.output_dir, key, self.plot[key]["plots"], 
                                    self.plot[key]["iter"], self.plot[key]["ylab"])

    def plot_reconstructions_curvature(self):
        for c in self.curvatures:
            scatter_plot(self.output_dir, c + ".png", [self.ages] * 2 + [[mean(self.ages)]] * 2, 
                         [self.get_curvature(c, "Original"), self.get_curvature(c), 
                          self.get_curvature(c, "Template (original)"),
                          self.get_curvature(c, "Template (final)")], 
                         ["True values", "Reconstructed values", 
                          "Template (original)", "Template (final)"],
                         labels = [i for i in range(self.n_obs)],
                         xlab = "Age (GW)", ylab = c)
            self.clean_curvature(c)
            self.clean_curvature(c, "Template (final)")
        
    def plot_regression_curvature(self):
        for c in self.curvatures:
            if self.get_curvature(c, "Regression"):
                ages = [a for a in range(int(min(self.ages)), int(max(self.ages)))]
                                
                scatter_plot(self.output_dir, "Regression_{}.png".format(c), ages, 
                            [self.get_curvature(c, "Regression")], [c],
                            xlab = "Age (GW)", ylab = c)
                
                self.clean_curvature(c, "Regression")
    
    def write(self, to_write, i = None):
        for c in self.curvatures:
            to_write.append(["\nCurvature: {}".format(c)])

            if i is not None:
                to_write.append(["Subject curvature: {}".format(self.get_curvature(c, "Original")[i])])
                to_write.append(["Deformed template to subject curvature: {}".format(self.get_curvature(c, "Current")[i])])
            else:
                to_write.append(["Template curvature: {}".format(self.get_curvature(c, "Template (final)")[0])])


        return to_write