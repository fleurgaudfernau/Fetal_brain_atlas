from ...support.utilities.plot_tools import scatter_plot
from numpy import mean

class Curvature():
    def __init__(self, dataset, model, ext, name, ages, n_obs):
        self.name = name
        self.ages = ages
        self.model = model
        self.n_obs = n_obs
        
        self.curvatures = dict()
        if ext == ".vtk":
            self.curvatures = ["Curvature_surface_area", "Curvature_GI"] #"Curvature_gaussian"
            self.curvatures = {c : {"Original" : [], "Current" : [], 
                                    "Template (original)" : [], "Template (final)" : [],
                                "Regression" : []} for c in self.curvatures}

        self.compute_mesh_curvatures(dataset, 0)
    
    def add_curvature(self, c, value, k = "Current"):
        self.curvatures[c][k].append(value)
    
    def get_curvature(self, c, k = "Current"):
        return self.curvatures[c][k]
    
    def clean_curvature(self, c, k = "Current"):
        self.curvatures[c][k] = []
    
    def compute_regression_curvature(self, dataset):
        if "Regression" in self.name:
            for c in self.curvatures.keys():
                curvature = c.replace("Curvature_", "")
                for time in (range(int(min(self.ages)), int(max(self.ages)))):
                    results = self.model.compute_flow_curvature(dataset, time, curvature)
                    for object in results.object_list:
                        self.add_curvature(c, object.curv[curvature]["mean"], "Regression")

    def compute_mesh_curvatures(self, dataset, iter, individual_RER = None):
        """
            At iter 0, compute object curvature. 
            Otherwise deformed template to subject curvature. 
        """
        k = "Original" if iter == 0 else "Current"
        for j in range(self.n_obs):  
            for c in self.curvatures.keys():
                curvature = c.replace("Curvature_", "")
                results = self.model.compute_curvature(dataset, j, individual_RER, 
                                                        curvature, iter)   
                for object in results.object_list:
                    self.add_curvature(c, object.curv[curvature]["mean"], k = k)
        
        # Template curvature
        k = "Template (original)" if iter == 0 else "Template (final)"
        for c in self.curvatures.keys():
            curvature = c.replace("Curvature_", "")
            results = self.model.compute_curvature(dataset, None, individual_RER, 
                                                    curvature, iter)   
            for object in results.object_list:
                self.add_curvature(c, object.curv[curvature]["mean"], k = k)

    def plot_reconstructions_curvature(self, output_dir):
        for c in self.curvatures.keys():
            scatter_plot(output_dir, c + ".png", 
                         [self.ages] * 2 + [[mean(self.ages)]] * 2, 
                         [self.get_curvature(c, "Original"), self.get_curvature(c), 
                          self.get_curvature(c, "Template (original)"),
                          self.get_curvature(c, "Template (final)")], 
                         ["True values", "Reconstructed values", 
                          "Template (original)", "Template (final)"],
                         labels = [i for i in range(self.n_obs)],
                         xlab = "Age (GW)", ylab = c)
            self.clean_curvature(c)
            self.clean_curvature(c, "Template (final)")
        
    def plot_regression_curvature(self, output_dir):
        for c in self.curvatures.keys():
            if self.get_curvature(c, "Regression"):
                ages = [a for a in range(int(min(self.ages)), int(max(self.ages)))]
                                
                scatter_plot(output_dir, "Regression_{}.png".format(c), ages, 
                            [self.get_curvature(c, "Regression")], [c],
                            xlab = "Age (GW)", ylab = c)
                
                self.clean_curvature(c, "Regression")
    
    def write(self, to_write, i = None):
        for c in self.curvatures.keys():
            to_write.append(["\nCurvature: {}".format(c)])

            if i is not None:
                to_write.append(["Subject curvature: {}".format(self.get_curvature(c, "Original")[i])])
                to_write.append(["Deformed template to subject curvature: {}".format(self.get_curvature(c, "Current")[i])])
            else:
                to_write.append(["Template curvature: {}".format(self.get_curvature(c, "Template (final)")[0])])


        return to_write