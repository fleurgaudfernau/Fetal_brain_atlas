import vtk
import os.path as op
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import matplotlib.patches as mpatches
import matplotlib.colors as cm
from copy import deepcopy

class PlotTracker:
    def __init__(self):
        self.plot = {}

    def initialize_plot_entry(self, key, template=None):
        template = template or {"v": [], "condition": True, "ylab": [""], "plots": [], "iter": []}
        self.plot[key] = deepcopy(template)

    def initialize_values(self, k, v = {}):
        self.plot[k]["v"] = v
    
    def set_condition(self, k, v = False):
        self.plot[k]["condition"] = v

    def set_ylab(self, k, v):
        self.plot[k]["ylab"] = list(v) if not isinstance(v, list) else v

    def add_iter(self, k, iteration):
        if iteration not in self.plot[k]["iter"]:
            self.plot[k]["iter"].append(iteration)
            
    def set_value(self, key, value, index=None):
        if index is not None:
            self.plot[key]["v"][index] = value
        else:
            self.plot[key]["v"] = value

    def add_value(self, key, value, index=None):
        if index is not None:
            self.plot[key]["v"][index].append(value)
        else:
            self.plot[key]["v"].append(value)
        
    def get_values(self, key, index=None):
        values = self.plot[key]["v"]
        if index is not None:
            return deepcopy(values[index])
        if isinstance(values, dict):
            return deepcopy(list(values.values()))
        return deepcopy(values)

    def set_to_plot(self):
        for key in self.plot.keys():
            self.plot[key]["plots"] = [v for v in self.get_values(key)]
    
    def to_plot(self, k):
        return self.plot[k]["condition"]

#####################################################################################
    
def scatter_plot(outputdir, filename, x_list, y_list, y_labels = None, labels = {}, colors = {}, 
                 xlab = "", ylab = "", lim = "", xlim = None, ylim = None, overwrite = True):
    """
    x: a list = the x data
    y_list : a list of lists : the y data to plot
    y_labels: a list: the label for each y data
    labels: the label to annotate each point
    """
    filename = op.join(outputdir, filename)

    if not overwrite and op.exists(filename):
        return
    
    if type(y_list[0]) != list:
        y_list = [y_list]
    
    if type(x_list[0]) != list:
        x_list = [x_list]
    
    if len(x_list) != len(y_list):
        x_list = x_list * len(y_list)
    
    if bool(labels) and type(labels[0]) != list:
        labels = [labels]
    
    if bool(labels) and len(labels) != len(y_list):
        labels = labels * len(y_list)
            
    colors = ["black", "blue", "purple", "pink", "red", "orange", "yellow", "green"]
    
    fig, ax = plt.subplots()
    for i, (x, y, c) in enumerate(zip(x_list, y_list, colors)):
        for obs in range(len(y)):
            ax.scatter(x[obs], y[obs], marker='s', c = c)

            if bool(labels):
                ax.annotate(labels[i][obs], (x[obs], y[obs]), color = c)

    if bool(y_labels):
        handles = []
        for ylabel, c in zip(y_labels, colors):
            handles.append(mpatches.Patch(color=c, label=ylabel))
        ax.legend(handles=handles)

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    elif lim == "sim":
        l = max([abs(max(sum(y_list, []), key=abs)), abs(max(sum(x_list, []), key=abs))]) + 0.1
        ax.set_ylim([- l, l])
        ax.set_xlim([- l, l])

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    fig.savefig(filename)
    plt.close()

def plot_sources(output_dir, sources, ages, overwrite = True, xlim = None, ylim = None):
    
    for s1, s2 in combinations(range(sources.shape[1]), r=2):                
        # Group data by age range:
        x_list, y_list, labels, y_labels = [[0]] * 8, [[0]] * 8, [[0]] * 8, [0] * 8
        for i, a in enumerate(range(22, 38, 2)):
            x, y, lab = [], [], []
            for obs in range(sources.shape[0]):
                if ages[obs] < a + 2 and ages[obs] >= a:
                    x.append(sources[obs, s1])
                    y.append(sources[obs, s2])
                    lab.append(obs)
            x_list[i], y_list[i], labels[i] = x, y, lab
            y_labels[i] = "{}-{}".format(a, a+2)
        
        name = "Sources__{}_{}.png".format(s1, s2)
        if xlim is not None:
            name = name.replace(".png", "_.png")
        
        scatter_plot(output_dir, name, x_list, y_list, 
                        y_labels = y_labels, labels = labels,
                        xlab = "source_{}".format(s1), ylab = "source_{}".format(s2),
                        lim = "sim", xlim = xlim, ylim = ylim)

def simple_plot(output_dir, name, x, y, xlabel):
    colors = list(cm.TABLEAU_COLORS.values())*10
    plt.plot(x, y, color = colors[0])
    plt.xlabel(xlabel)
    plt.ylim([0, max(y) + 0.1])
    plt.xlim([x[0], x[-1]])
    plt.savefig(op.join(output_dir, name))
    plt.close()

def simple_plot_error_bars(output_dir, name, x, y, sigma,
                           labels = [], xlabel = ""):
    colors = list(cm.TABLEAU_COLORS.values())*10

    if type(y[0]) != list:
        y = [y]

    for l, li, si in enumerate(zip(y, sigma)):
        #plt.plot(x, li, color = colors[l], label = labels[l])
        plt.errorbar(x, li, si, marker='^', label = labels[l], color = colors[l])

    plt.xlabel(xlabel)
    plt.ylim([0, max(y) + 0.1])
    plt.xlim([x[0], x[-1]])
    plt.savefig(op.join(output_dir, name))
    plt.close()

def plot_value_evolution(output_dir, name, plots, iterations, labels, 
                         vertical_lines = [], vertical_labels = []):
    colors = list(cm.TABLEAU_COLORS.values())*10
    my_colors = ["red", "blue", "green", "orange", "purple"]

    try:
        plots = [plots] if type(plots[0]) != list else plots

        for l, li in enumerate(plots):
            plt.plot(iterations, li, color = colors[l], label = labels[l])

        plt.xlabel('Iterations')
        plt.ylim([0, max(sum(plots, [])) + 0.1])
        plt.xlim([0, iterations[-1]])
        
        # vertical lines indicating CTF steps
        for iters, lab, col in zip(vertical_lines, vertical_labels, my_colors):        
            for j in iters: 
                if j:
                    plt.axvline(x = j, color = col, label = lab)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        plt.tight_layout()
        plt.legend(by_label.values(), by_label.keys(), loc = "best")
        plt.savefig(op.join(output_dir, name + ".png"))
        plt.close()
    
    except Exception as e:
            print("\nProblem plotting {}".format(name))
            print("Error {}".format(e))