import os
import os.path as op
from ..core import GpuMode
import matplotlib.pyplot as plt
gpu_mode = GpuMode.KERNEL

class PlotResiduals():
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.f, self.ax = dict(), dict()
        for k in range(7):
            self.f[k], self.ax[k] = plt.subplots()

    def plot(self, residuals, i, time, age):

        for j, (k, v) in enumerate(residuals.items()):

            path = op.join(self.output_dir, "Subject_{}_age_{}_{}.png".format(i, age, k))
            
            if not op.exists(path):
                if i > 0 and i % 5 == 0:
                    print("writing", i)
                    self.f[j].savefig(path)
                    plt.close()
                    self.f[j], self.ax[j] = plt.subplots()
                    
                self.ax[j].plot(list(v.keys()), list(v.values()), label = str(i))
                self.ax[j].scatter(x=int(time), y=v[int(time)])
                
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                self.ax[j].legend(by_label.values(), by_label.keys())
                plt.gca().set_ylim(bottom=0)
                if k == "distance_to_flow": self.ax[j].set_ylim(0, 1e6)
    
    def end(self, residuals, i, age):
        for j, (k, v) in enumerate(residuals.items()):
            path = op.join(self.output_dir, "Subject_{}_age_{}_{}.png".format(i, age, k))
            self.f[j].savefig(path)
            plt.close()

                

def make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def complete(dico, dico_):
    for k in set(dico.keys()).difference(dico_.keys()):
        dico_[k] = dico[k]
    
    return dico_

def options_for_registration(dico):
    dico_ = {"save_every_n_iters" : 100, "print_every_n_iters" : 100,
            "multiscale_meshes" : False, "multiscale_momenta" : False,
            "load_state_file" : False}
    
    return complete(dico, dico_)
            
    

