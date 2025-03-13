import torch
import logging
logger = logging.getLogger(__name__)
import pyvista as pv
import os.path as op
import numpy as np
from ..support.utilities import detach

####################################################################################################################
### Specific writing functions
####################################################################################################################

def cp_name(name = "", time = "", age = ""):
    if len(str(time)) > 0:
        return "{}ControlPoints__tp_{}.txt".format(name, time)

    return "{}__ControlPoints.txt".format(name)

def momenta_name(name = "", time = "", age = ""):
    if len(str(time)) > 0 and len(str(age)) > 0:
        return '{}Momenta__tp_{}__age_{:.2f}.txt'.format(name, time, age)
    
    elif len(str(time)) > 0:
        return "{}Momenta__tp_{}.txt".format(name, time)
    
    return "{}__Estimated__Momenta.txt".format(name)

def source_name(name):
    return '{}__Estimated__Sources.txt'.format(name)

def mod_matrix_name(name):
    return "{}__Estimated__ModulationMatrix.txt".format(name)

def mod_matrix_flow_name(name, t, time):
    return name + '__PiecewiseGeodesicFlow__ModulationMatrix__tp_' + str(t) + ('__age_%.2f' % time) + '.txt'

def paraview_name(name, iteration = "", comp = ""):
    if name in ["KernelRegression",  "DeformableTemplate"]:
        return "{}__Estimated__Fusion_CP_Momenta_{}.vtk".format(name, iteration)
    
    elif name in ["GeodesicRegression", "BayesianGeodesicRegression"] and len(str(comp)) > 0:
        return "{}__Estimated__Fusion_CP_Momenta__component_{}_iter_{}.vtk".format(name, comp, iteration)
    
    elif name == "GeodesicRegression" and len(str(iteration)) > 0:
        return "{}__Estimated__Fusion_CP_Momenta__iter_{}.vtk".format(name, iteration)

    elif name == "GeodesicRegression":
        return "{}__Estimated__Fusion_CP_Momenta.vtk".format(name)
        
    else:
        return name

def reconstruction_name(name, subject_id = "", time = "", age = "", ext = ""):
    if name == "KernelRegression":
        return "{}__Reconstruction__subject_{}_age_{}".format(name, subject_id, age, ext)

    elif name == "DeformableTemplate":
        return '{}__Reconstruction__subject_{}{}'.format(name, subject_id, ext)

    elif name == "GeodesicRegression":
        return '{}__Reconstruction____tp_{}__age_{}'.format(name, time, age, ext)
    
    elif name == "BayesianGeodesicRegression":
        return '{}__Reconstruction__subject__{}__tp_{}__age_{}{}'.format(name, subject_id, time, age, ext)

def template_name(name, time = "", age = "", t0 = "", iteration = "", 
                ext = "", freeze_template = False):
    if name == "KernelRegression":
        return "{}__Estimated__Template_time_{}{}".format(name, time, ext)
    
    elif name in ["DeformableTemplate", "BayesianAtlas"]:
        if not freeze_template and len(iteration) > 0:
            return "{}__Estimated__Template_{}{}".format(name, iteration, ext)
        else:
            return "{}__Estimated__Template_{}".format(name, ext)

    elif name == "GeodesicRegression":
        if not freeze_template:   
            return '{}__Estimated__Template__tp_{}__age_{}{}'.format(name, time, t0, ext)
        else:
            return '{}__Fixed__Template__tp_{}__age_{}{}'.format(name, time, t0, ext)

    elif name == "BayesianGeodesicRegression":
        if not freeze_template:
            return '{}__Estimated__Template___tp_{}__age_{}{}'.format(name, time, age, ext)
        else:
            return '{}__Fixed__Template___tp_{}__age_{}{}'.format(name, time, age, ext)

def flow_name(name, t, time, comp = "", ext = ""):
    if len(str(comp)) > 0:
        return '{}component_{}_tp_{}__age_{:.2f}{}'.format(name, comp, t, time, ext)
    
    return '{}tp_{}__age_{}{}'.format(name, t, time, ext)

def space_shift_name(name, source, t, time, ext, backward = False):
    if backward:
        return "{}__IndependentComponent_{}__tp_{}__age_{}__BackwardExponentialFlow{}"\
            .format(name, source, t, time, ext)
        
    return "{}__IndependentComponent_{}__tp_{}__age_{}__ForwardExponentialFlow{}"\
            .format(name, source, t, time, ext)

def write_cp(cp, output_dir, name, time = "", age = ""):
    write_2D_array(cp, output_dir, cp_name(name, time, age))

def write_momenta(momenta, output_dir, name, time = "", age = ""):
    write_3D_array(momenta, output_dir, momenta_name(name, time, age))

def write_sources(sources, output_dir, name):
    write_2D_array(sources, output_dir, source_name(name))

def write_mod_matrix(mod_matrix, output_dir, name):
    write_2D_array(mod_matrix, output_dir, mod_matrix_name(name))

def write_mod_matrix_flow(modulation_matrix, output_dir, name, t, time):
    write_2D_array(modulation_matrix, output_dir, mod_matrix_flow_name(name, t, time))

####################################################################################################################
### Write arrays
####################################################################################################################

def write_2D_array(array, output_dir, name, fmt='%f'):
    """
    Assuming 2-dim array here e.g. control points
    save_name = op.join(Settings().output_dir, name)
    np.savetxt(save_name, array)
    """
    save_name = op.join(output_dir, name)
    if len(array.shape) == 0:
        array = array.reshape(1,)
    np.savetxt(save_name, array, fmt=fmt)

def write_3D_array(array, output_dir, name):
    """
    Saving an array has dim (numsubjects, numcps, dimension), using deformetrica format
    """
    array = detach(array)

    # s = array.shape
    # if len(s) == 2:
    #     array = np.array([array])
    # save_name = op.join(output_dir, name)
    # with open(save_name, "w") as f:
    #     try:
    #         f.write(str(len(array)) + " " + str(len(array[0])) + " " + str(len(array[0, 0])) + "\n")
    #     except:
    #         pass
    #     for elt in array:
    #         f.write("\n")
    #         for elt1 in elt:
    #             for elt2 in elt1:
    #                 f.write(str(elt2) + " ")
    #             f.write("\n")
    array = np.atleast_3d(array)  # Ensure array has at least 3 dimensions
    shape = array.shape 
    save_name = op.join(output_dir, name)

    with open(save_name, "w") as f:
        f.write(f"{shape[0]} {shape[1]} {shape[2]}\n") 
        for subject in array:
            f.write("\n")
            for cp in subject:
                f.write(" ".join(map(str, cp)) + "\n") 


def concatenate_for_paraview(array_momenta, array_cp, output_dir, name, iteration = "", comp = "",
                            **kwargs):
    
    name = paraview_name(name, iteration, comp)

    array_momenta = detach(array_momenta)
    array_cp = detach(array_cp)
    kwargs = {k : detach(v) for k,v in kwargs.items()}

    if len(array_momenta.shape) == 1:
        array_momenta = np.array([array_momenta])
    if len(array_momenta.shape) == 2:
        array_momenta = np.array([array_momenta])
    
    # another option faster
    if array_cp.shape[1] == 2:
        col = np.zeros((array_cp.shape[0], 1))
        array_cp = np.hstack((array_cp, col))
        col = np.zeros((array_momenta.shape[0], array_momenta.shape[1], 1))
        array_momenta = np.concatenate((array_momenta, col), axis = 2)

    for sujet in range(array_momenta.shape[0]):
        momenta = array_momenta[sujet,:,:]
        polydata = pv.PolyData(array_cp)
        polydata.point_data["Momenta"] = momenta
        
        for k, v in kwargs.items():
            if len(v.shape) == 3:
                polydata.point_data[str(k)] = v[sujet,:,:]
            else:
                polydata.point_data[str(k)] = v
        
        save_name = op.join(output_dir, name).replace(".vtk", "_sujet_{}".format(sujet)) + ".vtk"
        polydata.save(save_name, binary=False)

####################################################################################################################
### Write lists
####################################################################################################################

def read_2D_list(path):
    """
    Reading a list of list.
    """
    with open(path, "r") as f:
        output_list = [[float(x) for x in line.split()] for line in f]
    return output_list


def write_2D_list(input_list, output_dir, name):
    """
    Saving a list of list.
    """
    input_list = [detach(l) for l in input_list]

    save_name = op.join(output_dir, name)
    with open(save_name, "w") as f:
        for elt_i in input_list:
            try:
                for elt_i_j in elt_i:
                    f.write(str(elt_i_j) + " ")
            except:
                pass
            f.write("\n")

def read_3D_list(path):
    """
    Reading a list of list of list.
    """
    with open(path, "r") as f:
        data = f.read().strip()  # Read and strip whitespace
        data = data.split("\n\n")
    
    return [[[float(x) for x in line.split()] for line in subject.splitlines()] 
            for subject in data]


    # with open(path, "r") as f:
    #     output_list = []
    #     subject_list = []
    #     for line in f:
    #         if not line == '\n':
    #             subject_list.append([float(x) for x in line.split()])
    #         else:
    #             output_list.append(subject_list)
    #             subject_list = []
    #     if not line == '\n':
    #         output_list.append(subject_list)
    #     return output_list

def write_3D_list(list, output_dir, name):
    """
    Saving a list of list of list.
    """
    save_name = op.join(output_dir, name)
    with open(save_name, "w") as f:
        for elt_i in list:
            for elt_i_j in elt_i:
                for elt_i_j_k in elt_i_j:
                    f.write(str(elt_i_j_k) + " ")
                if len(elt_i_j) > 1: 
                    f.write("\n")
            f.write("\n\n")
    
    save_name = op.join(output_dir, name)

    # with open(save_name, "w") as f:
    #     for sublist2 in list3d:
    #         for sublist1 in sublist2:
    #             f.write(" ".join(map(str, sublist1)) + "\n")
    #         f.write("\n\n") 

def flatten_3D_list(list3):
    return [elt for list2 in list3 for list1 in list2 for elt in list1]

####################################################################################################################
### Read arrays
####################################################################################################################

def read_2D_array(name):
    """
    Assuming 2-dim array here e.g. control points
    """
    return np.loadtxt(name)

def read_3D_array(name):
    """
    Loads a file containing momenta, old deformetrica syntax assumed
    """
    try:
        with open(name, "r") as f:
            lines = f.readlines()
            line0 = [int(elt) for elt in lines[0].split()]
            nbSubjects, nbControlPoints, dimension = line0[0], line0[1], line0[2]
            momenta = np.zeros((nbSubjects, nbControlPoints, dimension))
            lines = lines[2:]
            for i in range(nbSubjects):
                for c in range(nbControlPoints):
                    foo = lines[c].split()
                    assert (len(foo) == dimension)
                    foo = [float(elt) for elt in foo]
                    momenta[i, c, :] = foo
                lines = lines[1 + nbControlPoints:]
        if momenta.shape[0] == 1:
            return momenta[0]
        else:
            return momenta

    except ValueError:
        return read_2D_array(name)


