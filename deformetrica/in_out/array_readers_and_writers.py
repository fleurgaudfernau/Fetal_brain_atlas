import logging
logger = logging.getLogger(__name__)
import pyvista as pv
import os.path

import numpy as np


def write_2D_array(array, output_dir, name, fmt='%f'):
    """
    Assuming 2-dim array here e.g. control points
    save_name = os.path.join(Settings().output_dir, name)
    np.savetxt(save_name, array)
    """
    save_name = os.path.join(output_dir, name)
    if len(array.shape) == 0:
        array = array.reshape(1,)
    np.savetxt(save_name, array, fmt=fmt)


def write_3D_array(array, output_dir, name):
    """
    Saving an array has dim (numsubjects, numcps, dimension), using deformetrica format
    """
    s = array.shape
    if len(s) == 2:
        array = np.array([array])
    save_name = os.path.join(output_dir, name)
    with open(save_name, "w") as f:
        try:
            f.write(str(len(array)) + " " + str(len(array[0])) + " " + str(len(array[0, 0])) + "\n")
        except:
            pass
        for elt in array:
            f.write("\n")
            for elt1 in elt:
                for elt2 in elt1:
                    f.write(str(elt2) + " ")
                f.write("\n")

def concatenate_for_paraview(array_momenta, array_cp, output_dir, name, **kwargs):
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
        
        save_name = os.path.join(output_dir, name).replace(".vtk", "_sujet_{}.vtk".format(sujet))
        polydata.save(save_name, binary=False)


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
    save_name = os.path.join(output_dir, name)
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
        output_list = []
        subject_list = []
        for line in f:
            if not line == '\n':
                subject_list.append([float(x) for x in line.split()])
            else:
                output_list.append(subject_list)
                subject_list = []
        if not line == '\n':
            output_list.append(subject_list)
        return output_list

def write_3D_list(list, output_dir, name):
    """
    Saving a list of list of list.
    """
    save_name = os.path.join(output_dir, name)
    with open(save_name, "w") as f:
        for elt_i in list:
            for elt_i_j in elt_i:
                for elt_i_j_k in elt_i_j:
                    f.write(str(elt_i_j_k) + " ")
                if len(elt_i_j) > 1: f.write("\n")
            f.write("\n\n")

def flatten_3D_list(list3):
    out = []
    for list2 in list3:
        for list1 in list2:
            for elt in list1:
                out.append(elt)
    return out


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


def read_2D_array(name):
    """
    Assuming 2-dim array here e.g. control points
    """
    return np.loadtxt(name)
