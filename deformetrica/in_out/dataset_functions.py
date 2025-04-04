import logging
import warnings
import torch
from ..core import default
from copy import deepcopy
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from math import ceil, floor, trunc, prod
from ..support.kernels import factory
from ..support import kernels as kernel_factory 
from ..support.utilities.tools import gaussian_kernel
from ..core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from ..core.observations.datasets.longitudinal_dataset import LongitudinalDataset
from ..core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ..in_out.deformable_object_reader import ObjectReader, object_type
# from ..support.utilities.general_settings import Settings

logger = logging.getLogger(__name__)

def create_dataset(visit_ages=None, filenames=None, ids=None, interpolation = "", **kwargs):
    """
    Creates a longitudinal dataset object from xml parameters. 

    template_spec = ['object_id':]
    objects_dataset = [ [ [object a, object b for obs 0], [object a, object b for obs 1] ] ... [object list of a subject] ]
    """
    objects_dataset = []

    if filenames is not None:
        for i, subject in enumerate(filenames): #for each subject
            subject_objects = []
            for j, observations in enumerate(subject): #for each observation of i
                object_list = [ ObjectReader().create_object(observation, interpolation) \
                                for observation in observations.values() ]
                subject_objects.append(DeformableMultiObject(object_list))

            objects_dataset.append(subject_objects)

    return LongitudinalDataset(ids, visit_ages, objects_dataset)

def make_dataset_timeseries(dataset_specifications):
    # visit ages [[1], [2] ...] -> [[1, 2]]
    # filenames : [[{'obj_1':...}], [{'img':...}]] -> [[{'img':...}, {'img':...}]]
    # id : [1, 2, 3] -> [1]
    new_dataset_spec = deepcopy(dataset_specifications)
    new_dataset_spec["filenames"] = [sum(dataset_specifications["filenames"], [])]
    new_dataset_spec["visit_ages"] = [sum(dataset_specifications["visit_ages"], [])]
    new_dataset_spec["ids"] = [dataset_specifications["ids"][0]]

    return new_dataset_spec

def kernel_selection(dataset_spec, time, limit = 0.01):
    visit_ages = dataset_spec['visit_ages']
    weights = [ gaussian_kernel(time, age[0]) for age in visit_ages ]
    total_weights = np.sum(weights)
    weights_ = [w / total_weights for w in weights]
    
    selection = [ i for i, w in enumerate(weights_) if w > limit ] # 1% contribution
    weights = [round(weights[i], 2) for i in selection]
    weights_ = [round(weights_[i], 2) for i in selection]

    return selection, weights, weights_

def filter_dataset(dataset_spec, selection):
    new_dataset_spec = deepcopy(dataset_spec)   
    visit_ages = dataset_spec['visit_ages']     
    new_dataset_spec['visit_ages'] = [[visit_ages[i][0]] for i in selection]
    new_dataset_spec['ids'] = [dataset_spec['ids'][i] for i in selection]
    new_dataset_spec['filenames'] = [dataset_spec['filenames'][i] for i in selection]

    new_dataset_spec['n_subjects'] = len(new_dataset_spec['ids'])
    new_dataset_spec['n_observations'] = sum(len(v) for s in new_dataset_spec['filenames'] for v in s)
    return new_dataset_spec

def mean_object(dataset_spec, template_spec, weights):            
    if ".nii" in dataset_spec['filenames'][0][0]:

        data_list = [ nib.load(f[0]) for f in dataset_spec['filenames']]
        mean = np.zeros((data_list[0].get_fdata().shape))
        for i, f in enumerate(data_list):
            mean += f.get_fdata() * (weights[i]/total_weights)
        image_new = nib.nifti1.Nifti1Image(mean, data_list[0].affine, data_list[0].header)
        output_image = template_spec.replace("mean", "mean_age_{}".format(time))
        nib.save(image_new, output_image)   
        template_spec["Object_1"]["filename"] = output_image    

    else:
        #i = weights.index(max(weights))
        i = np.array(weights).argsort()[-2]
        template_spec["Object_1"]["filename"] = dataset_spec['filenames'][i][0]["Object_1"]
    
    return template_spec 

def id(dataset_specifications, i):
    return dataset_specifications["ids"][i]

def dataset_for_registration(subject, age, id):
    return {'filenames' :  [subject], "visit_ages" : [[age]], 'ids': [id]}

def age(dataset_specifications, i):
    return dataset_specifications['visit_ages'][i][0]

def mini(dataset_specifications):
    if isinstance(dataset_specifications["visit_ages"][0], list):
        return trunc(min(sum(dataset_specifications["visit_ages"], [])))
    
    return trunc(min(dataset_specifications["visit_ages"]))

def maxi(dataset_specifications):
    if isinstance(dataset_specifications["visit_ages"][0], list):
        return ceil(max(sum(dataset_specifications["visit_ages"], [])))
    
    return ceil(max(dataset_specifications["visit_ages"]))

def ages_histogram(dataset_specifications, path):

    if not op.exists(op.join(path, "ages_histogram.png")):
        if isinstance(dataset_specifications["visit_ages"][0], list):
            ages = sum(dataset_specifications["visit_ages"], [])
        else:
            ages = dataset_specifications["visit_ages"]
        
        fig, ax = plt.subplots(figsize=(4,3))

        # the histogram of the data
        bins = np.arange(int(min(ages)), int(min(ages)) + int(max(ages)) - int(min(ages)) + 1) #decalage vers gauche
        n, bins, patches = ax.hist(ages, bins = bins, rwidth = 0.8)

        # add a 'best fit' line
        ax.set_xlabel('Gestational age')
        ax.set_ylabel('Number of subjects')
        ticks = [k for k in range(int(min(ages)), int(min(ages)) + int(max(ages)) - int(min(ages)) + 1)]
        ax.set_xticks(ticks)
        fig.tight_layout()

        # Tweak spacing to prevent clipping of ylabel
        plt.savefig(op.join(path, "ages_histogram.png"))

def template_metadata(template_spec):
    """
    Creates a longitudinal dataset object from xml parameters.
    """   
    objects_list = [ ObjectReader().create_object( obj['filename'])
                    for obj in template_spec.values() ]
    
    objects_norm = [ _get_object_norm(obj, object_type(obj['filename']).lower())\
                     for obj in template_spec.values() ]

    objects_noise_variance = [-1.0 if obj['noise_std'] < 0 else obj['noise_std'] ** 2\
                                for obj in template_spec.values() ]

    objects_norm_kernels = [ factory(kernel_width = obj['kernel_width']) \
                            if object_norm in ['current', 'varifold'] \
                            else kernel_factory.Type.NO_KERNEL\
                            for obj, object_norm in zip(template_spec.values(), objects_norm) ]

    for i, obj in enumerate(template_spec.values()):
        logger.info("Attachment function: {}".format(objects_norm[i]))
        if objects_norm[i] in ['current', 'varifold']:
            logger.info("Attachment kernel width: {}".format(obj['kernel_width']))

        obj_type = object_type(obj['filename']).lower()

        if obj_type == 'image' and 'downsampling_factor' in list(obj.keys()):
            objects_list[-1].downsampling_factor = obj['downsampling_factor']
        if obj_type == 'image' and 'interpolation' in list(obj.keys()): #ajout fg
            objects_list[-1].interpolation = obj['interpolation']

    objects_attachment = MultiObjectAttachment(objects_norm, objects_norm_kernels)

    return objects_list, objects_noise_variance, objects_attachment

def compute_noise_dimension(template, objects_attachment):
    """
    Compute the dimension of the spaces where the norm are computed, for each object.
    """
    assert len(template.object_list) == len(objects_attachment.types)
    assert len(template.object_list) == len(objects_attachment.kernels)

    objects_noise_dimension = []
    dimension = template.object_list[0].dimension

    for k, (obj, attachment_type, kernel) in enumerate(zip(template.object_list, 
                    objects_attachment.types, objects_attachment.kernels)):

        if attachment_type in ['current', 'varifold']:
            noise_dimension = dimension * \
                    int(prod(floor((template.bounding_box[d, 1] - template.bounding_box[d, 0]) \
                    / kernel.kernel_width + 1) for d in range(dimension)))

        elif attachment_type == 'landmark':
            noise_dimension = dimension * obj.points.shape[0]

        elif attachment_type == 'L2':
            noise_dimension = obj.intensities.size
        
        objects_noise_dimension.append(noise_dimension)

    if objects_name is not None:
        logger.info('>> Objects noise dimension:')
        for (object_name, object_noise_dimension) in zip(objects_name, objects_noise_dimension):
            logger.info('\t\t[ %s ]\t%d' % (object_name, int(object_noise_dimension)))

    return objects_noise_dimension

def _get_object_norm(object, object_type):
    """
    object is a dictionary containing the deformable object properties.
    Here we make sure it is properly set, and deduce the right norm to use.
    """
    if object_type == 'SurfaceMesh'.lower():
        try:
            object_norm = object['attachment_type'].lower()
            assert object_norm in ['varifold', 'current', 'landmark']

        except KeyError as e:
            object_norm = 'varifold'

    elif object_type == 'Image'.lower():
        object_norm = 'L2'

    else:
        assert False, "Unknown object type {e}".format(e=type)

    return object_norm
