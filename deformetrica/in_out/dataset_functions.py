import logging
import warnings
import torch
from copy import deepcopy
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from math import ceil, floor, trunc, prod
from ..support import kernels as kernel_factory
from ..core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from ..core import default
from ..core.observations.datasets.longitudinal_dataset import LongitudinalDataset
from ..core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ..in_out.deformable_object_reader import DeformableObjectReader, object_type
from ..support.utilities.general_settings import Settings

logger = logging.getLogger(__name__)

def create_dataset(visit_ages=None, dataset_filenames=None, subject_ids=None, 
                    interpolation = "linear", kernel_width = None, **kwargs):
    """
    Creates a longitudinal dataset object from xml parameters. 

    template_spec = ['object_id':]
    objects_dataset = [ [ [object a, object b for obs 0], [object a, object b for obs 1] ] ... [object list of a subject] ]
    """
    objects_dataset = []

    if dataset_filenames is not None:
        for i, subject in enumerate(dataset_filenames): #for each subject
            subject_objects = []
            for j, observations in enumerate(subject): #for each observation of i
                object_list = [ DeformableObjectReader().create_object(observation, interpolation, kernel_width=kernel_width) \
                                for observation in observations.values() ]
                subject_objects.append(DeformableMultiObject(object_list))

            objects_dataset.append(subject_objects)

    return LongitudinalDataset(subject_ids, visit_ages, objects_dataset)

def make_dataset_timeseries(dataset_specifications):
    # visit ages [[1], [2] ...] -> [[1, 2]] / filenames : [[{'img':...}], [{'img':...}]] -> [[{'img':...}, {'img':...}]]
    # id : [1, 2, 3] -> [1]

    new_dataset_spec = deepcopy(dataset_specifications)
    new_dataset_spec["dataset_filenames"] = [sum(dataset_specifications["dataset_filenames"], [])]
    new_dataset_spec["visit_ages"] = [sum(dataset_specifications["visit_ages"], [])]
    new_dataset_spec["subject_ids"] = [dataset_specifications["subject_ids"][0]]

    return new_dataset_spec

def filter_dataset(dataset_specifications, age_limit):
    new_dataset_spec = {k:[] for k in dataset_specifications.keys()}
    new_dataset_spec["subject_ids"] = dataset_specifications["subject_ids"]

    for age, name in zip(dataset_specifications["visit_ages"][0], 
                       dataset_specifications["dataset_filenames"][0]):
        if age < age_limit:
            new_dataset_spec["visit_ages"].append(age)
            new_dataset_spec["dataset_filenames"].append(name)

    new_dataset_spec["visit_ages"] = [new_dataset_spec["visit_ages"]]
    new_dataset_spec["dataset_filenames"] = [new_dataset_spec["dataset_filenames"]]

    return new_dataset_spec

def id(dataset_specifications, i):
    return dataset_specifications["subject_ids"][i]

def dataset_for_registration(subject, age, id):
    return {'dataset_filenames' :  [subject], "visit_ages" : [[age]], 'subject_ids': [id]}

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
        #ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

        # Tweak spacing to prevent clipping of ylabel
        plt.savefig(op.join(path, "ages_histogram.png"))

def extension(filename: str):
    """
    Extract file root name and extension form known extension list
    :param filename:    filename to extract extension from
    :return:    tuple containing filename root and extension
    """
    known_extensions = ['.png', '.nii', '.nii.gz', '.pny', '.vtk', '.stl']

    for extension in known_extensions:
        if filename.endswith(extension):
            return extension

    raise RuntimeError('Unknown extension for file %s' % (filename,))

def create_template_metadata(template_spec):
    """
    Creates a longitudinal dataset object from xml parameters.
    """   
    objects_list = [ DeformableObjectReader().create_object(
                    obj['filename'], kernel_width=obj['kernel_width'])
                    for obj in template_spec.values() ]
    
    objects_extensions = [ extension(obj['filename']) for obj in template_spec.values() ]

    objects_norm = [ _get_object_norm(obj, object_type(obj['filename']).lower())\
                     for obj in template_spec.values() ]

    objects_noise_variance = [-1.0 if obj['noise_std'] < 0 else obj['noise_std'] ** 2\
                                for obj in template_spec.values() ]

    objects_norm_kernels = [ kernel_factory.factory(kernel_width=obj['kernel_width']) \
                            if object_norm in ['current', 'varifold']
                            else kernel_factory.Type.NO_KERNEL\
                            for obj, object_norm in zip(template_spec.values(), objects_norm) ]

    for obj in template_spec.values():
        type = object_type(obj['filename']).lower()

        if type == 'image' and 'downsampling_factor' in list(obj.keys()):
            objects_list[-1].downsampling_factor = obj['downsampling_factor']
        if type == 'image' and 'interpolation' in list(obj.keys()): #ajout fg
            objects_list[-1].interpolation = obj['interpolation']

    objects_attachment = MultiObjectAttachment(objects_norm, objects_norm_kernels)

    return objects_list, objects_extensions, objects_noise_variance, objects_attachment

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

        else:
            raise RuntimeError('Unknown noise dimension for the attachment type: {}'.format(objects_attachment.types[k]))
        
        objects_noise_dimension.append(noise_dimension)

    if objects_name is not None:
        logger.info('>> Objects noise dimension:')
        for (object_name, object_noise_dimension) in zip(objects_name, objects_noise_dimension):
            logger.info('\t\t[ %s ]\t%d' % (object_name, int(object_noise_dimension)))

    return objects_noise_dimension

def _get_object_norm(object, type):
    """
    object is a dictionary containing the deformable object properties.
    Here we make sure it is properly set, and deduce the right norm to use.
    """

    if type == 'SurfaceMesh'.lower():
        try:
            object_norm = object['attachment_type'].lower()
            assert object_norm in ['Varifold'.lower(), 'Current'.lower(), 'Landmark'.lower()]

        except KeyError as e:
            warnings.warn("Watch out, I did not get a distance type for the object")
            object_norm = 'none'

    elif type == 'Image'.lower():
        object_norm = 'L2'
        if 'attachment_type' in object.keys() and not object['attachment_type'].lower() == 'L2'.lower():
            msg = 'Only the "L2" attachment is available for image objects so far. ' \
                  'Overwriting the  invalid attachment: "%s"' % object['attachment_type']
            warnings.warn(msg)

    else:
        assert False, "Unknown object type {e}".format(e=type)

    return object_norm
