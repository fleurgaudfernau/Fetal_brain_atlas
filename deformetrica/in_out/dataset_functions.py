import logging
import warnings
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import os.path as op
import numpy as np
from torch.autograd import Variable
from math import ceil, floor, trunc

from ..support import kernels as kernel_factory
from ..core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from ..core import default
from ..core.observations.datasets.longitudinal_dataset import LongitudinalDataset
from ..core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ..in_out.deformable_object_reader import DeformableObjectReader
from ..support.utilities.general_settings import Settings

logger = logging.getLogger(__name__)


def create_dataset(template_specifications, visit_ages=None, dataset_filenames=None, subject_ids=None, 
                    dimension=None, interpolation = "linear"):
    """
    Creates a longitudinal dataset object from xml parameters. 

    template_specifications = ['object_id':]
    deformable_objects_dataset = [ [ [object a, object b for obs 0], [object a, object b for obs 1] ] ... [object list of a subject] ]
    """
    deformable_objects_dataset = []

    if dataset_filenames is not None:
        for i in range(len(dataset_filenames)): #for each subject
            deformable_objects_subject = []
            for j in range(len(dataset_filenames[i])): #for each observation of i
                object_list = []
                reader = DeformableObjectReader()
                for object_id in template_specifications.keys(): #for each object of i eg right hippocampus
                    if object_id not in dataset_filenames[i][j]:
                        raise RuntimeError('The template object with id ' + object_id + ' is not found for the visit '
                                           + str(j) + ' of subject ' + str(i) + '. Check the dataset xml.')
                    else:
                        object_type = template_specifications[object_id]['deformable_object_type']
                        #dataset_filenames[i][j][object_id]: path to file
                        object = template_specifications[object_id]
                        object_list.append(reader.create_object(dataset_filenames[i][j][object_id], object_type,
                                                                dimension, interpolation, 
                                                                kernel_width=object['kernel_width']))
                deformable_objects_subject.append(DeformableMultiObject(object_list))
            deformable_objects_dataset.append(deformable_objects_subject)

    longitudinal_dataset = LongitudinalDataset(
        subject_ids, times=visit_ages, deformable_objects=deformable_objects_dataset)

    return longitudinal_dataset

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


def dataset_for_registration(subject, age, id):
    return {'dataset_filenames' :  [subject], "visit_ages" : [[age]],
                'subject_ids': [id]}

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


def id(dataset_specifications, i):
    return dataset_specifications["subject_ids"][i]

def create_scalar_dataset(group, observations, timepoints):
    """
    Builds a dataset from the given data.
    """

    times = []
    subject_ids = []
    scalars = []

    for subject_id in group:
        if subject_id not in subject_ids:
            subject_ids.append(subject_id)
            times_subject = []
            scalars_subject = []
            for i in range(len(observations)):
                if group[i] == subject_id:
                    times_subject.append(timepoints[i])
                    scalars_subject.append(observations[i])
            assert len(times_subject) > 0, subject_id
            assert len(times_subject) == len(scalars_subject)
            times.append(np.array(times_subject))
            scalars.append(Variable(torch.from_numpy(np.array(scalars_subject)).type(Settings().tensor_scalar_type)))

    longitudinal_dataset = LongitudinalDataset()
    longitudinal_dataset.times = times
    longitudinal_dataset.subject_ids = subject_ids
    longitudinal_dataset.deformable_objects = scalars
    longitudinal_dataset.number_of_subjects = len(subject_ids)
    longitudinal_dataset.total_number_of_observations = len(timepoints)

    return longitudinal_dataset


def create_image_dataset(group, observations, timepoints):
    times = []
    subject_ids = []
    images = []

    for subject_id in group:
        if subject_id not in subject_ids:
            subject_ids.append(subject_id)
            times_subject = []
            images_subject = []
            for i in range(len(observations)):
                if group[i] == subject_id:
                    times_subject.append(timepoints[i])
                    images_subject.append(observations[i])
            assert len(times_subject) > 0, subject_id
            assert len(times_subject) == len(images_subject)
            times.append(np.array(times_subject))
            images.append(images_subject)

    longitudinal_dataset = LongitudinalDataset()
    longitudinal_dataset.times = times
    longitudinal_dataset.subject_ids = subject_ids
    longitudinal_dataset.deformable_objects = images
    longitudinal_dataset.number_of_subjects = len(subject_ids)
    longitudinal_dataset.total_number_of_observations = len(timepoints)
    longitudinal_dataset.check_image_shapes()
    longitudinal_dataset.order_observations()

    return longitudinal_dataset


def read_and_create_scalar_dataset(xml_parameters):
    """
    Read scalar observations e.g. from cognitive scores, and builds a dataset.
    """
    group = np.loadtxt(xml_parameters.group_file, delimiter=',', dtype=str)
    observations = np.loadtxt(xml_parameters.observations_file, delimiter=',')
    timepoints = np.loadtxt(xml_parameters.timepoints_file, delimiter=',')
    return create_scalar_dataset(group, observations, timepoints)


def read_and_create_image_dataset(dataset_filenames, visit_ages, subject_ids, template_specifications):
    """
    Builds a longitudinal dataset of images (non deformable images). Loads everything into memory. #TODO assert on the format of the images !
    """
    deformable_objects_dataset = []

    for i in range(len(dataset_filenames)):
        deformable_objects_subject = []
        for j in range(len(dataset_filenames[i])):
            for object_id in template_specifications.keys():
                if object_id not in dataset_filenames[i][j]:
                    raise RuntimeError('The template object with id ' + object_id + ' is not found for the visit '
                                       + str(j) + ' of subject ' + str(i) + '. Check the dataset xml.')
                else:
                    objectType = template_specifications[object_id]['deformable_object_type']
                    reader = DeformableObjectReader()
                    deformable_object_visit = reader.create_object(dataset_filenames[i][j][object_id], objectType)
                    deformable_object_visit.update()
            deformable_objects_subject.append(deformable_object_visit)
        if len(deformable_objects_subject) <= 1:
            msg = "I have only one observation for subject {}".format(str(i))
            warnings.warn(msg)
        deformable_objects_dataset.append(deformable_objects_subject)

    longitudinal_dataset = LongitudinalDataset()
    longitudinal_dataset.times = [np.array(elt) for elt in visit_ages]
    longitudinal_dataset.subject_ids = subject_ids
    longitudinal_dataset.deformable_objects = deformable_objects_dataset
    longitudinal_dataset.update()
    longitudinal_dataset.check_image_shapes()
    longitudinal_dataset.order_observations()

    return longitudinal_dataset


def split_filename(filename: str):
    """
    Extract file root name and extension form known extension list
    :param filename:    filename to extract extension from
    :return:    tuple containing filename root and extension
    """
    known_extensions = ['.png', '.nii', '.nii.gz', '.pny', '.vtk', '.stl']

    for extension in known_extensions:
        if filename.endswith(extension):
            return filename[:-len(extension)], extension

    raise RuntimeError('Unknown extension for file %s' % (filename,))


def create_template_metadata(template_specifications, dimension=None, gpu_mode=None):
    """
    Creates a longitudinal dataset object from xml parameters.
    """
    if gpu_mode is None:
        gpu_mode = default.gpu_mode
    elif gpu_mode == "GpuMode.NONE":
        gpu_mode = default.gpu_mode
    objects_list = []
    objects_name = []
    objects_noise_variance = []
    objects_name_extension = []
    objects_norm = []
    objects_norm_kernels = []

    for object_id, object in template_specifications.items():
        filename = object['filename']
        object_type = object['deformable_object_type'].lower()

        assert object_type in ['SurfaceMesh'.lower(), 'PolyLine'.lower(), 'PointCloud'.lower(), 'Landmark'.lower(),
                               'Image'.lower()], "Unknown object type."
        root, extension = split_filename(filename)
        reader = DeformableObjectReader()

        objects_list.append(reader.create_object(filename, object_type, dimension=dimension,
                            gpu_mode=gpu_mode,
                            kernel_width=object['kernel_width']))
        objects_name.append(object_id)
        objects_name_extension.append(extension)

        if object['noise_std'] < 0:
            objects_noise_variance.append(-1.0)
        else:
            objects_noise_variance.append(object['noise_std'] ** 2)

        object_norm = _get_norm_for_object(object, object_id) #=current

        objects_norm.append(object_norm)
        if object_norm in ['current', 'pointcloud', 'varifold']:
            objects_norm_kernels.append(kernel_factory.factory(
                                        gpu_mode=gpu_mode,
                                        kernel_width=object['kernel_width']))
        else:
            objects_norm_kernels.append(kernel_factory.Type.NO_KERNEL)

        # Optional grid downsampling parameter for image data.
        if object_type == 'image' and 'downsampling_factor' in list(object.keys()):
            objects_list[-1].downsampling_factor = object['downsampling_factor']
        if object_type == 'image' and 'interpolation' in list(object.keys()): #ajout fg
            objects_list[-1].interpolation = object['interpolation']
    multi_object_attachment = MultiObjectAttachment(objects_norm, objects_norm_kernels)

    return objects_list, objects_name, objects_name_extension, objects_noise_variance, multi_object_attachment

def create_mesh_attachements(template_specifications, gpu_mode=None):

    # ajout fg
    if gpu_mode is None:
        gpu_mode = default.gpu_mode
    elif gpu_mode == "GpuMode.NONE":
        gpu_mode = default.gpu_mode

    objects_norm = []
    mesh=False
    attachments = dict()

    user_kernel_width = int(max([o['kernel_width'] for o in template_specifications.values()]))

    #for kernel_width in range(user_kernel_width, user_kernel_width+5): #0 to  user 
    for kernel_width in np.arange(1, user_kernel_width + 2, 1):
        objects_norm_kernels = []

        for object_id, object in template_specifications.items():
            object_norm = _get_norm_for_object(object, object_id) #=current

            objects_norm.append(object_norm)
            if object_norm in ['current', 'pointcloud', 'varifold']:
                mesh = True
                objects_norm_kernels.append(kernel_factory.factory(
                    gpu_mode=gpu_mode, kernel_width=kernel_width))
        if mesh:
            attachments[kernel_width] = MultiObjectAttachment(objects_norm, objects_norm_kernels)
        else:
            return None
            
    return attachments

def compute_noise_dimension(template, multi_object_attachment, dimension, objects_name=None):
    """
    Compute the dimension of the spaces where the norm are computed, for each object.
    """
    assert len(template.object_list) == len(multi_object_attachment.attachment_types)
    assert len(template.object_list) == len(multi_object_attachment.kernels)

    objects_noise_dimension = []
    for k in range(len(template.object_list)):

        if multi_object_attachment.attachment_types[k] in ['current', 'varifold', 'pointcloud']:
            noise_dimension = 1
            for d in range(dimension):
                length = template.bounding_box[d, 1] - template.bounding_box[d, 0]
                assert length >= 0
                noise_dimension *= floor(length / multi_object_attachment.kernels[k].kernel_width + 1.0)
            noise_dimension *= dimension

        elif multi_object_attachment.attachment_types[k] in ['landmark']:
            noise_dimension = dimension * template.object_list[k].points.shape[0]

        elif multi_object_attachment.attachment_types[k] in ['L2']:
            noise_dimension = template.object_list[k].intensities.size

        else:
            raise RuntimeError('Unknown noise dimension for the attachment type: '
                               + multi_object_attachment.attachment_types[k])

        objects_noise_dimension.append(noise_dimension)

    if objects_name is not None:
        logger.info('>> Objects noise dimension:')
        for (object_name, object_noise_dimension) in zip(objects_name, objects_noise_dimension):
            logger.info('\t\t[ %s ]\t%d' % (object_name, int(object_noise_dimension)))

    return objects_noise_dimension


def _get_norm_for_object(object, object_id):
    """
    object is a dictionary containing the deformable object properties.
    Here we make sure it is properly set, and deduce the right norm to use.
    """
    object_type = object['deformable_object_type'].lower()

    if object_type == 'SurfaceMesh'.lower() or object_type == 'PolyLine'.lower():
        try:
            object_norm = object['attachment_type'].lower()
            assert object_norm in ['Varifold'.lower(), 'Current'.lower(), 'Landmark'.lower()]

        except KeyError as e:
            msg = "Watch out, I did not get a distance type for the object {e}, Please make sure you are running " \
                  "shooting or a parallel transport, otherwise distances are required.".format(e=object_id)
            warnings.warn(msg)
            object_norm = 'none'

    elif object_type == 'PointCloud'.lower():
        object_norm = 'PointCloud'.lower()  # it's automatic for point cloud

    elif object_type == 'Landmark'.lower():
        object_norm = 'Landmark'.lower()

    elif object_type == 'Image'.lower():
        object_norm = 'L2'
        if 'attachment_type' in object.keys() and not object['attachment_type'].lower() == 'L2'.lower():
            msg = 'Only the "L2" attachment is available for image objects so far. ' \
                  'Overwriting the user-specified invalid attachment: "%s"' % object['attachment_type']
            warnings.warn(msg)

    else:
        assert False, "Unknown object type {e}".format(e=object_type)

    return object_norm
