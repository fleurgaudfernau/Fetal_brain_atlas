#!/usr/bin/env python
# -*- encoding: utf-8 -*-


"""

ShapeMI at MICCAI 2018
https://shapemi.github.io/


Benchmark CPU vs GPU on small (500 points) and large (5000 points) meshes.

"""


import os
import matplotlib.pyplot as plt
import numpy as np
import support.kernels as kernel_factory
import torch

from in_out.deformable_object_reader import DeformableObjectReader
from core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from support.utilities.general_settings import Settings

path_to_small_surface_mesh_1 = 'data/landmark/surface_mesh/hippocampus_500_cells_1.vtk'
path_to_small_surface_mesh_2 = 'data/landmark/surface_mesh/hippocampus_500_cells_2.vtk'
path_to_large_surface_mesh_1 = 'data/landmark/surface_mesh/hippocampus_5000_cells_1.vtk'
path_to_large_surface_mesh_2 = 'data/landmark/surface_mesh/hippocampus_5000_cells_2.vtk'


class ProfileAttachments:
    def __init__(self, kernel_type, kernel_width, kernel_device='CPU', full_cuda=False, data_size='small'):

        if full_cuda:
            Settings().tensor_scalar_type = torch.cuda.FloatTensor
        else:
            Settings().tensor_scalar_type = torch.FloatTensor

        self.multi_object_attachment = MultiObjectAttachment()
        self.kernel = kernel_factory.factory(kernel_type, kernel_width)
        self.kernel.kernel_device = kernel_device

        reader = DeformableObjectReader()

        if data_size == 'small':
            self.surface_mesh_1 = reader.create_object(path_to_small_surface_mesh_1, 'SurfaceMesh')
            self.surface_mesh_2 = reader.create_object(path_to_small_surface_mesh_2, 'SurfaceMesh')
            self.surface_mesh_1_points = Settings().tensor_scalar_type(self.surface_mesh_1.get_points())

        elif data_size == 'large':
            self.surface_mesh_1 = reader.create_object(path_to_large_surface_mesh_1, 'SurfaceMesh')
            self.surface_mesh_2 = reader.create_object(path_to_large_surface_mesh_2, 'SurfaceMesh')
            self.surface_mesh_1_points = Settings().tensor_scalar_type(self.surface_mesh_1.get_points())

    def current_attachment(self):
        self.multi_object_attachment._current_distance(
            self.surface_mesh_1_points, self.surface_mesh_1, self.surface_mesh_2, self.kernel)

    def varifold_attachment(self):
        self.multi_object_attachment._varifold_distance(
            self.surface_mesh_1_points, self.surface_mesh_1, self.surface_mesh_2, self.kernel)


class BenchRunner:
    def __init__(self, kernel, kernel_width, method_to_run):
        self.obj = ProfileAttachments(kernel[0], kernel_width, kernel[1], method_to_run[0])
        self.to_run = getattr(self.obj, method_to_run[1])

        # run once for warm-up: cuda pre-compile with keops
        self.run()
        # print('BenchRunner::__init()__ done')

    """ The method that is to be benched must reside within the run() method """
    def run(self):
        self.to_run()

        print('.', end='', flush=True)    # uncomment to show progression

    def __exit__(self):
        print('BenchRunner::__exit()__')


def build_setup():
    kernels = [('torch', 'CPU'), ('keops', 'CPU'), ('torch', 'GPU'), ('keops', 'GPU')]
    # kernels = [('torch', 'GPU'), ('keops', 'GPU')]
    # method_to_run = [('small', 'current_attachment'), ('small', 'varifold_attachment')]
    method_to_run = [('large', 'current_attachment'), ('large', 'varifold_attachment')]
    setups = []

    for k, m in [(k, m) for k in kernels for m in method_to_run]:
        bench_setup = '''
from __main__ import BenchRunner
import torch
bench = BenchRunner({kernel}, 1.0, {method_to_run})
'''.format(kernel=k, method_to_run=m)

        setups.append({'kernel': k, 'method_to_run': m, 'bench_setup': bench_setup})
    return setups, kernels, method_to_run


if __name__ == "__main__":
    import timeit

    results = []

    build_setup, kernels, method_to_run = build_setup()

    # prepare and run bench
    for setup in build_setup:
        print('running setup ' + str(setup))

        res = {}
        res['setup'] = setup
        res['data'] = timeit.repeat("bench.run()", number=10, repeat=3, setup=setup['bench_setup'])
        res['min'] = min(res['data'])
        res['max'] = max(res['data'])

        print('')
        print(res['data'])
        results.append(res)

    fig, ax = plt.subplots()
    # plt.ylim(ymin=0)
    # ax.set_yscale('log')

    index = np.arange(len(method_to_run))
    bar_width = 0.2
    opacity = 0.4

    # extract data from raw data and add to plot
    i = 0
    for k in [(k) for k in kernels]:

        extracted_data = [r['max'] for r in results
                          if r['setup']['kernel'] == k]

        assert(len(extracted_data) > 0)
        assert(len(extracted_data) == len(index))

        ax.bar(index + bar_width * i, extracted_data, bar_width, alpha=opacity, label=k[0] + ':' + k[1])
        i = i+1

    # bar1 = ax.bar(index, cpu_res, bar_width, alpha=0.4, color='b', label='cpu')
    # bar2 = ax.bar(index + bar_width, cuda_res, bar_width, alpha=0.4, color='g', label='cuda')

    ax.set_xlabel('TODO')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('TODO')
    ax.set_xticks(index + bar_width * ((len(kernels))/2) - bar_width/2)
    ax.set_xticklabels([r['setup']['method_to_run'][1] for r in results])
    ax.legend()

    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(45)

    fig.tight_layout()

    plt.show()
