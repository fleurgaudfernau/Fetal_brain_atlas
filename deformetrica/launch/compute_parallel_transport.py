import math
import torch
from copy import deepcopy
import os.path as op
from ..core import default
from ..core.model_tools.deformations.exponential import Exponential
from ..core.model_tools.deformations.geodesic import Geodesic
from ..core.model_tools.deformations.piecewise_geodesic import PiecewiseGeodesic
from ..core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from ..in_out.array_readers_and_writers import *
from ..in_out.dataset_functions import template_metadata, create_dataset
from ..support.utilities import n_time_points, move_data, get_best_device
from ..support.kernels import factory
from ..core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from ..support.utilities.vtk_tools import screenshot_vtk

import logging
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")


class ParallelTransport():
    def __init__(self, template_specifications, tmin, tmax, time_concentration, 
                t0, start_time, target_time, output_dir, flow_path, initial_cp = None):

        self.data = {}
        
        self.tmin = tmin
        self.tmax = tmax
        self.t0 = t0
        self.start_time = start_time if start_time is not None else t0
        self.target_time = target_time if target_time is not None else tmax

        self.time_concentration = time_concentration
        self.initial_cp = initial_cp

        self.output_dir = output_dir
        self.flow_path = flow_path

        self.template_specifications = template_specifications
 
        length = abs(self.target_time - self.start_time)
        self.times = np.linspace(min(self.target_time, self.start_time), 
                                max(self.target_time, self.start_time), 
                                num = n_time_points(length, time_concentration)).tolist()
        
        self.objects, self.extensions, _, self.attachment = template_metadata(template_specifications)
        
        logger.info("Transport from start = {} to target = {}".format(self.start_time, self.target_time))
        logger.info("Trajectory length: {} time points".format(n_time_points(length, time_concentration)))

    def initialize(self, deformation_kernel_width, initial_cp, initial_momenta,
                    initial_momenta_to_transport, n_time_points, n_components = None):
                
        self.template = DeformableMultiObject(self.objects)

        cp = read_2D_array(initial_cp) # n_cp x d
        initial_momenta = read_3D_array(initial_momenta) # n_comp x n_cp x d
        initial_momenta_to_transport = read_3D_array(initial_momenta_to_transport)

        if n_components is None and len(initial_momenta.shape) > len(cp.shape):
            initial_momenta = initial_momenta.reshape(initial_momenta.shape[:-1])
        if n_components is None and len(initial_momenta_to_transport.shape) > len(cp.shape):
            initial_momenta_to_transport = initial_momenta_to_transport.reshape(initial_momenta_to_transport.shape[:-1])

        self.device = get_best_device()
        self.cp = move_data(cp, device=self.device)
        self.initial_momenta = move_data(initial_momenta, device=self.device)
        self.initial_momenta_to_transport = move_data(initial_momenta_to_transport, device=self.device)

        template_points = self.template.get_points()
        self.template_points = {key: move_data(value, device=self.device) for key, value in template_points.items()}

        template_data = self.template.get_data()
        self.template_data = {key: move_data(value, device=self.device) for key, value in template_data.items()}

        self.deformation_kernel = factory(kernel_width=deformation_kernel_width)
        
        self.exponential = Exponential(kernel=self.deformation_kernel, n_time_points=n_time_points) # by default n_tp = 11/ctp = 10
            
    def set_output_dir(self, output_dir):
        self.output_dir = output_dir

    def set_momenta_to_transport(self, initial_momenta_to_transport):
        initial_momenta_to_transport = read_3D_array(initial_momenta_to_transport)
        self.initial_momenta_to_transport = move_data(initial_momenta_to_transport, device=self.device)  

    def get_output(self):
        self.transported_mom_path = {
                time: op.join(self.output_dir, "Transported_Momenta_tp_{}__age_{}.txt".format(i, time))
                for i, time in enumerate(self.times) }
        self.transported_object_path = {
                time: op.join(self.output_dir, "Parallel_curve_tp_{}__age_{}{}".format(i, time, self.extensions[0]))
                for i, time in enumerate(self.times) }

    def check(self):
        self.get_output()
        if not op.exists(self.transported_mom_path[self.start_time]):
            return False
        else:   
            return True   
    
    def check_pt_for_ica(self):
        self.get_output_ica()

        if not op.exists(self.transported_mom_path[self.start_time]):
            return False
        else:
            self.cp_traj = read_3D_array(self.initial_cp)
            self.transport_trajectory = [None] * len(self.times)
            for i, (time) in enumerate(self.times):
                if op.exists(self.transported_mom_path[time]):
                    self.transport_trajectory[i] = read_3D_array(self.transported_mom_path[time])
                
            return True   

    def write_geodesic(self):
        print("write_geodesic", self.template)
        self.geodesic.write(self.template, self.template_data, self.output_dir, write_all = True)
        self.geodesic.output_path(self.output_dir)
        self.flow_path = self.geodesic.flow_path

    def get_flow(self):
        if self.flow_path is None:
            self.geodesic.output_path(self.output_dir)
            self.flow_path = self.geodesic.flow_path

    def shoot_registration(self):
        """
            Checks that shooting the registration momenta from template to target is ok
        """
        print("self.template", self.template)
        initial_template = self.geodesic.get_template_points(self.start_time)
        self.exponential.prepare_and_update(self.cp, self.initial_momenta_to_transport, initial_template)
        self.exponential.write_flow(self.extensions, self.template, self.template_data, 
                                    self.output_dir, write_only_last = True)

    def shoot_exponential(self, time, cp, mom):
        # Shoot exponential and get last image
        
        # Shooting from the geodesic:
        initial_template = self.geodesic.get_template_points(time)
        self.exponential.prepare_and_update(cp, mom, initial_template)

        parallel_points = self.exponential.get_template_points() #only the last template points
        parallel_data = self.template.get_deformed_data(parallel_points, self.template_data)

        return parallel_data
    
    def transport(self, is_orthogonal = False):
        concatenate_for_paraview(self.initial_momenta_to_transport, self.cp, 
                                self.output_dir, "ForParaview_momenta_to_transport.vtk")
        
        ### PARALLEL TRANSPORT
        self.transport_trajectory = self.geodesic.transport_(self.initial_momenta_to_transport,
                                                            self.start_time, self.target_time,
                                                            is_orthogonal) #ajout fg
    
    def write(self, perform_shooting = False):
        for i, (time, transported_mom) in enumerate(zip(self.times, self.transport_trajectory)):

            if time.is_integer():
                # Shooting from the geodesic
                cp = self.geodesic.get_cp_t(time, transform = lambda elt: elt)
                mom = self.geodesic.get_momenta_t(time, transform = lambda elt: elt)
                parallel_data = self.shoot_exponential(time, cp, transported_mom)

                # Regression Momenta
                if perform_shooting:
                    concatenate_for_paraview(mom, cp, self.output_dir, 
                                            "Regression_Momenta_tp_{0:d}__age_{1:.2f}.vtk".format(i, time))
                # Transported Momenta
                write_3D_array(transported_mom, self.output_dir, self.transported_mom_path[time].split("/")[-1])
                concatenate_for_paraview(transported_mom, cp, self.output_dir, 
                                        "Transported_Momenta_tp_{0:d}__age_{1:.2f}.vtk".format(i, time))
                # Parallel data
                names = [self.transported_object_path[time].split("/")[-1]]

                self.template.write(self.output_dir, names, parallel_data)

    def get_output_ica(self, sigma = 1):
        si = "+" if sigma > 0 else "-"
        self.transported_mom_path = {time: op.join(self.output_dir, 
                                    f"Transported_Space_Shift_tp_{i:d}__age_{time:.2f}.txt")\
                                    for i, time in enumerate(self.times) }
        self.transported_object_path = { time: op.join(self.output_dir, 
                                    f"GeometricMode__sigma_{si}{sigma}_tp_{i:d}__age_{time:.2f}{self.extensions[0]}")\
                                    for i, time in enumerate(self.times) }

    def write_for_ica(self):
        self.get_output_ica()

        for time, transported_mom in zip(self.times, self.transport_trajectory):
            if time.is_integer():
                if not op.exists(self.transported_mom_path[time]):
                                            
                    write_3D_array(transported_mom, self.output_dir, self.transported_mom_path[time].split("/")[-1])
                    # concatenate_for_paraview(transported_mom_, cp, self.output_dir, 
                    #                         self.transported_mom_path_)

    def shoot_for_ica(self, sigma = 1):

        self.get_output_ica(sigma)

        # Set number of time points to flow up to 3 standard deviations 
        # old_ntp = self.exponential.n_time_points
        # ntp = np.abs(1 + (sigma * self.exponential.n_time_points - 1))
        # self.exponential.n_time_points = ntp

        for time, cp, transported_mom in zip(self.times, self.cp_traj, self.transport_trajectory):
            if time.is_integer():
                if not op.exists(self.transported_object_path[time]):
                    transported_mom = read_3D_array(self.transported_mom_path[time])
                    transported_mom = move_data(transported_mom, device=self.device)  
                    transported_mom = transported_mom * sigma
                    
                    parallel_data = self.shoot_exponential(time, cp, transported_mom)
                    names = [self.transported_object_path[time].split("/")[-1]]
                    self.template.write(self.output_dir, names, parallel_data, transported_mom, cp, self.deformation_kernel)
    
    def screenshot_ica(self):
        for time in self.times:
            if time.is_integer():
                names = self.transported_object_path[time].split("/")[-1]
                new_name = self.transported_object_path[time].split("/")[-1].replace(".vtk", ".png")
                new_name = "Screenshot_" + new_name
                screenshot_vtk(names, new_name)

    def compute_distance_to_flow(self):
        self.data["distance_to_flow"] = {}
        for (time) in self.times:

            if time.is_integer():
                
                # Shooting from the geodesic:
                spec = deepcopy(self.template_specifications)
                spec["Object_1"]["filename"] = self.transported_object_path[time]

                objects, _, _, _ = template_metadata(spec)
                deformed_template = DeformableMultiObject(objects)
                parallel_data = deformed_template.get_data()
                parallel_data = {key: move_data(value, device=self.device) for key, value in parallel_data.items()}

                # Create the geodesic flow object 
                spec = {"dataset_filenames" : [[{"Object_1" : self.flow_path[time]}]], 
                        "visit_ages" : [[0]], "ids" : ["sub"]}
                dataset = create_dataset(self.template_specifications, **spec)
                flow = dataset.objects[0][0]

                # Plot distance to geodesic flow
                self.data["distance_to_flow"][time] = self.attachment.compute_distances(parallel_data, self.template, 
                                                                       flow).cpu().numpy()[0]
    def compute_norm(self):
        self.data["norm"] = {}
        self.data["squared_norm"] = {}
        self.data["scalar_product_with_original_registration_momenta"] = {}
        self.data["current_distance_with_original_registration_momenta"] = {}
        original_momenta = read_3D_array(self.transported_mom_path[self.start_time])
        original_momenta = move_data(original_momenta, device=self.device)

        for (time) in self.times:
            if time.is_integer():
                momenta = read_3D_array(self.transported_mom_path[time])
                momenta = move_data(momenta, device=self.device)
                self.data["squared_norm"][time] = self.exponential.scalar_product(self.cp, momenta, momenta)
                self.data["norm"][time] = torch.sqrt(self.data["squared_norm"][time])
                self.data["scalar_product_with_original_registration_momenta"][time] = self.exponential.scalar_product(self.cp, original_momenta,
                                                            momenta)
                norm = self.exponential.scalar_product(self.cp, original_momenta, original_momenta)
                self.data["current_distance_with_original_registration_momenta"][time] = norm + self.data["squared_norm"][time] - 2 * self.data["scalar_product_with_original_registration_momenta"][time]

                self.data["squared_norm"][time] = self.data["squared_norm"][time].cpu().numpy()
                self.data["norm"][time] = self.data["norm"][time].cpu().numpy()
                self.data["scalar_product_with_original_registration_momenta"][time] = self.data["scalar_product_with_original_registration_momenta"][time].cpu().numpy()
                self.data["current_distance_with_original_registration_momenta"][time] = self.data["current_distance_with_original_registration_momenta"][time].cpu().numpy()

class SimpleParallelTransport(ParallelTransport):
    def __init__(self, template_specifications, tmin, tmax, time_concentration, 
                t0, start_time, target_time, output_dir, flow_path):
        
        super().__init__(template_specifications, tmin, tmax, time_concentration, 
                        t0, start_time, target_time, output_dir, flow_path) 
        
    
    def initialize_(self, deformation_kernel_width, initial_cp, initial_momenta,
                    initial_momenta_to_transport, n_time_points):
        
        self.initialize(deformation_kernel_width, initial_cp, initial_momenta,
                        initial_momenta_to_transport, n_time_points)
        
        # Deformation tools
        self.geodesic = Geodesic(time_concentration=self.time_concentration,
                                kernel=self.deformation_kernel, t0=self.t0,
                                extensions = self.extensions, root_name = "GeodesicRegression")

    def set_geodesic(self):
        logger.info("Reference geodesic defined from tmin={} to tmax={}".format(self.tmin, self.tmax))
        logger.info("Reference momenta and control points defined at t0={}".format(self.t0))
        
        self.geodesic.set_tmin(self.tmin)
        self.geodesic.set_tmax(self.tmax)
        if self.t0 is None:
            self.geodesic.set_t0(self.geodesic.tmin)
        else:
            self.geodesic.set_t0(self.t0)
        
        self.geodesic.set_template_points_t0(self.template_points)
        self.geodesic.set_cp_t0(self.cp)
        self.geodesic.set_momenta_t0(self.initial_momenta)
        self.geodesic.update()

class PiecewiseParallelTransport(ParallelTransport):
    def __init__(self, template_specifications, tmin, tmax, time_concentration, 
                t0, start_time, target_time, tR, output_dir, flow_path, initial_cp = None):
        
        super().__init__(template_specifications, tmin, tmax, time_concentration, 
                        t0, start_time, target_time, output_dir, flow_path, initial_cp) 

        self.tR = tR
    
    def initialize_(self, deformation_kernel_width, initial_cp, initial_momenta,
                    initial_momenta_to_transport, n_time_points,  nb_components):
        
        self.initialize(deformation_kernel_width, initial_cp, initial_momenta,
                        initial_momenta_to_transport, n_time_points, nb_components)
        
        # Deformation tools
        self.geodesic = PiecewiseGeodesic(time_concentration=self.time_concentration,
                                        kernel=self.deformation_kernel, t0=self.t0, 
                                        nb_components = nb_components, extensions = self.extensions,
                                        root_name = "GeodesicRegression")
        
    def set_geodesic(self):
        logger.info("Reference geodesic defined from tmin={} to tmax={}".format(self.tmin, self.tmax))
        logger.info("Reference momenta and control points defined at t0={}".format(self.t0))
        logger.info("Rupture times: {}".format(self.tR))
        
        self.geodesic.set_tR(self.tR)
        self.geodesic.set_tmin(self.tmin)
        self.geodesic.set_tmax(self.tmax)
        t0 = self.t0 if self.t0 is not None else self.geodesic.tmin
        self.geodesic.set_t0(t0)

        self.geodesic.set_template_points_tR(self.template_points)
        self.geodesic.set_cp_tR(self.cp)
        self.geodesic.set_momenta_tR(self.initial_momenta)
        self.geodesic.update()

####################################################################################################################
### Launch functions
####################################################################################################################

def launch_parallel_transport(template_specifications,
                               deformation_kernel_width=None,
                               initial_cp=None, initial_momenta=None,
                               initial_momenta_to_transport=default.initial_momenta_to_transport,
                               tmin=default.tmin, tmax=default.tmax,
                               time_concentration=default.time_concentration, t0=default.t0, 
                               start_time = default.t0, target_time = default.t0,
                               n_time_points=default.n_time_points, output_dir=default.output_dir, 
                               perform_shooting = True, overwrite = True, flow_path = None, 
                               **kwargs):

    pt = SimpleParallelTransport(template_specifications, tmin, tmax, time_concentration, 
                                t0, start_time, target_time, output_dir, flow_path)
    
    if pt.check() and not overwrite:
        logger.info("\n Parallel Transport already performed -- Stopping")
        return pt.transported_mom_path
    
    pt.initialize_(deformation_kernel_width, initial_cp, initial_momenta,
                    initial_momenta_to_transport, n_time_points)
    pt.set_geodesic()
    
    if perform_shooting:
        pt.write_geodesic()
        pt.shoot_registration()
    
    pt.transport()

    pt.write(perform_shooting)
    
    return pt.transported_mom_path

def launch_piecewise_parallel_transport(template_specifications,
                               deformation_kernel_width=None,
                               initial_cp=None, initial_momenta=None,
                               initial_momenta_to_transport=default.initial_momenta_to_transport,
                               tmin=default.tmin, tmax=default.tmax,
                               time_concentration=default.time_concentration, t0=default.t0, 
                               start_time = default.t0, target_time = default.t0,
                               num_component = 4, tR = [], n_time_points=default.n_time_points,
                               output_dir=default.output_dir, perform_shooting = True, 
                               overwrite = True, flow_path = None, **kwargs):
    """
    Compute parallel transport
    PT in piececewise_SPT_reference_frame: 
    1- self.geodesic.update : update each expo with initial momenta and cp
    2 - gets a space shift at t0
    3- self.geodesic.parallel_transport(space_shift) along all trajectory
    """
    pt = PiecewiseParallelTransport(template_specifications, tmin, tmax, 
                                    time_concentration, t0, start_time, 
                                    target_time, tR, output_dir, flow_path)
    
    if pt.check() and not overwrite:
        logger.info("\n Parallel Transport already performed -- Stopping")
        return pt.transported_mom_path
    
    pt.initialize_(deformation_kernel_width, initial_cp, initial_momenta,
                    initial_momenta_to_transport, n_time_points, num_component)
    pt.set_geodesic()
    
    if perform_shooting:
        pt.write_geodesic()
        pt.shoot_registration()
    
    pt.transport()

    pt.write(perform_shooting)
    
    return pt.transported_mom_path

def compute_distance_to_flow(template_specifications,
                        deformation_kernel_width=None,
                        initial_cp=None, initial_momenta_tR=None,
                        initial_momenta_to_transport=default.initial_momenta_to_transport,
                        tmin=default.tmin, tmax=default.tmax,
                        time_concentration = default.time_concentration,
                        t0=default.t0, start_time = default.t0, target_time = default.t0,
                        num_component = 4, tR = [], n_time_points=default.n_time_points,

                        output_dir=default.output_dir, 
                        flow_path = None,
                        **kwargs):
    
    pt = PiecewiseParallelTransport(template_specifications, tmin, tmax, 
                                    time_concentration, t0, start_time, 
                                    target_time, tR, output_dir, flow_path)
    
    pt.get_output()
    pt.initialize_(deformation_kernel_width, initial_cp, 
                initial_momenta_tR, initial_momenta_to_transport, 
                n_time_points, num_component)
    pt.get_flow()

    pt.compute_distance_to_flow()
    pt.compute_norm()
    
    return pt.data