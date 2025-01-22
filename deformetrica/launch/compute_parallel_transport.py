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
from ..in_out.dataset_functions import create_template_metadata, create_dataset
from ..support import utilities
from ..support import kernels as kernel_factory
from ..core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from ..support.utilities.vtk_tools import screenshot_vtk

import logging
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")

class ParallelTransport():
    def __init__(self, template_specifications, dimension, 
                  tmin, tmax, concentration_of_time_points, 
                t0, start_time, target_time, gpu_mode, output_dir, flow_path,
                initial_control_points = None):
        
        self.data = {}
        self.dimension = dimension
        self.gpu_mode = gpu_mode
        
        self.tmin = tmin
        self.tmax = tmax
        self.t0 = t0

        self.start_time = start_time
        target_time = target_time if target_time is not None else tmax
        self.target_time = target_time

        self.concentration_of_time_points = concentration_of_time_points
        self.initial_control_points = initial_control_points

        self.output_dir = output_dir
        self.flow_path = flow_path

        self.template_specifications = template_specifications

        # Get times
        print("self.target_time", self.target_time)
        print("start_time", start_time)
        print("tmax", tmax)
        try:
            length = np.abs(self.target_time - start_time)
            nb_of_tp = max(1, int(length * concentration_of_time_points + 1.5))
        except:
            self.target_time = tmax
            length = np.abs(self.target_time - start_time)
            nb_of_tp = max(1, int(length * concentration_of_time_points + 1.5))


        self.times = np.linspace(min(target_time, start_time), max(target_time, start_time), 
                            num=nb_of_tp).tolist()
        
        self.objects_list, self.objects_name, self.objects_name_extension, _, self.multi_object_attachment = \
        create_template_metadata(template_specifications, dimension, gpu_mode=gpu_mode)
        
    def initialize(self, deformation_kernel_width, 
                 initial_control_points, initial_momenta,
                initial_momenta_to_transport, number_of_time_points):
        print("initial_momenta", initial_momenta)
        print("initial_momenta_to_transport", initial_momenta_to_transport)
        
        self.template = DeformableMultiObject(self.objects_list)

        control_points = read_2D_array(initial_control_points)
        initial_momenta = read_3D_array(initial_momenta) # n_comp x n_cp x d
        initial_momenta_to_transport = read_3D_array(initial_momenta_to_transport)

        device, _ = utilities.get_best_device(self.gpu_mode)
        self.device = device
        self.control_points = utilities.move_data(control_points, device=device)
        self.initial_momenta = utilities.move_data(initial_momenta, device=device)
        self.initial_momenta_to_transport = utilities.move_data(initial_momenta_to_transport, device=device)

        template_points = self.template.get_points()
        self.template_points = {key: utilities.move_data(value, device=device) for key, value in template_points.items()}

        template_data = self.template.get_data()
        self.template_data = {key: utilities.move_data(value, device=device) for key, value in template_data.items()}

        self.deformation_kernel = kernel_factory.factory(gpu_mode=self.gpu_mode, 
                                                         kernel_width=deformation_kernel_width)
        
        self.exponential = Exponential(kernel=self.deformation_kernel, 
                                       number_of_time_points=number_of_time_points,
                                        transport_cp = False) # by default n_tp = 11/ctp = 10
        
        self.geodesic = None
        
    def set_output_dir(self, output_dir):
        self.output_dir = output_dir

    def set_momenta_to_transport(self, initial_momenta_to_transport):
        initial_momenta_to_transport = read_3D_array(initial_momenta_to_transport)
        self.initial_momenta_to_transport = utilities.move_data(initial_momenta_to_transport, device=self.device)  

    def get_output(self):
        self.transported_mom_path = {}
        self.transported_object_path = {}
        for i, (time) in enumerate(self.times):
            self.transported_mom_path[time] = op.join(self.output_dir, 
                                                "Transported_Momenta_tp_{:d}__age_{:.2f}.txt".format(i, time))    
            self.transported_object_path[time] = op.join(self.output_dir, 
                                                 "{}_parallel_curve_tp_{:d}__age_{:.2f}{}".format(self.objects_name[0], i, time, self.objects_name_extension[0]))

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
            self.control_points_traj = read_3D_array(self.initial_control_points)
            self.parallel_transport_trajectory = [None]*len(self.times)
            for i, (time) in enumerate(self.times):
                if op.exists(self.transported_mom_path[time]):
                    self.parallel_transport_trajectory[i] = read_3D_array(self.transported_mom_path[time])
                
            return True   

    def write_geodesic(self):
        self.geodesic.write("Regression", self.objects_name, self.objects_name_extension, self.template, self.template_data, 
                   self.output_dir, write_adjoint_parameters=False, write_all = True)
        self.geodesic.output_path("Regression", self.objects_name, self.objects_name_extension, self.output_dir)
        self.flow_path = self.geodesic.flow_path

    def get_flow(self):
        if self.flow_path is None:
            self.geodesic.output_path("Regression", self.objects_name, self.objects_name_extension, self.output_dir)
            self.flow_path = self.geodesic.flow_path

    def shoot_registration(self):
        initial_template = self.geodesic.get_template_points(self.start_time)
        self.exponential.set_initial_template_points(initial_template)
        self.exponential.set_initial_control_points(self.control_points)
        self.exponential.set_initial_momenta(self.initial_momenta_to_transport)
        self.exponential.update()
        self.exponential.write_flow(["Shooting_with_registration_momenta"], [".vtk"], self.template, 
                            self.template_data, self.output_dir, write_only_last = True)

    def shoot_exponential(self, time, cp, mom):

        # Shoot exponential and get last image
        
        # Shooting from the geodesic:
        initial_template = self.geodesic.get_template_points(time)
        self.exponential.set_initial_template_points(initial_template)
        self.exponential.set_initial_control_points(cp)
        self.exponential.set_initial_momenta(mom)
        self.exponential.update()

        parallel_points = self.exponential.get_template_points(self.exponential.number_of_time_points-1) #returns only the last template point
        parallel_data = self.template.get_deformed_data(parallel_points, self.template_data)

        return parallel_data
    
    def transport(self, is_orthogonal = False):
        concatenate_for_paraview(self.initial_momenta_to_transport.cpu().numpy(), 
                                 self.control_points.cpu().numpy(), 
                                self.output_dir, "For_paraview_momenta_to_transport.vtk")
        
        ### PARALLEL TRANSPORT
        self.parallel_transport_trajectory = self.geodesic.parallel_transport_(self.initial_momenta_to_transport,
                                                                 self.start_time, self.target_time,
                                                                is_orthogonal) #ajout fg

        # Getting trajectory caracteristics:
        self.control_points_traj = self.geodesic.get_control_points_trajectory()
        self.momenta_traj = self.geodesic.get_momenta_trajectory() # divided by length
    
    def write(self, perform_shooting = False):
        for i, (time, cp, mom, transported_mom) in enumerate(
            zip(self.times, self.control_points_traj, self.momenta_traj, 
                self.parallel_transport_trajectory)):

            if time.is_integer():
                
                # Regression Momenta
                if perform_shooting:
                    concatenate_for_paraview(mom.detach().cpu().numpy(), cp.detach().cpu().numpy(), 
                                        self.output_dir, "For_paraview_regression_Momenta_tp_{0:d}__age_{1:.2f}.vtk".format(i, time))
                
                # Transported Momenta
                write_3D_array(transported_mom.detach().cpu().numpy(), self.output_dir,
                                self.transported_mom_path[time].split("/")[-1])
                concatenate_for_paraview(transported_mom.detach().cpu().numpy(), cp.detach().cpu().numpy(), 
                                        self.output_dir, "For_paraview_transported_Momenta_tp_{0:d}__age_{1:.2f}.vtk".format(i, time))
                
                # Shooting from the geodesic:
                parallel_data = self.shoot_exponential(time, cp, transported_mom)

                names = [self.transported_object_path[time].split("/")[-1]]

                self.template.write(self.output_dir, names, {key: value.detach().cpu().numpy() for key, value in parallel_data.items()})

                #objects_name_ = ["Parallel_transport_flow_{}_time_{}".format(i, time)]
                # exponential.write_flow(objects_name_, objects_name_extension, template, template_data, output_dir,
                #                     write_adjoint_parameters=False, write_only_last = False)

                # Scalar product
                if perform_shooting:
                    norm = torch.sqrt(self.exponential.scalar_product_at_points(cp, transported_mom, transported_mom))
                    norm = norm.detach().cpu().numpy()
                    norm = norm.reshape((norm.shape[0], 1))

                    # Scalar product with initial momenta
                    sp = self.exponential.scalar_product_at_points(cp, self.initial_momenta_to_transport, transported_mom)
                    sp_norm = sp / (torch.sqrt(self.exponential.scalar_product_at_points(cp, transported_mom, transported_mom)) * torch.sqrt(self.exponential.scalar_product_at_points(cp, self.initial_momenta_to_transport, self.initial_momenta_to_transport)))
                    
                    sp = sp.detach().cpu().numpy()
                    sp = sp.reshape((sp.shape[0], 1))

                    sp_norm = sp_norm.detach().cpu().numpy()
                    sp_norm = sp_norm.reshape((sp.shape[0], 1))

                    norm = np.nan_to_num(norm)
                    sp_norm = np.nan_to_num(sp_norm)
                    sp = np.nan_to_num(sp)

                    concatenate_for_paraview(transported_mom.detach().cpu().numpy(), cp.detach().cpu().numpy(), 
                                            self.output_dir, "For_paraview_sp__tp_{0:d}__age_{1:.2f}.vtk".format(i, time),
                                            norm = norm, sp = sp, sp_norm = sp_norm)

    def get_output_ica(self, sigma = 1):
        si = "+" if sigma > 0 else "-"
        self.transported_mom_path = {}
        self.transported_mom_path_ = {}
        self.transported_object_path = {}
        for i, (time) in enumerate(self.times):
            self.transported_mom_path[time] = op.join(self.output_dir, 
                                                "Transported_Space_Shift_tp_{:d}__age_{:.2f}.txt".format(i, time))    
            # self.transported_mom_path_[time] = op.join(self.output_dir, 
            #                                         "For_paraview_transported_Space_shift__sigma_{}{}_tp_{:d}__age_{:.2f}.vtk".format(si, sigma, i, time))
            self.transported_object_path[time] = op.join(self.output_dir, 
                                                "GeometricMode__sigma_{}{}_tp_{:d}__age_{:.2f}".format(si, sigma, i, time) \
                                                + self.objects_name_extension[0])

    def write_for_ica(self):
        self.get_output_ica()

        for time, transported_mom in zip(self.times, self.parallel_transport_trajectory):
            if time.is_integer():
                if not op.exists(self.transported_mom_path[time]):
                    
                    if torch.is_tensor(transported_mom):
                        transported_mom = transported_mom.detach().cpu().numpy()
                        
                    write_3D_array(transported_mom, self.output_dir, self.transported_mom_path[time].split("/")[-1])
                    # concatenate_for_paraview(transported_mom_, cp.detach().cpu().numpy(), self.output_dir, 
                    #                         self.transported_mom_path_)

    def shoot_for_ica(self, sigma = 1):

        self.get_output_ica(sigma)

        # Set number of time points to flow up to 3 standard deviations 
        # old_ntp = self.exponential.number_of_time_points
        # ntp = np.abs(1 + (sigma * self.exponential.number_of_time_points - 1))
        # self.exponential.number_of_time_points = ntp

        for time, cp, transported_mom in zip(self.times, self.control_points_traj, self.parallel_transport_trajectory):
            if time.is_integer():
                if not op.exists(self.transported_object_path[time]):
                    transported_mom = read_3D_array(self.transported_mom_path[time])
                    transported_mom = utilities.move_data(transported_mom, device=self.device)  
                    transported_mom = transported_mom * sigma
                    
                    parallel_data = self.shoot_exponential(time, cp, transported_mom)
                    names = [self.transported_object_path[time].split("/")[-1]]
                    self.template.write(self.output_dir, names, {key: value.detach().cpu().numpy() for key, value in parallel_data.items()},
                                    transported_mom, cp, self.deformation_kernel)
    
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
                spec[self.objects_name[0]]["filename"] = self.transported_object_path[time]

                objects_list, _, _, _, _ = create_template_metadata(spec, self.dimension, gpu_mode=self.gpu_mode)
                deformed_template = DeformableMultiObject(objects_list)
                parallel_data = deformed_template.get_data()
                parallel_data = {key: utilities.move_data(value, device=self.device) for key, value in parallel_data.items()}

                # Create the geodesic flow object 
                spec = {"dataset_filenames" : [[{self.objects_name[0] : self.flow_path[time]}]], 
                        "visit_ages" : [[0]], "subject_ids" : ["sub"]}
                dataset = create_dataset(self.template_specifications, **spec)
                flow = dataset.deformable_objects[0][0]

                # Plot distance to geodesic flow
                self.data["distance_to_flow"][time] = self.multi_object_attachment.compute_distances(parallel_data, self.template, 
                                                                       flow).cpu().numpy()[0]
    def compute_norm(self):
        self.data["norm"] = {}
        self.data["squared_norm"] = {}
        self.data["scalar_product_with_original_registration_momenta"] = {}
        self.data["current_distance_with_original_registration_momenta"] = {}
        original_momenta = read_3D_array(self.transported_mom_path[self.start_time])
        original_momenta = utilities.move_data(original_momenta, device=self.device)

        for (time) in self.times:
            if time.is_integer():
                momenta = read_3D_array(self.transported_mom_path[time])
                momenta = utilities.move_data(momenta, device=self.device)
                self.data["squared_norm"][time] = self.exponential.scalar_product(self.control_points, momenta, momenta)
                self.data["norm"][time] = torch.sqrt(self.data["squared_norm"][time])
                self.data["scalar_product_with_original_registration_momenta"][time] = self.exponential.scalar_product(self.control_points, original_momenta,
                                                            momenta)
                norm = self.exponential.scalar_product(self.control_points, original_momenta, original_momenta)
                self.data["current_distance_with_original_registration_momenta"][time] = norm + self.data["squared_norm"][time] - 2 * self.data["scalar_product_with_original_registration_momenta"][time]

                self.data["squared_norm"][time] = self.data["squared_norm"][time].cpu().numpy()
                self.data["norm"][time] = self.data["norm"][time].cpu().numpy()
                self.data["scalar_product_with_original_registration_momenta"][time] = self.data["scalar_product_with_original_registration_momenta"][time].cpu().numpy()
                self.data["current_distance_with_original_registration_momenta"][time] = self.data["current_distance_with_original_registration_momenta"][time].cpu().numpy()

class SimpleParallelTransport(ParallelTransport):
    def __init__(self, template_specifications, dimension, 
                  tmin, tmax, concentration_of_time_points, 
                t0, start_time, target_time, gpu_mode, output_dir, flow_path):
        
        super().__init__(template_specifications, dimension, 
                  tmin, tmax, concentration_of_time_points, 
                t0, start_time, target_time, gpu_mode, output_dir, flow_path) 
        
    
    def initialize_(self, deformation_kernel_width, 
                 initial_control_points, initial_momenta,
                initial_momenta_to_transport, number_of_time_points):
        
        self.initialize(deformation_kernel_width, 
                  initial_control_points, initial_momenta,
                initial_momenta_to_transport, number_of_time_points)
        
        # Deformation tools
        self.geodesic = Geodesic(concentration_of_time_points=self.concentration_of_time_points,
                        kernel=self.deformation_kernel,
                        t0=self.t0, transport_cp = False)

    def set_geodesic(self):
        
        assert math.fabs(self.tmin) != float("inf"), "Please specify a minimum time for the geodesic trajectory"
        assert math.fabs(self.tmax) != float("inf"), "Please specify a maximum time for the geodesic trajectory"

        self.geodesic.set_tmin(self.tmin)
        self.geodesic.set_tmax(self.tmax)
        if self.t0 is None:
            self.geodesic.set_t0(self.geodesic.tmin)
        else:
            self.geodesic.set_t0(self.t0)

        self.geodesic.set_template_points_t0(self.template_points)
        self.geodesic.set_control_points_t0(self.control_points)
        self.geodesic.set_momenta_t0(self.initial_momenta)
        self.geodesic.update()

        self.control_points_traj = self.geodesic.get_control_points_trajectory()
        

class PiecewiseParallelTransport(ParallelTransport):
    def __init__(self, template_specifications, dimension, 
                  tmin, tmax, concentration_of_time_points, 
                t0, start_time, target_time, tR, gpu_mode, output_dir, flow_path, 
                initial_control_points = None):
        
        super().__init__(template_specifications, dimension, 
                  tmin, tmax, concentration_of_time_points, 
                t0, start_time, target_time, gpu_mode, output_dir, flow_path,
                initial_control_points) 

        self.tR = tR
    
    def initialize_(self, deformation_kernel_width, 
                  initial_control_points, initial_momenta,
                initial_momenta_to_transport, number_of_time_points,
                nb_components):
        
        self.initialize(deformation_kernel_width, 
                  initial_control_points, initial_momenta,
                initial_momenta_to_transport, number_of_time_points)
        
        # Deformation tools
        self.geodesic = PiecewiseGeodesic(
                        concentration_of_time_points=self.concentration_of_time_points,
                        kernel=self.deformation_kernel, 
                        t0=self.t0, nb_components = nb_components)
        
    def set_geodesic(self):
        
        self.geodesic.set_tR(self.tR)

        assert math.fabs(self.tmin) != float("inf"), "Please specify a minimum time for the geodesic trajectory"
        assert math.fabs(self.tmax) != float("inf"), "Please specify a maximum time for the geodesic trajectory"

        self.geodesic.set_tmin(self.tmin)
        self.geodesic.set_tmax(self.tmax)
        if self.t0 is None:
            self.geodesic.set_t0(self.geodesic.tmin)
        else:
            self.geodesic.set_t0(self.t0)

        self.geodesic.set_template_points_tR(self.template_points)
        self.geodesic.set_control_points_tR(self.control_points)
        self.geodesic.set_momenta_tR(self.initial_momenta)
        self.geodesic.update()

        self.control_points_traj = self.geodesic.get_control_points_trajectory()
    

def compute_parallel_transport(template_specifications, dimension=default.dimension,
                               deformation_kernel_width=default.deformation_kernel_width,
                               initial_control_points=default.initial_control_points,
                               initial_momenta=default.initial_momenta,
                               initial_momenta_to_transport=default.initial_momenta_to_transport,
                               tmin=default.tmin, tmax=default.tmax,
                               concentration_of_time_points=default.concentration_of_time_points,
                               t0=default.t0, start_time = default.t0, target_time = default.t0,
                               number_of_time_points=default.number_of_time_points,
                               gpu_mode=default.gpu_mode,
                               output_dir=default.output_dir, 
                               perform_shooting = default.perform_shooting, 
                               overwrite = True, flow_path = None, **kwargs):
    
    pt = SimpleParallelTransport(template_specifications,
                                    dimension, tmin, tmax,
                                    concentration_of_time_points, t0, start_time, target_time,
                                    gpu_mode, output_dir, flow_path)
    
    if not overwrite and pt.check():
        logger.info("\n Parallel Transport already performed -- Stopping")
        return pt.transported_mom_path
    
    pt.initialize_(deformation_kernel_width, 
                  initial_control_points, initial_momenta,
                initial_momenta_to_transport, number_of_time_points)
    pt.set_geodesic()
    
    if perform_shooting:
        pt.write_geodesic()
        pt.shoot_registration()
    
    pt.transport()

    pt.write(perform_shooting)
    
    return pt.transported_mom_path

def compute_piecewise_parallel_transport(template_specifications, dimension=default.dimension,
                               deformation_kernel_width=default.deformation_kernel_width,
                               initial_control_points=default.initial_control_points,
                               initial_momenta_tR=default.initial_momenta,
                               initial_momenta_to_transport=default.initial_momenta_to_transport,
                               tmin=default.tmin, tmax=default.tmax,
                               concentration_of_time_points=default.concentration_of_time_points,
                               t0=default.t0, start_time = default.t0, target_time = default.t0,
                               num_component = 4, tR = [],
                               number_of_time_points=default.number_of_time_points,
                               gpu_mode=default.gpu_mode,
                               output_dir=default.output_dir, 
                               perform_shooting = default.perform_shooting, 
                               overwrite = True, flow_path = None, 
                               **kwargs):
    """
    Compute parallel transport
    PT in piececewise_SPT_reference_frame: 
    1- self.geodesic.update : update each expo with initial momenta and cp
    2 - gets a space shift at t0
    3- self.geodesic.parallel_transport(space_shift) along all trajectory
    """
    print("initial_momenta_tR", initial_momenta_tR)
    pt = PiecewiseParallelTransport(template_specifications, dimension, 
                                    tmin, tmax, concentration_of_time_points, t0, start_time, 
                                    target_time, tR, gpu_mode, output_dir, flow_path)
    
    if not overwrite and pt.check():
        logger.info("\n Parallel Transport already performed -- Stopping")
        return pt.transported_mom_path
    
    pt.initialize_(deformation_kernel_width, 
                  initial_control_points, initial_momenta_tR,
                initial_momenta_to_transport, number_of_time_points)
    pt.set_geodesic()
    
    if perform_shooting:
        pt.write_geodesic()
        pt.shoot_registration()
    
    pt.transport()

    pt.write(perform_shooting)
    
    return pt.transported_mom_path


def compute_distance_to_flow(template_specifications,
                        dimension=default.dimension,
                        deformation_kernel_width=default.deformation_kernel_width,
                        initial_control_points=default.initial_control_points,
                        initial_momenta_tR=default.initial_momenta,
                        initial_momenta_to_transport=default.initial_momenta_to_transport,
                        tmin=default.tmin, tmax=default.tmax,
                        concentration_of_time_points=default.concentration_of_time_points,
                        t0=default.t0, start_time = default.t0, target_time = default.t0,
                        num_component = 4,
                        tR = [],
                        number_of_time_points=default.number_of_time_points,

                        gpu_mode=default.gpu_mode,
                        output_dir=default.output_dir, 
                        flow_path = None,
                        **kwargs):
    
    pt = PiecewiseParallelTransport(template_specifications, dimension, 
                                    tmin, tmax, concentration_of_time_points, t0, start_time, 
                                    target_time, tR, gpu_mode, output_dir, flow_path)
    
    pt.get_output()
    pt.initialize_(deformation_kernel_width, 
                  initial_control_points, initial_momenta_tR,
                initial_momenta_to_transport, number_of_time_points,
                num_component)
    pt.get_flow()

    pt.compute_distance_to_flow()
    pt.compute_norm()
    
    return pt.data