import copy
import logging
import itertools
import os.path as op
from ...support import utilities
import numpy as np
from ...core import GpuMode
import torch
from ...support.utilities.tools import residuals_change
from ...support.utilities.wavelets import haar_backward_transpose
from ...support import kernels as kernel_factory

logger = logging.getLogger(__name__)

class MultiscaleMomenta():
    def __init__(self, multiscale, multiscale_images, multiscale_meshes, model, 
                ctf_interval, ctf_max_interval,
                points_per_axis, start_scale, gamma = 1, naive = True):
                
        self.model = model
        self.model_name = model.name
        self.n_momenta = model.get_momenta().shape[0]
        self.n_cp = model.get_momenta().shape[1]
        self.dimension = model.dimension

        # Model options
        self.freeze_template = model.freeze_template
        
        self.initial_convergence_threshold = 0.001
        self.convergence_threshold = 0.001
        self.ctf_interval = ctf_interval
        self.ctf_max_interval = ctf_max_interval

        self.multiscale = multiscale
        self.multiscale_images = multiscale_images
        self.multiscale_meshes = multiscale_meshes
        self.iter = [0] if multiscale else []
        self.gamma = gamma
        self.naive = naive
        self.zones = dict()
        self.silent_haar_coef_momenta_subjects = dict()

        self.smoothing_kernel = kernel_factory.factory('keops', gpu_mode=GpuMode.KERNEL, 
                                                       kernel_width=3.5)
        self.smoothing_kernel_2 = kernel_factory.factory('keops', gpu_mode=GpuMode.KERNEL, 
                                                       kernel_width=2.5)

        # if we want to stop before scale 0
        self.end_scale = 1
        self.start_scale = start_scale

        self.coarser_scale = None 

        self.points_per_axis = points_per_axis
        
        if self.dimension == 3:
            self.points_per_axis = [self.points_per_axis[1], self.points_per_axis[0], self.points_per_axis[2]]

    def initialize(self):
        """
            Initialization to perform the multiscale optimization of the momentum vectors
            Initializes the current scale as the maximum scale 
        """
        # This is used to initialize the other multiscales
        haar_coef_momenta = self.compute_haar_transform(self.model.fixed_effects['momenta'])
        self.coarser_scale = haar_coef_momenta[0][0].J - 1

        if self.multiscale:
            if self.start_scale is not None and self.start_scale <= haar_coef_momenta[0][0].J:
                self.scale = self.start_scale
            else:
                self.scale = haar_coef_momenta[0][0].J

            logger.info("\nInitialisation - coarse to fine on momenta at scale {}".format(self.scale))    
            logger.info("Momenta end scale {}".format(self.end_scale))

            for s in range(self.n_momenta): #!! for local adaptation
                self.silent_haar_coef_momenta_subjects[s] = dict()
                self.silent_haar_coef_momenta_subjects[s][0] = []
                self.silent_haar_coef_momenta_subjects[s][self.scale] = []
            
            self.zones[self.scale] = dict() 
    
    def compute_haar_transform(self, momenta):
        if len(momenta.shape) == 2:
            #momenta = list(momenta)
            momenta = momenta[np.newaxis, :, :]
        
        haar_coef_momenta = [[haar_backward_transpose(momenta[s, :, d].reshape(tuple(self.points_per_axis))) \
                            for d in range(self.dimension)] for s in range(len(momenta))]
        
        return haar_coef_momenta
    
    ####################################################################################################################
    ### Coarse to fine on momenta
    ####################################################################################################################
    def coarse_to_fine_condition(self, iteration, avg_residuals, end):
        """
        Authorizes coarse to fine if more than 2 iterations after previous CTF (or beginning of algo)
        and if residuals diminution is low
        """
        if self.multiscale_images: return False # dual multiscale handled later
        
        if self.multiscale_meshes: return False

        if not self.multiscale: return False

        if self.scale == 1: self.scale = 0

        if self.scale in [self.end_scale, 0]: return False
        
        if end: return True
                
        if self.too_much(iteration): return True
            
        if self.enough(iteration): return residuals_change(avg_residuals) < self.convergence_threshold
        
        return False

    def coarse_to_fine_step(self, iteration, step = 1):
        #go to smaller scale #beginning : self.scale = J
        self.scale = max(1, self.scale - step) #we only go to scale 0 so that the local adaptation is not called again ... 
        
        if step > 0:
            logger.info("Coarse to fine on momenta")
        else:
            logger.info("\nReverse coarse to fine on momenta")
            
        logger.info("Scale {}".format(self.scale))
        self.iter.append(iteration)
        
        # if self.scale in range(1, self.coarser_scale):
        #     #self.local_adaptation(iteration, new_parameters, output_dir, dataset, residuals)
        #     self.local_adaptation_subject(iteration, new_parameters, output_dir, current_dataset, residuals)

    def coarse_to_fine(self, new_parameters, current_dataset, iteration, 
                        avg_residuals, end):

        if self.coarse_to_fine_condition(iteration, avg_residuals, end):
            self.coarse_to_fine_step(iteration)
        
        return current_dataset, new_parameters
    
    ####################################################################################################################
    ### Gradient tools
    ####################################################################################################################         
    def compute_haar_momenta_gradient(self, gradient):
        """
            Wavelet transform gradient of momenta
            Set to 0 fine-scale coefficients
        """
        if len(gradient['momenta'].shape) == 2:
            gradient['momenta'] = gradient['momenta'][np.newaxis, :, :]

        haar_gradient = []
        for s, momenta_gradient in enumerate(gradient['momenta']): #for each subject
            subject_gradient = []

            for d in range(self.dimension):
                gradient_of_momenta = momenta_gradient[:, d].reshape(tuple(self.points_per_axis))
                
                # Compute haar transform of gradient
                gradient_of_coef = haar_backward_transpose(gradient_of_momenta, J = None, gamma =self.gamma) #before 13/12
                
                # Silence some coefficients
                gradient_of_coef = self.silence_fine_or_smooth_zones(gradient_of_coef, s)   
                subject_gradient.append(gradient_of_coef)

            haar_gradient.append(subject_gradient)
        gradient["haar_coef_momenta"] = haar_gradient

        return gradient

    def smooth_gradient(self, gradient):
        return gradient
    
        if "regression" in self.model_name.lower() and "kernel" not in self.model_name.lower():
            points = self.model.fixed_effects['control_points']
            cp = torch.tensor(points, dtype=torch.float32, device='cuda:0')
            try:
                n_comp = len(self.model.fixed_effects['rupture_time']) + 1
            except:
                return gradient
                
            if n_comp > 1:
                for comp, t in enumerate(self.model.fixed_effects['rupture_time']):
                    if t < 30 and comp < n_comp - 1:
                        print("smooth comp", comp)
                        momenta = gradient["momenta"][comp]
                        ones = np.ones(momenta.shape)
                        if not isinstance(momenta, torch.Tensor):
                            momenta = torch.tensor(momenta, dtype=torch.float32, device='cuda:0')
                        smooth_gradient = self.smoothing_kernel.convolve(cp, cp, momenta)  
                        
                        # normalize 
                        ones = torch.from_numpy(ones).float()
                        sum_kernel_along_rows = self.smoothing_kernel.convolve(cp, cp, ones)
                        gradient["momenta"][comp] = smooth_gradient.cpu().numpy() / sum_kernel_along_rows.numpy()
                    # elif comp < n_comp - 1:
                    #     print("smooth comp", comp)
                    #     comp = comp + 1
                    #     momenta = gradient["momenta"][comp]
                    #     ones = np.ones(momenta.shape)
                    #     if not isinstance(momenta, torch.Tensor):
                    #         momenta = torch.tensor(momenta, dtype=torch.float32, device='cuda:0')
                    #     smooth_gradient = self.smoothing_kernel_2.convolve(cp, cp, momenta)  
                        
                    #     # normalize 
                    #     ones = torch.from_numpy(ones).float()
                    #     sum_kernel_along_rows = self.smoothing_kernel_2.convolve(cp, cp, ones)
                    #     gradient["momenta"][comp] = smooth_gradient.cpu().numpy() / sum_kernel_along_rows.numpy()

        return gradient   

    
    def compute_haar_gradient(self, gradient):
        gradient = self.smooth_gradient(gradient)

        if self.ctf_is_happening():
            gradient = self.compute_haar_momenta_gradient(gradient)
        
        return gradient
    
    def gradient_ascent(self, parameters, new_parameters, gradient, step):
        """
            Wavelet transform the momenta
            Update the momenta with wavelet-transformed gradient
            Backward transform the momenta
        """
        haar_coef_momenta = self.compute_haar_transform(parameters['momenta'])

        if parameters["momenta"].shape[0] > 100:
            for c in range(self.n_momenta): 
                for d in range(self.dimension):
                    # Haar transform of momenta update
                    haar_coef_momenta[c][d].wc += gradient["haar_coef_momenta"][c][d].wc * step["haar_coef_momenta"]
                    
                    # Momenta update
                    momenta_recovered = haar_coef_momenta[c][d].haar_backward()
                    print(new_parameters['momenta'].shape)
                    print(momenta_recovered.shape)
                    print(momenta_recovered.flatten().shape)
                    print(new_parameters['momenta'][c][d].shape)
                    new_parameters['momenta'][c][d] = momenta_recovered.flatten()

        else:
            for c in range(self.n_momenta): 
                for d in range(self.dimension):
                    # Haar transform of momenta update
                    haar_coef_momenta[c][d].wc += gradient["haar_coef_momenta"][c][d].wc * step["haar_coef_momenta"]
                    
                    # Momenta update
                    momenta_recovered = haar_coef_momenta[c][d].haar_backward()
                    new_parameters['momenta'][c, :, d] = momenta_recovered.flatten()
        
        # def vect_norm(array, order = 2):
        #     return np.linalg.norm(array, ord = order, axis = 1)

        # if self.multiscale and self.scale > 1:
        #     logger.info("Smoothing momenta... ")

        #     tensor_scalar_type = utilities.get_torch_scalar_type('float32')
        #     device, _ = utilities.get_best_device(gpu_mode=GpuMode.KERNEL)

        #     mom = torch.from_numpy(new_parameters['momenta'])
        #     cp = torch.from_numpy(self.model.fixed_effects['control_points'])
        #     mom = utilities.move_data(mom, dtype=tensor_scalar_type, requires_grad=False, device=device)
        #     cp = utilities.move_data(cp, dtype=tensor_scalar_type, requires_grad=False, device=device)
        #     self.smoothing_kernel_ = kernel_factory.factory("keops", gpu_mode = GpuMode.KERNEL,
        #                                                kernel_width=10)
        #     filtered_mom = self.smoothing_kernel_.convolve(cp, cp, mom, mode = "gaussian_weighted")
        #     filtered_mom = filtered_mom.detach().cpu().numpy()
        #     old_norm = vect_norm(new_parameters['momenta'])
        #     new_norm = vect_norm(filtered_mom)
        #     ratio = np.expand_dims(old_norm/new_norm, axis = 1)

        #     new_parameters['momenta'] = filtered_mom * ratio

        # Update momenta
        # haar_coef_momenta = self.compute_haar_transform(parameters['momenta'])
        # updated_momenta = [np.zeros((self.n_cp, self.dimension))] * self.n_momenta
                
        # if self.n_momenta != 1:
        #     updated_momenta = np.array(updated_momenta)
        # elif updated_momenta[0].shape == parameters['momenta'].shape:
        #     updated_momenta = updated_momenta[0] 
        # else:
        #     updated_momenta = updated_momenta[0].reshape(parameters['momenta'].shape)

        # new_parameters['momenta'] = updated_momenta
        
        return new_parameters

    ####################################################################################################################
    ### Local adaptation
    ####################################################################################################################     

    def silence_fine_or_smooth_zones(self, gradient_of_coef, sujet):
        """
            Filters the values of the Haar coefficients of the gradient of the momenta
            Coefficients whose scale is lower than the current scale are set to 0

        zones: dictionary containing information about zone 
        silent_haar_coef_momenta_subject: zones to silence 
        gradient_of_coef: Haar coefficients of the gradient of the momenta
        """
        silent_haar_coef_momenta_subject = self.silent_haar_coef_momenta_subjects[sujet]
        
        indices_to_browse_along_dim = [list(range(e)) for e in list(gradient_of_coef.wc.shape)]

        for indices in itertools.product(*indices_to_browse_along_dim):
            position, type, scale = gradient_of_coef.haar_coeff_pos_type([i for i in indices])

            #silence finer zones that we haven't reached yet
            if scale < self.scale: #the higher the scale, the coarser
                gradient_of_coef.wc[indices] = 0 #fine scales : only ad, da, dd
                
            #silence smooth zones
            elif silent_haar_coef_momenta_subject and scale in silent_haar_coef_momenta_subject.keys():
                zones_to_silence_at_scale = silent_haar_coef_momenta_subject[scale]
                positions_to_silence_at_scale = [self.zones[scale][k]["position"] for k in zones_to_silence_at_scale]
                if position in positions_to_silence_at_scale and type != ['L', 'L']:
                    gradient_of_coef.wc[indices] = 0
                        
        return gradient_of_coef  

    ####################################################################################################################
    ### Optimization tools
    ####################################################################################################################     
    def enough(self, iteration):
        return iteration - self.iter[-1] > self.ctf_interval

    def too_much(self, iteration):
        return iteration - self.iter[-1] > self.ctf_max_interval

    def ctf_is_happening(self):
        """
        Check the algorithm still performs CTF
        (avoid useless computations when the end scale is reached)
        """
        return self.multiscale and self.scale > self.end_scale

    def convergence(self, iteration, cond):
        """
        Check that convergence conditions fulfilled
        """
        if not self.multiscale: return cond

        return self.scale <= self.end_scale and self.enough(iteration)
        
    def after_ctf(self, iteration):
        """
        Check that CTF just happened (to prevent convergence)
        """

        return self.multiscale and (self.iter[-1] == iteration)
    
    def folder_name(self, name):
        if not self.multiscale: return name

        return name + "scale_{}".format(self.scale)

    ####################################################################################################################
    ### Figure tools
    ####################################################################################################################
    
    # def save_heat_map(objects_extension, template, iteration, deformed_template, residuals_by_point, output_dir):
    #     names = "Heat_map_" + str(iteration) + objects_extension[0]
    #     deformed_template['image_intensities'] = residuals_by_point
    #     template.write(output_dir, [names], 
    #     {key: value.data.cpu().numpy() for key, value in deformed_template.items()})


    # def compute_area_around_point(deformation_kernel_width, dimension, template_data, point_position):
    #     limits = [[max(0, int(point_position[d] - deformation_kernel_width/2)), 
    #             min(template_data['image_intensities'].shape[d]-1, int(point_position[d] + deformation_kernel_width/2))] \
    #             for d in range(dimension)]
    #     return limits

    # def draw_silenced_point(self, limits, template_data, point_position):
    #     template_data['image_intensities'][max(0, int(point_position[0]-1)):min(template_data['image_intensities'].shape[0], int(point_position[0]+1)),
    #                                         max(0, int(point_position[1]-1)):min(template_data['image_intensities'].shape[1], int(point_position[1]+1)),
    #                                         max(0, int(point_position[2]-1)):min(template_data['image_intensities'].shape[2], int(point_position[2]+1))] = 10000
    #     #give low values around the point zone
    #     template_data['image_intensities'][limits[0][0]:limits[0][1], limits[1][0]:limits[1][1], limits[2][0]:limits[2][1]] = (-1)*max_residuals

    #     return template_data

    # def draw_silenced_coarser_zones_3d(momenta_scale, zones, deformation_kernel_width, 
    #                                 dimension, template_data, silent_haar_coef_momenta):
    #     for scale in range(max([int(k) for k in silent_haar_coef_momenta.keys()]), momenta_scale, -1):
    #         for zone in silent_haar_coef_momenta[scale]:
    #             points_in_zone = zones[scale][zone]["points"]

    #             for point_position in points_in_zone.tolist():
    #                 limits = compute_area_around_point(deformation_kernel_width, dimension, template_data, point_position)
    #                 template_data = draw_silenced_point(limits, template_data, point_position)
                    
    #     return template_data

    # def draw_silenced_finer_zones_3d(momenta_scale, zones, template_data, deformation_kernel_width, 
    #                                 dimension, silent_haar_coef_momenta):
    #     for (zone, _) in zones[momenta_scale].items():
    #         points_in_zone = zones[momenta_scale][zone]["points"]

    #         if "coarser_zone" not in zones[momenta_scale][zone].keys(): #not in silenced coarser zone
    #             if momenta_scale in silent_haar_coef_momenta.keys() \
    #                 and zone in silent_haar_coef_momenta[momenta_scale]:

    #                 for point_position in points_in_zone:
    #                     limits = compute_area_around_point(deformation_kernel_width, dimension, template_data, point_position)
    #                     template_data = draw_silenced_point(limits, template_data, point_position)
        
    #     return template_data

    # def draw_silenced_finer_zones_2d(momenta_scale, zones, deformation_kernel_width, dimension, max_residuals,
    #                                 template_data, silent_haar_coef_momenta):
    #     for (zone, _) in zones[momenta_scale].items():
    #         points_in_zone = zones[momenta_scale][zone]["points"]
    #         if momenta_scale in silent_haar_coef_momenta.keys() \
    #             and zone in silent_haar_coef_momenta[momenta_scale]:
    #             for point_position in points_in_zone:
    #                 limits = [[max(0, int(point_position[d] - deformation_kernel_width/2)), 
    #                             min(template_data['image_intensities'].shape[d]-1, int(point_position[d] + deformation_kernel_width/2))] \
    #                                 for d in range(dimension)]
    #                 #delineate zone 
    #                 template_data['image_intensities'][limits[0][0]:limits[0][1], limits[1][0]] = max_residuals
    #                 template_data['image_intensities'][limits[0][0]:limits[0][1], limits[1][1]] = max_residuals
    #                 template_data['image_intensities'][limits[0][0], limits[1][0]:limits[1][1]] = max_residuals
    #                 template_data['image_intensities'][limits[0][1], limits[1][0]:limits[1][1]] = max_residuals
        
    #     return template_data

    # def save_heat_map_with_zones(momenta_scale, zones, template, objects_extension,
    #                             iteration, output_dir, template_data, dataset, subject):
    #     zones = len(zones[momenta_scale].items())

    #     if np.any(template_data['image_intensities'].cpu().numpy()):#check that at least one value != 0 
    #         if isinstance(subject, int):
    #             subject = dataset.subject_ids[subject]
    #         names = "Sujet_" + str(subject) + "_heat_map_scale_" + str(momenta_scale) + "_residuals_zones_" \
    #                 + str(zones+1) + "iter_" + str(iteration) + objects_extension[0]

    #         template.write(output_dir, [names], {key: value.data.cpu().numpy() for key, value in template_data.items()})

    # def save_silenced_zones(output_dir, iteration, dataset, subject, silent_haar_coef_momenta_subjects,
    #                         dimension):
    #     """
    #     Save an image of the silenced zones, overimposed with residuals heat map
    #     """
    #     template_data, _ = initialize_template_before_transformation()
    #     template_data['image_intensities'] = torch.zeros(template_data['image_intensities'].shape)

    #     silent_haar_coef_momenta = silent_haar_coef_momenta_subjects[subject]

    #     if dimension == 3:
    #         template_data = draw_silenced_coarser_zones_3d(template_data, silent_haar_coef_momenta)
    #         template_data = draw_silenced_finer_zones_3d(template_data, silent_haar_coef_momenta) 
    #     else:                        
    #         template_data= draw_silenced_finer_zones_2d(template_data, silent_haar_coef_momenta)

    #     save_heat_map_with_zones(iteration, output_dir, template_data, dataset, subject)

    # def save_silenced_zones_summary(output_dir, iteration, dataset, dimension,
    #                             silent_haar_coef_momenta_subjects):
    #     """
    #     Save a summary image of the silenced zones, overimposed with residuals heat map
    #     """
    #     template_data, _ = initialize_template_before_transformation()
    #     template_data['image_intensities'] = torch.zeros(template_data['image_intensities'].shape)

    #     for subject in silent_haar_coef_momenta_subjects.keys():
    #         silent_haar_coef_momenta = silent_haar_coef_momenta_subjects[subject]

    #         if dimension == 3:
    #             template_data = draw_silenced_coarser_zones_3d(template_data, silent_haar_coef_momenta)
    #             template_data = draw_silenced_finer_zones_3d(template_data, silent_haar_coef_momenta) 
    #         else:                        
    #             template_data= draw_silenced_finer_zones_2d(template_data, silent_haar_coef_momenta)
        
    #     save_heat_map_with_zones(iteration, output_dir, template_data, dataset, "All_subjects")

    # def initialize_template_before_transformation(self):
    #     # Deform template
    #     device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)
    #     template_data, template_points, control_points, momenta, _ = \
    #         self._fixed_effects_to_torch_tensors(False, device = device)
        
    #     #####compute residuals
    #     self.exponential.set_initial_template_points(template_points)
    #     self.exponential.set_initial_control_points(control_points)

    #     return template_data, momenta

    # # def print_information_silenced_zones(self, subject = None):
    # #     if subject:
    # #         print("\nsubject", subject)
    # #         if len(self.silent_haar_coef_momenta[self.momenta.scale]) == 0:
    # #             return

    # #     print("Out of", len(self.zones[self.momenta.scale].keys()), "zones", "silence", len(self.silent_haar_coef_momenta[self.momenta.scale]))
    # #     print(self.sum_already_silent, "were already silenced by coarser zone")

    # def number_of_pixels(self, dataset):
    #     objet_intensities = dataset.deformable_objects[0][0].get_data()["image_intensities"]
    #     number_of_pixels = 1
    #     for k in range(self.dimension):
    #         number_of_pixels = number_of_pixels * objet_intensities.shape[k]
        
    #     return number_of_pixels
        
    # def compute_wavelets_position_at_momenta_scale(self, haar_coef_momenta):
    #     """
    #     Compute a list of all wavelets positions at the current scale
    #     """
    #     self.list_wavelet_positions = []
    #     if self.dimension == 3:
    #         for x in range(haar_coef_momenta[0][0].wc.shape[0]): #browse shape of cp
    #             for y in range(haar_coef_momenta[0][0].wc.shape[1]):
    #                 for z in range(haar_coef_momenta[0][0].wc.shape[2]):
    #                     position, _, scale = haar_coef_momenta[0][0].haar_coeff_pos_type((x, y, z))
    #                     if scale == self.momenta.scale and position not in self.list_wavelet_positions:
    #                         self.list_wavelet_positions.append(position)
    #     else:
    #         for x in range(haar_coef_momenta[0][0].wc.shape[0]): #browse shape of cp
    #             for y in range(haar_coef_momenta[0][0].wc.shape[1]):
    #                 position, _, scale = haar_coef_momenta[0][0].haar_coeff_pos_type((x, y))
    #                 if scale == self.momenta.scale and position not in self.list_wavelet_positions:
    #                     self.list_wavelet_positions.append(position)

    #     #in coarse scale, we can miss the small zone at the corner, which has no dd, ad, or da (and of course no aa)
    #     #no importance in the variability (it will never be divided)
        
    #     #nb_of_voxels = np.product([residuals.shape[k] for k in range(self.dimension)])

    # def fetch_control_points_in_zone(self, position, limites, control_points):
    #     """
    #         Fetch coordinates of the controls points in a specific zone
    #     """ 

    #     if self.dimension == 3:
    #         points_in_zone = control_points[position[0]:limites[0], position[1]:limites[1], position[2]:limites[2], :]
    #     else:
    #         points_in_zone = control_points[position[0]:limites[0], position[1]:limites[1], :]
        
    #     #reshape points in zone for easy browsing : n_cp x dim
    #     points_in_zone = np.reshape(points_in_zone, (np.product(points_in_zone.shape[:-1]), points_in_zone.shape[-1]))

    #     return points_in_zone
    
    # def residuals_around_point(self, model, point_position, residuals):
    #     """
    #         Given a point position,  
    #     """
    #     #limits of the zone around point (in voxels coordinates)
    #     limits = [[max(0, int(point_position[d] - model.deformation_kernel_width)), 
    #                 min(residuals.shape[d], int(point_position[d] + model.deformation_kernel_width))] for d in range(self.dimension)]
        
    #     #voxels positions around point
    #     if self.dimension == 3:
    #         voxels_pos = self.voxels_pos_image[limits[0][0]:limits[0][1], limits[1][0]:limits[1][1], limits[2][0]:limits[2][1]]
    #     else:
    #         voxels_pos = self.voxels_pos_image[limits[0][0]:limits[0][1], limits[1][0]:limits[1][1]]

    #     voxels_pos = np.reshape(voxels_pos, (np.product(voxels_pos.shape[:-1]), self.dimension)) #n voxels x 3
    #     voxels_pos = torch.tensor(voxels_pos, dtype = torch.float)

    #     #residuals around point
    #     if self.dimension == 3:
    #         voxels_res = residuals[limits[0][0]:limits[0][1], limits[1][0]:limits[1][1], limits[2][0]:limits[2][1]]
    #     else:
    #         voxels_res = residuals[limits[0][0]:limits[0][1], limits[1][0]:limits[1][1]]
        
    #     voxels_res = np.reshape(voxels_res.flatten(), (len(voxels_res.flatten()), 1))
    #     voxels_res = np.concatenate(tuple([voxels_res for k in range(self.dimension)]), axis = 1) #n voxels x 3
    #     voxels_res = torch.tensor(voxels_res, dtype = torch.float)

    #     point_position = torch.tensor(np.reshape(point_position, (1, self.dimension)), dtype = torch.float) 
        
    #     #convolve residuals around point
    #     residuals_around_point = self.model.exponential.kernel.convolve(point_position, voxels_pos, voxels_res).cpu().numpy()[0][0]
        
    #     return residuals_around_point

    # def store_zones_information(self, model, control_points, residuals):
    #     """
    #         Store information about a specific zone in a dict
    #     """
    #     self.zones[self.momenta.scale] = dict()
    #     for (zone, position) in enumerate(self.list_wavelet_positions):

    #         #fetch control points in zone - position of cp in voxels coordinates
    #         limites = [min(position[d] + self.wavelets_size[d], control_points.shape[d]) for d in range(self.dimension)]
    #         points_in_zone = self.fetch_control_points_in_zone(position, limites, control_points) 

    #         #compute residuals in zone
    #         residuals_in_zone = [self.residuals_around_point(model, position, residuals) for position in points_in_zone]
            
    #         #compute nb of voxels
    #         number_voxels_in_zone = np.product([limites[d]-position[d] for d in range(self.dimension)])

    #         self.zones[self.momenta.scale][zone] = dict()
    #         self.zones[self.momenta.scale][zone]["position"] = position
    #         self.zones[self.momenta.scale][zone]["points"] = points_in_zone
    #         self.zones[self.momenta.scale][zone]["size"] = self.wavelets_size
    #         self.zones[self.momenta.scale][zone]["residuals"] = sum(residuals_in_zone) 
    #         self.zones[self.momenta.scale][zone]["residuals_ratio"] = sum(residuals_in_zone)/number_voxels_in_zone
    
    # def residuals_subject_zone(self, position, control_points, residuals):
    #     """
    #         Store information about a specific zone in a dict
    #     """           
    #     #fetch control points in zone - position of cp in voxels coordinates
    #     limites = [min(position[d] + self.wavelets_size[d], control_points.shape[d]) for d in range(self.dimension)]
    #     points_in_zone = self.fetch_control_points_in_zone(position, limites, control_points) 

    #     #compute residuals in zone
    #     residuals_in_zone = [self.residuals_around_point(point_position, residuals) for point_position in points_in_zone]
    
    #     return sum(residuals_in_zone)

    # def compute_maximum_residuals(self):
    #     """
    #         Maximum residuals: maximum of the sum of average subjects residuals inside a zone
    #     """
    #     self.max_residuals = max([self.zones[self.momenta.scale][z]["residuals"] \
    #                             for z, _ in enumerate(self.list_wavelet_positions)])
    
    # def compute_subject_residuals(self, model, subject, dataset):
    #     """
    #     Compute residuals at each pixel/voxel between one subject and deformed template.
    #     """
    #     template_data, momenta = model.initialize_template_before_transformation()
    #     subject_residuals, _ = model.subject_residuals(subject, dataset, template_data, momenta)

    #     return subject_residuals.cpu().numpy()

    # def compute_maximum_residuals_subject(self, control_points, subjects_residuals):
    #     """
    #         Maximum residuals: maximum of the sum of average subjects residuals inside a zone
    #     """
    #     residus = []
    #     for (_, position) in enumerate(self.list_wavelet_positions):
    #         residuals_subject_zone = self.residuals_subject_zone(position, control_points, subjects_residuals)
    #         residus.append(residuals_subject_zone)
    #     self.max_residuals_subject = max(residus)

    # def silence_zones_in_silent_coarser_zone(self, zone):
    #     """
    #         Silence zones belonging to a coarser zone that was already silenced in previous coarse to fine iteration
    #     """
    #     points_in_zone = self.zones[self.momenta.scale][zone]["points"]
                        
    #     if self.momenta.scale+1 < self.coarser_scale:
    #         for coarse_zone in self.silent_haar_coef_momenta[self.momenta.scale+1]:
    #             points_in_coarse_zone = self.zones[self.momenta.scale+1][coarse_zone]["points"]
    #             #pos = self.zones[self.momenta.scale+1][coarse_zone]["position"]
    #             common_points = [point for point in points_in_zone.tolist() if point in points_in_coarse_zone.tolist()]
                
    #             if common_points != []:
    #                 self.silent_haar_coef_momenta[self.momenta.scale].append(zone)
    #                 self.zones[self.momenta.scale][zone]["coarser_zone"] = coarse_zone
    #                 self.sum_already_silent += 1

    # def silence_smooth_zones(self, zone):
    #     """
    #         Silence zones with low residuals
    #     """
    #     if self.zones[self.momenta.scale][zone]["residuals"] < 0.01*self.max_residuals \
    #     and zone not in self.silent_haar_coef_momenta[self.momenta.scale]:
    #         self.silent_haar_coef_momenta[self.momenta.scale].append(zone)
    
    # def silence_smooth_zones_subject(self, zone, residuals_zone):
    #     """
    #         Silence zones with low residuals for a subject
    #     """
    #     if residuals_zone < 0.001*self.max_residuals_subject and zone not in self.silent_haar_coef_momenta[self.momenta.scale]:
    #         self.silent_haar_coef_momenta[self.momenta.scale].append(zone)
    

    # def get_points_and_momenta(self, model, new_parameters):
    #     #get control points
    #     device, _ = utilities.get_best_device(gpu_mode=self.gpu_mode)
    #     _, _, control_points, _, haar_coef_momenta = model._fixed_effects_to_torch_tensors(False, device = device)

    #     control_points = np.reshape(control_points.cpu().numpy(), tuple(self.points_per_axis + [self.dimension]))
    #     haar_coef_momenta = new_parameters["haar_coef_momenta"]

    #     return control_points, haar_coef_momenta

    # def compute_wavelet_size(self):
    #     self.wavelets_size = [2** self.momenta.scale for d in range(self.dimension)]
    #     check = [w <= x for (w, x) in zip(self.wavelets_size, self.points_per_axis)]
        
    #     if False in check:
    #         self.wavelets_size = self.points_per_axis
    
    # def compute_voxels_position_on_image(self, residuals):
    #     #array of np.indices = self.dim x shape_1, x shape 2 (x shape 3) -> array[:, x, y, z] = [x, y, z]
    #     if self.dimension == 3:
    #         self.voxels_pos_image = np.indices((residuals.shape[0], residuals.shape[1], residuals.shape[2])).transpose((1,2,3,0))
    #     else:
    #         self.voxels_pos_image = np.indices((residuals.shape[0], residuals.shape[1])).transpose((1,2,0))

    # def initialize_silent_momenta(self, subject):
    #     self.sum_already_silent = 0
    #     self.silent_haar_coef_momenta_subjects[subject][self.momenta.scale] = []

    # def local_adaptation_subject(self, model, iteration, new_parameters, 
    #                             output_dir, dataset, residuals):
        
    #     control_points, haar_coef_momenta = self.get_points_and_momenta(model, new_parameters)

    #     #compute size : number of points in biggest wavelet
    #     self.compute_wavelet_size()

    #     #compute list of unique wavelet positions
    #     self.compute_wavelets_position_at_momenta_scale(haar_coef_momenta)
    
    #     #compute voxels positions on image
    #     self.compute_voxels_position_on_image(residuals)
        
    #     self.store_zones_information(model, control_points, residuals)              
    #     self.compute_maximum_residuals()

    #     if not self.naive:
    #         print("search smooth zones ...")
    #         for s in range(len(haar_coef_momenta)):
    #             self.initialize_silent_momenta(s)

    #             subjects_residuals = self.compute_subject_residuals(model, s, dataset)
    #             self.compute_maximum_residuals_subject(control_points, subjects_residuals)
                
    #             for (zone, position) in enumerate(self.list_wavelet_positions):
    #                 residuals_subject_zone = self.residuals_subject_zone(position, control_points, subjects_residuals)  
    #                 self.silence_zones_in_silent_coarser_zone(zone) #update self.silent_haar_coef_momenta
    #                 self.silence_smooth_zones_subject(zone, residuals_subject_zone)             
            
    #             self.silent_haar_coef_momenta_subjects[s][self.momenta.scale] = self.silent_haar_coef_momenta[self.momenta.scale].copy()

    #             self.save_silenced_zones(output_dir, iteration, dataset, s)
        
    #         self.save_silenced_zones_summary(output_dir, iteration, dataset)

    # def local_adaptation(self, iteration, new_parameters, output_dir, dataset, residuals):
    #     self.iter.append(iteration)
    #     control_points, haar_coef_momenta = self.get_points_and_momenta(new_parameters)
        
    #     print("self.momenta.scale", self.momenta.scale, "max (coarser) scale", self.coarser_scale)

    #     #compute size : number of points in biggest wavelet
    #     self.compute_wavelet_size()

    #     #compute list of unique wavelet positions
    #     self.compute_wavelets_position_at_momenta_scale(haar_coef_momenta)
    
    #     #compute voxels positions on image
    #     self.compute_voxels_position_on_image(residuals)
        
    #     self.store_zones_information(control_points, residuals)               
    #     self.compute_maximum_residuals()

    #     self.initialize_silent_momenta()

    #     if not self.naive:
    #         print("search smooth zones ...")
    #         for (zone, _) in enumerate():
    #             self.silence_zones_in_silent_coarser_zone(zone)
    #             self.silence_smooth_zones(zone)
            
    #     self.save_silenced_zones(output_dir, iteration, dataset)
    #     #self.print_information_silenced_zones()


