import torch
import time
from ....core import default
from ....core.model_tools.deformations.exponential import Exponential
from ....core.model_tools.deformations.piecewise_geodesic import PiecewiseGeodesic
from ....in_out.array_readers_and_writers import *
#from ....support import utilities
from ....support.utilities import interpolate


class SpatialPiecewiseGeodesic:
    """
    Control-point-based LDDMM spatio-temporal reference frame, based on exp-parallelization.
    See "Learning distributions of shape trajectories from longitudinal datasets: a hierarchical model on a manifold
    of diffeomorphisms", Bône et al. (2018), in review.

    """
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, kernel=None, tR=default.t0, time_concentration=default.time_concentration,
                 n_time_points=default.n_time_points, template_tR=None, nb_components=2, 
                 num_components=None, root_name = ""):

        self.exponential = Exponential(kernel=kernel, n_time_points=n_time_points)

        self.geodesic = PiecewiseGeodesic(kernel=kernel, 
                                    time_concentration=time_concentration, template_tR=template_tR,
                                    nb_components=nb_components, num_components=num_components,
                                    root_name = root_name)

        self.mod_matrix_tR = None
        self.projected_mod_matrix_tR = None
        self.projected_mod_matrix_t = None
        self.number_of_sources = None
        self.nb_components = nb_components

        self.transport_is_modified = True

        self.times = None
        self.template_points_t = None
        self.cp_t = None

        self.root_name = root_name

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_kernel(self, kernel):
        self.geodesic.set_kernel(kernel)
        self.exponential.set_kernel(kernel)

    def get_kernel_width(self):
        return self.exponential.kernel.kernel_width

    def get_time_concentration(self):
        return self.geodesic.time_concentration

    def set_time_concentration(self, ctp):
        self.geodesic.time_concentration = ctp

    def set_n_time_points(self, ntp):
        self.exponential.n_time_points = ntp
    
    def nb_of_tp(self):
        return self.exponential.n_time_points

    def set_template_points_tR(self, td):
        self.geodesic.set_template_points_tR(td)

    def set_cp_tR(self, cp):
        self.geodesic.set_cp_tR(cp)
        self.transport_is_modified = True

    def set_momenta_tR(self, mom):
        self.geodesic.set_momenta_tR(mom)
        self.transport_is_modified = True
        self.momenta_size = mom[0].size()

    def set_modulation_matrix_tR(self, mm):
        self.mod_matrix_tR = mm
        self.number_of_sources = mm.size()[1]
        self.transport_is_modified = True

    def set_t0(self, t0):
        self.geodesic.set_t0(t0)
        self.transport_is_modified = True

    def set_tR(self, tR):
        self.geodesic.set_tR(tR)
        self.transport_is_modified = True

    def get_tmin(self):
        return self.geodesic.get_tmin()

    def set_tmin(self, tmin, optimize=False):
        self.geodesic.set_tmin(tmin)
        self.transport_is_modified = True

    def get_tmax(self):
        return self.geodesic.get_tmax()

    def set_tmax(self, tmax, optimize=False):
        self.geodesic.set_tmax(tmax)
        self.transport_is_modified = True
    
    def add_component(self):
        self.geodesic.add_component()
    
    def get_exponential(l):
        return self.geodesic.exponential[l]
    
    def exponential_initial_cp(l):
        return self.get_exponential(l).get_initial_cp()

    def add_exponential(self, c):
        self.geodesic.add_exponential(c)
    
    def get_space_shift(self, s):
        return self.projected_mod_matrix_tR[:, s].contiguous().view(self.momenta_size)
    
    def get_template_points(self, time, sources, device=None):
        """
            Get the template points deformed at a given time + shot by given sources
        """

        # Assert for coherent length of attribute lists.
        assert len(self.template_points_t[list(self.template_points_t.keys())[0]]) == len(self.cp_t) \
               == len(self.projected_mod_matrix_t) == len(self.times)

        # Deal with the special case of a geodesic reduced to a single point.
        if len(self.times) == 1:
            logger.info('>> The spatiotemporal reference frame geodesic is reduced to a single point.')
            self.exponential.set_initial_template_points({k: v[0] for k, v in self.template_points_t.items()})
            self.exponential.set_initial_cp(self.cp_t[0])
            self.exponential.set_initial_momenta(torch.mm(self.projected_mod_matrix_t[0],
                                                          sources.unsqueeze(1)).view(self.momenta_size))

        # Standard case.
        else:
            # Get the template points at time t
            # get modulation matrix at time t -> time shift
            # Shoot the template points at time t with space shift using the Exp
            i = next((i for i, t in enumerate(self.times[1:]) if time <  t), None)
            time_diff = self.times[i] - self.times[i - 1]
            weight_L = (self.times[i] - time) / time_diff
            weight_R = (time - self.times[i - 1]) / time_diff

            template_points = {k: interpolate(weight_L, weight_R, v, i)
                               for k, v in self.template_points_t.items()}
            control_points = interpolate(weight_L, weight_R, self.cp_t, i)
            modulation_matrix = interpolate(weight_L, weight_R, self.projected_mod_matrix_t, i)
            space_shift = torch.mm(modulation_matrix, sources.unsqueeze(1)).view(self.momenta_size)

            self.exponential.set_initial_template_points(template_points)
            self.exponential.set_initial_cp(control_points)
            self.exponential.set_initial_momenta(space_shift)

        if device is not None:
            self.exponential.move_data_to_(device)
        self.exponential.update()

        return self.exponential.get_template_points()

    def _get_interpolation_index_and_weights(self, time):
        for i in range(1, len(self.times)):
            if time - self.times[i] < 0: #before : time.data.cpu().numpy()
                break

        time_diff = self.times[i] - self.times[i - 1]
        weight_L = (self.times[i] - time) / time_diff
        weight_R = (time - self.times[i - 1]) / time_diff

        return i, weight_L, weight_R

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Update the geodesic, and compute the parallel transport of each column of the modulation matrix along
        this geodesic, ignoring the tangential components.
        """
        device = self.geodesic.control_points.device

        # Update the geodesic.
        self.geodesic.update()

        # Convenient attributes for later use.
        self.times = self.geodesic.get_times()
        self.template_points_t = self.geodesic.get_template_points_trajectory()
        self.cp_t = self.geodesic.get_cp_trajectory()

        if self.transport_is_modified:
            # Projects the modulation_matrix_t0 attribute columns (orthogonal to geodesic momenta)
            self._update_projected_mod_matrix_tR(device=device)

            # Initializes the projected_mod_matrix_t attribute size.
            self.projected_mod_matrix_t = [torch.zeros(self.mod_matrix_tR.size(), 
                                                dtype=self.mod_matrix_tR.dtype, device=device)
                                                for _ in range(len(self.cp_t))]

            # Transport each column, ignoring the tangential components.
            for s in range(self.number_of_sources):
                #print("\nPT for source", s)
                space_shift_tR = self.get_space_shift(s)
                space_shift_t = self.geodesic.transport(space_shift_tR, is_orthogonal=False)

                # Set the result correctly in the projected_mod_matrix_t attribute.
                for t, space_shift in enumerate(space_shift_t):
                    try:
                        print("space_shift.view(-1) working ?")
                        ss = space_shift.view(-1)
                    except:
                        print("no: space_shift.contiguous().view(-1) working")
                        ss = space_shift.contiguous().view(-1)
                    self.projected_mod_matrix_t[t][:, s] = ss

            self.transport_is_modified = False

            t3 = time.perf_counter()

        assert len(self.template_points_t[list(self.template_points_t.keys())[0]]) == len(self.cp_t) \
               == len(self.times) == len(self.projected_mod_matrix_t), \
            "That's weird: len(self.template_points_t[list(self.template_points_t.keys())[0]]) = %d, " \
            "len(self.cp_t) = %d, len(self.times) = %d,  len(self.projected_mod_matrix_t) = %d" % \
            (len(self.template_points_t[list(self.template_points_t.keys())[0]]), len(self.cp_t),
             len(self.times), len(self.projected_mod_matrix_t))

    ####################################################################################################################
    ### Auxiliary methods:
    ####################################################################################################################

    def _update_projected_mod_matrix_tR(self, device='cpu'):
        self.projected_mod_matrix_tR = self.mod_matrix_tR.clone()
        momenta_orth = self.ortho()   

        for l in range(self.nb_components):
            initial_cp_l = self.exponential_initial_cp(l)
            norm_squared = self.get_exponential(l).norm(initial_cp_l, momenta_orth[l])

            if norm_squared != 0:
                for s in range(self.number_of_sources):
                    space_shift_tR = self.get_space_shift(s).clone()
                    sp = self.get_exponential(l).scalar_product(initial_cp_l, momenta_orth[l], space_shift_tR) / norm_squared
                    # orthogonal projection of SS WR to ortho momenta
                    projected_space_shift_tR = space_shift_tR - sp * momenta_orth[l]
                    self.projected_mod_matrix_tR[:, s] = projected_space_shift_tR.view(-1).contiguous()

    def ortho(self):
        """
            Compute for each component, the momenta orthogonal to all the others momenta components
        """
        # old code: for stochastic optimization = no gradient computation = variables not on torch
        if torch.is_tensor(self.geodesic.momenta):
            print('spatial_piecewise_geo-> ortho 1')
            momenta = self.geodesic.momenta.clone()
            momenta_ortho = self.geodesic.momenta.clone()
        else:
            print('spatial_piecewise_geo-> ortho 2')
            momenta = self.geodesic.momenta.copy()
            momenta_ortho = self.geodesic.momenta.copy()

        for k in range(1, self.n_components): # for each component/subject (before: momenta.__len__())
            for l in range(k): #
                # Make momenta k ortho to the momenta that precede it
                initial_cp_l = self.exponential_initial_cp()
                norm_squared = self.get_exponential(l).norm(initial_cp_l, momenta[l])
                if norm_squared != 0:
                    sp_to_ortho = self.get_exponential(l).scalar_product(initial_cp_l, momenta[k], momenta[l]) / norm_squared
                    momenta_ortho.data[k] = momenta_ortho.data[k] - sp_to_ortho * momenta_ortho.data[l]

        return momenta_ortho

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, template, template_data, output_dir,
              write_adjoint_parameters = False, write_exponential_flow=False, write_all = True):

        # Write the geodesic -------------------------------------------------------------------------------------------
        self.geodesic.write(template, template_data, output_dir, write_adjoint_parameters, write_all) 
        #step = 10 if not write_all else 1

        # Write the orthogonal flow ------------------------------------------------------------------------------------
        # Plot the flow up to three standard deviations.
        if write_all:
            step = self.nb_of_tp() - 1
            self.set_n_time_points(1 + 3 * (self.nb_of_tp() - 1))
            for s in range(self.number_of_sources):
                for sign, si in  zip([1, -1], ["+", "-"]): # Direct and indirect flows
                    space_shift = self.get_space_shift(s)
                    self.exponential.set_initial_template_points(self.geodesic.template_points_tR)
                    self.exponential.set_initial_cp(self.geodesic.control_points)
                    self.exponential.set_initial_momenta(sign * space_shift)
                    self.exponential.update()

                    concatenate_for_paraview(sign * space_shift, self.geodesic.control_points, output_dir, 
                                            "For_paraview__GeometricMode_{}__sign_{}.vtk".format(s, si))

                    for j in range(step, self.nb_of_tp(), step):
                        names = ['{}__GeometricMode_{}__{}__{}{}_sigma'\
                                .format(self.root_name, s, self.nb_of_tp() - 1 + j,
                                 si, (3. * float(j) / (self.nb_of_tp() - 1)))]
                        deformed_points = self.exponential.get_template_points(j)
                        deformed_data = template.get_deformed_data(deformed_points, template_data)
                        template.write(output_dir, names, deformed_data)

        # Correctly resets the initial number of time points.
        self.set_n_time_points(1 + (self.nb_of_tp() - 1) // 3)

        # Optionally write the projected modulation matrices along the geodesic flow -----------------------------------
        if write_adjoint_parameters:
            times = self.geodesic.get_times()
            for t, (time, modulation_matrix) in enumerate(zip(times, self.projected_mod_matrix_t)):
                write_mod_matrix_flow(modulation_matrix, output_dir, self.root_name, t, time)

        # Optionally write the exp-parallel curves and associated flows (massive writing) ------------------------------
        if write_exponential_flow:
            times = self.geodesic.get_times()
            for t, (time, modulation_matrix) in enumerate(zip(times, self.projected_mod_matrix_t)):
                for s in range(self.number_of_sources):

                    # Forward: uses the shooting exponential
                    # takes the template and cp at time t and shoot them using space shift at time t
                    space_shift = self.get_space_shift(s)
                    self.exponential.set_initial_template_points({key: v[t] for key, v in self.template_points_t.items()})
                    self.exponential.set_initial_cp(self.cp_t[t])
                    self.exponential.set_initial_momenta(space_shift)
                    self.exponential.update()

                    names = [ space_shift_name(self.root_name, s, t, time)]
                    self.exponential.write_flow(names, template, template_data, output_dir,
                                                write_adjoint_parameters, write_only_last = True)

                    # Backward
                    self.exponential.set_initial_momenta(- space_shift)
                    self.exponential.update()

                    names = [ space_shift_name(self.root_name, s, t, time, backward = True)]
                    self.exponential.write_flow(names, template, template_data, output_dir,
                                                write_adjoint_parameters, write_only_last = True)
