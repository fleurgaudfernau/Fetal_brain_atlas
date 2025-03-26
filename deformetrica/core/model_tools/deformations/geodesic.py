import time
import warnings
import os.path as op

from ....core import default
from ....core.model_tools.deformations.exponential import Exponential
from ....in_out.array_readers_and_writers import *
from ....support.utilities import get_best_device, move_data, detach, interpolate, reverse_if

import logging
logger = logging.getLogger(__name__)


def _parallel_transport(*args):

    # read args
    compute_backward, exponential, momenta_to_transport, is_ortho = args

    # compute
    if compute_backward:
        return compute_backward, exponential.transport(momenta_to_transport, is_ortho)
    else:
        return compute_backward, exponential.transport(momenta_to_transport, is_ortho)

def concatenate(backward, forward):
    return backward[::-1] + forward[1:]

class Geodesic:
    """
    Control-point-based LDDMM geodesic.
    See "Morphometry of anatomical shape complexes with dense deformations and sparse parameters",
    Durrleman et al. (2013).
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, kernel=None, t0=default.t0, time_concentration=default.time_concentration,
                 root_name = ""):

        self.root_name = '{}__GeodesicFlow__'.format(root_name)
        self.time_concentration = time_concentration
        self.t0 = t0
        self.tmax = None
        self.tmin = None

        self.cp_t0 = None
        self.momenta_t0 = None
        self.template_points_t0 = None

        self.bw_exponential = Exponential(kernel=kernel, use_rk2_for_shoot = True)
        self.fw_exponential = Exponential(kernel=kernel, use_rk2_for_shoot = True)

        # Flags to save extra computations that have already been made in the update methods.
        self.shoot_is_modified = True
        self.flow_is_modified = True
        self.backward_extension = 0
        self.forward_extension = 0
    
    def new_exponential(self):
        return Exponential(kernel = self.bw_exponential.kernel, use_rk2_for_shoot = True)

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################
    def set_kernel(self, kernel):
        self.bw_exponential.kernel = kernel
        self.fw_exponential.kernel = kernel
        
    def best_device(self):
        return get_best_device(self.bw_exponential.kernel.gpu_mode)

    def set_t0(self, t0):
        self.t0 = t0
        self.shoot_is_modified = True
    
    def set_t1(self, t1): #ajout fg : time of momenta to transport
        self.t1 = t1

    def get_tmin(self):
        return self.tmin

    def set_tmin(self, tmin, optimize=False):
        if not optimize:
            self.tmin = tmin
            self.shoot_is_modified = True

        else:
            if self.tmin is None:
                self.tmin = tmin

            elif tmin < self.tmin:
                if self.bw_exponential.is_long():
                    dt = (self.t0 - self.tmin) / float(self.bw_exponential.n_time_points - 1)
                    self.backward_extension = int((self.tmin - tmin) / dt)
                    self.tmin -= self.backward_extension * dt
                else:
                    self.tmin = tmin
                    length = self.t0 - self.tmin
                    self.backward_extension = max(0, int(length * self.time_concentration + 0.5))
                    self.bw_exponential.set_initial_momenta(- self.momenta_t0 * length)

    def get_tmax(self):
        return self.tmax

    def set_tmax(self, tmax, optimize=False):
        if not optimize:
            self.tmax = tmax
            self.shoot_is_modified = True

        else:
            if self.tmax is None:
                self.tmax = tmax

            elif tmax > self.tmax:
                if self.fw_exponential.is_long():
                    dt = (self.tmax - self.t0) / float(self.fw_exponential.n_time_points - 1)
                    self.forward_extension = int((tmax - self.tmax) / dt)
                    self.tmax += self.forward_extension * dt
                else:
                    self.tmax = tmax
                    length = self.tmax - self.t0
                    self.forward_extension = max(0, int(length * self.time_concentration + 0.5))
                    self.fw_exponential.set_initial_momenta(self.momenta_t0 * length)

    def get_template_points_t0(self):
        return self.template_points_t0

    def set_template_points_t0(self, td):
        self.template_points_t0 = td
        self.flow_is_modified = True

    def set_cp_t0(self, cp):
        self.cp_t0 = cp
        self.shoot_is_modified = True

    def set_momenta_t0(self, mom):
        self.momenta_t0 = mom
        self.shoot_is_modified = True
        
    def get_momenta_t(self, time = None, transform = detach):
        j = self.get_time_index(time)

        momenta_t = [transform(elt) for elt in self.get_momenta_trajectory()]

        return momenta_t[j] if j is not None else momenta_t[-1]
    
    def get_cp_t(self, time=None, transform = detach):
        j = self.get_time_index(time)

        cp_t = [transform(elt) for elt in self.get_cp_trajectory()]

        return cp_t[j] if j is not None else cp_t[-1]

    def get_template_points(self, time):
        """
        Returns the position of the landmark points, at the given time.
        Performs a linear interpolation between the two closest available data points.
        """
        assert self.tmin <= time <= self.tmax
        if self.shoot_is_modified or self.flow_is_modified:
            warnings.warn("Asking for deformed template data but geodesic was modified and not updated")

        times = self.get_times()
        j = self.get_time_index(time)

        if j is None:
            return self.template_points_t0

        delta = times[j] - times[j - 1]
        weight_l = move_data((times[j] - time) / delta, device = self.best_device() )
        weight_r = move_data((time - times[j - 1]) / delta, device = self.best_device())
        template_t = {key: [move_data(v, device = self.best_device()) for v in value]\
                    for key, value in self.get_template_points_trajectory().items()}
        deformed_points = {k: interpolate(weight_l, weight_r, v, j) for k, v in template_t.items()}

        return deformed_points

    ####################################################################################################################
    ### Main methods:
    ####################################################################################################################

    def update(self):
        """
        Compute the time bounds, accordingly sets the number of points and momenta of the attribute exponentials,
        then shoot and flow them.
        """
        assert self.t0 >= self.tmin, "tmin should be smaller than t0"
        assert self.t0 <= self.tmax, "tmax should be larger than t0"

        if self.shoot_is_modified or self.flow_is_modified:

            # Backward exponential -------------------------------------------------------------------------------------
            length = self.t0 - self.tmin
            self.bw_exponential.n_time_points = self.nb_of_tp(length)
            if self.shoot_is_modified:
                self.bw_exponential.set_initial_momenta(- self.momenta_t0 * length)
                self.bw_exponential.set_initial_cp(self.cp_t0)

            if self.flow_is_modified:
                self.bw_exponential.set_initial_template_points(self.template_points_t0)

            if self.bw_exponential.is_long():
                self.bw_exponential.move_data_to_(device = self.best_device())
                self.bw_exponential.update()

            # Forward exponential --------------------------------------------------------------------------------------
            length = self.tmax - self.t0
            self.fw_exponential.n_time_points = self.nb_of_tp(length)

            if self.shoot_is_modified:
                self.fw_exponential.set_initial_momenta(self.momenta_t0 * length)
                self.fw_exponential.set_initial_cp(self.cp_t0)

            if self.flow_is_modified:
                self.fw_exponential.set_initial_template_points(self.template_points_t0)

            if self.fw_exponential.is_long():
                self.fw_exponential.move_data_to_(device = self.best_device())
                self.fw_exponential.update() 

        else:
            if self.backward_extension > 0:
                self.bw_exponential.extend(self.backward_extension)

            if self.forward_extension > 0:
                self.fw_exponential.extend(self.forward_extension)
        
        self.shoot_is_modified, self.flow_is_modified = False, False
        self.backward_extension, self.forward_extension = 0, 0
    
    def nb_of_tp(self, length):
        return max(1, int(length * self.time_concentration + 1.5))

    def get_norm_squared(self):
        """
        Get the norm of the geodesic.
        """
        return self.fw_exponential.scalar_product(self.cp_t0, self.momenta_t0, self.momenta_t0)

    ####################################################################################################################
    ### Parallel transport:
    ####################################################################################################################

    def transport(self, momenta_to_transport, target_time = None, is_ortho=False):
        """
        :param momenta_to_transport: vector to transport, given at t0 and carried at cp_t0
        :returns: the full trajectory of the parallel transport, from tmin to tmax.
        """
        logger.info("Classical parallel transport: momenta to transport defined at t0")
        
        if target_time is None:
            logger.info("Combining backward + forward transports from t0")

            backward_transport = ( self.bw_exponential.transport(momenta_to_transport, is_ortho)
                                    if self.bw_exponential.is_long()
                                    else [momenta_to_transport] )

            forward_transport = (self.fw_exponential.transport(momenta_to_transport, is_ortho)
                                if self.fw_exponential.is_long() else [] )

            return concatenate(backward_transport, forward_transport)
        
        if target_time < self.t0:
            logger.info("-> -> Backward transport from t0={} to {}".format(self.t0, target_time))
            return self.bw_exponential.transport(momenta_to_transport, is_ortho)
        
        elif target_time >= self.t0:
            logger.info("<- <- Forward transport from t0={} to {}".format(self.t0, target_time))
            return self.fw_exponential.transport(momenta_to_transport, is_ortho)

    def transport_(self, momenta_to_transport, start_time, target_time, is_ortho=False):
        """
        Special case of a transport between start_time and target_time,
        instead of transport from t0 to tmax and tmin
        """
        if self.shoot_is_modified:
            msg = "Trying to parallel transport but the geodesic object was modified, please update before."
            warnings.warn(msg)
            
        if start_time == self.t0:
            return self.transport(momenta_to_transport, target_time, is_ortho)
        
        transport = [momenta_to_transport]
        
        backward = (start_time > target_time)
        
        # Prepare first Transport-------------------------------------------------------------------------------
        cp = self.get_cp_t(start_time)
        momenta = self.get_momenta_t(start_time)
        new_expo = self.new_exponential()

        # Only one bwd (/ fwd) transport along bwd (/fwd) exponential --------------------------------------------------------------------------
        if (backward and start_time < self.t0) or (not backward and start_time > self.t0):
            logger.info("Only one transport from {} to {}".format(start_time, target_time))
            length = abs(start_time - target_time)
            new_expo.prepare_and_update(cp, momenta * length, length = self.nb_of_tp(length), 
                                        device = self.best_device())
            transport = new_expo.transport(momenta_to_transport, is_ortho)

            return reverse_if(transport, backward)

        length = abs(start_time - self.t0)

        if backward:
            # First backward transport along fw exponential --------------------------------------------------------------------------            
            logger.info("First transport from {} to {}".format(start_time, self.t0))
            new_expo.prepare_and_update(cp, -momenta * length, length = self.nb_of_tp(length), 
                                        device = self.best_device())
            transport = new_expo.transport(momenta_to_transport, is_ortho)
            last_transported_mom = transport[-1] 
            
            # Second Backward Transport ------------------------------------------------------------------------------------------------
            if self.bw_exponential.is_long():
                logger.info("<- <- Backward transport from {} to {}".format(self.t0, target_time))
                transport += self.bw_exponential.transport(last_transported_mom, is_ortho)[1:]
        else:
            # First Forward Transport along bw exponential --------------------------------------------------------------------------            
            logger.info("First transport from {} to {}".format(start_time, self.t0))
            new_expo.prepare_and_update(cp, -momenta * length, length = self.nb_of_tp(length), 
                                        device = self.best_device())
            transport = new_expo.transport(last_transported_mom, is_ortho)
            last_transported_mom = transport[-1]

            # Second Forward Transport ------------------------------------------------------------------------------------------------
            if self.fw_exponential.is_long():
                logger.info("-> -> Forward transport from {} to {}".format(self.t0, target_time))
                transport += self.fw_exponential.transport(last_transported_mom, is_ortho)[1:]

        return reverse_if(transport, backward)

    def get_times(self):
        times_backward = np.linspace(self.t0, self.tmin, 
                                        num=self.bw_exponential.n_time_points).tolist()\
                        if self.bw_exponential.is_long() else [self.t0]

        times_forward = np.linspace(self.t0, self.tmax, 
                                        num=self.fw_exponential.n_time_points).tolist()\
                        if self.fw_exponential.is_long() else [self.t0]

        return concatenate(times_backward, times_forward)
    
    def get_time_index(self, time = None):
        # Deal with the special case of a geodesic reduced to a single point.
        if time is None or len(self.get_times()) == 1:
            return None

        # Standard case.
        for j, t in enumerate(self.get_times()):
            if time <= t:
                return j

    def get_cp_trajectory(self):
        if self.shoot_is_modified:
            warnings.warn("Trying to get cp trajectory in a non updated geodesic.")

        backward_cp_t = self.bw_exponential.cp_t if self.bw_exponential.is_long()\
                        else [self.bw_exponential.initial_cp]

        forward_cp_t = self.fw_exponential.cp_t if self.fw_exponential.is_long()\
                        else [self.fw_exponential.initial_cp]

        return concatenate(backward_cp_t, forward_cp_t)

    def get_momenta_trajectory(self):
        if self.shoot_is_modified:
            warnings.warn("Trying to get mom trajectory in non updated geodesic.")

        backward_momenta_t = [self.momenta_t0]
        if self.bw_exponential.is_long():
            backward_length = self.tmin - self.t0
            backward_momenta_t = [elt / backward_length for elt in self.bw_exponential.momenta_t]

        forward_momenta_t = [self.momenta_t0]
        if self.fw_exponential.is_long():
            forward_length = self.tmax - self.t0
            forward_momenta_t = [elt / forward_length for elt in self.fw_exponential.momenta_t]

        return concatenate(backward_momenta_t, forward_momenta_t)

    def get_template_points_trajectory(self):
        """
        Return list of template points along trajectory
        """
        if self.shoot_is_modified or self.flow_is_modified:
            warnings.warn("Trying to get template trajectory in non updated geodesic.")

        template_t = {}
        for key in self.template_points_t0.keys():

            backward_template_t = self.bw_exponential.template_points_t[key]\
                                    if self.bw_exponential.is_long()\
                                    else [self.bw_exponential.get_initial_template_points()[key]]

            forward_template_t = self.fw_exponential.template_points_t[key]\
                                    if self.fw_exponential.is_long()\
                                    else [self.fw_exponential.get_initial_template_points()[key]]

            template_t[key] = concatenate(backward_template_t, forward_template_t)

        return template_t

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def output_path(self, output_dir, write_all = False):
        self.flow_path = {}
        self.momenta_flow_path = {}

        if self.time_concentration > 2:
            step = self.time_concentration
        
        for t, time in enumerate(self.get_times()):
            name = flow_name(self.root_name, t, time)
            
            if write_all and  t % step == 0:
                self.flow_path[time] = op.join(output_dir, name)
            
            self.momenta_flow_path[time] = op.join(output_dir, momenta_name(self.root_name, time, age))
        
        if not write_all:
            self.flow_path[time] = op.join(output_dir, name)

    def write(self, template, template_data, output_dir, write_adjoint_parameters=False, write_all = True):

        # Core loop ----------------------------------------------------------------------------------------------------
        if write_all:
            step = self.time_concentration if self.time_concentration > 1 else 0.5
            
            for t, time in enumerate(self.get_times()):
                names = [flow_name(self.root_name, t, time)]
                deformed_points = self.get_template_points(time)
                deformed_data = template.get_deformed_data(deformed_points, template_data)

                if t % step == 0:
                    template.write(output_dir, names, deformed_data)

        # Optional writing of the control points and momenta -----------------------------------------------------------
        if write_adjoint_parameters:
            for t, (time, cp, momenta) in enumerate(zip(self.get_times(), self.get_cp_trajectory(), 
                                                        self.get_momenta_trajectory())):
                write_momenta(momenta, output_dir, self.root_name, time = t, age = time)

                concatenate_for_paraview(momenta, cp, output_dir, self.root_name)
