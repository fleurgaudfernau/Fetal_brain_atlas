import time
import torch
from torch import is_tensor
import warnings
import os.path as op
from ....core import default
from ....core.model_tools.deformations.exponential import Exponential
from ....in_out.array_readers_and_writers import *
from ....support.utilities import move_data, get_best_device, detach, interpolate, reverse_if
import logging
logger = logging.getLogger(__name__)

def concatenate(transport, reverse = False):
    if reverse:
        transport = reverse_if(transport, True)

    return transport[0] + [item for sublist in transport[1:] for item in sublist[1:]]

def _parallel_transport(*args):
    # read args
    compute_backward, exponential, momenta_to_transport, is_ortho = args

    # compute
    if compute_backward:
        return compute_backward, exponential.transport(momenta_to_transport, is_ortho=is_ortho)
    else:
        return compute_backward, exponential.transport(momenta_to_transport, is_ortho=is_ortho)

class PiecewiseGeodesic:

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, kernel=None, t0=default.t0, 
                time_concentration=default.time_concentration, nb_components=2, 
                num_components = None, template_tR=None, root_name = ''):

        self.root_name = '{}__GeodesicFlow__'.format(root_name)
        # usual t0 replaced by tR
        self.time_concentration = time_concentration
        self.tmax = None # Geodesic tmax
        self.tmin = None #Geodesic tmin
        self.t0 = t0 #Geodesic starting point (i.e. template defined at t0)

        self.cp = None
        self.momenta = None
        self.nb_components = int(nb_components)

        #contains t0 and tmax in addition to the tR
        # tR[0]=tmin, then all the rupture times then tR[-1]=tmax
        self.tR = [self.tmin] * (self.nb_components + 1) 
        self.exponential = [ Exponential(kernel=kernel) for i in range(self.nb_components)]

        # Flags to save extra computations that have already been made in the update methods.
        self.shoot_is_modified = [True] * self.nb_components

        self.template_index = None

    def new_exponential(self):
        return Exponential(kernel=self.exponential[0].kernel, use_rk2_for_shoot = True)

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################
    def best_device(self):
        return get_best_device(self.exponential[0].kernel.gpu_mode)

    def set_kernel(self, kernel):
        for l in range(self.nb_components):
            self.exponential[l].kernel = kernel

    def set_tR(self, tR):
        """
        Set the rupture times between each dynamic
        """
        for i in range(1, self.nb_components): #ignore tR[0]=tmin
            self.tR[i] = tR[i-1]

        self.shoot_is_modified = [True] * self.nb_components
    
    def get_t0(self):
        #ajout fg
        return self.t0
    
    def set_t0(self, t0 = None):
        #ajout fg
        self.t0 = t0
        assert (self.t0 in self.tR), "Template at {} must be defined at a rupture time {}".format(self.t0, self.tR)
        self.shoot_is_modified = [True] * self.nb_components

        # ajout fg: get the index of the tR=t0
        self.get_template_index()
        
    def get_template_index(self):
        for l in range(self.nb_components):
            if self.tR[l] == self.t0: 
                self.template_index = l

    def set_t1(self, t1): #ajout fg : time of momenta to transport
        self.t1 = t1

    def get_tmin(self):
        return self.tmin

    def set_tmin(self, tmin):
        self.tmin = tmin
        self.tR[0] = tmin
        self.shoot_is_modified[0] = True

    def get_tmax(self):
        return self.tmax

    def set_tmax(self, tmax):
        self.tmax = tmax
        self.tR[-1] = tmax
        self.shoot_is_modified[-1] = True

    def expo_length(self, l):
        return self.tR[l + 1] - self.tR[l]

    def exponential_n_time_points(self, l):
        return self.exponential[l].n_time_points
    
    def nb_of_tp(self, length):
        return max(1, int(length * self.time_concentration + 1.5))
    
    def set_exponential_n_time_points(self, l, length):
        #self.exponential[l].n_time_points = v
        self.exponential[l].n_time_points = self.nb_of_tp(length)

    def match_time_with_exponential(self, time):
        return self.tR.index([t for t in self.tR if t > time][0]) - 1

    def get_template_points_tR(self):
        return self.template_points_tR

    def set_template_points_tR(self, td):
        self.template_points_tR = td
        self.flow_is_modified = True

    def set_cp_tR(self, cp):
        self.cp = cp
        self.shoot_is_modified = [True]*self.nb_components

    def set_momenta_tR(self, mom):
        self.momenta = mom # an array shape n_components * n_cp * dim
        self.shoot_is_modified = [True]*self.nb_components
    
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
        if any(self.shoot_is_modified) or self.flow_is_modified:
            warnings.warn("Asking for deformed template data but the geodesic was modified and not updated")

        # All the trajectory times (defined by C° of tp)
        times = self.get_times()

        # Deal with the special case of a geodesic reduced to a single point.
        if len(times) == 1:
            return self.template_points_tR

        # Fetch index j of time closest (and above) 'time'
        for j in range(1, len(times)):
            if time - times[j] < 0: break
        
        # Mean of the two closest template points
        delta = times[j] - times[j - 1] #np.float64
        weight_l = move_data((times[j] - time) / delta, device = self.best_device())
        weight_r = move_data((time - times[j - 1]) / delta, device = self.best_device())
        template_t = {k: [move_data(v, device = self.best_device()) for v in value] \
                    for k, value in self.get_template_points_trajectory().items()}
        deformed_points = {k: interpolate(weight_l, weight_r, v, j) for k, v in template_t.items()}

        return deformed_points

    ####################################################################################################################
    ### Main methods:
    ####################################################################################################################

    def update_exponential(self, l, device):
        if self.exponential[l].is_long():
            self.exponential[l].move_data_to_(device = self.best_device())
            self.exponential[l].update()
            self.shoot_is_modified[l] = False

    def add_component(self):
        self.nb_components += 1
        self.tR = self.tR + [self.tR[-1]]

        self.shoot_is_modified = [True]*self.nb_components
        self.flow_is_modified = True
        
    def update(self):
        """
        Compute the time bounds, accordingly sets the number of points and momenta of the attribute exponentials,
        then shoot and flow them.
        """
        self.get_template_index()
                    
        # NB: exp.cp_t computed with exp.shoot() (called by exp.update())
        # Update the forward and backward exponentials at t0
        for l in range(self.template_index - 1, self.template_index + 1):
            
            length = self.expo_length(l)
            self.set_exponential_n_time_points(l, length)
                
            if self.shoot_is_modified[l]:
                self.exponential[l].set_initial_momenta(self.momenta[l] * length)
                self.exponential[l].set_initial_cp(self.cp)
            
            if self.flow_is_modified:
                self.exponential[l].set_initial_template_points(self.template_points_tR)
            
            self.update_exponential(l, self.best_device() )

        # Backward shoot                 
        for l in range(0, self.template_index - 1):
            length = self.expo_length(l)
            
            self.set_exponential_n_time_points(l, length)
                
            if self.shoot_is_modified[l]:
                self.exponential[l].set_initial_momenta(self.momenta[l] * length)
                self.exponential[l].set_initial_cp(self.exponential[l+1].last_cp())
            
            if self.flow_is_modified:
                self.exponential[l].set_initial_template_points(
                                                    self.exponential[l + 1].last_template_points())
            
            self.update_exponential(l, self.best_device() )

        # Forward shoot
        for l in range(self.template_index + 1, self.nb_components):
            length = self.expo_length(l)
            self.set_exponential_n_time_points(l, length)

            if self.shoot_is_modified[l]:
                self.exponential[l].set_initial_momenta(self.momenta[l] * length)
                self.exponential[l].set_initial_cp(self.exponential[l-1].last_cp())
            
            if self.flow_is_modified:
                self.exponential[l].set_initial_template_points(
                                                self.exponential[l - 1].last_template_points())

            self.update_exponential(l, self.best_device() )
        
        self.flow_is_modified = False

    def get_norm_squared(self, l):
        """
        Get the norm of the geodesic.
        Modif fg to handle a more flexible t0
        """
        if l < self.template_index:
            cp = self.exponential[l+1].initial_cp
        elif l == self.template_index:
            cp = self.cp
        elif l > self.template_index:
            cp = self.exponential[l-1].last_cp()

        return self.exponential[l].norm(cp, self.momenta[l])

    def get_component(self, time):
        for tR in self.tR[1:]:
            if time < tR:
                break
        return self.tR[1:].index(tR)

    ####################################################################################################################
    ### Transport:
    ####################################################################################################################

    def transport_along_expo(self, is_ortho, backward = False):
        self.transport_expo.use_rk2_for_shoot = True

        transport = ( self.transport_expo.transport(self.last_transported_mom, is_ortho)\
                    if self.transport_expo.is_long()\
                    else [self.last_transported_mom] )
        self.last_transported_mom = transport[-1]

        return reverse_if(transport, backward) # reverse order of a SINGLE transport

    def full_backward_transport(self, transport, is_ortho):
        for l in range(self.template_index -1, -1, -1):
            self.transport_expo = self.exponential[l].light_copy()
            transport.append(self.transport_along_expo(is_ortho, backward = True))

        return reverse_if(transport, True) # reverse order of transportS

    def full_forward_transport(self, transport, is_ortho):
        for l in range(self.template_index, self.nb_components):
            self.transport_expo = self.exponential[l].light_copy()
            transport.append(self.transport_along_expo(is_ortho))
        
        return transport

    def transport(self, is_ortho = False, target_time = None): #ajout fg
        """
        CLassical parallel transport: momenta_to_transport defined at t0
        """
        logger.info("Classical parallel transport: momenta to transport defined at t0")
        
        transport = []

        if target_time is None:
            logger.info("Combining full backward + forward transports from t0")
            transport = self.full_backward_transport(transport, is_ortho)
            transport = self.full_forward_transport(transport, is_ortho)

            return concatenate(transport)

        if self.t0 > target_time:
            # Backward transport
            transport = self.full_backward_transport(transport, is_ortho)
            return concatenate(transport, reverse = True)
        
        else:
            # Forward transport
            transport = self.full_forward_transport(transport, is_ortho)
            return concatenate(transport)
        
    def transport_(self, momenta_to_transport, start_time, target_time, is_ortho=False):
        """
        Parallel transport: handles special case where momenta to transport not defined
        at t0: need to get regression momenta at start time, define a new geodesic,
        transport towards the next rupture time
        """
        if any(self.shoot_is_modified):
            warnings.warn("Trying to parallel transport but the geodesic object was modified, please update before.")

        self.last_transported_mom = momenta_to_transport
        
        if start_time == self.t0:
            return self.transport(is_ortho, target_time)

        transport = []
        backward = (start_time > target_time)
        
        # Prepare first Transport-------------------------------------------------------------------------------
        l = self.match_time_with_exponential(start_time)
        logger.info("Start time {} belongs to exponential {}".format(start_time, l))  

        cp = self.get_cp_t(start_time)
        momenta = self.get_momenta_t(start_time)
        self.transport_expo = self.new_exponential()

        if backward:
            # First backward Transport------------------------------------------------------------------------            
            logger.info("<- <- First Backward transport from {} to {}".format(start_time, self.tR[l]))
            
            length = abs(start_time - self.tR[l])
            momenta *= -length if start_time > self.t0 else length
            self.transport_expo.prepare_and_update(cp, momenta, length = self.nb_of_tp(length), 
                                                    device = self.best_device())
            transport.append(self.transport_along_expo(is_ortho, backward))

            # Backward Transport------------------------------------------------------------------------            
            for m in range(l - 1, -1, -1):
                if target_time < self.tR[m + 1]:
                    print("\n<- <- Backward transport from {} to {} (expo {})".format(self.tR[m + 1], self.tR[m], m))
                    self.transport_expo = self.exponential[m]
                    transport.append(self.transport_along_expo(is_ortho, backward))

        else:
            # First forward Transport------------------------------------------------------------------------            
            logger.info("-> -> First Forward transport from {} to {}".format(start_time, self.tR[l+1]))

            length = abs(start_time - self.tR[l+1])
            momenta *= -length if start_time < self.t0 else length
            self.transport_expo.prepare_and_update(cp, momenta, length = self.nb_of_tp(length), 
                                                    device = self.best_device())
            transport.append(self.transport_along_expo(is_ortho))
            
            # Forward Transport------------------------------------------------------------------------            
            for m in range(l+1, self.nb_components):
                if target_time > self.tR[m]:
                    print("-> -> Forward transport from {} to {} (expo {})".format(self.tR[m], self.tR[m+1], m))
                    self.transport_expo = self.exponential[m].light_copy()
                    transport.append(self.transport_along_expo(is_ortho))

        return concatenate(transport, reverse = backward)

    ####################################################################################################################
    ### Extension methods:
    ####################################################################################################################

    def convert_to_array(self, tR_l):
        if torch.is_tensor(tR_l) and tR_l.device != "cpu":
            tR_l = tR_l.cpu()
        if torch.is_tensor(tR_l) and tR_l.requires_grad:
            tR_l = tR_l.detach()
        return tR_l

    def get_times(self):
        times = [ np.linspace( self.convert_to_array(self.tR[l]), self.convert_to_array(self.tR[l + 1]),
                                num = self.exponential[l].n_time_points).tolist() \
                    if self.exponential[l].is_long() \
                    else [ self.convert_to_array(self.tR[l]) ]\
                    for l in range(self.nb_components) ]
                        
        times_concat = times[0] + [t for sublist in times[1:] for t in sublist[1:]]

        return times_concat
    
    def get_time_index(self, time = None):
        if time is None or len(self.get_times()) == 1:
            return None

        # Standard case.
        for j, t in enumerate(self.get_times()):
            if time <= t:
                return j

    def get_cp_trajectory(self):
        if any(self.shoot_is_modified):
            warnings.warn("Trying to get cp trajectory in a non updated geodesic.")

        cp_t = []
        for l in range(self.nb_components):
            if self.exponential[l].is_long():
                ajout = reverse_if(self.exponential[l].cp_t, self.tR[l] < self.t0)
                ajout = ajout[1:] if l > 0 else ajout
                cp_t += ajout
                    
        return cp_t

    def get_momenta_trajectory(self):
        if any(self.shoot_is_modified):
            warnings.warn("Trying to get mom trajectory in non updated geodesic.")

        # modif fg: flexible t0
        momenta_t = []
        for l in range(self.nb_components):
            if self.exponential[l].is_long():
                mom = reverse_if(self.exponential[l].momenta_t, self.tR[l] < self.t0)
                mom = mom[1:] if l > 0 else mom

                momenta_t += [elt/self.expo_length(l) for elt in mom]

        return momenta_t

    def get_template_points_trajectory(self):
        """
        Return list of template points along trajectory
        by adding all template points along all exponentials
        """
        if any(self.shoot_is_modified) or self.flow_is_modified:
            warnings.warn("Trying to get template trajectory in non updated geodesic.")

        # to work it is necessary to have tmin < t0 = at least one subject younger that t0...
        template_t = {k : [] for k in self.template_points_tR.keys()}
        
        for key in self.template_points_tR.keys():
            # Get all template points from 1st expo + all template points but 1st one
            for l in range(self.nb_components):
                if self.exponential[l].is_long():
                    ajout = reverse_if(self.exponential[l].template_points_t[key], self.tR[0] < self.t0)
                    ajout = ajout[1:] if l > 0 else ajout
                    template_t[key] += ajout # append: wrong!

        return template_t
    
    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def output_path(self, output_dir):
        self.flow_path = {}
        self.momenta_flow_path = {}

        step = self.time_concentration if self.time_concentration > 1 else 0.5
        step = 1
            
        for t, time in enumerate(self.get_times()):
            component = self.get_component(time)

            names = [flow_name(self.root_name, t, time, component)]
            
            if (t % step == 0) or time == self.tR[component + 1]:
                self.flow_path[time] = op.join(output_dir, names[0])

            self.momenta_flow_path[time] = op.join(output_dir, momenta_name(self.root_name, t, time))

    def write(self, template, template_data, output_dir, write_adjoint_parameters = False, write_all = True):
        
        # Core loop ----------------------------------------------------------------------------------------------------
        if write_all:
            step = self.time_concentration if self.time_concentration > 1 else 0.5
            for t, time in enumerate(self.get_times()):
                component = self.get_component(time)
                
                if (t % step == 0) or time == self.tR[component + 1]:
                    names = [ flow_name(self.root_name, t, time, component)]
                    deformed_points = self.get_template_points(time)
                    deformed_data = template.get_deformed_data(deformed_points, template_data)

                    template.write(output_dir, names, deformed_data, momenta = self.get_momenta_t(time), 
                                    cp = self.get_cp_t(time), kernel = self.exponential[0].kernel)

        # Optional writing of the control points and momenta -----------------------------------------------------------
        #if write_adjoint_parameters:
        if True:
            for t, (time, cp, momenta) in enumerate(zip(self.get_times(), 
                                    self.get_cp_trajectory(), self.get_momenta_trajectory())):
                concatenate_for_paraview(momenta, cp, output_dir, self.root_name)
