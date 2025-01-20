import time
import torch
from torch import is_tensor
import warnings
import os.path as op
from ....core import default
from ....core.model_tools.deformations.exponential import Exponential
from ....in_out.array_readers_and_writers import *
from ....support import utilities

import logging
logger = logging.getLogger(__name__)


def _parallel_transport(*args):
    # read args
    compute_backward, exponential, momenta_to_transport_tR, is_orthogonal = args

    # compute
    if compute_backward:
        return compute_backward, exponential.parallel_transport(momenta_to_transport_tR, is_orthogonal=is_orthogonal)
    else:
        return compute_backward, exponential.parallel_transport(momenta_to_transport_tR, is_orthogonal=is_orthogonal)


class PiecewiseGeodesic:

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, kernel=default.deformation_kernel,
                 t0=default.t0, concentration_of_time_points=default.concentration_of_time_points,
                 use_rk2_for_shoot=default.use_rk2_for_shoot, use_rk2_for_flow=default.use_rk2_for_flow,
                 nb_components=2, num_components = None, template_tR=None, transport_cp = True):

        # usual t0 replaced by tR
        self.concentration_of_time_points = concentration_of_time_points
        self.tmax = None # Geodesic tmax
        self.tmin = None #Geodesic tmin
        self.t0 = t0 #Geodesic starting point (i.e. template defined at t0)

        self.control_points = None
        self.momenta = None
        self.template_points_tR0 = None
        self.nb_components = int(nb_components)

        #contains t0 and tmax in addition to the tR
        # tR[0]=tmin, then all the rupture times then tR[-1]=tmax
        self.tR = [self.tmin] * (self.nb_components + 1) 
        self.exponential = []
        for i in range(self.nb_components):
            self.exponential.append(
                Exponential(kernel=kernel,
                            use_rk2_for_shoot=use_rk2_for_shoot, use_rk2_for_flow=use_rk2_for_flow,
                            transport_cp = transport_cp))

        # Flags to save extra computations that have already been made in the update methods.
        self.shoot_is_modified = [True]*self.nb_components
        self.backward_extension = 0
        self.forward_extension = 0

        self.template_index = None

    def new_exponential(self):
        return Exponential(kernel=self.exponential[0].kernel, 
                            use_rk2_for_shoot=self.exponential[0].use_rk2_for_shoot, 
                            use_rk2_for_flow=self.exponential[0].use_rk2_for_flow,
                            transport_cp = self.exponential[0].transport_cp)


    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_use_rk2_for_shoot(self, flag):
        for l in range(self.nb_components):
            self.exponential[l].set_use_rk2_for_shoot(flag)

    def set_use_rk2_for_flow(self, flag):
        for l in range(self.nb_components):
            self.exponential[l].set_use_rk2_for_flow(flag)

    def set_kernel(self, kernel):
        for l in range(self.nb_components):
            self.exponential[l].kernel = kernel

    def set_tR(self, tR):
        """
        Set the rupture times between each dynamic
        """
        for i in range(1, self.nb_components): #ignore tR[0]=tmin
            self.tR[i] = tR[i-1]

        self.shoot_is_modified = [True]*self.nb_components
    
    def get_t0(self):
        #ajout fg
        return self.t0
    
    def set_t0(self, t0 = None):
        #ajout fg
        self.t0 = t0
        assert (self.t0 in self.tR), "Template at {} must be defined at a rupture time {}".format(self.t0, self.tR)
        self.shoot_is_modified = [True]*self.nb_components

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

    def get_template_points_tR(self):
        return self.template_points_tR

    def set_template_points_tR(self, td):
        self.template_points_tR = td
        self.flow_is_modified = True

    def set_control_points_tR(self, cp):
        self.control_points = cp
        self.shoot_is_modified = [True]*self.nb_components

    def set_momenta_tR(self, mom):
        self.momenta = mom # an array shape n_components * n_cp * dim
        self.shoot_is_modified = [True]*self.nb_components

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
            logger.info('>> The geodesic seems to be reduced to a single point.')
            return self.template_points_tR

        # Fetch index j of time closest (and above) 'time'
        for j in range(1, len(times)):
            if time - times[j] < 0: break
        
        # Mean of the two closest template points
        device, _ = utilities.get_best_device(self.exponential[0].kernel.gpu_mode)

        weight_left = utilities.move_data([(times[j] - time) / (times[j] - times[j - 1])], device=device, dtype=self.momenta[0].dtype)
        weight_right = utilities.move_data([(time - times[j - 1]) / (times[j] - times[j - 1])], device=device, dtype=self.momenta[0].dtype)
        template_t = {key: [utilities.move_data(v, device=device) for v in value] \
                for key, value in self.get_template_points_trajectory().items()}
        deformed_points = {key: weight_left * value[j - 1] + weight_right * value[j]
                           for key, value in template_t.items()}

        return deformed_points

    ####################################################################################################################
    ### Main methods:
    ####################################################################################################################

    def update_exponential(self, l, device):
        if self.exponential[l].number_of_time_points > 1:
            self.exponential[l].move_data_to_(device=device)
            self.exponential[l].update()
            self.shoot_is_modified[l] = False

    def nb_of_tp(self, length):
        return max(1, int(length * self.concentration_of_time_points + 1.5))

    def add_component(self):
        self.nb_components += 1
        self.tR = self.tR + [self.tR[-1]]

        self.shoot_is_modified = [True]*self.nb_components
        self.flow_is_modified = True

    def add_exponential(self, c):
        # Recomputre template index
        self.get_template_index()

        # Update momenta
        exponentials= [None] * self.nb_components

        logger.info("Add expo in position {}".format(c+1))

        for i in range(self.nb_components):
            if i <= c:
                exponentials[i] = self.exponential[i]
            elif i == c+1:
                exponentials[i] = self.new_exponential()
            else:
                exponentials[i] = self.exponential[i-1]

        # Add an exponential
        self.exponential = exponentials
        
        
    def update(self):
        """
        Compute the time bounds, accordingly sets the number of points and momenta of the attribute exponentials,
        then shoot and flow them.
        """
        device, _ = utilities.get_best_device(self.exponential[0].kernel.gpu_mode)

        self.get_template_index()
                    
        # NB: exp.control_points_t computed with exp.shoot() (called by exp.update())
        
        # Update the forward and backward exponentials at t0
        for l in range(self.template_index - 1, self.template_index + 1):
            
            length = self.tR[l+1] - self.tR[l]
            
            self.exponential[l].number_of_time_points = self.nb_of_tp(length)
                
            if self.shoot_is_modified[l]:
                self.exponential[l].set_initial_momenta(self.momenta[l] * length)
                self.exponential[l].set_initial_control_points(self.control_points)
            
            if self.flow_is_modified:
                self.exponential[l].set_initial_template_points(self.template_points_tR)
            
            self.update_exponential(l, device)

        # Backward shoot                 
        for l in range(0, self.template_index - 1):
            length = self.tR[l+1] - self.tR[l]
            
            self.exponential[l].number_of_time_points = self.nb_of_tp(length)
                
            if self.shoot_is_modified[l]:
                self.exponential[l].set_initial_momenta(self.momenta[l] * length)
                self.exponential[l].set_initial_control_points(self.exponential[l+1].control_points_t[-1])
            
            if self.flow_is_modified:
                self.exponential[l].set_initial_template_points(
                        self.exponential[l + 1].get_template_points(
                        self.exponential[l + 1].momenta_t.__len__() - 1))
            
            self.update_exponential(l, device)

        # Forward shoot
        
        for l in range(self.template_index + 1, self.nb_components):
            length = self.tR[l+1] - self.tR[l]
            self.exponential[l].number_of_time_points = self.nb_of_tp(length)

            if self.shoot_is_modified[l]:
                self.exponential[l].set_initial_momenta(self.momenta[l] * length)
                self.exponential[l].set_initial_control_points(self.exponential[l-1].control_points_t[-1])
            
            if self.flow_is_modified:
                self.exponential[l].set_initial_template_points(
                        self.exponential[l - 1].get_template_points(
                        self.exponential[l - 1].momenta_t.__len__() - 1))

            self.update_exponential(l, device)
        
        self.flow_is_modified = False

    def get_norm_squared(self, l):
        """
        Get the norm of the geodesic.
        Modif fg to handle a more flexible t0
        """
        if l < self.template_index:
            cp = self.exponential[l+1].get_initial_control_points()
        elif l == self.template_index:
            cp = self.control_points
        elif l > self.template_index:
            cp = self.exponential[l-1].control_points_t[-1]

        return self.exponential[l].scalar_product(cp, self.momenta[l], self.momenta[l])

    def get_component(self, time):
        # tR = 24, 24, 28, 32, 36
        for tR in self.tR[1:]:
            if time < tR:
                break
        l = self.tR[1:].index(tR)
        return l


    def transport_along_exponential(self, last_transported_mom, transport, is_orthogonal,
                                    l = None, exponential = None, initial_time_point = 0,
                                    backward = False):

        if l is not None: exponential = self.exponential[l]

        if exponential.number_of_time_points > 1:
            transported = exponential.parallel_transport(last_transported_mom,
                                                        is_orthogonal=is_orthogonal,
                                                        initial_time_point = initial_time_point)
            last_transported_mom = transported[-1]
            if not backward:
                transport.append(transported)
            else:
                transport.append(transported[::-1])
        else:
            transport.append([last_transported_mom])
        
        return transport, last_transported_mom

    def concatenate_transport(self, transport):
        transport_concat = transport[0]
        n_transport = len(transport)
        for m in range(1, n_transport):
            transport_concat += transport[m][1:]
        
        return transport_concat
    
    def parallel_transport(self, momenta_to_transport_tR, is_orthogonal = False, 
                           target_time = None): #ajout fg
        """
        CLassical parallel transport: momenta_to_transport_tR defined at t0
        """
        if any(self.shoot_is_modified):
            msg = "Trying to parallel transport but the geodesic object was modified, please update before."
            warnings.warn(msg)
                        
        transport = []
        last_transported_mom = momenta_to_transport_tR

        if target_time is not None:
            if self.t0 > target_time:
                # Backward transport
                for l in range(self.template_index -1, -1, -1):
                    transport, last_transported_mom = self.transport_along_exponential(last_transported_mom, 
                                                                        transport, is_orthogonal, l,
                                                                        backward=True)
                transport_concat = self.concatenate_transport(transport[::-1])

                return transport_concat
            else:
                # Forward transport
                for l in range(self.template_index, self.nb_components):
                    transport, last_transported_mom = self.transport_along_exponential(last_transported_mom, 
                                                                                transport, is_orthogonal, l)

                transport_concat = self.concatenate_transport(transport)
                return transport_concat
                
        # Backward transport
        for l in range(self.template_index -1, -1, -1):
            transport, last_transported_mom = self.transport_along_exponential(last_transported_mom, 
                                                                        transport, is_orthogonal, l,
                                                                        backward=True)

        transport = transport[::-1]
        # Forward transoport
        last_transported_mom = momenta_to_transport_tR
        for l in range(self.template_index, self.nb_components):
            transport, last_transported_mom = self.transport_along_exponential(last_transported_mom, 
                                                                        transport, is_orthogonal, l)

        transport_concat = self.concatenate_transport(transport)
                    
        return transport_concat
    
    def parallel_transport_(self, momenta_to_transport_tR, start_time, target_time,
                            is_orthogonal=False):
        """
        Parallel transport: handles special case where momenta to transport not defined
        at t0: need to get regression momenta at start time, define a new geodesic,
        transport towards the next rupture time
        """
        device, _ = utilities.get_best_device(self.exponential[0].kernel.gpu_mode)

        if start_time == self.t0:
            return self.parallel_transport(momenta_to_transport_tR, is_orthogonal, target_time)
        else:

            if start_time > target_time:
                # Get the exponential that start_time belongs to
                l = self.tR.index([t for t in self.tR if t >= start_time][0]) - 1
                
                length = start_time - self.tR[l]
                start_time_ = self.nb_of_tp(length) -1

                # /!\ Momenta is divided by old length in here (mandatory)
                template_at_start_time = self.exponential[l].get_template_points(start_time_)
                start_time_t = self.nb_of_tp(start_time - self.tR[0]) -1            
                momenta_at_start_time = self.get_momenta_trajectory()[start_time_t]
                cp_at_start_time = self.get_control_points_trajectory()[start_time_t]

                new_expo = self.new_exponential()

                transport = []
                last_transported_mom = momenta_to_transport_tR

                # Backward Transport
                length = start_time - self.tR[l]
                new_expo.number_of_time_points = self.nb_of_tp(length)
                print("\nBackward transport from {} to {} (length {})".format(start_time, self.tR[l], length + 1))
                
                if start_time > self.t0: momenta_at_start_time = - momenta_at_start_time

                new_expo.set_initial_momenta(momenta_at_start_time * length)
                new_expo.set_initial_control_points(cp_at_start_time)
                new_expo.set_initial_template_points(template_at_start_time)
                new_expo.move_data_to_(device=device)
                new_expo.update()

                # Transport momenta to self.tR[l]
                transport, last_transported_mom = self.transport_along_exponential(last_transported_mom, 
                                                                                    transport, is_orthogonal,
                                                                                    exponential=new_expo,
                                                                                    backward=True)                
                for m in range(l-1, -1, -1):
                    if target_time < self.tR[m+1]:
                        print("\nBackward transport from {} to {}".format(self.tR[m+1], self.tmin))
                        transport, last_transported_mom = self.transport_along_exponential(last_transported_mom, 
                                                        transport, is_orthogonal, m, backward=True)
                transport = transport[::-1]

            else:

                l = self.tR.index([t for t in self.tR if t > start_time][0]) - 1
            
                length = self.tR[l+1] - start_time
                length_exp = self.tR[l+1] - self.tR[l]
                start_time_ = self.nb_of_tp(length_exp) - self.nb_of_tp(length)

                momenta_at_start_time = self.get_momenta_trajectory()[start_time_]

                if start_time < self.t0: momenta_at_start_time = - momenta_at_start_time
                
                # Transport momenta to self.tR[l]
                transport = []
                last_transported_mom = momenta_to_transport_tR
                print("\nForward transport from {} to {}".format(start_time, self.tR[l+1]))

                print("Transport on expo with {} tp starting from time {}".format(self.nb_of_tp(length_exp), start_time_))
                transport, last_transported_mom = self.transport_along_exponential(last_transported_mom, 
                                                                                    transport, is_orthogonal,
                                                                                    l=l, initial_time_point=start_time_)    

                for m in range(l+1, self.nb_components):
                    if target_time > self.tR[m]:
                        print("Forward transport from {} to {}".format(self.tR[m], self.tR[m+1]))
                        transport, last_transported_mom = self.transport_along_exponential(last_transported_mom, 
                                                                                transport, is_orthogonal, m)
            transport_concat = self.concatenate_transport(transport)

            return transport_concat
    
                            



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
        times = []

        for l in range(self.nb_components):
            tR_l = self.convert_to_array(self.tR[l])
            times.append([tR_l])

            if self.exponential[l].number_of_time_points > 1:
                tR_l_ = self.convert_to_array(self.tR[l+1])
                times[l] = np.linspace(tR_l, tR_l_, 
                    num=self.exponential[l].number_of_time_points).tolist()
                
        times_concat = times[0]
        
        for l in range(1, self.nb_components):
            times_concat += times[l][1:]

        return times_concat

    def get_control_points_trajectory(self):
        if any(self.shoot_is_modified):
            msg = "Trying to get cp trajectory in a non updated geodesic."
            warnings.warn(msg)

        #control_points_t = [self.exponential[0].get_initial_control_points()]
        # if self.exponential[0].number_of_time_points > 1:
        #     control_points_t = self.exponential[0].control_points_t[::-1]

        # for l in range(1, self.nb_components):
        #     if self.exponential[l].number_of_time_points > 1:
        #         control_points_t += self.exponential[l].control_points_t[1:]

        # Modif fg
        control_points_t = []
        if self.exponential[0].number_of_time_points > 1:
            if self.tR[0] < self.t0:
                control_points_t += self.exponential[0].control_points_t[::-1]
            else:
                control_points_t += self.exponential[0].control_points_t

        for l in range(1, self.nb_components):
            if self.exponential[l].number_of_time_points > 1:
                if self.tR[l] < self.t0:
                    control_points_t += self.exponential[l].control_points_t[::-1][1:]
                else:
                    control_points_t += self.exponential[l].control_points_t[1:]

        return control_points_t

    def get_momenta_trajectory(self):
        if any(self.shoot_is_modified):
            msg = "Trying to get mom trajectory in non updated geodesic."
            warnings.warn(msg)

        # modif fg: flexible t0
        momenta_t = []
        for l in range(self.nb_components):
            length = self.tR[l+1] - self.tR[l]
            if self.exponential[l].number_of_time_points > 1:
                if self.tR[l] < self.t0 :
                    mom = self.exponential[l].momenta_t[::-1]
                else:
                    mom = self.exponential[l].momenta_t
                if l > 0:
                    mom = mom[1:]
                #print("Adding expo {} with {} tp, divided by length {}".format(l, len(mom), length))

                momenta_t += [elt/length for elt in mom]


        return momenta_t

    def get_template_points_trajectory(self):
        """
        Return list of template points along trajectory
        by adding all template points along all exponentials
        """
        if any(self.shoot_is_modified) or self.flow_is_modified:
            warnings.warn("Trying to get template trajectory in non updated geodesic.")

        # template_t = {}
        # for key in self.template_points_tR.keys():
        #     if self.exponential[0].number_of_time_points > 1:
        #         template_t[key] = self.exponential[0].template_points_t[key][::-1]
        #     else: 
        #         template_t[key] = [self.exponential[0].get_initial_template_points()[key]]

        #     for l in range(1,self.nb_components):
        #         if self.exponential[l].number_of_time_points > 1:
        #             template_t[key] += self.exponential[l].template_points_t[key][1:]
        
        # Modif fg / flexible t0
        # to work it is necessary to have tmin < t0 = at least one subject younger that t0...
        template_t = {k:[] for k in self.exponential[0].template_points_t.keys()}
        
        for key in self.template_points_tR.keys():
            # Get all template points from 1st expo
            if self.exponential[0].number_of_time_points > 1:
                if self.tR[0] < self.t0:
                    template_t[key] += self.exponential[0].template_points_t[key][::-1]
                else:
                    template_t[key] += self.exponential[0].template_points_t[key]
            
            # Get all template points but 1st one (already in previous exp)
            for l in range(1, self.nb_components):
                if self.exponential[l].number_of_time_points > 1:
                    if self.tR[l] < self.t0:
                        template_t[key] += self.exponential[l].template_points_t[key][::-1][1:]
                    else:
                        template_t[key] += self.exponential[l].template_points_t[key][1:]


        return template_t

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def output_path(self, root_name, objects_name, objects_extension, output_dir):
        self.flow_path = {}
        self.momenta_flow_path = {}

        times = self.get_times()
        component = 0
        step = self.concentration_of_time_points if self.concentration_of_time_points > 1 else 0.5
        step = 1
        for t, time in enumerate(times):
            rupture_time = self.tR[component + 1]
            names = []
            if time > rupture_time: component += 1

            for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                name = root_name + '__GeodesicFlow__' + object_name + '__component_{}'.format(component) \
                        + '__tp_' + str(t) + ('__age_%.2f' % time) + object_extension
                names.append(name)
            
            if (t % step == 0) or time == rupture_time:
                self.flow_path[time] = op.join(output_dir, names[0])

            self.momenta_flow_path[time] = op.join(output_dir, root_name + '__GeodesicFlow__Momenta__tp_' + str(t)
                                            + ('__age_%.2f' % time) + '.txt')

    def write(self, root_name, objects_name, objects_extension, template, template_data, output_dir,
              write_adjoint_parameters = False, write_all = True):
        
        momenta_t = [elt.detach().cpu().numpy() for elt in self.get_momenta_trajectory()]
        control_points_t = [elt.detach().cpu().numpy() for elt in self.get_control_points_trajectory()]

        # Core loop ----------------------------------------------------------------------------------------------------
        if write_all:
            times = self.get_times()
            component = 0
            step = self.concentration_of_time_points if self.concentration_of_time_points > 1 else 0.5
            step = 10
            #step = 1
            for t, time in enumerate(times):
                rupture_time = self.tR[component + 1]
                names = []
                if time > rupture_time: component += 1

                for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
                    name = root_name + '__GeodesicFlow__' + object_name + '__component_{}'.format(component) \
                            + '__tp_' + str(t) + ('__age_%.2f' % time) + object_extension
                    names.append(name)
                deformed_points = self.get_template_points(time)
                deformed_data = template.get_deformed_data(deformed_points, template_data)
                
                if (t % step == 0) or time == rupture_time:
                    template.write(output_dir, names,
                            {key: value.detach().cpu().numpy() for key, value in deformed_data.items()},
                            momenta = momenta_t[t], cp = control_points_t[t], kernel = self.exponential[0].kernel)

        # Optional writing of the control points and momenta -----------------------------------------------------------
        #if write_adjoint_parameters:
        if True:
            times = self.get_times()
            
            for t, (time, control_points, momenta) in enumerate(zip(times, control_points_t, momenta_t)):
                #write_2D_array(momenta, output_dir, root_name + '__GeodesicFlow__Momenta__tp_' + str(t)
                #               + ('__age_%.2f' % time) + '.txt')
                concatenate_for_paraview(momenta, control_points, output_dir, 
                            root_name + "__EstimatedParameters__Fusion_CP_Momenta_iter_{}.vtk")
