import warnings
from copy import deepcopy
import torch
from ....core import default
from ....in_out.array_readers_and_writers import *
from ....support.utilities import move_data, detach, assert_same_device
from ....support.kernels import factory

import logging
logger = logging.getLogger(__name__)

def cuda_used():
    return torch.cuda.memory_allocated(0)/torch.cuda.memory_reserved(0)

class Exponential:
    """
    Control-point-based LDDMM exponential, that transforms the template objects according to initial control points
    and momenta parameters.
    See "Morphometry of anatomical shape complexes with dense deformations and sparse parameters",
    Durrleman et al. (2013).
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################
    def __init__(self, 
                 kernel=default.deformation_kernel, n_time_points=None,
                 initial_control_points=None, control_points_t=None,
                 initial_momenta=None, momenta_t=None,
                 initial_template_points=None, template_points_t=None,
                 shoot_is_modified=True, flow_is_modified=True, use_rk2_for_shoot=False, 
                 use_rk2_for_flow=False, transport_cp = True):

        self.kernel = kernel

        if self.kernel is None:
            self.shoot_kernel = factory(gpu_mode=kernel.gpu_mode, kernel_width=kernel.kernel_width)
        else:
            self.shoot_kernel = self.kernel

        self.n_time_points = n_time_points
        # Initial position of control points
        self.initial_cp = initial_control_points
        # Initial momenta
        self.initial_momenta = initial_momenta
        # Control points and momenta trajectory
        self.cp_t = control_points_t
        self.momenta_t = momenta_t
        
        # Initial template points
        self.initial_template_points = initial_template_points
        # Trajectory of the whole template points
        self.template_points_t = template_points_t
        # If the cp or mom have been modified:
        self.shoot_is_modified = shoot_is_modified
        # If the template points has been modified
        self.flow_is_modified = flow_is_modified
        # Wether to use a RK2 or a simple euler for shooting or flowing respectively.
        self.use_rk2_for_shoot = use_rk2_for_shoot
        self.use_rk2_for_flow = use_rk2_for_flow

        # Contains the inverse kernel matrices for the time points 1 to self.n_time_points
        self.cometric_matrices = {}
        self.kernel_matrix = {}
        self.kernel_matrix_inv = {}

        # ajout fg
        self.transport_cp = transport_cp
    
    def move_data_to_(self, device):
        if self.initial_cp is not None:
            self.initial_cp = move_data(self.initial_cp, device)

        if self.initial_momenta is not None:
            self.initial_momenta = move_data(self.initial_momenta, device)

        if self.initial_template_points is not None:
            self.initial_template_points = {key: move_data(value, device) for key, value in
                                            self.initial_template_points.items()}

    def light_copy(self):
        light_copy = Exponential(deepcopy(self.kernel),
                                 self.n_time_points,
                                 self.initial_cp, self.cp_t,
                                 self.initial_momenta, self.momenta_t,
                                 self.initial_template_points, self.template_points_t,
                                 self.shoot_is_modified, self.flow_is_modified)
        return light_copy

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################
    def is_long(self):
        return self.n_time_points > 1
        
    def get_kernel_width(self):
        return self.kernel.kernel_width

    def set_kernel(self, kernel):
        # TODO which kernel to set ?
        self.kernel = kernel

    def set_initial_template_points(self, td):
        self.initial_template_points = td
        self.flow_is_modified = True

    def get_initial_template_points(self):
        return self.initial_template_points

    def set_initial_cp(self, cps):
        self.shoot_is_modified = True
        self.initial_cp = cps

    def get_initial_cp(self):
        return self.initial_cp

    def get_initial_momenta(self):
        return self.initial_momenta

    def set_initial_momenta(self, mom):
        self.shoot_is_modified = True
        self.initial_momenta = mom

    def get_initial_momenta(self):
        return self.initial_momenta

    def scalar_product(self, cp, mom1, mom2):
        """
        returns the scalar product 'mom1 K(cp) mom 2'
        """
        return torch.sum(mom1 * self.kernel.convolve(cp, cp, mom2))

    def norm(self, cp, mom):
        return self.scalar_product(cp, mom, mom)

    def get_norm_squared(self):
        return self.scalar_product(self.initial_cp, self.initial_momenta, self.initial_momenta)

    def scalar_product_at_points(self, cp, mom1, mom2):
        """
        returns the scalar product 'mom1 K(cp) mom 2'
        """
        return torch.sum(mom1 * self.kernel.convolve(cp, cp, mom2), dim = 1)
    
    def norm_at_points(self, cp, mom):
        return torch.sum(mom * self.kernel.convolve(cp, cp, mom), dim = 1)

    def get_template_points(self, time_index=None):
        """
        Returns the position of the landmark points, at the given time_index in the Trajectory
        """
        if self.flow_is_modified:
            warnings.warn("The flow was modified, the exponential should be updated")

        if time_index is None:
            return {key: self.template_points_t[key][-1] for key in self.initial_template_points.keys()}
        
        return {key: self.template_points_t[key][time_index] for key in self.initial_template_points.keys()}

    ####################################################################################################################
    ### Main methods:
    ####################################################################################################################

    def update(self):
        """
        Update the state of the object, depending on what's needed.
        This is the only clean way to call shoot or flow on the deformation.
        """
        assert self.n_time_points > 0
        if self.shoot_is_modified:
            if self.transport_cp:
                self.cometric_matrices.clear()
            self.shoot()
            if self.initial_template_points is not None:
                self.flow()
            else:
                logger.warning("In exponential update,  I don't have any template points to flow")

        if self.flow_is_modified:
            if self.initial_template_points is not None:
                self.flow()
            else:
                logger.warning("In exponential update,  I don't have any template points to flow")

    def shoot(self):
        """
        Computes the flow of momenta and control points.
        """
        assert len(self.initial_cp) > 0, "Control points not initialized in shooting"
        assert len(self.initial_momenta) > 0, "Momenta not initialized in shooting"

        # Integrate the Hamiltonian equations.
        self.cp_t = [self.initial_cp]
        self.momenta_t = [self.initial_momenta]

        dt = 1.0 / float(self.n_time_points - 1)

        if self.use_rk2_for_shoot:
            for i in range(self.n_time_points - 1):
                new_cp, new_mom = self._rk2_step(self.shoot_kernel, self.cp_t[i], self.momenta_t[i], dt)
                self.cp_t.append(new_cp)
                self.momenta_t.append(new_mom)

        else:
            #self.cp_t et self.momenta_t listes de longueur 1 t -> 11t après la boucle
            for i in range(self.n_time_points - 1): #10 time points for discretization   
                new_cp, new_mom = self._euler_step(self.shoot_kernel, self.cp_t[i], self.momenta_t[i], dt)
                self.cp_t.append(new_cp)
                self.momenta_t.append(new_mom)

        # Correctly resets the attribute flag.
        self.shoot_is_modified = False

    def flow(self):
        """
        Flow the trajectory of the landmark and/or image points.
        """
        assert not self.shoot_is_modified, "CP or momenta were modified and the shoot not computed, and now you are asking me to flow ?"
        assert len(self.cp_t) > 0, "Shoot before flow"
        assert len(self.momenta_t) > 0, "Control points given but no momenta"

        # Initialization.
        dt = 1.0 / float(self.n_time_points - 1)
        self.template_points_t = {}

        # Flow landmarks points.
        if 'landmark_points' in self.initial_template_points.keys():
            landmark_points = [self.initial_template_points['landmark_points']]

            for i in range(self.n_time_points - 1):
                d_pos = self.kernel.convolve(landmark_points[i], self.cp_t[i], self.momenta_t[i])
                landmark_points.append(landmark_points[i] + dt * d_pos)

                # In this case improved euler (= Heun's method)
                if i < self.n_time_points - 2:
                    landmark_points[-1] = landmark_points[i] + dt / 2 * \
                                        (self.kernel.convolve(landmark_points[i + 1],
                                        self.cp_t[i + 1], self.momenta_t[i + 1]) + d_pos)

            self.template_points_t['landmark_points'] = landmark_points

        # Flow image points.
        if 'image_points' in self.initial_template_points.keys():
            image_points = [self.initial_template_points['image_points']]

            for i in range(self.n_time_points - 1):
                vf = self.kernel.convolve_image_points(image_points[0], 
                                self.cp_t[i], self.momenta_t[i])
                dY = self._compute_image_explicit_euler_step_at_order_1(image_points[i], vf)
                image_points.append(image_points[i] - dt * dY)

            self.template_points_t['image_points'] = image_points

        assert len(self.template_points_t) > 0

        # Correctly resets the attribute flag.
        self.flow_is_modified = False

    def transport(self, i, h, parallel_transport_t, initial_norm_squared, eps, norm_squared):
        """
        transport_cp: ajout fg to avoid recomputing cometric matrices for the non moving cp
        """
        # Shoot the two perturbed geodesics ------------------------------------------------------------------------
        cp_eps_pos = self._rk2_step(self.shoot_kernel, self.cp_t[i],
                                    self.momenta_t[i] + eps * parallel_transport_t[-1], 
                                    h, return_mom=False)
        cp_eps_neg = self._rk2_step(self.shoot_kernel, self.cp_t[i],
                                    self.momenta_t[i] - eps * parallel_transport_t[-1], 
                                    h, return_mom=False)

        # Compute J/h ----------------------------------------------------------------------------------------------
        approx_velocity = (cp_eps_pos - cp_eps_neg) / (2 * eps * h)
        
        # We need to find the cotangent space version of this vector -----------------------------------------------
        # If we don't have already the cometric matrix, we compute and store it-Consumes a lot of memory!
        
        # Need to get the kernel matrix everytime: the points move
        kernel_matrix = self.shoot_kernel.get_kernel_matrix(self.cp_t[i + 1])
        #kernel_matrix_reg = kernel_matrix + 100 * torch.eye(kernel_matrix.size()[0], kernel_matrix.size()[1], 
        #                                       device = kernel_matrix.device)
        #kernel_matrix_inv = torch.inverse(kernel_matrix_reg)
        kernel_matrix_reg = kernel_matrix + 1 * torch.eye(kernel_matrix.size()[0], kernel_matrix.size()[1], 
                                                device = kernel_matrix.device)
        #u = torch.linalg.cholesky(kernel_matrix_reg)
        kernel_matrix_inv = torch.cholesky_inverse(kernel_matrix_reg, upper=True)
        #kernel_matrix_inv = torch.inverse(cholesky)
        
        print("\ni", i)
        # print("Condition number of K", np.linalg.cond(kernel_matrix.cpu().numpy()))
        # print("Condition number of K", np.linalg.cond(kernel_matrix.cpu().numpy()))
            
        # Solve the linear system - option 1 (original hack)
        #approx_momenta = torch.mm(cometric_matrix, approx_velocity)
        
        # # Option 2 - do nothing (no so good)
        # approx_momenta = approx_velocity
            
        approx_momenta = torch.mm(kernel_matrix_inv, approx_velocity)

        # We get rid of the component of this momenta along the geodesic velocity:
        scalar_prod_with_velocity = self.scalar_product(self.cp_t[i + 1], approx_momenta,
                                                        self.momenta_t[i + 1]) / norm_squared

        approx_momenta_norm_squared_ = self.scalar_product(self.cp_t[i + 1], approx_momenta,
                                                            approx_momenta)
        approx_momenta = approx_momenta - scalar_prod_with_velocity * self.momenta_t[i + 1]

        # # Renormalization ------------------------------------------------------------------------------------------
        approx_momenta_norm_squared = self.scalar_product(self.cp_t[i + 1], approx_momenta,
                                                            approx_momenta)
        renormalization_factor = torch.sqrt(initial_norm_squared / approx_momenta_norm_squared)
        renormalized_momenta = approx_momenta * renormalization_factor

        # print("renormalization_factor", renormalization_factor)
        # print("approx_momenta_norm_squared before ortho", approx_momenta_norm_squared_)
        # print("approx_momenta_norm_squared", approx_momenta_norm_squared)
        # print("target and final norm squared", self.scalar_product(self.cp_t[i + 1], renormalized_momenta,
        #                                                     renormalized_momenta))
        # print("target and final norm squared 2", self.scalar_product(self.cp_t[0], renormalized_momenta,
        #                                                     renormalized_momenta))
        
        # norm = torch.norm(renormalized_momenta, dim = 1) # norm of each vector
        # indices = (norm > 5).nonzero(as_tuple=False)
        # if len(indices) > 0:
        #     renormalized_momenta[indices] = renormalized_momenta[indices] * 5 / norm[indices].unsqueeze(1)
        #     renormalized_momenta[indices] = renormalized_momenta[indices] * 10 * (norm[indices].unsqueeze(1)/ maximum)

        #     #print("approx_momenta_norm_squared 2", torch.norm(renormalized_momenta))
        #     #print(indices.size()[0])

        # if abs(detach(renormalization_factor) - 1.) > 0.1:
        #     raise ValueError('Absurd required renormalization factor during parallel transport: %.4f. '
        #                      'Exception raised.' % renormalization_factor.detach().cpu().numpy())
        # elif abs(renormalization_factor.detach().cpu().numpy() - 1.) > abs(worst_renormalization_factor - 1.):
        #     worst_renormalization_factor = renormalization_factor.detach().cpu().numpy()

        # Finalization ---------------------------------------------------------------------------------------------
        parallel_transport_t.append(renormalized_momenta)

        return parallel_transport_t

    def parallel_transport(self, momenta_to_transport, initial_time_point=0, is_ortho=False):
        """
        Parallel transport of the initial_momenta along the exponential.
        momenta_to_transport is assumed to be a torch Variable, carried at the control points on the diffeo.
        if is_ortho is on, then the momenta to transport must be orthogonal to the momenta of the geodesic.
        Note: uses shoot kernel
        
        """
        # Sanity checks ------------------------------------------------------------------------------------------------
        assert not self.shoot_is_modified, "You want to parallel transport but the shoot was modified, please update."
        assert self.use_rk2_for_shoot, "The shoot integration must be done with a 2nd order numerical scheme in order to use parallel transport."
        assert (momenta_to_transport.size() == self.initial_momenta.size())
        
        # Special cases, where the transport is simply the identity ----------------------------------------------------
        #       1) Nearly zero initial momenta yield no motion.
        if (detach(torch.norm(self.initial_momenta)) < 1e-6 or
            detach(torch.norm(momenta_to_transport)) < 1e-6):
            parallel_transport_t = [momenta_to_transport] * (self.n_time_points - initial_time_point)
            return parallel_transport_t

        # Step sizes ---------------------------------------------------------------------------------------------------
        h = 1. / (self.n_time_points - 1.)
        eps = h

        # For #printing -------------------------------------------------------------------------------------------------
        worst_renormalization_factor = 1.0
        self.sp = []
        # Optional initial orthogonalization ---------------------------------------------------------------------------
        norm_squared = self.get_norm_squared()

        if not is_ortho: # Always goes in this loop
            sp = self.scalar_product(self.cp_t[initial_time_point], momenta_to_transport,
                                     self.momenta_t[initial_time_point]) / norm_squared
            momenta_to_transport_orthogonal = momenta_to_transport - sp * self.momenta_t[initial_time_point]
            parallel_transport_t = [momenta_to_transport_orthogonal]
        else:
            sp = detach(( self.scalar_product( self.cp_t[initial_time_point], momenta_to_transport,
                                            self.momenta_t[initial_time_point]) / norm_squared ))

            parallel_transport_t = [momenta_to_transport]

        # Then, store the initial norm of this orthogonal momenta ------------------------------------------------------
        initial_norm_squared_before_ortho = self.scalar_product(self.cp_t[initial_time_point], momenta_to_transport,
                                                                momenta_to_transport)
        initial_norm_squared = self.scalar_product(self.cp_t[initial_time_point], parallel_transport_t[0],
                                                   parallel_transport_t[0])

        print("Transport from {} to {}".format(initial_time_point, self.n_time_points -1))
        for i in range(initial_time_point, self.n_time_points - 1):
            parallel_transport_t = self.transport(i, h, parallel_transport_t, initial_norm_squared, eps,
                                                norm_squared)

        # We now need to add back the component along the velocity to the transported vectors.
        #the norm is modified here 
        if not is_ortho:
            parallel_transport_t_ = []
            for i in range(initial_time_point, self.n_time_points):
                parallel_transport_t_.append(parallel_transport_t[i-initial_time_point] + sp * self.momenta_t[i])
            parallel_transport_t =parallel_transport_t_

        if abs(worst_renormalization_factor - 1.) > 0.05:
            msg = ("Watch out, a large renormalization factor %.4f is required during the parallel transport. "
                   "Try using a finer discretization." % worst_renormalization_factor)
            logger.warning(msg)

        return parallel_transport_t

    ####################################################################################################################
    ### Utility methods:
    ####################################################################################################################

    @staticmethod
    def _euler_step(kernel, cp, mom, h):
        """
        simple euler step of length h, with cp and mom. It always returns mom.
        """
        assert_same_device(cp = cp, mom = mom)

        return cp + h * kernel.convolve(cp, cp, mom), \
               mom - h * kernel.convolve_gradient(mom, cp)

    @staticmethod
    def _rk2_step(kernel, cp, mom, h, return_mom=True):
        """
        perform a single mid-point rk2 step on the geodesic equation with initial cp and mom.
        also used in parallel transport.
        return_mom: bool to know if the mom at time t+h is to be computed and returned
        """
        assert_same_device(cp = cp, mom = mom)
        
        mid_cp = cp + h / 2 * kernel.convolve(cp, cp, mom)
        mid_mom = mom - h / 2 * kernel.convolve_gradient(mom, cp)
        res_cp = cp + h * kernel.convolve(mid_cp, mid_cp, mid_mom)
        
        if return_mom:
            res_mom = mom - h * kernel.convolve_gradient(mid_mom, mid_cp)
            return res_cp, res_mom

        return res_cp


    # TODO. Wrap pytorch of an efficient C code ? Use keops ? Called ApplyH in PyCa. Check Numba as well.
    # @jit(parallel=True)
    @staticmethod
    def _compute_image_explicit_euler_step_at_order_1(Y, vf):
        assert_same_device(Y = Y, vf = vf)

        dY = torch.zeros(Y.shape, dtype=vf.dtype, device=vf.device)
        dimension = len(Y.shape) - 1

        if dimension == 2:
            ni, nj = Y.shape[:2]

            # Center.
            dY[1:ni - 1, :] = dY[1:ni - 1, :] + 0.5 * vf[1:ni - 1, :, 0].view(ni - 2, nj, 1).expand(ni - 2, nj, 2) * (
                    Y[2:ni, :] - Y[0:ni - 2, :])
            dY[:, 1:nj - 1] = dY[:, 1:nj - 1] + 0.5 * vf[:, 1:nj - 1, 1].view(ni, nj - 2, 1).expand(ni, nj - 2, 2) * (
                    Y[:, 2:nj] - Y[:, 0:nj - 2])

            # Borders.
            dY[0, :] = dY[0, :] + vf[0, :, 0].view(nj, 1).expand(nj, 2) * (Y[1, :] - Y[0, :])
            dY[ni - 1, :] = dY[ni - 1, :] + vf[ni - 1, :, 0].view(nj, 1).expand(nj, 2) * (Y[ni - 1, :] - Y[ni - 2, :])

            dY[:, 0] = dY[:, 0] + vf[:, 0, 1].view(ni, 1).expand(ni, 2) * (Y[:, 1] - Y[:, 0])
            dY[:, nj - 1] = dY[:, nj - 1] + vf[:, nj - 1, 1].view(ni, 1).expand(ni, 2) * (Y[:, nj - 1] - Y[:, nj - 2])

        elif dimension == 3:
            ni, nj, nk = Y.shape[:3]

            # Center.
            dY[1:ni - 1, :, :] = dY[1:ni - 1, :, :] + 0.5 * vf[1:ni - 1, :, :, 0].view(ni - 2, nj, nk, 1).expand(ni - 2,
                                                                                                                 nj, nk,
                                                                                                                 3) * (
                                         Y[2:ni, :, :] - Y[0:ni - 2, :, :])
            dY[:, 1:nj - 1, :] = dY[:, 1:nj - 1, :] + 0.5 * vf[:, 1:nj - 1, :, 1].view(ni, nj - 2, nk, 1).expand(ni,
                                                                                                                 nj - 2,
                                                                                                                 nk,
                                                                                                                 3) * (
                                         Y[:, 2:nj, :] - Y[:, 0:nj - 2, :])
            dY[:, :, 1:nk - 1] = dY[:, :, 1:nk - 1] + 0.5 * vf[:, :, 1:nk - 1, 2].view(ni, nj, nk - 2, 1).expand(ni, nj,
                                                                                                                 nk - 2,
                                                                                                                 3) * (
                                         Y[:, :, 2:nk] - Y[:, :, 0:nk - 2])

            # Borders.
            dY[0, :, :] = dY[0, :, :] + vf[0, :, :, 0].view(nj, nk, 1).expand(nj, nk, 3) * (Y[1, :, :] - Y[0, :, :])
            dY[ni - 1, :, :] = dY[ni - 1, :, :] + vf[ni - 1, :, :, 0].view(nj, nk, 1).expand(nj, nk, 3) * (
                    Y[ni - 1, :, :] - Y[ni - 2, :, :])

            dY[:, 0, :] = dY[:, 0, :] + vf[:, 0, :, 1].view(ni, nk, 1).expand(ni, nk, 3) * (Y[:, 1, :] - Y[:, 0, :])
            dY[:, nj - 1, :] = dY[:, nj - 1, :] + vf[:, nj - 1, :, 1].view(ni, nk, 1).expand(ni, nk, 3) * (
                    Y[:, nj - 1, :] - Y[:, nj - 2, :])

            dY[:, :, 0] = dY[:, :, 0] + vf[:, :, 0, 2].view(ni, nj, 1).expand(ni, nj, 3) * (Y[:, :, 1] - Y[:, :, 0])
            dY[:, :, nk - 1] = dY[:, :, nk - 1] + vf[:, :, nk - 1, 2].view(ni, nj, 1).expand(ni, nj, 3) * (
                    Y[:, :, nk - 1] - Y[:, :, nk - 2])

        else:
            raise RuntimeError('Invalid dimension of the ambient space: %d' % dimension)

        return dY

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write_flow(self, objects_names, objects_extensions, template, template_data, output_dir,
                   write_adjoint_parameters=False, write_only_last = False):
        assert not self.flow_is_modified, "You are trying to write flow data, but it has been modified and not updated."

        for j in range(self.n_time_points):

            if (not write_only_last) or (write_only_last and j == self.n_time_points -1):
                names = []
                for k, elt in enumerate(objects_names):
                    names.append(elt + "__tp_" + str(j) + objects_extensions[k])

                deformed_points = self.get_template_points(j)
                deformed_data = template.get_deformed_data(deformed_points, template_data)
                
                #modif fg: do not write flow
                template.write(output_dir, names, {key: detach(value) for key, value in deformed_data.items()})

                if write_adjoint_parameters:
                    cp = detach(self.cp_t[j])
                    mom = detach(self.momenta_t[j])
                    if self.transport_cp:
                        write_2D_array(cp, output_dir, elt + "__ControlPoints__tp_" + str(j) + ".txt")
                    write_3D_array(mom, output_dir, elt + "__Momenta__tp_" + str(j) + ".txt")

    def write_control_points_and_momenta_flow(self, name):
        """
        Write the flow of cp and momenta
        names are expected without extension
        """
        assert not self.shoot_is_modified, \
            "You are trying to write data relative to the shooting, but it has been modified and not updated."
        assert len(self.cp_t) == len(self.momenta_t), \
            "Something is wrong, not as many cp as momenta in diffeo"
        
        for j, (control_points, momenta) in enumerate(zip(self.cp_t, self.momenta_t)):
            write_2D_array(detach(control_points), name + "__control_points_" + str(j) + ".txt")
            write_2D_array(detach(momenta), name + "__momenta_" + str(j) + ".txt")
