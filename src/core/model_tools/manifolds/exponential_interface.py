import os.path
import sys
import warnings
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../../../')
import torch


"""
An implementation of this interface must implement the inverse metric method, and optionnaly, a closed form (arg is velocity) or a closed form for dp.
Any exponential object is best used through a generic_geodesic.

Note: to use the parallel transport with a closed form geodesic, closed_form_velocity must be implemented
"""

# Possible improvements:
#   1) Store matrices if a lot of transports are required
#   2) Do not save the momenta trajectory if no transport is required
#   3) Do not go back and forth between velocity and momenta when only momenta are used !
# (maybe higher level management of this)
#   4) Maybe more subtle management of the rk2 operation: do not return momenta if no transport is used !
#   5) Add the possibility to implement the metric, if a closed form can be obtained (instead of inverting)
#   6) Manage exact transport formulas.


class ExponentialInterface:

    def __init__(self):
        self.number_of_time_points = 10
        self.position_t = None
        self.momenta_t = None
        self.velocity_t = None

        self.initial_momenta = None
        self.initial_position = None
        self.initial_velocity = None

        self.is_modified = True

        self.norm_squared = None

        self.has_closed_form = None
        self.has_closed_form_dp = None

    def get_initial_position(self):
        return self.initial_position

    def set_initial_position(self, q):
        self.initial_position = q
        self.is_modified = True

    def velocity_to_momenta(self, v, q=None):
        """
        Must be called at the initial position.
        """
        if q is None:
            return torch.matmul(torch.inverse(self.inverse_metric(self.initial_position).view(1, 1)), v)
        else:
            return torch.matmul(torch.inverse(self.inverse_metric(q).view(1, 1)), v)

    def momenta_to_velocity(self, p):
        """
        Must be called at the initial position.
        """
        return torch.matmul(self.inverse_metric(self.initial_position), p)

    def set_initial_momenta(self, p):
        self.initial_momenta = p
        self.initial_velocity = self.momenta_to_velocity(p)
        self.is_modified = True

    def set_initial_velocity(self, v):
        self.initial_velocity = v
        self.initial_momenta = self.velocity_to_momenta(v)
        self.is_modified = True

    def inverse_metric(self, q):
        raise ValueError("Inverse metric must be implemented in the child classes of the exponential interface.")

    def dp(self, q, p):
        raise ValueError("Dp must be implemented in the child classes of the exponential interface. "
                         "Alternatively, the flag has_closed_form_dp must be set to off.")

    def get_final_position(self):
        if self.initial_position is None:
            msg = "In get_final_position, I am not flowing because I don't have an initial position"
            warnings.warn(msg)
        if self.has_closed_form:
            return self.closed_form(self.position_t0, self.velocity_t0, 1.)
        else:
            if self.is_modified:
                msg = "Update should be called on a non closed-form geodesic before getting final position"
                warnings.warn(msg)
            else:
                return self.position_t[-1]

    def _flow(self):
        """
        Generic flow of an exponential.
        """
        if self.initial_position is None:
            msg = "In exponential update, I am not flowing because I don't have an initial position"
            warnings.warn(msg)
        if self.has_closed_form:
            raise ValueError("Flow should not be called on a closed form exponential. Set has_closed_form to True.")
        elif self.initial_momenta is None:
            msg = "In exponential update, I am not flowing because I don't have an initial momenta"
            warnings.warn(msg)
        else:
            """
            Standard flow using the Hamiltonian equation
            if dp is not provided, autodiff is used (expensive)
            """
            if self.has_closed_form_dp:
                self.position_t, self.momenta_t = ExponentialInterface.exponential(
                    self.initial_position, self.initial_momenta,
                    inverse_metric=self.inverse_metric,
                    nb_steps=self.number_of_time_points,
                    dp=self.dp)
            else:
                self.position_t, self.momenta_t = ExponentialInterface.exponential(
                    self.initial_position, self.initial_momenta,
                    inverse_metric=self.inverse_metric,
                    nb_steps=self.number_of_time_points)

    def update(self):
        """
        Update the exponential object. Only way to properly flow.
        """
        if self.has_closed_form:
            # Because we don't need, we'll get the closed form values that are directly required.
            return
        assert self.number_of_time_points > 0
        if self.is_modified:
            self._flow()
            self._update_norm_squared()
            self.is_modified = False

    def _update_norm_squared(self):
        self.norm_squared = ExponentialInterface.hamiltonian(
            self.initial_position, self.initial_momenta, self.inverse_metric)

    def parallel_transport(self, vector_to_transport, with_tangential_component=True):
        """
        Computes the parallel transport, using the Jacobi scheme.
        It is much faster if dp is given !
        # Note that if there is a closed form formula for the geodesic, everything can be done in terms of velocity...
        ACHTUNG: if a closed form exists for the geodesic, this method returns a list of velocities.
        Otherwise, it returns a list of momenta (because the velocities are not really needed for any computations)
        """

        # Closed form case, Jacobi fan with velocities only (fast) :)
        if self.has_closed_form:
            return self._parallel_transport_with_closed_form(vector_to_transport, with_tangential_component)

        # Second case: no closed form available. We use RK2 integration of the Hamiltonian equations + Jacobi field.
        else:
            return self._parallel_transport_without_closed_form(vector_to_transport, with_tangential_component)

    def _parallel_transport_with_closed_form(self, vector_to_transport, with_tangential_component=True):

        # Special cases, where the transport is simply the identity:
        #       1) Nearly zero initial momenta yield no motion.
        #       2) Nearly zero momenta to transport.
        if (torch.norm(self.initial_momenta).data.numpy()[0] < 1e-15 or
                    torch.norm(vector_to_transport).data.numpy()[0] < 1e-15):
            parallel_transport_t = [vector_to_transport] * self.number_of_time_points
            return parallel_transport_t

        h = 1. / (self.number_of_time_points - 1.)
        epsilon = h

        # First get the scalar product between the initial velocity and the vector to transport
        sp = ExponentialInterface.velocity_scalar_product(self.initial_position, self.initial_velocity, vector_to_transport)
        vector_to_transport_orthogonal = vector_to_transport - sp * self.initial_velocity

        sp_for_assert = ExponentialInterface.velocity_scalar_product(self.initial_position, self.initial_velocity,
                                                                         vector_to_transport_orthogonal)
        assert sp_for_assert < 1e-5, "Projection onto orthogonal not orthogonal {e}".format(e=sp_for_assert)

        initial_norm_squared = ExponentialInterface.velocity_scalar_product(self.initial_position, vector_to_transport_orthogonal,
                                                                                vector_to_transport_orthogonal)

        parallel_transport_t = [vector_to_transport_orthogonal]

        for i in range(self.number_of_time_points - 1):
            # Get the two perturbed geodesics points
            velocity_ti = self.closed_form_velocity(self.position_t[i], self.velocity_t[i], h) # Could also be saved, in a perfect world.
            position_eps_pos = self.closed_form(self.position_t[i], velocity_ti + epsilon * parallel_transport_t[i], h)
            position_eps_neg = self.closed_form(self.position_t[i], velocity_ti - epsilon * parallel_transport_t[i], h)

            # Approximation of J / h
            approx_velocity = (position_eps_pos - position_eps_neg) / (2. * epsilon * h)
            approx_velocity_norm_squared = ExponentialInterface.velocity_scalar_product(self.position_t[i+1], approx_velocity, approx_velocity)
            renormalization_factor = torch.sqrt(initial_norm_squared / approx_velocity_norm_squared)
            renormalized_velocity = approx_velocity * renormalization_factor

            if abs(renormalization_factor.data.numpy()[0] - 1.) > 0.5:
                raise ValueError(
                    'Absurd required renormalization factor during parallel transport. Exception raised.')
            elif abs(renormalization_factor.data.numpy()[0] - 1.) > 0.02:
                msg = (
                        "Watch out, a large renormalization factor %.4f is required during the parallel transport, "
                        "please use a finer discretization." % renormalization_factor.data.numpy()[0])
                warnings.warn(msg)

            # Finalization
            parallel_transport_t.append(renormalized_velocity)

        assert len(parallel_transport_t) == len(self.position_t) == len(self.momenta_t), "Something went wrong"

        if with_tangential_component:
            parallel_transport_t = [parallel_transport_t[i] + sp * self.velocity_t[i] for i in range(self.number_of_time_points)]

        return parallel_transport_t

    def _parallel_transport_without_closed_form(self, vector_to_transport, with_tangential_component=True):

        momenta_to_transport = self.velocity_to_momenta(vector_to_transport)

        # Special cases, where the transport is simply the identity:
        #       1) Nearly zero initial momenta yield no motion.
        #       2) Nearly zero momenta to transport.
        if (torch.norm(self.initial_momenta).data.numpy()[0] < 1e-15 or
                    torch.norm(vector_to_transport).data.numpy()[0] < 1e-15):
            parallel_transport_t = [momenta_to_transport] * self.number_of_time_points
            return parallel_transport_t

        h = 1. / (self.number_of_time_points - 1.)
        epsilon = h

        # First get the scalar product between the initial velocity and the vector to transport.
        sp = ExponentialInterface.momenta_scalar_product(self.initial_position, self.initial_momenta,
                                                             momenta_to_transport)
        momenta_to_transport_orthogonal = momenta_to_transport - sp * self.initial_momenta

        sp_for_assert = ExponentialInterface.momenta_scalar_product(self.initial_position, self.initial_momenta,
                                                                        momenta_to_transport_orthogonal)
        assert sp_for_assert < 1e-5, "Projection onto orthogonal not orthogonal {e}".format(e=sp_for_assert)

        # Store the norm of this initial orthogonal momenta
        initial_norm_squared = ExponentialInterface.momenta_scalar_product(self.initial_position,
                                                                               momenta_to_transport_orthogonal,
                                                                               momenta_to_transport_orthogonal)

        parallel_transport_t = [momenta_to_transport_orthogonal]

        for i in range(self.number_of_time_points - 1):
            # Shoot the two perturbed geodesics:

            # Case where closed_dp is available
            if self.has_closed_form_dp:
                position_eps_pos, _ = ExponentialInterface._rk2_step_with_dp(self.position_t[i],
                                                                                 self.momenta_t[i] + epsilon *
                                                                                 parallel_transport_t[i - 1],
                                                                                 h, self.inverse_metric(),
                                                                                 self.dp)
                position_eps_neg, _ = ExponentialInterface._rk2_step_with_dp(self.position_t[i],
                                                                                 self.momenta_t[i] - epsilon *
                                                                                 parallel_transport_t[i - 1],
                                                                                 h, self.inverse_metric(),
                                                                                 self.dp)
            # Case where autodiff is required (expensive :( )
            else:
                position_eps_pos, _ = ExponentialInterface._rk2_step_without_dp(self.position_t[i],
                                                                                 self.momenta_t[i] + epsilon *
                                                                                 parallel_transport_t[i - 1],
                                                                                 h, self.inverse_metric())
                position_eps_neg, _ = ExponentialInterface._rk2_step_without_dp(self.position_t[i],
                                                                                 self.momenta_t[i] - epsilon *
                                                                                 parallel_transport_t[i - 1],
                                                                                 h, self.inverse_metric())

            # Approximation of J / h
            approx_velocity = (position_eps_pos - position_eps_neg) / (2. * epsilon * h)
            # Corresponding momenta, to continue the computations
            approx_momenta = self.velocity_to_momenta(approx_velocity, p=self.position_t[i + 1])

            # Renormalization
            approx_momenta_norm_squared = ExponentialInterface.momenta_scalar_product(self.position_t[i + 1],
                                                                                          approx_momenta,
                                                                                          approx_momenta)
            renormalization_factor = torch.sqrt(initial_norm_squared / approx_momenta_norm_squared)
            renormalized_momenta = approx_momenta * renormalization_factor

            if abs(renormalization_factor.data.numpy()[0] - 1.) > 0.5:
                raise ValueError(
                    'Absurd required renormalization factor during parallel transport. Exception raised.')
            elif abs(renormalization_factor.data.numpy()[0] - 1.) > 0.02:
                msg = (
                        "Watch out, a large renormalization factor %.4f is required during the parallel transport, "
                        "please use a finer discretization." % renormalization_factor.data.numpy()[0])
                warnings.warn(msg)

            # Finalization
            parallel_transport_t.append(renormalized_momenta)

        assert len(parallel_transport_t) == len(self.position_t) == len(self.momenta_t), "Something went wrong"

        if with_tangential_component:
            parallel_transport_t = [parallel_transport_t[i] + sp * self.momenta_t[i] for i in range(self.number_of_time_points)]

        return parallel_transport_t

    def set_parameters(self, extra_parameters):
        """
        Used to set any extra parameters of the exponential object.
        """
        msg = 'Set parameters called, but not implemented ! Is this right ?'
        warnings.warn(msg)

    #################################################################################################
    ####################    Static methods for generic manifold computations ########################
    #################################################################################################

    @staticmethod
    def _dp_autodiff(h, q):
        """
        if dp is not given on the manifold, we get it using automatic differentiation (more expensive of course)
        """
        return torch.autograd.grad(h, q, create_graph=True, retain_graph=True)[0]

    @staticmethod
    def _rk2_step_with_dp(q, p, dt, inverse_metric, dp):
            mid_q = q + 0.5 * dt * torch.matmul(inverse_metric(q), p)
            mid_p = p - 0.5 * dt * dp(q, p)
            return q + dt * torch.matmul(inverse_metric(mid_q), mid_p), p - dt * dp(q, p)

    @staticmethod
    def _rk2_step_without_dp(q, p, dt, inverse_metric):
        # Intermediate step
        h1 = ExponentialInterface.hamiltonian(q, p, inverse_metric)
        mid_q = q + 0.5 * dt * torch.matmul(inverse_metric(q), p)
        mid_p = p - 0.5 * dt * ExponentialInterface._dp_autodiff(h1, q)

        # Final step
        h2 = ExponentialInterface.hamiltonian(mid_q, mid_p, inverse_metric)
        return q + dt * torch.matmul(inverse_metric(mid_q), mid_p), p - dt * ExponentialInterface._dp_autodiff(h2, mid_q)

    @staticmethod
    def hamiltonian(q, p, inverse_metric):
        return ExponentialInterface.momenta_scalar_product(q, p, p, inverse_metric) * 0.5

    @staticmethod
    def momenta_scalar_product(q, p1, p2, inverse_metric):
        return torch.dot(p1, torch.matmul(inverse_metric(q), p2))

    @staticmethod
    def velocity_scalar_product(q, v1, v2, inverse_metric):
        return torch.dot(v1, torch.matmul(torch.inverse(inverse_metric(q)), v2))

    @staticmethod
    def exponential(self, q, p, inverse_metric, nb_steps=10, dp=None):
        """
        Use the given inverse_metric to compute the Hamiltonian equations.
        OR a given closed-form expression for the geodesic.
        """

        if dp is None:
            q.requires_grad = True

        traj_q, traj_p = [], []
        traj_q.append(q)
        traj_p.append(p)
        dt = 1. / float(nb_steps)
        times = np.linspace(dt, 1., nb_steps-1)

        if dp is None:
            for _ in times:
                new_q, new_p = self._rk2_step_without_dp(traj_q[-1], traj_p[-1], dt, inverse_metric)
                traj_q.append(new_q)
                traj_p.append(new_p)
        else:
            for _ in times:
                new_q, new_p = self._rk2_step_with_dp(traj_q[-1], traj_p[-1], dt, inverse_metric, dp)
                traj_q.append(new_q)
                traj_p.append(new_p)

        return traj_q, traj_p

