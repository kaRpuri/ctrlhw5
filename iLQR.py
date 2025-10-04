import numpy as np
import time
from scipy.signal import cont2discrete
from typing import List, Tuple

# import quad_sim
import matplotlib.pyplot as plt
import sympy as sm
import sympy.physics.mechanics as me
from scipy.integrate import solve_ivp


class iLQR(object):

    def __init__(
        self,
        x_goal: np.ndarray,
        N: int,
        dt: float,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
    ):
        """
        Constructor for the iLQR solver
        :param N: iLQR horizon
        :param dt: timestep
        """

        # Quadrotor dynamics parameters
        self.m = 1
        self.m1 = 0.2
        self.a = 0.25
        self.I = 0.0625
        self.g = 9.81
        self.L = 1.0

        self.nx = 8
        self.nu = 2

        # iLQR constants
        self.N = N
        self.dt = dt

        # regularization
        self.rho = 1e-8
        self.alpha_step = 1
        self.max_iter = 1e3
        self.tol = 1e-4

        # target state
        self.x_goal = x_goal
        self.u_goal = 0.5 * 9.81 * np.ones((2,))

        # Cost terms
        self.Q = Q
        self.R = R
        self.Qf = Qf

        # Symbolic linearized dynamics
        self.A, self.B, self.xdot = self.get_symbolic_linearized_dynamics()

    def simulate_dynamics(self, xc, uc, dt):
        def f(_, x):
            sub_vals = [
                self.m,
                self.m1,
                self.a,
                self.I,
                self.L,
                self.g,
                x[4],
                x[5],
                x[6],
                x[7],
                x[0],
                x[1],
                x[2],
                x[3],
                uc[0],
                uc[1],
            ]
            return self.xdot(*sub_vals).ravel()

        sol = solve_ivp(f, (0, dt), xc, first_step=dt)
        return sol.y[:, -1].ravel()

    def get_symbolic_linearized_dynamics(self):
        start_time = time.perf_counter()
        print(f"Start creating symbolic linearized dynamics")
        t = me.dynamicsymbols._t
        m, m1, a, I, L, g = sm.symbols("m, m1, a, I, L, g")
        theta, phi, y, z = me.dynamicsymbols("theta, phi, y, z")

        q = sm.Matrix([y, z, theta, phi])
        qd = q.diff(t)
        qdd = qd.diff(t)

        x = sm.Matrix([q, qd])
        u1, u2 = sm.symbols("u1, u2")
        u = sm.Matrix([u1, u2])

        # Derive the manipulator equation using Lagrange's method
        T = (
            m * (qd[0] ** 2 + qd[1] ** 2) / 2
            + m1 * (qd[0] + qd[3] * L * sm.cos(phi)) ** 2 / 2
            + m1 * (qd[1] + qd[3] * L * sm.sin(phi)) ** 2 / 2
            + I * qd[2] ** 2 / 2
        )
        V = m * g * z + m1 * g * (z - L * sm.cos(phi))
        K = T - V

        K_as_matrix = sm.Matrix([K])
        Fs = sm.trigsimp(
            (K_as_matrix.jacobian(qd).diff(t) - K_as_matrix.jacobian(q)).transpose()
        )

        M = sm.trigsimp(Fs.jacobian(qdd))
        C = sm.Matrix(
            [
                -m1 * L * sm.sin(phi) * qd[3] ** 2,
                m1 * L * sm.cos(phi) * qd[3] ** 2,
                0,
                0,
            ]
        )
        Tg = sm.Matrix([0, -(m + m1) * g, 0, -m1 * g * L * sm.sin(phi)])

        Bu = sm.Matrix(
            [
                [-sm.sin(theta), -sm.sin(theta)],
                [sm.cos(theta), sm.cos(theta)],
                [a, -a],
                [0, 0],
            ]
        )
        qdd_inverse_dyn = sm.Matrix(M.inv() * (-C + Tg + sm.Matrix(Bu * u)))
        xd = sm.Matrix([qd, qdd_inverse_dyn])

        A = sm.trigsimp(xd.jacobian(x))
        B = sm.trigsimp(xd.jacobian(u))
        sym_vars = {
            "m": m,
            "m1": m1,
            "a": a,
            "I": I,
            "L": L,
            "g": g,
            "ydot": qd[0],
            "zdot": qd[1],
            "thetadot": qd[2],
            "phidot": qd[3],
            "y": y,
            "z": z,
            "theta": theta,
            "phi": phi,
            "u1": u1,
            "u2": u2,
        }
        A_lambdified = sm.lambdify(list(sym_vars.values()), A)
        B_lambdified = sm.lambdify(list(sym_vars.values()), B)
        xdot_lambdified = sm.lambdify(list(sym_vars.values()), xd)
        print(
            f"Finished creating symbolic linearized dynamics after {time.perf_counter() - start_time:.1f} seconds"
        )
        return A_lambdified, B_lambdified, xdot_lambdified

    def total_cost(self, xx, uu):
        J = sum([self.running_cost(xx[k], uu[k]) for k in range(self.N - 1)])
        return J + self.terminal_cost(xx[-1])

    def get_linearized_dynamics(
        self, x: np.ndarray, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        sub_vals = [
            self.m,
            self.m1,
            self.a,
            self.I,
            self.L,
            9.81,
            x[4],
            x[5],
            x[6],
            x[7],
            x[0],
            x[1],
            x[2],
            x[3],
            u[0],
            u[1],
        ]
        return self.A(*sub_vals), self.B(*sub_vals)

    def get_linearized_discrete_dynamics(
        self, x: np.ndarray, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param x: state
        :param u: input
        :return: the discrete linearized dynamics matrices, A, B as a tuple
        """
        A, B = self.get_linearized_dynamics(x, u)
        C = np.eye(A.shape[0])
        D = np.zeros((A.shape[0],))
        [Ad, Bd, _, _, dt] = cont2discrete((A, B, C, D), self.dt)
        return Ad, Bd

    def running_cost(self, xk: np.ndarray, uk: np.ndarray) -> float:
        """
        :param xk: state
        :param uk: input
        :return: l(xk, uk), the running cost incurred by xk, uk
        """

        # Standard LQR cost on the goal state
        lqr_cost = (
            self.dt
            * 0.5
            * (
                (xk - self.x_goal).T @ self.Q @ (xk - self.x_goal)
                + (uk - self.u_goal).T @ self.R @ (uk - self.u_goal)
            )
        )

        return lqr_cost

    def grad_running_cost(self, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        """
        :param xk: state
        :param uk: input
        :return: [∂l/∂xᵀ, ∂l/∂uᵀ]ᵀ, evaluated at xk, uk
        """
        grad = np.zeros((self.nx + self.nu,))

        # TODO: Compute the gradient
        grad[:self.nx] = self.Q @ (xk - self.x_goal)
        grad[self.nx:] = self.R @ (uk - self.u_goal)

        return grad

    def hess_running_cost(self, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        """
        :param xk: state
        :param uk: input
        :return: The hessian of the running cost
        [[∂²l/∂x², ∂²l/∂x∂u],
         [∂²l/∂u∂x, ∂²l/∂u²]], evaluated at xk, uk
        """
        H = np.zeros((self.nx + self.nu, self.nx + self.nu))


        # TODO: Compute the hessian
        H[:self.nx, :self.nx] = self.Q
        H[self.nx:, self.nx:] = self.R    

        return H

    def terminal_cost(self, xf: np.ndarray) -> float:
        """
        :param xf: state
        :return: Lf(xf), the running cost incurred by xf
        """
        return 0.5 * (xf - self.x_goal).T @ self.Qf @ (xf - self.x_goal)

    def grad_terminal_cost(self, xf: np.ndarray) -> np.ndarray:
        """
        :param xf: final state
        :return: ∂Lf/∂xf
        """

        grad = np.zeros((self.nx))
        
        # TODO: Compute the gradient
        grad = self.Qf @ (xf - self.x_goal)


        return grad

    def hess_terminal_cost(self, xf: np.ndarray) -> np.ndarray:
        """
        :param xf: final state
        :return: The hessian of the running cost
        [[∂²l/∂x², ∂²l/∂x∂u],
         [∂²l/∂u∂x, ∂²l/∂u²]]
        """

        H = np.zeros((self.nx, self.nx))
        H = self.Qf 


        # TODO: Compute H

        return H

    def forward_pass(
        self,
        xx: List[np.ndarray],
        uu: List[np.ndarray],
        dd: List[np.ndarray],
        KK: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        :param xx: list of states, should be length N
        :param uu: list of inputs, should be length N-1
        :param dd: list of "feed-forward" components of iLQR update, should be length N-1
        :param KK: list of "Feedback" LQR gain components of iLQR update, should be length N-1
        :return: A tuple (xx, uu) containing the updated state and input
                 trajectories after applying the iLQR forward pass
        """

        assert len(xx) == self.N
        assert len(uu) == self.N - 1
        assert len(dd) == self.N - 1
        assert len(KK) == self.N - 1

        xtraj = [np.zeros((self.nx,))] * self.N
        utraj = [np.zeros((self.nu,))] * (self.N - 1)
        xtraj[0] = xx[0]

        # TODO: Compute forward pass
        alpha = self.alpha_step
        for k in range(self.N - 1):
            xk = xtraj[k]
            uk = uu[k] + alpha * dd[k] + KK[k] @ (xk - xx[k])
            xk_next = self.simulate_dynamics(xk, uk, self.dt)
            xtraj[k + 1] = xk_next
            utraj[k] = uk

        return xtraj, utraj

    def backward_pass(
        self, xx: List[np.ndarray], uu: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        :param xx: state trajectory guess, should be length N
        :param uu: input trajectory guess, should be length N-1
        :return: KK and dd, the feedback and feedforward components of the iLQR update
        """
        assert len(xx) == self.N
        assert len(uu) == self.N - 1

        dd = [np.zeros((self.nu,))] * (self.N - 1)
        KK = [np.zeros((self.nu, self.nx))] * (self.N - 1)

        # TODO: compute backward pass

    
        V_x = self.grad_terminal_cost(xx[-1])
        V_xx = self.hess_terminal_cost(xx[-1])

        for k in range(self.N - 2, -1, -1):
            xk = xx[k]
            uk = uu[k]
            A, B = self.get_linearized_discrete_dynamics(xk, uk)








            grad = self.grad_running_cost(xk, uk)
            hess = self.hess_running_cost(xk, uk)

            l_x = grad[:self.nx]
            l_u = grad[self.nx:]

            l_xx = hess[:self.nx, :self.nx]
            l_uu = hess[self.nx:, self.nx:]
            l_xu = hess[:self.nx, self.nx:]

            Q_x = l_x + A.T @ V_x
            Q_u = l_u + B.T @ V_x
            Q_xx = l_xx + A.T @ V_xx @ A
            Q_uu = l_uu + B.T @ V_xx @ B
            Q_ux = l_xu.T + B.T @ V_xx @ A  
            Q_xu = Q_ux.T

        

            # Q_uu_reg = Q_uu + self.rho * np.eye(self.nu)

            K_k = -np.linalg.inv(Q_uu) @ Q_ux
            d_k = -np.linalg.inv(Q_uu) @ Q_u


            V_x = Q_x  - K_k.T @ Q_uu @ d_k 
            V_xx = Q_xx - K_k.T @ Q_uu @ K_k

            KK[k] = K_k
            dd[k] = d_k

        return dd, KK

    def calculate_optimal_trajectory(
        self, x: np.ndarray, uu_guess: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Calculate the optimal trajectory using iLQR from a given initial condition x,
        with an initial input sequence guess uu
        :param x: initial state
        :param uu_guess: initial guess at input trajectory
        :return: xx, uu, KK, the input and state trajectory and associated sequence of LQR gains
        """
        assert len(uu_guess) == self.N - 1

        # Get an initial, dynamically consistent guess for xx by simulating the quadrotor
        xx = [x]
        for k in range(self.N - 1):
            xx.append(self.simulate_dynamics(xx[k], uu_guess[k], self.dt))

        Jprev = np.inf
        Jnext = self.total_cost(xx, uu_guess)
        uu = uu_guess
        KK = None

        i = 0
        while np.abs(Jprev - Jnext) > self.tol and i < self.max_iter:
            dd, KK = self.backward_pass(xx, uu)
            xx, uu = self.forward_pass(xx, uu, dd, KK)

            Jprev = Jnext
            Jnext = self.total_cost(xx, uu)
            print(f"Iteration: {i}, Cost: {Jnext}, Previous Cost: {Jprev}")
            i += 1
        return xx, uu, KK
