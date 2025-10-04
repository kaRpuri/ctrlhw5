import numpy as np
import importlib
from iLQR import iLQR

import matplotlib.pyplot as plt

from quad_visualizer import Quadrotor2DWithPendulumVisualizer, create_animation
from IPython.display import HTML

quad_visualizer = Quadrotor2DWithPendulumVisualizer()


# Setup the iLQR problem
N = 200
dt = 0.02
x_goal = np.array([-0.5, 1.5, 0, np.pi, 0, 0, 0, 0])

# TODO: Adjust the costs as needed for convergence
Q = np.eye(8)
Qf = 1e2 * np.eye(8)
R = 1e-3 * np.eye(2)

ilqr = iLQR(x_goal, N, dt, Q, R, Qf)

# Initial state at rest at the origin
x0 = np.zeros((8,))

# initial guess for the input is just hovering in place
u_guess = [0.5 * 9.81 * ilqr.m * np.ones((2,))] * (N-1)

x_sol, u_sol, K_sol = ilqr.calculate_optimal_trajectory(x0, u_guess)

# Visualize the solution
xx = np.array(x_sol)
ani = create_animation(quad_visualizer, xx.T, dt=0.02)

HTML(ani.to_html5_video())