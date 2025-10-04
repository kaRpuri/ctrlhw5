"""Code for visualizing the quadrotor with a pendulum (largely based on
https://github.com/RussTedrake/underactuated/blob/master/underactuated/quadrotor2d.py)"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


class Quadrotor2DWithPendulumVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-2.0, 3.0)
        self.ax.set_ylim(-2.0, 3.0)

        self.length = 0.25  # moment arm (meters)
        self.rod_length = 0.9
        self.ball_radius = 0.1

        self.base = np.vstack(
            (
                1.2 * self.length * np.array([1, -1, -1, 1, 1]),
                0.025 * np.array([1, 1, -1, -1, 1]),
            )
        )
        self.pin = np.vstack(
            (
                0.005 * np.array([1, 1, -1, -1, 1]),
                0.1 * np.array([1, 0, 0, 1, 1]),
            )
        )
        a = np.linspace(0, 2 * np.pi, 50)
        self.prop = np.vstack(
            (self.length / 1.5 * np.cos(a), 0.1 + 0.02 * np.sin(2 * a))
        )
        self.rod = np.vstack(
            (
                self.rod_length * np.array([1, 0, 0, 1, 1]),
                0.01 * np.array([1, 1, -1, -1, 1]),
            )
        )
        a = np.linspace(0, 2 * np.pi, 50)
        self.ball = np.vstack(
            (
                self.rod_length + self.ball_radius * (1 + np.cos(a)),
                self.ball_radius * np.sin(a),
            )
        )

        # yapf: disable
        self.base_fill = self.ax.fill(
            self.base[0, :], self.base[1, :], zorder=1, edgecolor="k",
            facecolor=[.6, .6, .6])
        self.left_pin_fill = self.ax.fill(
            self.pin[0, :], self.pin[1, :], zorder=0, edgecolor="k",
            facecolor=[0, 0, 0])
        self.right_pin_fill = self.ax.fill(
            self.pin[0, :], self.pin[1, :], zorder=0, edgecolor="k",
            facecolor=[0, 0, 0])
        self.left_prop_fill = self.ax.fill(
            self.prop[0, :], self.prop[0, :], zorder=0, edgecolor="k",
            facecolor=[0, 0, 1])
        self.right_prop_fill = self.ax.fill(
            self.prop[0, :], self.prop[0, :], zorder=0, edgecolor="k",
            facecolor=[0, 0, 1])
        self.rod_fill = self.ax.fill(
            self.rod[0, :], self.rod[1, :], zorder=1, edgecolor="k",
            facecolor=[0, 0, 1])
        self.ball_fill = self.ax.fill(
            self.ball[0, :], self.ball[0, :], zorder=0, edgecolor="k",
            facecolor=[0, 0, 1])
        # yapf: enable

    def draw(self, x, t):
        R = np.array([[np.cos(x[2]), -np.sin(x[2])], [np.sin(x[2]), np.cos(x[2])]])

        p = np.dot(R, self.base)
        self.base_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.base_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        p = np.dot(R, np.vstack((-self.length + self.pin[0, :], self.pin[1, :])))
        self.left_pin_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.left_pin_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]
        p = np.dot(R, np.vstack((self.length + self.pin[0, :], self.pin[1, :])))
        self.right_pin_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.right_pin_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        p = np.dot(R, np.vstack((-self.length + self.prop[0, :], self.prop[1, :])))
        self.left_prop_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.left_prop_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        p = np.dot(R, np.vstack((self.length + self.prop[0, :], self.prop[1, :])))
        self.right_prop_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.right_prop_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        R1 = np.array(
            [
                [np.cos(-np.pi / 2 + x[3]), -np.sin(-np.pi / 2 + x[3])],
                [np.sin(-np.pi / 2 + x[3]), np.cos(-np.pi / 2 + x[3])],
            ]
        )
        p = np.dot(R1, self.rod)
        self.rod_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.rod_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]

        p = np.dot(R1, self.ball)
        self.ball_fill[0].get_path().vertices[:, 0] = x[0] + p[0, :]
        self.ball_fill[0].get_path().vertices[:, 1] = x[1] + p[1, :]
        self.ax.set_title("t = {:.1f}".format(t))


def create_animation(quad_vis, x_traj, dt):
    def update(i):
        quad_vis.draw(x_traj[:, i], i * dt)

    ani = animation.FuncAnimation(
        quad_vis.fig, update, x_traj.shape[1], interval=dt * 1000
    )
    return ani
