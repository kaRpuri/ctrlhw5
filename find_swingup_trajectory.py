import matplotlib.pyplot as plt
import numpy as np
import importlib

from pydrake.all import (
    DiagramBuilder,
    Simulator,
    FindResourceOrThrow,
    MultibodyPlant,
    PiecewisePolynomial,
    SceneGraph,
    Parser,
    JointActuatorIndex,
    MathematicalProgram,
    Solve,
)
from kinematic_constraints import AddBoxCollisionConstraints
from dynamics_constraints import AddCollocationConstraints, EvaluateDynamics


def find_swingup_trajectory(N, initial_state, final_configuration, box_centers, box_width, box_height):
    """
    Parameters:
      N - number of knot points
      initial_state - starting configuration
      distance - target distance to throw the ball

    """

    builder = DiagramBuilder()
    plant = builder.AddSystem(MultibodyPlant(0.0))
    file_name = "acrobot.urdf"
    Parser(plant=plant).AddModels(file_name)
    plant.Finalize()
    acrobot = plant.ToAutoDiffXd()

    plant_context = plant.CreateDefaultContext()
    context = acrobot.CreateDefaultContext()

    # Dimensions specific to the acrobot
    n_q = acrobot.num_positions()
    n_v = acrobot.num_velocities()
    n_x = n_q + n_v
    n_u = acrobot.num_actuators()

    # Store the actuator limits here
    effort_limits = np.zeros(n_u)
    for act_idx in range(n_u):
        effort_limits[act_idx] = acrobot.get_joint_actuator(
            JointActuatorIndex(act_idx)
        ).effort_limit()
    joint_limits = np.pi * np.ones(n_q)
    vel_limits = 15 * np.ones(n_v)

    # Create the mathematical program
    prog = MathematicalProgram()
    x = np.zeros((N, n_x), dtype="object")
    u = np.zeros((N, n_u), dtype="object")
    for i in range(N):
        x[i] = prog.NewContinuousVariables(n_x, "x_" + str(i))
        u[i] = prog.NewContinuousVariables(n_u, "u_" + str(i))

    t0 = 0.0
    dt = 0.7
    timesteps = np.linspace(t0, N*dt, N)
    x0 = x[0]
    xf = x[-1]

    #############################################################    
    # DO NOT MODIFY THE LINES ABOVE
    #############################################################

    # Add the kinematic constraints (initial state, final state)
    # TODO: Add constraints on the initial state
    prog.AddLinearEqualityConstraint(x0, initial_state)
    qf = xf[:n_q]
    prog.AddLinearEqualityConstraint(qf, final_configuration)

    # TODO: Add the collision constraints to every timestep, for every box
    # for every timestep
    for i in range(N):
        # for every box
        for box_center in box_centers:
            box_params = (box_center, box_width, box_height)
            AddBoxCollisionConstraints(prog, x[i][:n_q], box_params)
            

    # Add the collocation aka dynamics constraints
    AddCollocationConstraints(prog, acrobot, context, N, x, u, timesteps)

    # TODO: Add the cost function here
    cost = 0
    for i in range(N - 1):
        prog.AddQuadraticCost((dt / 2) * np.eye(n_u), np.zeros(n_u), u[i])
        prog.AddQuadraticCost((dt / 2) * np.eye(n_u), np.zeros(n_u), u[i + 1])

    # TODO: Add bounding box constraints on the inputs and qdot
    for i in range(N):
        prog.AddBoundingBoxConstraint(-effort_limits, effort_limits, u[i])
        prog.AddBoundingBoxConstraint(-joint_limits, joint_limits, x[i][:n_q])
        prog.AddBoundingBoxConstraint(-vel_limits, vel_limits, x[i][n_q:])

    # TODO: give the solver an initial guess for x and u using prog.SetInitialGuess(var, value)
    for i in range(N):
        prog.SetInitialGuess(x[i], np.zeros(n_x))
        prog.SetInitialGuess(u[i], np.zeros(n_u))
        prog.SetInitialGuess(x[i][0], np.pi)
        prog.SetInitialGuess(x[i][2], 0.0)
        prog.SetInitialGuess(x[i][3], 0.0)  
        prog.SetInitialGuess(x[i][1], (final_configuration[1] - initial_state[1]) * i / (N - 1))


    #############################################################
    # DO NOT MODIFY THE LINES BELOW
    #############################################################

    # Set up solver
    result = Solve(prog)

    x_sol = result.GetSolution(x)
    u_sol = result.GetSolution(u)

    print("optimal cost: ", result.get_optimal_cost())
    print("x_sol: ", x_sol)
    print("u_sol: ", u_sol)

    print(result.get_solution_result())

    # Reconstruct the trajectory
    u_sol = np.expand_dims(u_sol, axis=1)
    xdot_sol = np.zeros(x_sol.shape)
    for i in range(N):
        xdot_sol[i] = EvaluateDynamics(plant, plant_context, x_sol[i], u_sol[i])

    x_traj = PiecewisePolynomial.CubicHermite(timesteps, x_sol.T, xdot_sol.T)
    u_traj = PiecewisePolynomial.ZeroOrderHold(timesteps, u_sol.T)

    return x_traj, u_traj, prog, result, prog.GetInitialGuess(x), prog.GetInitialGuess(u)


if __name__ == "__main__":
    N = 5
    initial_state = np.zeros(4)
    final_configuration = np.array([np.pi, 0])
    box_center = np.array([2.0, 0.0])
    box_width = 1.0
    box_height = 1.0
    x_traj, u_traj, prog, _, _, _ = find_swingup_trajectory(
        N, initial_state, final_configuration,
        box_center, box_width, box_height
    )
