import numpy as np

from pydrake.math import inv
from pydrake.autodiffutils import AutoDiffXd


def EvaluateDynamics(acrobot, context, x, u):
  # Computes the dynamics xdot = f(x,u)

  acrobot.SetPositionsAndVelocities(context, x)
  n_v = acrobot.num_velocities()

  M = acrobot.CalcMassMatrixViaInverseDynamics(context)
  B = acrobot.MakeActuationMatrix()
  g = acrobot.CalcGravityGeneralizedForces(context)
  C = acrobot.CalcBiasTerm(context)

  M_inv = np.zeros((n_v,n_v)) 
  if(x.dtype == AutoDiffXd):
    M_inv = inv(M)
  else:
    M_inv = np.linalg.inv(M)

  v_dot = M_inv @ (B @ u + g - C)
  return np.hstack((x[-n_v:], v_dot))

def CollocationConstraintEvaluator(acrobot, context, dt, x_i, u_i, x_ip1, u_ip1):
  h_i = np.zeros(4,)
  # TODO: Add a dynamics constraint using x_i, u_i, x_ip1, u_ip1, dt
  # You should make use of the EvaluateDynamics() function to compute f(x,u)


  f_i = EvaluateDynamics(acrobot, context, x_i, u_i)
  f_ip1 = EvaluateDynamics(acrobot, context, x_ip1, u_ip1)

  h_i = (3/(2*dt)) * (x_ip1 - x_i) - (1/4) * (f_i - f_ip1) - EvaluateDynamics(acrobot, context, (x_i + x_ip1)/2 - (dt/8)*(f_ip1 - f_i), (u_i + u_ip1)/2)

  return h_i

def AddCollocationConstraints(prog, acrobot, context, N, x, u, timesteps):
  n_u = acrobot.num_actuators()
  n_x = acrobot.num_positions() + acrobot.num_velocities()
    
  for i in range(N - 1):
    def CollocationConstraintHelper(vars):
      x_i = vars[:n_x]
      u_i = vars[n_x:n_x + n_u]
      x_ip1 = vars[n_x + n_u: 2*n_x + n_u]
      u_ip1 = vars[-n_u:]
      return CollocationConstraintEvaluator(acrobot, context, timesteps[i+1] - timesteps[i], x_i, u_i, x_ip1, u_ip1)
      
    # TODO: Within this loop add the dynamics constraints for segment i (aka collocation constraints)
    #       to prog
    # Hint: use prog.AddConstraint(CollocationConstraintHelper, lb, ub, vars)
    # where vars = hstack(x[i], u[i], ...)
    vars = np.hstack((x[i], u[i], x[i+1], u[i+1]))
    lb = np.zeros((4,))
    ub = np.zeros((4,))

    prog.AddConstraint(CollocationConstraintHelper, lb, ub, vars)
