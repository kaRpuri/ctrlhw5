import numpy as np
from pydrake.autodiffutils import AutoDiffXd

def cos(theta):
  return AutoDiffXd.cos(theta)
def sin(theta):
  return AutoDiffXd.sin(theta)

def box_obstacle(ee_pos, box_center, width, height):
  '''
  Calculate signed (squared) distance to box obstacle
    ee_pos: [x,z] np.array
    box_center: [x,z] np.array
    width and height: from end to end
  '''
  # TODO: Write signed distance formula for box
  xb = box_center[0]
  zb = box_center[1]
  xe = ee_pos[0]
  ze = ee_pos[1]


  dx = AutoDiffXd.abs(xe - xb) - width / 2
  dz = AutoDiffXd.abs(ze - zb) - height / 2

  dx_pos = AutoDiffXd.max(dx, AutoDiffXd(0.0))
  dz_pos = AutoDiffXd.max(dz, AutoDiffXd(0.0))
  
  outside_dist = AutoDiffXd.sqrt(dx_pos * dx_pos + dz_pos * dz_pos)
  inside_dist = AutoDiffXd.min(AutoDiffXd.max(dx, dz), AutoDiffXd(0.0))
  
  return outside_dist + inside_dist


def collision_constraint_function(box_params):
  '''
  Impose a constraint such that the EE does not collide with the obstacle
  pos (x_EE, z_EE):
    Matrix([[-l*sin(q0) - l*sin(q0 + q1)], [-l*cos(q0) - l*cos(q0 + q1)]])

  '''
  l = 1
  g = 9.81
  box_center = box_params[0]
  box_width = box_params[1]
  box_height = box_params[2]
  
  def collision_box_fn(vars):
    q = vars[:2]
    constraint_eval = np.zeros((1,), dtype=AutoDiffXd)
    pos = np.array([[-l*sin(q[0]) - l*sin(q[0] + q[1])],
                    [-l*cos(q[0]) - l*cos(q[0] + q[1])]])
    
    constraint_eval[0] = box_obstacle(pos.flatten(), box_center, box_width, box_height)


    return constraint_eval
    
  return collision_box_fn

def AddBoxCollisionConstraints(prog, q, box_params):
  '''
  q: joint pos (q0, q1)
  box_params: tuple ([center_x, center_z], width, height)
  '''
  coll_constr = collision_constraint_function(box_params)

  # TODO: Add collision constraint as an inequality constraint using prog.AddConstraint
  radius = 0.05
  
  def constraint_fn(vars):
      return np.array([coll_constr(vars) - radius])
  
  prog.AddConstraint(constraint_fn, np.array([0.0]), np.array([np.inf]), q)
