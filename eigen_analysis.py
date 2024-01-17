import sys
sys.path.insert(1, '/home/rc13011/projects/mxs/pymxs/models')

import argparse
import json
import os
import numpy as np
from collections import namedtuple

from stable_baselines3 import PPO as MlAlg

import matplotlib.pyplot as plt
import gym
import gym_mxs

import subprocess

from prettyeqn import Equation, Symbol, Matrix, SymbolVector

def namespace_object_hook(dict):
  container = argparse.Namespace()
  for k, v in dict.items():
    setattr(container, k, v)
  return container

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("run_name")

  output_args = parser.add_argument_group("Output options")
  output_args.add_argument("--no-save", dest="save", action="store_false")
  output_args.add_argument("-d", "--directory", default="./runs", help="Destination for saving runs")

  args = parser.parse_args()

  run_dir = f"{args.directory}/{args.run_name}"

  with open(f"{run_dir}/metadata.json") as f:
    metadata = json.load(f)
    print(metadata)
    f.seek(0)
    run_args = json.load(
      f,
      object_hook=namespace_object_hook
    )
  arg_defaults = [
    ("use_reduced_observation", False),
    ("multi_manoeuvre", False)
  ]
  for (arg,default) in arg_defaults:
    if not hasattr(run_args, arg):
      setattr(run_args, arg, default) 

  # Attempt to checkout correct version of reward function
  try:
    commit = run_args.commit
    if "dirty" in commit:
      commit = commit[0:-6] # Remove -dirty
    subprocess.check_call(f"git show {commit}:testgym.py > testgym_previous.py", shell=True)
    from testgym_previous import create_reward_func
    os.remove("testgym_previous.py")
  except Exception as e:
    print(f"Error importing previous version of reward function: {e}")
    from testgym import create_reward_func
  from testgym import evaluate_model, LongitudinalStateWrapper, MultiManoeuvreWrapper, StartStateWrapper

  reward_func = create_reward_func(run_args)

  model = MlAlg.load(f"{run_dir}/model.zip")
  env = gym.make('gym_mxs/MXS-v0', reward_func=reward_func, timestep_limit=run_args.episode_length)

  if run_args.use_reduced_observation:
    env = LongitudinalStateWrapper(env)

  if run_args.manoeuvre in ["hoverexit", "hoversus"]:
    angle_by_2 = (np.pi / 2) / 2
    env = StartStateWrapper(
      env,
      start_velocity=[0,0,0],
      start_attitude=[0,np.sin(angle_by_2),0,np.cos(angle_by_2)]
    )

  if run_args.multi_manoeuvre:
    manoeuvre_names = ["hover", "descent"]
    env = MultiManoeuvreWrapper(
      env,
      manoeuvre_names,
      create_reward_func,
      run_args
    )


  obs = env.reset()
  obs = np.array(obs)
  state = np.array(env.unwrapped._get_obs())
  action, _state = model.predict(obs, deterministic=True)

  next_obs_list, _reward, _done, _info = env.step(action)
  
  def get_derivative(x):
    """
    Return the derivative of the system (x_dot) at state x
    """
    env.unwrapped.vehicle.statevector = x
    y = env.observation(x)

    action, _state = model.predict(y, deterministic=True)
    env.step(action)
    x_t1 = env.unwrapped._get_obs()
    # next_obs = obs + x_dot * env.dT
    # Therefore:
    return (np.array(x_t1) - x) / env.dT

  # Get derivative at the operating point
  x_dot_0 = get_derivative(state)

  dX = 0.001

  a = np.zeros((state.size, state.size))

  for state_dim in range(len(state)):
    deltaX = np.zeros_like(state)
    deltaX[state_dim] += dX

    x_dot = get_derivative(state + deltaX)
    x_dot_dot = (x_dot - x_dot_0) / dX

    a[:,state_dim] = x_dot_dot

  print(f"=== Full Matrix ===\n")
  # print(a)
  eigvals, eigvecs = np.linalg.eig(a)
  print(f"Eigenvalues:\n{eigvals}")
  
  plt.figure("Full Matrix")
  plt.scatter(
    np.real(eigvals),
    np.imag(eigvals)
  )
  
  plt.xlabel('Re')
  plt.ylabel('Im')
  # plt.xscale('log')
  # plt.yscale('log')
  plt.minorticks_on()
  plt.grid(True,'both')
  plt.axis('equal')
  
  print(Equation([
    Symbol('x\u0307'),
    Symbol('='),
    Matrix(a),
    SymbolVector(['x','y','z','u','v','w','qi','qj','qk','qw','p','q','r',])
  ]))
  
  # Now do the same thing but with fiddling for the quaternion
  a_red = np.zeros((6,6))
  longitudinal_indices = [0, 2, 3, 5, 7, 9, 11]
  state_indices = [0, 2, 3, 5, 11, None]
  
  theta = np.arcsin(2*(state[9]*state[7]))
  print(f"Initial theta: {theta}")
  
  # Get theta_dot from quaternion_dot
  theta_dot_0 = 2.0 * x_dot_0[9] / -state[7]
  
  for aj_index,state_index in enumerate(state_indices):
    deltaX = np.zeros_like(state)
    if state_index is None:
        # Theta is represented by a quaternion so needs special handling
        # Calculate the equivalent theta
        theta = np.arcsin(2*(state[9]*state[7]))
        # Set quaternion elements to appropriate values by calculating required deltaX
        deltaX[7] = np.sin((theta+dX)/2) - state[7]
        deltaX[9] = np.cos((theta+dX)/2) - state[9]
    else:
      deltaX[state_index] += dX

    if run_args.use_reduced_observation:
      obs = (state + deltaX)[longitudinal_indices]
    else:
      obs = state + deltaX

    x_dot = get_derivative(state + deltaX)
    x_dot_dot = (x_dot - x_dot_0) / dX

    for ai_index, deriv_index in enumerate(state_indices):
        if deriv_index is None:
            # Compute the rate of change of theta from the rate of change of quaternion
            # https://ahrs.readthedocs.io/en/latest/filters/angular.html
            # qx = qz = 0
            # wx = wz = 0
            # dqw = 0.5 * (-wy*qy)
            # dqy = dqj = 0.5 * (wy*qw - wz*qx + wx*qz)
            #           = 0.5 * wy*qw
            
            # dqw
            wy = 2.0 * x_dot[9] / -state[7]
            # dqj
            wy2 = 2.0 * x_dot[7] / state[9]
            try:
              delta_wy = wy - wy2
              assert delta_wy < 0.0001
            except AssertionError:
              print(f"wy results differ: {wy=} {wy2=}")
              print(f"For A[{ai_index}, {aj_index}]")
              raise
            
            a_red[ai_index,aj_index] = (wy - theta_dot_0) / dX
        else:
            a_red[ai_index,aj_index] = x_dot_dot[deriv_index]
  
  print(f"=== Reduced Matrix ===\n")
  # print(a_red)
  eigvals, eigvecs = np.linalg.eig(a_red)
  print("Eigenvalues:")
  for val in eigvals:
    print(f"  {val.real:9.4f} {val.imag:+9.4f}j")
  
  print("Eigenvectors:")
  formatstr = "[{} ]".format(" {:8.4f}" * 6)
  print("[{} ]".format(" {:>8s}" * 6).format("x","z","u","w","q","\u03b8"))
  for i in range(eigvecs.shape[0]):
    vec = eigvecs[:,i]
    print(formatstr.format(*[abs(x) for x in vec]))
  
  plt.figure("Reduced Matrix")
  plt.scatter(
    np.real(eigvals),
    np.imag(eigvals)
  )
  
  plt.xlabel('Re')
  plt.ylabel('Im')
  # plt.xscale('log')
  # plt.yscale('log')
  plt.minorticks_on()
  plt.grid(True,'both')
  # plt.axis('equal')

  print(Equation([
    SymbolVector(['x\u0307', 'z\u0307', 'u\u0307','w\u0307','q\u0307','\u03b8\u0307']),
    Symbol('='),
    Matrix(a_red),
    SymbolVector(['x', 'z', 'u','w','q','\u03b8'])
  ]))

  print(a_red[:,0])
  
  plt.show()
