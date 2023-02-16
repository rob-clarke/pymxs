import sys
sys.path.insert(1, '/home/rc13011/projects/mxs/pymxs/models')

import json
import os
import subprocess
from contextlib import nullcontext

from stable_baselines3 import PPO as MlAlg
from numba import jit

import gym
import gym_mxs

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_X_LIMIT = 30 # m
DEFAULT_U_LIMIT = 25 # m/s
DEFAULT_TIME_LIMIT = 1000 # s*100
DEFAULT_CLIMB_WEIGHT = 1
DEFAULT_PITCH_WEIGHT = 0

def get_pitch(qx,qy,qz,qw):
  if True:
    # Can't use scipy if 'jit'ting
    rot = Rotation.from_quat(np.array([qx,qy,qz,qw]).T)
    [yaw, pitch, roll] = rot.as_euler('zyx', degrees=False)
    # print([yaw,pitch,roll])
    if yaw != 0:
      if pitch > 0:
        pitch = np.pi/2 + (np.pi/2 - pitch)
      else:
        pitch = -np.pi/2 + (-np.pi/2 - pitch)
  if False:
    sinp = np.sqrt(1 + 2 * (qw*qy - qx*qz))
    cosp = np.sqrt(1 - 2 * (qw*qy - qx*qz))
    pitch = 2 * np.arctan2(sinp, cosp) - np.pi / 2
  return pitch

def create_reward_func(args):
  # Split these out so numba can jit the reward function
  u_limit = args.u_limit
  x_limit = args.x_limit
  pitch_weight = args.pitch_weight
  climb_weight = args.climb_weight
  waypoint_weight = args.waypoint_weight
  waypoints = args.waypoints

  # @jit
  def descent_reward_func(obs, max_z):
    [x,y,z, u,v,w, qx,qy,qz,qw, p,q,r] = obs

    if max_z == None:
      max_z = 0
    # Update max_z
    max_z = max(z, max_z)

    pitch = get_pitch(qx,qy,qz,qw)

    if x < 0 or u > u_limit or u < 0 or pitch > np.radians(89) or pitch < np.radians(-270):
      return -1000, True, max_z

    if x > x_limit:
      total_ke_sqd = u**2+v**2+w**2
      ke_fraction = total_ke_sqd / (15**2)

      climb = max_z - z

      pitch_error = abs(pitch)

      return (1 - pitch_weight*pitch_error) * z / (ke_fraction * (1+climb*climb_weight)), True, max_z

    reward = 0
    if waypoint_weight != 0:
      for [wp_x,wp_z] in waypoints:
        if wp_x < x:
          continue
        reward += waypoint_weight / (np.hypot(x-wp_x, z-wp_z) + 0.01)

    return reward, False, max_z

  def within(value, lower, upper):
    if value < lower:
      return False
    if value > upper:
      return False
    return True

  def hover_reward_func(obs, reward_state):
    if reward_state is None:
      reward_state = 0
    reward_state += 1

    [x,y,z, u,v,w, qx,qy,qz,qw, p,q,r] = obs
    pitch = get_pitch(qx,qy,qz,qw)

    is_hover = within(q, -0.01, 0.01) \
      and within(pitch, np.radians(85), np.radians(95)) \
      and within(u, -0.1, 0.1) \
      and within(w, -0.1, 0.1) \

    if is_hover:
      # Reward is based on hover position
      return 100, True, reward_state

    if reward_state >= 250:
      # Time limited episode, reward progress to hover
      q_progress = 1 / (1 + abs(q))
      pitch_progress = 1 / (1+abs(np.radians(90) - pitch))
      u_progress = 1 / (1 + abs(u))
      w_progress = 1 / (1 + abs(w))
      hover_progress = q_progress * pitch_progress * u_progress * w_progress
      return 100 * hover_progress, True, None

    return 0, False, reward_state

  if not hasattr(args, "manoeuvre"):
    manoeuvre = "descent"
  else:
    manoeuvre = args.manoeuvre

  if manoeuvre == "hover":
    return hover_reward_func
  else:
    return descent_reward_func

def evaluate_model(model, env, output_path=False):
  obs = env.reset()
  done = False
  simtime = 0
  with open(output_path, "w") if output_path else nullcontext() as outfile:
    if outfile:
      outfile.write("time,x,y,z,u,v,w,qx,qy,qz,qw,p,q,r,alpha,airspeed,elevator,throttle\n")
    while not done:
      action, _state = model.predict(obs)
      obs, reward, done, info = env.step(action)
      if outfile:
        outfile.write(f"{simtime},{env.render('ansi')[1:-1]}\n")
      simtime += env.dT

  return obs, reward, done, info, simtime

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("run_name")

  output_args = parser.add_argument_group("Output options")
  output_args.add_argument("--no-save", dest="save", action="store_false", help="Don't save this run")
  output_args.add_argument("-d", "--directory", default="./runs", help="Destination for saving runs")
  output_args.add_argument("-o", "--output", action="store_true", help="Generate CSV for final output")
  output_args.add_argument("--plot", action="store_true", help="Show plots at end of training. (Will generate CSV as if -o specified)")
  output_args.add_argument("--ignore-dirty", action="store_true", help="Ignore dirty tree when saving run")

  training_args = parser.add_argument_group("Training options")
  training_args.add_argument("-s", "--steps", help="Total timesteps to train for", type=int, default=500_000)
  training_args.add_argument("-l", "--episode-length", help="Episode timestep limit", type=int, default=DEFAULT_TIME_LIMIT)

  reward_args = parser.add_argument_group("Reward function options")
  reward_args.add_argument("-x", "--x-limit", help="x coordinate limit", type=float, default=DEFAULT_X_LIMIT)
  reward_args.add_argument("-u", "--u-limit", help="u velocity limit", type=float, default=DEFAULT_U_LIMIT)
  reward_args.add_argument("-c", "--climb-weight", help="Weight for climb cost", type=float, default=DEFAULT_CLIMB_WEIGHT)
  reward_args.add_argument("-p", "--pitch-weight", help="Weight for pitch cost", type=float, default=DEFAULT_PITCH_WEIGHT)
  reward_args.add_argument("-w", "--waypoint-weight", help="Weight for waypoints", type=float, default=0)
  reward_args.add_argument("-f", "--waypoint-file", help="File for waypoints", default=0)
  reward_args.add_argument("-m", "--manoeuvre", help="Manoeuvre to use", type=str)

  args = parser.parse_args()

  # Check if on clean commit
  diff_result = subprocess.run(["git", "diff", "-s", "--exit-code", "HEAD"])
  git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], encoding="UTF-8")
  git_sha = git_sha.strip()
  if diff_result.returncode == 0:
    args.commit = git_sha
  else:
    if args.ignore_dirty or not args.save:
      args.commit = f"{git_sha}-dirty"
    else:
      print("Error: Current tree not committed.")
      print("Prevent saving with --no-save, or explicitly ignore dirty tree with --ignore-dirty")
      sys.exit(1)

  # Attempt to load any waypoint file
  if args.waypoint_file:
    with open(args.waypoint_file) as f:
      args.waypoints = json.load(f)
  else:
    args.waypoints = []

  reward_func = create_reward_func(args)

  env = gym.make('gym_mxs/MXS-v0', reward_func=reward_func, timestep_limit=1000)

  model = MlAlg("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=args.steps)

  run_dir = f"{args.directory}/{args.run_name}"
  if args.save:
    os.makedirs(run_dir)
    model.save(f"{run_dir}/model.zip")
    with open(f"{run_dir}/metadata.json", "w") as f:
      json.dump(vars(args), f, indent=2)

  if args.output or args.plot:
    output_file = f"{run_dir}/output.csv"
    evaluate_model(model, env, output_file)
    
  if args.plot:
    subprocess.call(["python", f"{os.path.dirname(os.path.realpath(__file__))}/plotting/unified_plot.py", "-d", args.directory, args.run_name])
