import sys
sys.path.insert(1, '/home/rc13011/projects/mxs/pymxs/models')

import argparse
import copy
import json
import os
import importlib
from collections import namedtuple
from contextlib import nullcontext
import numpy as np

from stable_baselines3 import PPO as MlAlg

import gym
import gym_mxs

import subprocess

def namespace_object_hook(dict):
  container = argparse.Namespace()
  for k, v in dict.items():
    setattr(container, k, v)
  return container

def evaluate_model(model, env, output_path=False, transformer=lambda x: x, initial_obs=None, initial_sim=0):
  if initial_obs is None:
    obs = env.reset()
  else:
    obs = initial_obs
  done = False
  simtime = initial_sim
  with open(output_path, "w") if output_path else nullcontext() as outfile:
    if outfile:
      outfile.write("time,x,y,z,u,v,w,qx,qy,qz,qw,p,q,r,alpha,airspeed,elevator,throttle\n")
    while not done:
      action, _state = model.predict(transformer(obs), deterministic=True)
      obs, reward, done, info = env.step(action)
      if outfile:
        outfile.write(f"{simtime},{env.render('ansi')[1:-1]}\n")
      simtime += env.dT

  return obs, reward, done, info, simtime

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  # parser.add_argument("run_name")

  output_args = parser.add_argument_group("Output options")
  output_args.add_argument("--no-save", dest="save", action="store_false")
  output_args.add_argument("-d", "--directory", default="./runs", help="Destination for saving runs")
  output_args.add_argument("-o", "--output", action="store_true", help="Generate CSV for final output")
  output_args.add_argument("--plot", action="store_true", help="Show plots at end of training. (Will act as if -o specified)")
  output_args.add_argument("--save-plots", action="store_true", help="Save generated plots. (Will act as if --plot and -o are specified)")

  args = parser.parse_args()

  runarg_defaults = [
    ("use_reduced_observation", False),
    ("multi_manoeuvre", False)
  ]

  # runs = ["2023-03-10T15-57-12", "2023-12-13T10-28-02"]
  # runs = ["2023-03-10T16-28-39", "2023-12-13T10-28-02"]
  runs = ["2023-03-10T15-57-12", "2023-12-08T15-06-19"]

  run_datas = []
  for run in runs:
    run_dir = f"{args.directory}/{run}"
    
    with open(f"{run_dir}/metadata.json") as f:
      metadata = json.load(f)
      print(metadata)
      f.seek(0)
      run_args = json.load(
        f,
        object_hook=namespace_object_hook
      )
    for (arg,default) in runarg_defaults:
      if not hasattr(run_args, arg):
        setattr(run_args, arg, default) 
    
    # Attempt to checkout correct version of reward function
    try:
      commit = run_args.commit
      if "dirty" in commit:
        commit = commit[0:-6] # Remove -dirty
      subprocess.check_call(f"git show {commit}:testgym.py > testgym_previous.py", shell=True)
      import testgym_previous
      importlib.reload(testgym_previous)
      from testgym_previous import create_reward_func
      os.remove("testgym_previous.py")
    except Exception as e:
      print(f"Error importing previous version of reward function: {e}")
      from testgym import create_reward_func

    reward_func = create_reward_func(run_args)

    model = MlAlg.load(f"{run_dir}/model.zip")

    run_datas.append((run_args, model, reward_func))

  from testgym import LongitudinalStateWrapper, MultiManoeuvreWrapper, StartStateWrapper

  env = gym.make('gym_mxs/MXS-v0', reward_func=reward_func, timestep_limit=10_000)
  env = LongitudinalStateWrapper(env)

  if run_datas[0][0].manoeuvre in ['hoversus', 'hoverexit']:
    angle_by_2 = (np.pi / 2) / 2
    env = StartStateWrapper(
      env,
      start_velocity=[0,0,0],
      start_attitude=[0,np.sin(angle_by_2),0,np.cos(angle_by_2)]
    )


  simtime = 0
  output_path = f"{run_dir}/output.csv"
  obs = env.reset()
  origin = copy.copy(obs)
  done = False

  with open(output_path, "w") if output_path else nullcontext() as outfile:
    if outfile:
      outfile.write("time,x,y,z,u,v,w,qx,qy,qz,qw,p,q,r,alpha,airspeed,elevator,throttle,reward,run_idx\n")

    for (run_idx, (run_args, model, reward_func)) in enumerate(run_datas):
      print(f"Processing: {run_args.run_name}")
      def model_transformer(obs):
        obs[0] = obs[0] - origin[0]
        obs[1] = obs[1] - origin[1]
        return obs
      
      def reward_transformer(obs):
        # Reward sees full state so need to modify [2] for z
        obs[0] = obs[0] - origin[0]
        obs[2] = obs[2] - origin[1]
        return obs

      # Create transfomed reward func and assign to env
      def transformed_reward_func(observation, state):
        transformed_obs = reward_transformer(observation)
        result = reward_func(transformed_obs, state)
        return result

      env.unwrapped.reward_state = None
      env.unwrapped.reward_func = transformed_reward_func

      segment_start = simtime
      while not done:
        action, _state = model.predict(model_transformer(obs), deterministic=True)
        obs, reward, done, info = env.step(action)
        if outfile:
          outfile.write(f"{simtime},{env.render('ansi')[1:-1]},{reward},{run_idx}\n")
        simtime += env.dT
        
        if simtime - segment_start > 3:
          done = True
      
      done = False
      # Update origin to point at current observation
      origin = copy.copy(obs)
      print(f"End of stage state: {obs}")

  if args.plot or args.save_plots:
    plot_command = [
      "python",
      f"{os.path.dirname(os.path.realpath(__file__))}/plotting/unified_plot.py",
      "-d", args.directory,
      run
    ]
    if args.save_plots:
      plot_command.append("--save")

    subprocess.call(plot_command)
