import sys
sys.path.insert(1, '/home/rc13011/projects/mxs/pymxs/models')

import argparse
import copy
import json
import os
from collections import namedtuple
from contextlib import nullcontext

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
  parser.add_argument("run_name")

  output_args = parser.add_argument_group("Output options")
  output_args.add_argument("--no-save", dest="save", action="store_false")
  output_args.add_argument("-d", "--directory", default="./runs", help="Destination for saving runs")
  output_args.add_argument("-o", "--output", action="store_true", help="Generate CSV for final output")
  output_args.add_argument("--plot", action="store_true", help="Show plots at end of training. (Will act as if -o specified)")
  output_args.add_argument("--save-plots", action="store_true", help="Save generated plots. (Will act as if --plot and -o are specified)")

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
  from testgym import LongitudinalStateWrapper, MultiManoeuvreWrapper

  reward_func = create_reward_func(run_args)

  model = MlAlg.load(f"{run_dir}/model.zip")
  env = gym.make('gym_mxs/MXS-v0', reward_func=reward_func, timestep_limit=1000)
  if run_args.use_reduced_observation:
    env = LongitudinalStateWrapper(env)

  if run_args.multi_manoeuvre:
    manoeuvre_names = ["hover", "descent"]
    env = MultiManoeuvreWrapper(
      env,
      manoeuvre_names,
      create_reward_func,
      run_args
    )

  if args.output or args.plot or args.save_plots:

    if run_args.multi_manoeuvre:
      # Run the descent case first
      obs = env.reset()
      env.manoeuvre_index = 1
      output_file = f"{run_dir}/output.descent.seq.csv"
      obs, reward, done, info, simtime = evaluate_model(model, env, output_file, initial_obs=obs)

      print(f"Manoeuvre: descent")
      print(f"{obs=}")
      print(f"{reward=}")

      midpoint_obs = copy.copy(obs)

      def transformer(raw_obs):
        raw_obs[0] = raw_obs[0] - midpoint_obs[0]
        raw_obs[1] = raw_obs[1] - midpoint_obs[1]
        return raw_obs

      # Run the hover case
      env.manoeuvre_index = 0
      output_file = f"{run_dir}/output.hover.seq.csv"
      obs, reward, done, info, simtime = evaluate_model(model, env, output_file, transformer=transformer, initial_obs=obs, initial_sim=simtime)

      print(f"Manoeuvre: hover")
      print(f"{obs=}")
      print(f"tf_obs={transformer(obs)}")
      print(f"{reward=}")

    else:
      output_file = f"{run_dir}/output.csv"
      obs, reward, done, info, simtime = evaluate_model(model, env, output_file)

      print(f"{obs=}")
      print(f"{reward=}")


  if args.plot or args.save_plots:
    plot_command = [
      "python",
      f"{os.path.dirname(os.path.realpath(__file__))}/plotting/unified_plot.py",
      "-d", args.directory,
      args.run_name
    ]
    if args.save_plots:
      plot_command.append("--save")
    if run_args.multi_manoeuvre:
      plot_command.extend(["--multi-manoeuvre", ','.join(manoeuvre_names)])

    subprocess.call(plot_command)
