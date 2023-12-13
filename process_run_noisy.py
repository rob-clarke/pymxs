import sys
sys.path.insert(1, '/home/rc13011/projects/mxs/pymxs/models')

import argparse
import json
import os
import numpy as np
from collections import namedtuple

from stable_baselines3 import PPO as MlAlg

import gym
import gym_mxs

import subprocess

def namespace_object_hook(dict):
  container = argparse.Namespace()
  for k, v in dict.items():
    setattr(container, k, v)
  return container

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("run_name")
  parser.add_argument("-s", "--samples", help="Samples for noise", type=int, required=False)
  parser.add_argument("-n", "--noise", help="Noise levels", required=False)

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
  from testgym import evaluate_model, LongitudinalStateWrapper, MultiManoeuvreWrapper, StartStateWrapper, StartNoiseWrapper

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

  if args.samples:
    if args.noise is not None:
      noise_amplitutdes = list(map(float, args.noise.split(",")))
    else:
      noise_amplitutdes = list(map(float, run_args.noise.split(",")))
    env = StartNoiseWrapper(env, noise_amplitutdes)

  if run_args.multi_manoeuvre:
    manoeuvre_names = ["hover", "descent"]
    env = MultiManoeuvreWrapper(
      env,
      manoeuvre_names,
      create_reward_func,
      run_args
    )


  sample_ids = [ "" ]
  if args.samples:
    sample_ids = [ f".{sample_id}" for sample_id in range(args.samples) ]

  if args.output or args.plot or args.save_plots:

    for sample_id in sample_ids:
      if run_args.multi_manoeuvre:
        for (idx,manoeuvre_name) in enumerate(manoeuvre_names):
          # Set to 1 less as reset will increment
          env.manoeuvre_index = idx - 1
          output_file = f"{run_dir}/output.{manoeuvre_name}{sample_id}.csv"
          obs, reward, done, info, simtime = evaluate_model(model, env, output_file)

          print(f"Manoeuvre: {manoeuvre_name} ({idx})")
          print(f"{obs=}")
          print(f"{reward=}")

      else:
        output_file = f"{run_dir}/output{sample_id}.csv"
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
      plot_command.extend(["--save", "--no-show"])
    if args.samples:
      if run_args.multi_manoeuvre:
        output_file_ids = []
        for sample_id in sample_ids:
          output_file_ids.extend([ f"{manoeuvre_name}{sample_id}" for manoeuvre_name in manoeuvre_names ])
        plot_command.extend(["--multi-manoeuvre", ','.join(output_file_ids)])
      else:
        plot_command.extend(["--multi-manoeuvre", ','.join([ sid[1:] for sid in sample_ids ])])

    subprocess.call(plot_command)
