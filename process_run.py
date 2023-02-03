import sys
sys.path.insert(1, '/home/rc13011/projects/mxs/pymxs/models')

import json
import os
from collections import namedtuple

from stable_baselines3 import PPO as MlAlg

import gym
import gym_mxs

import subprocess

from testgym import create_reward_func, evaluate_model

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("run_name")

  output_args = parser.add_argument_group("Output options")
  output_args.add_argument("--no-save", dest="save", action="store_false")
  output_args.add_argument("-d", "--directory", default="./runs", help="Destination for saving runs")
  output_args.add_argument("-o", "--output", action="store_true", help="Generate CSV for final output")
  output_args.add_argument("--plot", action="store_true", help="Show plots at end of training. (Will act as if -o specified)")

  args = parser.parse_args()

  run_dir = f"{args.directory}/{args.run_name}"

  with open(f"{run_dir}/metadata.json") as f:
    metadata = json.load(f)
    print(metadata)
    f.seek(0)
    run_args = json.load(
      f,
      object_hook=lambda d: namedtuple('X', d.keys())(*d.values())
    )

  reward_func = create_reward_func(run_args)

  model = MlAlg.load(f"{run_dir}/model.zip")
  env = gym.make('gym_mxs/MXS-v0', reward_func=reward_func, timestep_limit=1000)

  if args.output or args.plot:
    output_file = f"{run_dir}/output.csv"
    obs, reward, done, info, simtime = evaluate_model(model, env, output_file)

    print(f"{obs=}")
    print(f"{reward=}")
    
  if args.plot:
    subprocess.call(["python", f"{os.path.dirname(os.path.realpath(__file__))}/plotting/unified_plot.py", "-d", args.directory, args.run_name])
