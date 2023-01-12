import sys
sys.path.insert(1, '/home/rc13011/projects/mxs/pymxs/models')

from stable_baselines3 import DDPG

import gym
import gym_mxs

import numpy as np

def reward_func(obs):
  [x,y,z, u,v,w, qx,qy,qz,qw, p,q,r] = obs
  pitch = np.arcsin(2*(qx*qy+qz*qw))
  if x < 0 or u > 25 or u < 0 or pitch < np.radians(-80):
    return -1000, True
  if x > 20:
    total_ke_sqd = u**2+v**2+w**2
    ke_fraction = total_ke_sqd / (15**2)
    return z / ke_fraction, True
  return 0, False

env = gym.make('gym_mxs/MXS-v0', reward_func=reward_func, timestep_limit=1000)

model = DDPG("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)

obs = env.reset()
print(obs)
done = False
simtime = 0

with open(sys.argv[1], "w") as f:
  f.write("time,x,y,z,u,v,w,qx,qy,qz,qw,p,q,r,alpha,airspeed,elevator\n")
  while not done:
    action, _state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    f.write(f"{simtime},{env.render('ansi')[1:-1]},{action[1]}\n")
    simtime += env.dT

print(obs)
