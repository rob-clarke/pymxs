import gym
from gym import spaces

import numpy as np

from pyaerso import AffectedBody, AeroBody, Body
from gym_mxs.model import Combined, calc_state, inertia, trim_points

class MxsEnv(gym.Env):
    metadata = {
        "render_modes": ["ansi"],
        "render_fps": 4
    }

    def __init__(self, render_mode=None, reward_func=lambda obs: 0.5, timestep_limit=100):
        self.observation_space = spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(13,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1,np.radians(-30),0,-1], dtype=np.float32),
            high=np.array([1,np.radians(30),1,1], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )

        selected_trim_point = 15

        mass = 1.221
        position,velocity,attitude,rates = calc_state(
            np.radians(trim_points[selected_trim_point][0]),
            selected_trim_point,
            False
        )

        self.dT = 0.01
        self.initial_state = [position,velocity,attitude,rates]

        body = Body(mass,inertia,*self.initial_state)
        aerobody = AeroBody(body)
        self.vehicle = AffectedBody(aerobody,[Combined()])

        self.steps = 0

        self.reward_func = reward_func
        self.timestep_limit = timestep_limit

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return self.vehicle.statevector

    def reset(self, seed=None, return_info=False, options=None):
        self.vehicle.statevector = [
            *self.initial_state[0],
            *self.initial_state[1],
            *self.initial_state[2],
            *self.initial_state[3]
        ]

        self.steps = 0

        observation = self._get_obs()
        info = {}
        return (observation,info) if return_info else observation

    def step(self, action):
        self.vehicle.step(self.dT,action)
        observation = self._get_obs()

        reward, ep_done = self.reward_func(observation)
        self.steps += 1
        done = ep_done or self.steps >= self.timestep_limit
        return observation, reward, done, {}

    def render(self, mode):
        if mode == "ansi":
            elements = ", ".join([f"{v:.{4}f}" for v in self._get_obs()])
            return f"[{elements},{self.vehicle.airstate[0]},{self.vehicle.airstate[2]}]"

    def close(self):
        pass
