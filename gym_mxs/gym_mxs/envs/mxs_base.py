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
            low=np.array([np.radians(-60),0], dtype=np.float32),
            high=np.array([np.radians(60),1], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        
        self.selected_trim_point = 15

        mass = 1.221
        position,velocity,attitude,rates = calc_state(
            np.radians(trim_points[self.selected_trim_point][0]),
            self.selected_trim_point,
            False
        )

        self.dT = 0.01
        self.initial_state = [position,velocity,attitude,rates]

        body = Body(mass,inertia,*self.initial_state)
        aerobody = AeroBody(body)
        self.vehicle = AffectedBody(aerobody,[Combined(self.dT)])

        self.steps = 0

        self.reward_func = reward_func
        self.reward_state = None
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

        # self.vehicle.set_state(np.array([[
        #     *self.initial_state[0],
        #     *self.initial_state[1],
        #     *self.initial_state[2],
        #     *self.initial_state[3]
        # ]]).T)

        self.elevator = np.radians(trim_points[self.selected_trim_point][1])
        self.throttle = trim_points[self.selected_trim_point][2]

        self.steps = 0
        self.reward_state = None

        observation = self._get_obs()
        info = {}
        return (observation,info) if return_info else observation

    def step(self, action):
        self.elevator = np.clip(self.elevator + action[0] * self.dT, np.radians(-30), np.radians(30))
        self.throttle = action[1]
        self.vehicle.step(self.dT,[0,self.elevator,self.throttle,0])
        observation = self._get_obs()

        reward, ep_done, self.reward_state = self.reward_func(observation, self.reward_state)
        self.steps += 1
        done = ep_done or self.steps >= self.timestep_limit
        return observation, reward, done, {}

    def render(self, mode):
        if mode == "ansi":
            elements = ", ".join([f"{v:.{4}f}" for v in self._get_obs()])
            return f"[{elements},{self.vehicle.airstate[0]},{self.vehicle.airstate[2]},{self.elevator},{self.throttle}]"

    def close(self):
        pass
