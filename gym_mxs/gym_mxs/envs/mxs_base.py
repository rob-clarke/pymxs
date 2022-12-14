import gym
from gym import spaces

from gym_mxs.model import Combined

class MxsEnv(gym.Env):
    metadata = {
        "render_modes": [],
        "render_fps": 4
    }
    
    def __init__(self, render_mode=None):
        self.observation_space = spaces.Continuous(13)
        self.action_space = spaces.Continuous(4)
        
        self.dT = 0.01
        self.initial_state = [position,velocity,attitude,rates]
        
        body = Body(mass,inertia,*self.initial_state)
        aerobody = AeroBody(body)
        self.vehicle = AffectedBody(aerobody,[Combined()])
    
    
    def _get_obs(self):
        self.vehicle.statevector
    
    def reset(self, seed=None, return_info=False, options=None):
        self.vehicle.statevector = self.initial_state
        
        observation = self._get_obs()
        info = self._get_obs()
        return (observation,info) if return_info else observation

    def step(self, action):
        self.vehicle.step(self.dT,action)
        observation = self._get_obs()
        return observation
    
    def render(self):
        pass
    
    def close(self):
        pass
