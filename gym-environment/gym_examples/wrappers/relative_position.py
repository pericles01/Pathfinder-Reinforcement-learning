import gym
from gym.spaces import Box
import numpy as np


class RelativePosition(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(shape=(2,), low=-np.inf, high=np.inf)

    def observation(self, obs):
        if isinstance(obs, tuple):
            return obs[0]["target"] - obs[0]["agent"]
        elif isinstance(obs, dict):
            return obs["target"] - obs["agent"]
