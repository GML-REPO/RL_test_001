import os, time, sys
if __name__ == '__main__': sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')

from typing import Tuple, Dict, Any
_ActionType: np.ndarray
_OperationType: np.ndarray

import numpy as np
import gym
from gym import spaces

class BaselineEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        metadata = {"render.modes": []}
        reward_range = (-float("inf"), float("inf"))
        spec = None

        self.action_space = None#spaces.Box(-1, 1, shape=ACTION_SHAPE)
        self.observation_space = None#spaces.Box(-np.inf, np.inf, shape=STATE_SHAPE)


    # override
    def reset(self) -> Any:
        return super().reset()
    def step(self, action: _ActionType) -> Tuple[_OperationType, float, bool, Dict[str, Any]]:
        return super().step(action)
    def observation(self):
        return self._observation()
    

    # local method
    def _reset(self):
        return self.observation()

    def _step(self, action):
        pass

    def _observation(self):
        pass
