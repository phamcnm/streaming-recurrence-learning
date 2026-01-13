import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RepeatFirstBlankLastEnv(gym.Env):
    def __init__(self, seq_len=10, num_suits=3):
        super().__init__()
        assert seq_len >= 1
        assert num_suits >= 1

        self.seq_len = seq_len
        self.num_suits = num_suits
        self.penalty = -1.0 / seq_len

        # 0 = blank token / NOOP action
        # 1..num_suits = suit symbols
        self.action_space = spaces.Discrete(num_suits + 1)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(num_suits + 1),  # token
            spaces.Discrete(2),              # is_query
        ))

        self.rng = None
        self.t = 0
        self.target = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.target = int(self.rng.integers(1, self.num_suits + 1))
        self.t = 0
        return (self.target, 0 if self.seq_len > 1 else 1), {}

    def step(self, action):
        is_query = (self.t == self.seq_len - 1)

        if not is_query:
            reward = 0.0 if action == 0 else self.penalty
            self.t += 1
            # after t=0, everything is blank; last step is blank+query
            if self.t == self.seq_len - 1:
                obs = (0, 1)   # blank token, query signal
            else:
                obs = (0, 0)   # blank token, not query
            return obs, reward, False, False, {}

        reward = 1.0 if action == self.target else self.penalty
        self.t += 1
        return (0, 1), reward, True, False, {}
