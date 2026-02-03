import numpy as np
import gymnasium as gym
from gymnasium import spaces

# exact same as RepeatFirstEnv, but with penalize=True
class RepeatFirstPenEnv(gym.Env):
    def __init__(self, seq_len=10, num_suits=3, penalize=True, rng_obs=True):
        super().__init__()
        assert seq_len >= 1
        assert num_suits >= 1

        self.seq_len = seq_len
        self.num_suits = num_suits
        self.penalty = -1.0 / seq_len
        self.penalize = bool(penalize)
        self.rng_obs = bool(rng_obs)

        # 0 = blank token / NOOP action
        # 1..num_suits = suit symbols
        self.action_space = spaces.Discrete(num_suits + 1)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(num_suits + 1),  # token
            spaces.Discrete(2),              # is_first
            spaces.Discrete(2),              # is_query
        ))

        self.rng = None
        self.t = 0
        self.target = 0
        self.state = None

    def get_episode_length(self):
        return self.seq_len

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.target = int(self.rng.integers(1, self.num_suits + 1))
        self.t = 0
        self.state = (self.target, 1, 0 if self.seq_len > 1 else 1)
        return self.state, {}

    def step(self, action):
        is_query = (self.t == self.seq_len - 1)

        if not is_query:
            if self.penalize:
                reward = 0.0 if action == 0 else self.penalty
            else:
                reward = 0.0
            self.t += 1
            # after t=0, everything is blank; last step is blank+query
            if self.t == self.seq_len - 1:
                self.state = (0, 0, 1)   # blank token, not first, query signal
            else:
                if self.rng_obs:
                    token = int(self.rng.integers(1, self.num_suits + 1))
                else:
                    token = 0
                self.state = (token, 0, 0)   # token, not first, not query
            return self.state, reward, False, False, {}

        if self.penalize:
            reward = 1.0 if action == self.target else self.penalty
        else:
            reward = 1.0 if action == self.target else -1.0
        self.t += 1
        self.state = (0, 0, 1)
        return self.state, reward, True, False, {}
