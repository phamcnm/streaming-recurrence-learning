import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CopyTaskEnv(gym.Env):
    def __init__(self, seq_len=3, num_suits=2, S=3):
        super().__init__()
        assert seq_len >= 1
        assert num_suits >= 1
        assert S >= 1

        self.seq_len = seq_len
        self.num_suits = num_suits
        self.S = S
        self.blank = 0
        self.delim = 1
        self.min_suit = 2
        self.max_suit = 2 + num_suits
        self.penalty = -1.0 / (self.seq_len + 2 * self.S)
        self.correct_recall_reward = 1.0 / self.S

        self.action_space = spaces.Discrete(self.max_suit + 1)
        self.observation_space = spaces.Discrete(self.max_suit + 1)

        self.rng = None
        self.t = 0
        self.u = None
        self.sequence = None
        self.state = None

    def get_episode_length(self):
        return self.seq_len + self.S * 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.u = self.rng.integers(self.min_suit, self.max_suit + 1, size=self.S)
        L = self.seq_len + 2 * self.S
        self.sequence = np.full(L, self.blank, dtype=np.int64)
        self.sequence[:self.S] = self.u
        self.sequence[self.S + self.seq_len - 1] = self.delim

        self.t = 0
        self.state = self.sequence[: self.t + 1].copy()
        return int(self.sequence[self.t]), {}

    def step(self, action):
        L = self.seq_len + 2 * self.S
        before_recall = self.t < self.S + self.seq_len
        if before_recall:
            reward = 0.0 if action == 0 else self.penalty
        else:
            target = int(self.u[self.t - (self.S + self.seq_len)])
            reward = self.correct_recall_reward if action == target else self.penalty

        self.t += 1
        terminated = self.t >= L
        obs = int(self.sequence[-1]) if terminated else int(self.sequence[self.t])
        self.state = self.sequence[: self.t + 1].copy()
        return obs, reward, terminated, False, {}
