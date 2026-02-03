import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RepeatPreviousPopgymEnv(gym.Env):
    """
    Convention consistent with the user's original code:

    - seq_len = number of step() calls in an episode (including the terminal step)
    - reset() returns the first suit observation (to seed the history)
    - each step():
        reward = 0 for the first k steps
        reward = +1/seq_len if action == observation from k steps ago else -1/seq_len
        observation = random suit in [0, num_suits)
        terminated=True on the seq_len-th call to step()
    """

    def __init__(self, seq_len=10, num_suits=3, k=3):
        super().__init__()
        assert seq_len >= 1
        assert num_suits >= 1
        assert k >= 0
        assert k < seq_len

        self.seq_len = int(seq_len)
        self.num_suits = int(num_suits)
        self.k = int(k)

        # Observation/action are just suits: [0, num_suits)
        self.action_space = spaces.Discrete(self.num_suits)
        self.observation_space = spaces.Discrete(self.num_suits)

        self.rng = None
        self.t = 0              # counts how many step() calls have happened so far
        self.history = []       # list of past observations
        self.state = None       # keep state for external access

    def get_episode_length(self):
        # In this convention, episode length means number of step() calls.
        return self.seq_len

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.t = 0
        first_obs = int(self.rng.integers(0, self.num_suits))
        self.history = [first_obs]

        # Observation at reset is the first suit (seeds the history)
        self.state = first_obs
        return self.state, {}

    def step(self, action):
        action = int(action)
        recall_steps = self.seq_len - self.k
        reward_scale = 1.0 / recall_steps

        if self.t < self.k:
            reward = 0.0
        else:
            target = self.history[self.t - self.k]
            reward = reward_scale if action == target else -reward_scale

        # Consistent with your original code:
        # terminate on the step where t == seq_len - 1 (i.e., the seq_len-th step() call)
        terminated = (self.t == self.seq_len - 1)

        self.t += 1

        # Next observation is just a random suit (no blank/query/first flags)
        self.state = int(self.rng.integers(0, self.num_suits))
        self.history.append(self.state)

        return self.state, reward, terminated, False, {}
