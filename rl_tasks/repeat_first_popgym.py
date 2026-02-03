import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RepeatFirstPopgymEnv(gym.Env):
    """
    Convention consistent with the user's original code:

    - seq_len = number of step() calls in an episode (including the terminal step)
    - reset() returns the first suit observation (the one to remember)
    - each step():
        reward = +1/seq_len if action == first_suit else -1/seq_len
        observation = random suit in [0, num_suits)
        terminated=True on the seq_len-th call to step()
    """

    def __init__(self, seq_len=10, num_suits=4):
        super().__init__()
        assert seq_len >= 1
        assert num_suits >= 1

        self.seq_len = int(seq_len)
        self.num_suits = int(num_suits)

        # Observation/action are just suits: [0, num_suits)
        self.action_space = spaces.Discrete(self.num_suits)
        self.observation_space = spaces.Discrete(self.num_suits)

        self.rng = None
        self.t = 0              # counts how many step() calls have happened so far
        self.target = 0         # first suit to repeat
        self.state = None       # keep state for external access

    def get_episode_length(self):
        # In this convention, episode length means number of step() calls.
        return self.seq_len

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.t = 0
        self.target = int(self.rng.integers(0, self.num_suits))

        # Observation at reset is the first suit (must be remembered)
        self.state = self.target
        return self.state, {}

    def step(self, action):
        action = int(action)
        reward_scale = 1.0 / self.seq_len

        reward = reward_scale if action == self.target else -reward_scale

        # Consistent with your original code:
        # terminate on the step where t == seq_len - 1 (i.e., the seq_len-th step() call)
        terminated = (self.t == self.seq_len - 1)

        self.t += 1

        # Next observation is just a random suit (no blank/query/first flags)
        self.state = int(self.rng.integers(0, self.num_suits))

        return self.state, reward, terminated, False, {}
