import numpy as np
import gymnasium as gym

class SampleMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.p = np.ones(shape, "float64")
        self.count = 0

    def update(self, x):
        if self.count == 0:
            self.mean = x
            self.p = np.zeros_like(x)
        self.mean, self.var, self.p, self.count = self.update_mean_var_count_from_moments(self.mean, self.p, self.count, x*1.0)

    def update_mean_var_count_from_moments(self, mean, p, count, sample):
        new_count = count + 1
        new_mean = mean + (sample - mean) / new_count
        p = p + (sample - mean) * (sample - new_mean)
        new_var = 1 if new_count < 2 else p / (new_count - 1)
        return new_mean, new_var, p, new_count
    
    def get_stats(self):
        return {'mean': self.mean, 'var': self.var, 'p': self.p, 'count': self.count}

    def set_stats(self, stats):
        self.mean = stats.get('mean', self.mean)
        self.var = stats.get('var', self.var)
        self.p = stats.get('p', self.p)
        self.count = stats.get('count', self.count)

class NormalizeObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        if self.is_vector_env:
            self.obs_stats = SampleMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_stats = SampleMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon
        self._ensure_float_observation_space()

    def _ensure_float_observation_space(self):
        if isinstance(self.observation_space, gym.spaces.Box):
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.observation_space.shape,
                dtype=np.float32,
            )
        if hasattr(self, "single_observation_space") and isinstance(self.single_observation_space, gym.spaces.Box):
            self.single_observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.single_observation_space.shape,
                dtype=np.float32,
            )

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        self.obs_stats.update(obs)
        normalized = (obs - self.obs_stats.mean) / np.sqrt(self.obs_stats.var + self.epsilon)
        return normalized.astype(np.float32, copy=False)

class ScaleReward(gym.core.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False
        self.reward_stats = SampleMeanStd(shape=())
        self.reward_trace = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self._ensure_float_observation_space()

    def _ensure_float_observation_space(self):
        if isinstance(self.observation_space, gym.spaces.Box):
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.observation_space.shape,
                dtype=np.float32,
            )
        if hasattr(self, "single_observation_space") and isinstance(self.single_observation_space, gym.spaces.Box):
            self.single_observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.single_observation_space.shape,
                dtype=np.float32,
            )

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        term = terminateds or truncateds
        self.reward_trace = self.reward_trace * self.gamma * (1 - term) + rews
        rews = self.normalize(rews)
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, terminateds, truncateds, infos

    def normalize(self, rews):
        self.reward_stats.update(self.reward_trace)
        return rews / np.sqrt(self.reward_stats.var + self.epsilon)
