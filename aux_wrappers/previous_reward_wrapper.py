from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType
from gymnasium.vector.utils import batch_space

OBS = "obs"
STATE = "state"
PREV_REWARD = "prev_reward"


class PreviousReward(gym.Wrapper):
    """Wrapper that adds the last reward to the observation.

    Args:
        env: The environment.
        null_reward: Optional null reward used on reset. Defaults to 0.0.
        mode: "tuple" to append as tuple/dict entry, "concat" to append to Box.

    Returns:
        A gym environment.
    """

    def __init__(
        self,
        env: gym.Env,
        null_reward: Optional[float] = None,
        mode: str = "concat",
    ):
        super().__init__(env)
        if mode not in {"tuple", "concat"}:
            raise ValueError("mode must be 'tuple' or 'concat'")
        self.mode = mode
        self._init_vector_env_attrs()
        self._init_spaces()
        self.null_reward = 0.0 if null_reward is None else float(null_reward)

    @staticmethod
    def reward_space() -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    @staticmethod
    def add_reward_space_to_obs_space(
        observation_space: spaces.Space, mode: str = "tuple"
    ) -> spaces.Space:
        if mode == "concat":
            if not isinstance(observation_space, spaces.Box):
                raise NotImplementedError("concat mode requires Box observations")
            low = np.concatenate(
                [observation_space.low.flatten(), np.array([-np.inf], dtype=np.float32)]
            )
            high = np.concatenate(
                [observation_space.high.flatten(), np.array([np.inf], dtype=np.float32)]
            )
            return spaces.Box(low=low, high=high, dtype=np.float32)

        if isinstance(
            observation_space,
            (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        ):
            observation_space = spaces.Tuple((observation_space, PreviousReward.reward_space()))
        elif isinstance(observation_space, spaces.Tuple):
            observation_space = spaces.Tuple(
                tuple(observation_space.spaces) + (PreviousReward.reward_space(),)
            )
        elif isinstance(observation_space, spaces.Dict):
            observation_space = observation_space.spaces.copy()
            if set(observation_space.keys()) == {OBS, STATE}:
                observation_space[OBS] = PreviousReward.add_reward_space_to_obs_space(
                    observation_space[OBS], mode=mode
                )
            else:
                observation_space[PREV_REWARD] = PreviousReward.reward_space()
            observation_space = spaces.Dict(observation_space)
        else:
            raise NotImplementedError("Unknown observation space")
        return observation_space

    @staticmethod
    def add_reward_to_obs(
        observation_space: spaces.Space, obs: ObsType, reward: float, mode: str = "tuple"
    ) -> ObsType:
        if mode == "concat":
            if not isinstance(observation_space, spaces.Box):
                raise NotImplementedError("concat mode requires Box observations")
            obs_vec = np.asarray(obs, dtype=np.float32)
            reward_vec = np.asarray(reward, dtype=np.float32)
            if obs_vec.ndim == 1:
                return np.concatenate([obs_vec.flatten(), reward_vec.reshape(1)], axis=-1)
            reward_vec = reward_vec.reshape(obs_vec.shape[0], 1)
            return np.concatenate([obs_vec.reshape(obs_vec.shape[0], -1), reward_vec], axis=-1)

        if isinstance(
            observation_space,
            (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        ):
            obs = (obs, reward)
        elif isinstance(observation_space, spaces.Tuple):
            obs = (*obs, reward)
        elif isinstance(observation_space, spaces.Dict):
            if set(observation_space.keys()) == {OBS, STATE}:
                obs[OBS] = PreviousReward.add_reward_to_obs(
                    observation_space[OBS], obs[OBS], reward, mode=mode
                )
            else:
                obs[PREV_REWARD] = reward
        else:
            raise NotImplementedError("Unknown observation space")
        return obs

    def step(self, action) -> Tuple[ObsType, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = PreviousReward.add_reward_to_obs(
            self.env.observation_space, obs, reward, mode=self.mode
        )
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        null_reward = self.null_reward
        if self.is_vector_env:
            null_reward = np.full((self.num_envs,), null_reward, dtype=np.float32)
        obs = PreviousReward.add_reward_to_obs(
            self.env.observation_space, obs, null_reward, mode=self.mode
        )
        return obs, info

    def _init_vector_env_attrs(self) -> None:
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

    def _init_spaces(self) -> None:
        if self.is_vector_env and hasattr(self.env, "single_observation_space"):
            self.single_observation_space = PreviousReward.add_reward_space_to_obs_space(
                self.env.single_observation_space, mode=self.mode
            )

        if self.is_vector_env and hasattr(self, "single_observation_space"):
            self.observation_space = batch_space(
                self.single_observation_space, self.num_envs
            )
        else:
            self.observation_space = PreviousReward.add_reward_space_to_obs_space(
                self.env.observation_space, mode=self.mode
            )
