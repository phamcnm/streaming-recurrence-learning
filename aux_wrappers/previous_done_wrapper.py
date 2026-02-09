from typing import Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType
from gymnasium.vector.utils import batch_space

OBS = "obs"
STATE = "state"
PREV_DONE = "prev_done"


class PreviousDone(gym.Wrapper):
    """Wrapper that adds the last done signal to the observation.

    Args:
        env: The environment.
        mode: "tuple" to append as tuple/dict entry, "concat" to append to Box.

    Returns:
        A gym environment.
    """

    def __init__(self, env: gym.Env, mode: str = "concat"):
        super().__init__(env)
        if mode not in {"tuple", "concat"}:
            raise ValueError("mode must be 'tuple' or 'concat'")
        self.mode = mode
        self._init_vector_env_attrs()
        self._init_spaces()

    @staticmethod
    def done_space() -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    @staticmethod
    def add_done_space_to_obs_space(
        observation_space: spaces.Space, mode: str = "tuple"
    ) -> spaces.Space:
        if mode == "concat":
            if not isinstance(observation_space, spaces.Box):
                raise NotImplementedError("concat mode requires Box observations")
            low = np.concatenate(
                [observation_space.low.flatten(), np.array([0.0], dtype=np.float32)]
            )
            high = np.concatenate(
                [observation_space.high.flatten(), np.array([1.0], dtype=np.float32)]
            )
            return spaces.Box(low=low, high=high, dtype=np.float32)

        if isinstance(
            observation_space,
            (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        ):
            observation_space = spaces.Tuple((observation_space, PreviousDone.done_space()))
        elif isinstance(observation_space, spaces.Tuple):
            observation_space = spaces.Tuple(
                tuple(observation_space.spaces) + (PreviousDone.done_space(),)
            )
        elif isinstance(observation_space, spaces.Dict):
            observation_space = observation_space.spaces.copy()
            if set(observation_space.keys()) == {OBS, STATE}:
                observation_space[OBS] = PreviousDone.add_done_space_to_obs_space(
                    observation_space[OBS], mode=mode
                )
            else:
                observation_space[PREV_DONE] = PreviousDone.done_space()
            observation_space = spaces.Dict(observation_space)
        else:
            raise NotImplementedError("Unknown observation space")
        return observation_space

    @staticmethod
    def add_done_to_obs(
        observation_space: spaces.Space, obs: ObsType, done: np.ndarray, mode: str = "tuple"
    ) -> ObsType:
        if mode == "concat":
            if not isinstance(observation_space, spaces.Box):
                raise NotImplementedError("concat mode requires Box observations")
            obs_vec = np.asarray(obs, dtype=np.float32)
            done_vec = np.asarray(done, dtype=np.float32)
            if obs_vec.ndim == 1:
                return np.concatenate([obs_vec.flatten(), done_vec.reshape(1)], axis=-1)
            done_vec = done_vec.reshape(obs_vec.shape[0], 1)
            return np.concatenate([obs_vec.reshape(obs_vec.shape[0], -1), done_vec], axis=-1)

        if isinstance(
            observation_space,
            (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        ):
            obs = (obs, done)
        elif isinstance(observation_space, spaces.Tuple):
            obs = (*obs, done)
        elif isinstance(observation_space, spaces.Dict):
            if set(observation_space.keys()) == {OBS, STATE}:
                obs[OBS] = PreviousDone.add_done_to_obs(
                    observation_space[OBS], obs[OBS], done, mode=mode
                )
            else:
                obs[PREV_DONE] = done
        else:
            raise NotImplementedError("Unknown observation space")
        return obs

    def step(self, action) -> Tuple[ObsType, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = np.asarray(terminated or truncated, dtype=np.float32)
        if self.is_vector_env:
            done = np.asarray(terminated, dtype=np.float32) | np.asarray(
                truncated, dtype=np.float32
            )
        obs = PreviousDone.add_done_to_obs(
            self.env.observation_space, obs, done, mode=self.mode
        )
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        done = 0.0
        if self.is_vector_env:
            done = np.zeros((self.num_envs,), dtype=np.float32)
        obs = PreviousDone.add_done_to_obs(
            self.env.observation_space, obs, done, mode=self.mode
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
            self.single_observation_space = PreviousDone.add_done_space_to_obs_space(
                self.env.single_observation_space, mode=self.mode
            )

        if self.is_vector_env and hasattr(self, "single_observation_space"):
            self.observation_space = batch_space(
                self.single_observation_space, self.num_envs
            )
        else:
            self.observation_space = PreviousDone.add_done_space_to_obs_space(
                self.env.observation_space, mode=self.mode
            )
