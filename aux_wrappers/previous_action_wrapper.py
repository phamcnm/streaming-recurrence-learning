from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from gymnasium.vector.utils import batch_space

OBS = "obs"
STATE = "state"
PREV_ACTION = "prev_action"


class PreviousAction(gym.Wrapper):
    """Wrapper that adds the last action to the observation.

    Args:
        env: The environment
        null_action: Optional null action that is returned when resetting the
            environment. If not provided, the null action will be 0
            (int or vector) if it is in the action space, or the lowest action
            possible.

    Returns:
        A gym environment
    """

    def __init__(
        self,
        env: gym.Env,
        null_action: Optional[ActType] = None,
        mode: str = "tuple",
    ):
        super().__init__(env)
        if mode not in {"tuple", "concat"}:
            raise ValueError("mode must be 'tuple' or 'concat'")
        self.mode = mode
        self._init_vector_env_attrs()
        self._init_spaces()
        self.null_action = self._init_null_action(null_action)

    @staticmethod
    def add_act_space_to_obs_space(
        observation_space: spaces.Space, action_space: spaces.Space, mode: str = "tuple"
    ) -> spaces.Space:
        """
        Returns a modified observation space to account for the last action.
        Args:
            observation_space: Original observation space
            action_space: Action space
            mode: "tuple" to append as tuple/dict entry, "concat" to append to Box

        Returns:
            The new observation space
        """
        if mode == "concat":
            if not isinstance(observation_space, spaces.Box):
                raise NotImplementedError("concat mode requires Box observations")
            act_box = PreviousAction.action_space_to_box(action_space)
            low = np.concatenate(
                [observation_space.low.flatten(), act_box.low.flatten()]
            )
            high = np.concatenate(
                [observation_space.high.flatten(), act_box.high.flatten()]
            )
            return spaces.Box(low=low, high=high, dtype=np.float32)

        if isinstance(
            observation_space,
            (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        ):
            observation_space = spaces.Tuple((observation_space, action_space))
        elif isinstance(observation_space, spaces.Tuple):
            observation_space = spaces.Tuple(
                tuple(observation_space.spaces) + (action_space,)
            )
        elif isinstance(observation_space, spaces.Dict):
            observation_space = observation_space.spaces.copy()
            if set(observation_space.keys()) == {OBS, STATE}:
                # Observation comes from ObservabilityWrapper with
                # observability level FULL_AND_PARTIAL
                observation_space[OBS] = PreviousAction.add_act_space_to_obs_space(
                    observation_space[OBS], action_space, mode=mode
                )
            else:
                observation_space[PREV_ACTION] = action_space
            observation_space = spaces.Dict(observation_space)
        else:
            raise NotImplementedError("Unknown observation space")
        return observation_space

    @staticmethod
    def add_act_to_obs(
        observation_space: spaces.Space,
        obs: ObsType,
        action: ActType,
        action_space: Optional[spaces.Space] = None,
        mode: str = "tuple",
    ) -> ObsType:
        """
        Static method that adds the action to the observation.
        Args:
            observation_space: Original observation space of the environment.
            obs: The observation.
            action: The action.
            action_space: Action space (required for concat mode).
            mode: "tuple" to append as tuple/dict entry, "concat" to append to Box

        Returns:
            Modified observation.
        """
        if mode == "concat":
            if not isinstance(observation_space, spaces.Box):
                raise NotImplementedError("concat mode requires Box observations")
            if action_space is None:
                raise ValueError("action_space is required for concat mode")
            obs_vec = np.asarray(obs, dtype=np.float32)
            act_vec = PreviousAction.action_to_vector(action_space, action)
            if obs_vec.ndim == 1:
                return np.concatenate([obs_vec.flatten(), act_vec], axis=-1)
            return np.concatenate(
                [obs_vec.reshape(obs_vec.shape[0], -1), act_vec], axis=-1
            )

        if isinstance(
            observation_space,
            (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        ):
            obs = (obs, action)
        elif isinstance(observation_space, spaces.Tuple):
            obs = (*obs, action)
        elif isinstance(observation_space, spaces.Dict):
            if set(observation_space.keys()) == {OBS, STATE}:
                # Observation comes from ObservabilityWrapper with
                # observability level FULL_AND_PARTIAL
                obs[OBS] = PreviousAction.add_act_to_obs(
                    observation_space[OBS],
                    obs[OBS],
                    action,
                    action_space=action_space,
                    mode=mode,
                )
            else:
                obs[PREV_ACTION] = action
        else:
            raise NotImplementedError("Unknown observation space")
        return obs

    @staticmethod
    def get_null_action(action_space: spaces.Space) -> ActType:
        """
        Static method that generates a null action based on the action space.
        Args:
            action_space: The action space.

        Returns:
            The null action.
        """
        if isinstance(
            action_space,
            (spaces.Discrete, spaces.MultiBinary, spaces.MultiDiscrete, spaces.Box),
        ):
            action = np.zeros(action_space.shape, action_space.dtype)
            if not action_space.contains(action):
                action = action_space.low
        elif isinstance(action_space, spaces.Tuple):
            action = tuple(
                PreviousAction.get_null_action(action_space_)
                for action_space_ in action_space
            )
        elif isinstance(action_space, spaces.Dict):
            action = {
                key: PreviousAction.get_null_action(value)
                for key, value in action_space.items()
            }
        else:
            raise NotImplementedError
        return action

    @staticmethod
    def batch_action(
        action: ActType, num_envs: int, action_space: spaces.Space
    ) -> ActType:
        if isinstance(action_space, spaces.Discrete):
            return np.full((num_envs,), action, dtype=action_space.dtype)
        if isinstance(
            action_space, (spaces.Box, spaces.MultiBinary, spaces.MultiDiscrete)
        ):
            action = np.asarray(action, dtype=action_space.dtype)
            return np.repeat(action[None, ...], num_envs, axis=0)
        if isinstance(action_space, spaces.Tuple):
            return tuple(
                PreviousAction.batch_action(action_i, num_envs, space_i)
                for action_i, space_i in zip(action, action_space.spaces)
            )
        if isinstance(action_space, spaces.Dict):
            return {
                key: PreviousAction.batch_action(action[key], num_envs, space)
                for key, space in action_space.spaces.items()
            }
        raise NotImplementedError("Unknown action space")

    @staticmethod
    def action_space_to_box(action_space: spaces.Space) -> spaces.Box:
        if isinstance(action_space, spaces.Box):
            low = action_space.low.flatten()
            high = action_space.high.flatten()
        elif isinstance(action_space, spaces.Discrete):
            low = np.zeros((action_space.n,), dtype=np.float32)
            high = np.ones((action_space.n,), dtype=np.float32)
        elif isinstance(action_space, spaces.MultiDiscrete):
            low = np.zeros_like(action_space.nvec, dtype=np.float32)
            high = (action_space.nvec - 1).astype(np.float32)
        elif isinstance(action_space, spaces.MultiBinary):
            low = np.zeros((int(np.prod(action_space.shape)),), dtype=np.float32)
            high = np.ones((int(np.prod(action_space.shape)),), dtype=np.float32)
        else:
            raise NotImplementedError("Unsupported action space for concat")
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @staticmethod
    def action_to_vector(action_space: spaces.Space, action: ActType) -> np.ndarray:
        if isinstance(action_space, spaces.Box):
            act = np.asarray(action, dtype=np.float32)
            if act.ndim <= len(action_space.shape):
                return act.reshape(-1)
            return act.reshape(act.shape[0], -1)
        if isinstance(action_space, spaces.Discrete):
            if np.asarray(action).ndim == 0:
                vec = np.zeros((action_space.n,), dtype=np.float32)
                vec[int(action)] = 1.0
                return vec
            idx = np.asarray(action, dtype=np.int64)
            vec = np.zeros((idx.shape[0], action_space.n), dtype=np.float32)
            vec[np.arange(idx.shape[0]), idx] = 1.0
            return vec
        if isinstance(action_space, (spaces.MultiDiscrete, spaces.MultiBinary)):
            act = np.asarray(action, dtype=np.float32)
            if act.ndim == 1:
                return act.flatten()
            return act.reshape(act.shape[0], -1)
        raise NotImplementedError("Unsupported action space for concat")

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        base_action_space = self._base_action_space()
        obs = PreviousAction.add_act_to_obs(
            self.env.observation_space,
            obs,
            action,
            action_space=base_action_space,
            mode=self.mode,
        )
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base_action_space = self._base_action_space()
        obs = PreviousAction.add_act_to_obs(
            self.env.observation_space,
            obs,
            self.null_action,
            action_space=base_action_space,
            mode=self.mode,
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
            self.single_observation_space = PreviousAction.add_act_space_to_obs_space(
                self.env.single_observation_space,
                self.env.single_action_space,
                mode=self.mode,
            )
        if self.is_vector_env and hasattr(self.env, "single_action_space"):
            self.single_action_space = self.env.single_action_space

        if self.is_vector_env and hasattr(self, "single_observation_space"):
            self.observation_space = batch_space(
                self.single_observation_space, self.num_envs
            )
        else:
            self.observation_space = PreviousAction.add_act_space_to_obs_space(
                self.env.observation_space, self.env.action_space, mode=self.mode
            )

    def _init_null_action(self, null_action: Optional[ActType]) -> ActType:
        base_action_space = self._base_action_space()
        if null_action is None:
            null_action = PreviousAction.get_null_action(base_action_space)
            if self.is_vector_env:
                null_action = PreviousAction.batch_action(
                    null_action, self.num_envs, base_action_space
                )
        elif (
            self.is_vector_env
            and not self.action_space.contains(null_action)
            and self.env.single_action_space.contains(null_action)
        ):
            null_action = PreviousAction.batch_action(
                null_action, self.num_envs, self.env.single_action_space
            )
        assert self.action_space.contains(null_action)
        return null_action

    def _base_action_space(self) -> spaces.Space:
        if self.is_vector_env and hasattr(self.env, "single_action_space"):
            return self.env.single_action_space
        return self.env.action_space
