import jax
import jax.numpy as jnp
import chex
from flax import struct
from functools import partial
from typing import Any, Dict, Tuple, Union, Optional, Callable
from gymnax.environments import spaces


from nasimjax.envs.environment_base import Environment, EnvState, EnvParams
from nasimjax.envs.common import NASimJaxEnvState, NASimJaxEnvParams
from nasimjax.envs.generated_environment import Level, AUX_INFO_FLAGS
from nasimjax.envs.host_vector_batched import (
    get_host_vector_flat,
    HostVectorBatched,
)
from nasimjax.envs.utils import ActionType


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class AugmentedObservationsWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Define the sizes of the different observation components
        self.aux_info_size = len(ActionType) + AUX_INFO_FLAGS
        # The host_data_size is the size of ONE block of host information
        self.host_data_size = (
            self._env.observation_space(params=None).shape[0] - self.aux_info_size
        )

    def observation_space(self, params: NASimJaxEnvParams = None):
        """The new observation space is [raw_data, aggregated_data, aux_info]."""
        # The shape is two blocks of host data plus one block of auxiliary info
        obs_shape = ((self.host_data_size * 2) + self.aux_info_size,)
        return spaces.Box(
            low=self._env._obs_low,
            high=self._env._obs_high,
            shape=obs_shape,
            dtype=jnp.uint8,
        )

    # The aggregation logic remains the same as it's correct.
    def _aggregate_access(self, last: jnp.ndarray, current: jnp.ndarray) -> jnp.ndarray:
        last_idx = jnp.argmax(last, axis=1)
        curr_idx = jnp.argmax(current, axis=1)
        agg_idx = jnp.maximum(last_idx, curr_idx)
        return jnp.eye(last.shape[1], dtype=jnp.uint8)[agg_idx]

    def _aggregate_observations(
        self, current_obs_data: jnp.ndarray, last_aggregated_data: jnp.ndarray
    ) -> jnp.ndarray:
        # We can still use the base environment's reconstructor here because we are
        # feeding it simple [host_data] blocks, not our complex augmented observation.
        current_hosts = self._env.reconstruct_hosts_from_observation(current_obs_data)
        last_hosts = self._env.reconstruct_hosts_from_observation(last_aggregated_data)

        aggregated_hosts = HostVectorBatched(
            subnet_address=jnp.logical_or(
                current_hosts.subnet_address, last_hosts.subnet_address
            ),
            reachable=jnp.logical_or(current_hosts.reachable, last_hosts.reachable),
            discovered=jnp.logical_or(current_hosts.discovered, last_hosts.discovered),
            sensitive=jnp.logical_or(current_hosts.sensitive, last_hosts.sensitive),
            access_level=self._aggregate_access(
                current_hosts.access_level, last_hosts.access_level
            ),
            os=jnp.logical_or(current_hosts.os, last_hosts.os),
            services=jnp.logical_or(current_hosts.services, last_hosts.services),
            processes=jnp.logical_or(current_hosts.processes, last_hosts.processes),
        )
        return get_host_vector_flat(aggregated_hosts)

    @partial(jax.jit, static_argnums=(0))
    def step(
        self,
        key: jax.Array,
        state: NASimJaxEnvState,
        action: int,
        params: NASimJaxEnvParams,
    ) -> Tuple[jnp.ndarray, NASimJaxEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        # 1. Get the simple, raw observation from the base environment
        obs, next_state, reward, done, info = self._env.step(key, state, action, params)

        # 2. Extract the necessary components for aggregation
        # The new raw information from this step
        current_raw_data = obs[: self.host_data_size]
        obs_w_aux = obs[: self.host_data_size + self.aux_info_size]
        # The aggregated information from the *previous* step
        previous_aggregated_data = state.last_obs[self.host_data_size :]

        # 3. Perform the aggregation
        new_aggregated_data = self._aggregate_observations(
            obs_w_aux, previous_aggregated_data
        )

        # 4. Construct the new augmented observation: [raw | aux | aggregated]
        current_aux_info = obs[-self.aux_info_size :]
        augmented_obs = jnp.concatenate(
            [current_raw_data, current_aux_info, new_aggregated_data]
        )

        # 5. Store the full augmented observation in the state for the next step
        next_state = next_state.replace(last_obs=augmented_obs)

        return augmented_obs, next_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: jax.Array, params: NASimJaxEnvParams = None
    ) -> Tuple[jax.Array, NASimJaxEnvState]:
        obs, state = self._env.reset(key, params)

        # Deconstruct the initial observation
        initial_raw_data = obs[: self.host_data_size]
        initial_aux_info = obs[-self.aux_info_size :]

        # At reset, the "aggregated" data is identical to the raw data
        initial_aggregated_data = initial_raw_data

        # Construct the initial augmented observation: [raw | aux | aggregated]
        augmented_obs = jnp.concatenate(
            [initial_raw_data, initial_aux_info, initial_aggregated_data]
        )

        # IMPORTANT: Store this new augmented observation in the state
        state = state.replace(last_obs=augmented_obs)

        return augmented_obs, state

    def reset_to_level(
        self, rng: chex.PRNGKey, level: Level, params: NASimJaxEnvParams
    ) -> Tuple[jnp.array, NASimJaxEnvState]:
        obs, state = self._env.reset_to_level(rng, level, params)

        # Deconstruct the initial observation
        initial_raw_data = obs[: self.host_data_size]
        initial_aux_info = obs[-self.aux_info_size :]

        # At reset, the "aggregated" data is identical to the raw data
        initial_aggregated_data = initial_raw_data

        # Construct the initial augmented observation: [raw | aux | aggregated]
        augmented_obs = jnp.concatenate(
            [initial_raw_data, initial_aux_info, initial_aggregated_data]
        )

        # IMPORTANT: Store this new augmented observation in the state
        state = state.replace(last_obs=augmented_obs)

        return augmented_obs, state

    def action_space(self, params: NASimJaxEnvParams) -> Any:
        return self._env.action_space(params)


class NormalizeRewardWrapper(GymnaxWrapper):
    """
    Symlog (symmetric log) transformation for UED environments.
    It preserves sign and relative magnitudes while compressing large reward values.
    """

    def __init__(self, env):
        self._env = env

    @property
    def default_params(self) -> NASimJaxEnvParams:
        return self._env.default_params

    def step(
        self,
        rng: chex.PRNGKey,
        state: NASimJaxEnvState,
        action: Union[int, float],
        params: NASimJaxEnvParams,
    ) -> Tuple[chex.ArrayTree, NASimJaxEnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(rng, state, action, params)
        # Normalize by max possible reward
        norm_reward = reward / state.max_possible_reward
        return obs, state, norm_reward, done, info

    def reset_to_level(
        self, rng: chex.PRNGKey, level: Level, params: NASimJaxEnvParams
    ) -> Tuple[jnp.array, NASimJaxEnvState]:
        return self._env.reset_to_level(rng, level, params)

    def action_space(self, params: NASimJaxEnvParams) -> Any:
        return self._env.action_space(params)

    def observation_space(self, params: NASimJaxEnvParams) -> Any:
        return self._env.observation_space(params)


@struct.dataclass
class LogEnvState:
    env_state: EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns, lengths and achievements."""

    def __init__(self, env):
        self._env = env

    @property
    def default_params(self) -> EnvParams:
        return self._env.default_params

    @property
    def unwrapped(self) -> Environment:
        return self._env.unwrapped

    def reset_to_level(
        self, rng: chex.PRNGKey, level: Level, params: EnvParams
    ) -> Tuple[jnp.ndarray, EnvState]:
        obs, env_state = self._env.reset_to_level(rng, level, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done

        return obs, state, reward, done, info

    def get_obs(self, state: LogEnvState) -> chex.Array:
        return self._env.get_obs(state.env_state)

    def action_space(self, params: EnvParams) -> Any:
        return self._env.action_space(params)

    def observation_space(self, params: EnvParams) -> Any:
        return self._env.observation_space(params)


@struct.dataclass
class AutoReplayState:
    env_state: EnvState
    level: Level


class AutoReplayWrapper(GymnaxWrapper):
    """
    This wrapper replay the **same** level over and over again by resetting
    to the same level after each episode.
    This is useful for training/rolling out multiple times on the same level.
    """

    def __init__(self, env: Environment):
        self._env = env

    @property
    def default_params(self) -> EnvParams:
        return self._env.default_params

    @property
    def unwrapped(self) -> Environment:
        return self._env.unwrapped

    def step(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        rng_reset, rng_step = jax.random.split(rng)
        obs_re, env_state_re = self._env.reset_to_level(rng_reset, state.level, params)
        obs_st, env_state_st, reward, done, info = self._env.step(
            rng_step, state.env_state, action, params
        )
        env_state = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(done, x, y), env_state_re, env_state_st
        )
        obs = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st
        )

        # The state object for this wrapper also needs to be updated on reset
        new_state_re = AutoReplayState(env_state=env_state_re, level=state.level)
        new_state_st = state.replace(env_state=env_state_st)
        next_state = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(done, x, y), new_state_re, new_state_st
        )
        return obs, next_state, reward, done, info

    def reset_to_level(
        self, rng: chex.PRNGKey, level: Level, params: EnvParams
    ) -> Tuple[jnp.ndarray, EnvState]:
        obs, env_state = self._env.reset_to_level(rng, level, params)
        return obs, AutoReplayState(env_state=env_state, level=level)

    def action_space(self, params: EnvParams) -> Any:
        return self._env.action_space(params)

    def observation_space(self, params: EnvParams) -> Any:
        return self._env.observation_space(params)


@struct.dataclass
class AutoResetState:
    env_state: EnvState
    rng: chex.PRNGKey


class AutoResetWrapper(GymnaxWrapper):
    """
    This is a wrapper around an `UnderspecifiedEnv`, allowing for the environment
    to be automatically reset upon completion of an episode. This behaviour is
    similar to the default Gymnax interface. The user can specify a callable
    `sample_level` that takes in a PRNGKey and returns a level.

    Warning:
        To maintain compliance with UnderspecifiedEnv interface, user can reset to an
        arbitrary level. This includes levels outside the support of sample_level().
        Consequently, the tagged rng is defaulted to jax.random.PRNGKey(0).
        If your code relies on this, careful attention may be required.
    """

    def __init__(self, env: Environment, sample_level: Callable[[chex.PRNGKey], Level]):
        self._env = env
        self.sample_level = sample_level

    @property
    def default_params(self) -> EnvParams:
        return self._env.default_params

    @property
    def unwrapped(self) -> Environment:
        return self._env.unwrapped

    def step(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        rng_sample, rng_reset, rng_step = jax.random.split(rng, 3)

        new_level = self.sample_level(rng_sample)

        obs_re, env_state_re = self._env.reset_to_level(rng_reset, new_level, params)
        obs_st, env_state_st, reward, done, info = self._env.step(
            rng_step, state.env_state, action, params
        )

        env_state = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(done, x, y), env_state_re, env_state_st
        )
        obs = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st
        )
        level_rng = jax.lax.select(done, rng_sample, state.rng)

        info["rng"] = level_rng

        return (
            obs,
            AutoResetState(env_state=env_state, rng=level_rng),
            reward,
            done,
            info,
        )

    def reset_to_level(
        self, rng: chex.PRNGKey, level: Level, params: EnvParams
    ) -> Tuple[jnp.ndarray, EnvState]:
        obs, env_state = self._env.reset_to_level(rng, level, params)
        return obs, AutoResetState(env_state=env_state, rng=jax.random.PRNGKey(0))

    def action_space(self, params: EnvParams) -> Any:
        return self._env.action_space(params)

    def observation_space(self, params: EnvParams) -> Any:
        return self._env.observation_space(params)
