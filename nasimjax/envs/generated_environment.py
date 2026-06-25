# Copyright (c) 2026 The NASimJax Authors.
# Licensed under the MIT License. See LICENSE in the project root.
"""JAX-based NASim Environment implementation.

This module provides a JAX-compatible version of the NASim environment
that follows functional programming paradigms and can be JIT compiled.

We provide both the interface to seamlessly integrate with Gymnax, as
well as algorithms devloped for the JaxUED library.
"""

from collections.abc import Callable
from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import chex
from flax import struct
from gymnax.environments import spaces

from nasimjax.envs.environment_base import Environment
from nasimjax.envs.common import NASimJaxEnvState, NASimJaxEnvParams
from nasimjax.envs.action import (
    GeneratedFlatActionSpaceJAX,
    get_action_from_arrays,
)
from nasimjax.envs.host_vector_batched import (
    HostVectorBatched,
    get_host_vector_flat,
    reconstruct_host_vector_from_flat,
)
from nasimjax.envs.observation_function import get_observation, get_initial_observation
from nasimjax.envs.transition_logic import perform_action_on_host
from nasimjax.envs.network_generator import make_level_generator
from nasimjax.envs.utils import ActionType

AUX_INFO_FLAGS = 4


@struct.dataclass
class Level:
    hosts: HostVectorBatched
    topology: jnp.ndarray
    num_active_hosts: int
    is_flat_topo: bool
    num_sensitive_hosts: int


class GeneratedNASimEnvJAX(Environment[NASimJaxEnvState, NASimJaxEnvParams]):
    """JAX-compatible NASim environment.

    This environment implements the Gymnax interface and follows functional
    programming paradigms for JAX compatibility.
    """

    def __init__(
        self,
        key: jax.random.key,
        params: NASimJaxEnvParams = None,
        default_level_sampler: Callable[[jax.Array], Level] = None,
    ):
        """Initialize the JAX environment.

        Parameters
        ----------
        key : jax.random.key
            Random key for environment generation
        params : NASimJaxEnvParams, optional
            Environment parameters
        default_level_sampler : Callable[[jax.Array], Level], optional
            Function to sample default levels (default: None, will use internal generator)
        """
        if params is None:
            params = NASimJaxEnvParams()

        params = params.replace(max_steps_in_episode=params.step_limit)

        self._default_level_sampler = default_level_sampler or make_level_generator(
            params
        )

        # Store environment configuration
        self.fully_obs = params.fully_obs
        self.num_hosts = params.num_hosts
        self.step_limit = params.step_limit
        self.service_scan_cost = params.service_scan_cost
        self.process_scan_cost = params.process_scan_cost
        self.os_scan_cost = params.os_scan_cost
        self.subnet_scan_cost = params.subnet_scan_cost
        self.exploit_cost = params.exploit_cost
        self.privesc_cost = params.privesc_cost
        self._default_params = params

        # Sample once to get an initial set of hosts to derive the observation dimensions from
        dummy_level = self._default_level_sampler(key)
        self._initial_hosts = dummy_level.hosts

        # Initialize generated action space for MAXIMUM hosts (fixed for JIT)
        self.action_space_jax = GeneratedFlatActionSpaceJAX(
            num_hosts=self.num_hosts,  # Fixed maximum for JIT compatibility
            num_services=params.num_services,
            num_processes=params.num_processes,
            num_os=params.num_os,
            exploit_success_prob=0.9,
            privesc_success_prob=0.9,
            service_scan_cost=self.service_scan_cost,
            process_scan_cost=self.process_scan_cost,
            os_scan_cost=self.os_scan_cost,
            subnet_scan_cost=self.subnet_scan_cost,
            exploit_cost=self.exploit_cost,
            privesc_cost=self.privesc_cost,
        )
        self.action_array = self.action_space_jax.action_arrays
        self.num_action_types = len(ActionType)

        dummy_obs = get_host_vector_flat(self._initial_hosts)
        self._obs_low = jnp.min(dummy_obs)
        self._obs_high = jnp.max(dummy_obs)
        self._obs_shape = (dummy_obs.shape[0] + len(ActionType) + 4,)
        self._flat_obs_size = self._obs_shape  # Equal since we only provide plat obs.

        # Set scenario name for compatibility
        self._scenario_name = f"Gen_{self.num_hosts}H_{params.num_services}S_{params.num_processes}P_{params.num_os}O"

    @property
    def default_params(self) -> NASimJaxEnvParams:
        """Default environment parameters."""
        return self._default_params

    @property
    def name(self) -> str:
        """Environment name."""
        return self._scenario_name

    @property
    def num_actions(self) -> int:
        """Number of actions in the environment."""
        return self.action_space_jax.n

    def action_space(self, params: NASimJaxEnvParams):
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: NASimJaxEnvParams):
        """Observation space of the environment."""
        obs_shape = self._obs_shape

        return spaces.Box(
            low=self._obs_low, high=self._obs_high, shape=obs_shape, dtype=jnp.uint8
        )

    def state_space(self, params: NASimJaxEnvParams):
        """State space of the environment."""
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=self._obs_shape,  # Use same shape as observation for simplicity
            dtype=jnp.uint8,
        )

    @partial(jax.jit, static_argnames=("self",))
    def reset_env(
        self, key: chex.PRNGKey, params: NASimJaxEnvParams
    ) -> Tuple[jnp.ndarray, NASimJaxEnvState]:
        """Reset the environment to initial state with variable host count."""
        level_key, reset_key = jax.random.split(key)
        level = self._default_level_sampler(level_key)
        return self.reset_env_to_level(reset_key, level, params)

    def reset_env_to_level(
        self, key: chex.PRNGKey, level: Level, params: NASimJaxEnvParams
    ) -> Tuple[jnp.array, NASimJaxEnvState]:
        hosts_init = level.hosts
        traffic_rules = level.topology

        # Generate initial observation
        def flatten_state():
            return get_host_vector_flat(hosts_init)

        def flatten_obs():
            obs_hosts = get_initial_observation(hosts_init)
            return get_host_vector_flat(obs_hosts)

        initial_obs = jax.lax.cond(params.fully_obs, flatten_state, flatten_obs)
        # Used for giving information on the last executed action type,
        # and the last action result
        additional_info = jnp.zeros(len(ActionType) + AUX_INFO_FLAGS, dtype=jnp.uint8)

        obs = jnp.append(initial_obs, additional_info)

        max_possible_reward = self._compute_max_possible_reward(hosts_init, params)

        # Create initial environment state
        env_state = NASimJaxEnvState(
            time=0,
            hosts=hosts_init,
            last_obs=obs,
            steps=0,
            done=False,
            traffic_rules=traffic_rules,
            max_possible_reward=max_possible_reward,
        )

        return obs, env_state

    @partial(jax.jit, static_argnames=("self",))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: NASimJaxEnvState,
        action: int,
        params: NASimJaxEnvParams,
    ) -> Tuple[jnp.ndarray, NASimJaxEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Perform one environment step with bounds checking."""
        # Convert action index to action
        action_obj = get_action_from_arrays(self.action_array, action)

        def perform_valid_action():
            """Execute action on valid target host."""
            return perform_action_on_host(
                state.hosts,
                action_obj.target_host_idx,
                self.num_hosts,
                action_obj,
                state.traffic_rules,
                key,
                params,
            )

        next_hosts, action_result = perform_valid_action()

        def flatten_state():
            return get_host_vector_flat(next_hosts)

        def flatten_obs():
            obs_hosts = get_observation(next_hosts, action_obj, action_result)
            return get_host_vector_flat(obs_hosts)

        next_obs = jax.lax.cond(params.fully_obs, flatten_state, flatten_obs)

        action_type = jax.nn.one_hot(
            action_obj.action_type, num_classes=len(ActionType), dtype=jnp.uint8
        )
        # Add the auxiliary information to the observation. These are the flags:
        # success, connection error, permission error, undefined error
        aux_info = jnp.array(
            [
                action_result.success,
                action_result.connection_error,
                action_result.permission_error,
                action_result.undefined_error,
            ]
        )
        next_obs = jnp.append(next_obs, action_type)
        next_obs = jnp.append(next_obs, aux_info)

        # Check if goal is reached (all sensitive hosts have root access)
        goal_reached = self._check_goal_completion(next_hosts)

        # Check step limit
        step_limit_reached = state.steps >= (params.step_limit - 1)
        done = goal_reached | step_limit_reached

        # Calculate reward
        reward = action_result.value - action_result.cost

        # Create next state
        next_state = NASimJaxEnvState(
            time=state.time + 1,
            hosts=next_hosts,
            last_obs=next_obs,
            steps=state.steps + 1,
            done=done,
            traffic_rules=state.traffic_rules,
            max_possible_reward=state.max_possible_reward,
        )

        # Keep info empty for JIT compatibility
        info = {}

        return next_obs, next_state, reward, done, info

    @partial(jax.jit, static_argnames=("self",))
    def is_terminal(
        self, state: NASimJaxEnvState, params: NASimJaxEnvParams
    ) -> jnp.ndarray:
        """Check if state is terminal."""
        return state.done

    @partial(jax.jit, static_argnames=("self",))
    def get_obs(
        self, state: NASimJaxEnvState, params: NASimJaxEnvParams = None
    ) -> jnp.ndarray:
        """Get observation from state."""
        if params is None:
            params = self.default_params

        if params.flat_obs:
            return state.last_obs.flatten()
        else:
            return state.last_obs

    @partial(jax.jit, static_argnames=("self",))
    def get_valid_actions(self, state: NASimJaxEnvState, host_idx):
        host_os = state.hosts.os[host_idx]
        host_services = state.hosts.services[host_idx]
        host_processes = state.hosts.processes[host_idx]

        exploit_mask = jnp.outer(host_services, host_os).flatten()
        privesc_mask = jnp.outer(host_processes, host_os).flatten()

        # 4 scan actions per host (Service, OS, Subnet, Process)
        scan_mask = jnp.ones(4, dtype=exploit_mask.dtype)
        action_mask = jnp.concatenate([scan_mask, exploit_mask, privesc_mask])

        return action_mask

    def _state_as_ints(self, flattened: jnp.ndarray) -> jnp.ndarray:
        """
        Converts each hosts binary feature vector into an integer.
        Useful if you want one integer per host to track per-host changes.
        """
        # print("Flattened:", flattened)
        bits = jnp.rint(flattened).astype(jnp.uint64)
        # print("Bits:", bits)
        powers = 1 << jnp.arange(
            bits.shape[0], dtype=jnp.uint64
        )  # [1, 2, 4, ..., 2^(n-1)]
        # print("Powers", powers)
        return jnp.sum(bits * powers, axis=-1)  # shape: (N,)

    def _check_goal_completion(self, hosts: HostVectorBatched) -> jnp.ndarray:
        """Check if all sensitive hosts have been compromised with root access.

        Args:
            hosts: Current host vector state

        Returns:
            Boolean indicating if goal is reached
        """
        is_sensitive_host = hosts.sensitive
        has_root_access = jnp.argmax(hosts.access_level, axis=1) >= 2
        # Check whether the intersection between sensitive hosts and root access is
        # satisfied everywhere.
        return jnp.all(
            jnp.equal(
                is_sensitive_host, jnp.logical_and(is_sensitive_host, has_root_access)
            )
        )

    def _compute_max_possible_reward(
        self, hosts: HostVectorBatched, params: NASimJaxEnvParams
    ) -> int:
        """Compute the approximate maximum possible reward in the environment.
        It is an approximation since the actual maximum reward depends on the
        number of hosts that have to be compromised to reach all sensitive hosts.

        This is used for normalizing rewards in some algorithms.

        Args:
            hosts: Initial host vector state
            params: Environment parameters

        Returns:
            Maximum possible reward
        """
        # Each sensitive host gives a fixed reward when compromised with root access
        sensitive_host_reward = params.sensitive_host_value
        num_sensitive_hosts = jnp.sum(hosts.sensitive)

        max_reward = (
            sensitive_host_reward * num_sensitive_hosts
            - (params.exploit_cost + params.privesc_cost) * num_sensitive_hosts
        )

        return max_reward

    def reconstruct_hosts_from_observation(
        self, obs_array: jnp.ndarray
    ) -> HostVectorBatched:
        """Reconstruct HostVectorBatched from observation array.

        This method takes an observation array (including auxiliary info) and
        reconstructs it back into a HostVectorBatched structure for debugging
        and analysis purposes.

        Args:
            obs_array: Complete observation array from environment (includes aux info)

        Returns:
            Reconstructed HostVectorBatched structure
        """
        # Remove auxiliary info (last 4 elements)
        host_data = obs_array[: -(len(ActionType) + AUX_INFO_FLAGS)]

        # Use template from initial hosts to get dimensions
        return reconstruct_host_vector_from_flat(host_data, self._initial_hosts)
