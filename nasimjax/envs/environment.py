# Copyright (c) 2026 The NASimJax Authors.
# Licensed under the MIT License. See LICENSE in the project root.
"""JAX-based NASim Environment implementation.

This module provides a JAX-compatible version of the NASim environment
that follows functional programming paradigms and can be JIT compiled.
"""

from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import chex
from gymnax.environments import spaces

from nasimjax.envs.environment_base import Environment
from nasimjax.envs.common import NASimJaxEnvState, NASimJaxEnvParams
from nasimjax.envs.action import FlatActionSpaceJAX, get_action_from_arrays
from nasimjax.envs.host_vector_batched import (
    HostVectorBatched,
    set_reachable,
    set_discovered,
    set_access,
    create_host_vector_batched,
    get_host_vector_flat,
    reconstruct_host_vector_from_flat,
)
from nasimjax.envs.observation_function import get_observation, get_initial_observation
from nasimjax.envs.transition_logic import perform_action_on_host
from nasimjax.envs.utils import ActionType


AUX_INFO_FLAGS = 4


class NASimEnvJAX(Environment[NASimJaxEnvState, NASimJaxEnvParams]):
    """JAX-compatible NASim environment.

    This environment implements the Gymnax interface and follows functional
    programming paradigms for JAX compatibility.
    """

    def __init__(self, scenario, fully_obs=False):
        """Initialize the JAX environment.

        Parameters
        ----------
        scenario : Scenario
            The scenario definition
        """
        self.scenario = scenario
        self._scenario_name = scenario.name
        self.fully_obs = fully_obs

        # Initialize action space
        self.action_space_jax = FlatActionSpaceJAX(scenario)
        self.action_array = self.action_space_jax.action_arrays

        # Create topology matrix for action execution
        self.topology = jnp.array(scenario.topology, dtype=jnp.uint8)

        # Use the topology and the services.firewall dictionary to create the firewall rules
        traffic_rules = jnp.zeros(
            (
                len(scenario.topology),
                len(scenario.topology[0]),
                (1 + len(scenario.services)),
            ),
            dtype=jnp.uint8,
        )

        # Define serive map
        service_map = {
            service_name: service_idx
            for service_idx, service_name in enumerate(scenario.services)
        }

        # Set the connection between hosts
        for i in range(len(scenario.topology)):
            for j in range(len(scenario.topology)):
                traffic_rules = traffic_rules.at[i, j, 0].set(
                    jnp.uint8(scenario.topology[i][j])
                )

                # Set which services are allowed
                if (i, j) in scenario.firewall:
                    idx = jnp.array(
                        [service_map[s] for s in scenario.firewall[(i, j)]],
                        dtype=jnp.uint8,
                    )
                    idx += (
                        1  # Shifts idx by 1 to not overwrite connection between hosts.
                    )
                    traffic_rules = traffic_rules.at[i, j, idx].set(1)

        self.traffic_rules = traffic_rules
        self.address_space = jnp.array(scenario.address_space, dtype=jnp.uint8)
        self.num_hosts = len(self.address_space)

        # Pre-compute initial host structure with proper connectivity (outside JIT)
        self._initial_hosts = create_host_vector_batched(scenario)

        # TODO: Here we always assume that the first host is the starting host and the
        # second the one that is reachable next. This might not always be the case.
        # We should adapt this to the scenario definition.
        # Set initial host attributes. Sicne it's the attacker's machine, he controls it.
        self._initial_hosts = set_reachable(
            hosts=self._initial_hosts, host_idx=0, value=True
        )
        self._initial_hosts = set_discovered(
            hosts=self._initial_hosts, host_idx=0, value=True
        )
        self._initial_hosts = set_access(
            hosts=self._initial_hosts, host_idx=0, access_level=2
        )  # Give root access

        # Set the next host attributes, just put reachable
        self._initial_hosts = set_reachable(self._initial_hosts, host_idx=1, value=True)

        # Calculate observation dimensions for space definition
        dummy_obs = get_host_vector_flat(self._initial_hosts)
        self._obs_low = jnp.min(dummy_obs)
        self._obs_high = jnp.max(dummy_obs)
        self._obs_shape = ((dummy_obs.shape[0] + len(ActionType) + AUX_INFO_FLAGS),)
        self._flat_obs_size = self._obs_shape  # Equal since we only provide plat obs.

    @property
    def default_params(self) -> NASimJaxEnvParams:
        """Default environment parameters."""
        return NASimJaxEnvParams(
            step_limit=self.scenario.step_limit or 1000,
            fully_obs=self.fully_obs,
            num_hosts=self.num_hosts,
        )

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
        """Reset the environment to initial state."""
        # Simply copy the pre-computed initial host state (JIT-compatible)
        hosts_init = self._initial_hosts

        # Generate initial observation
        def flatten_state():
            return get_host_vector_flat(hosts_init)

        def flatten_obs():
            return get_host_vector_flat(get_initial_observation(hosts_init))

        initial_obs = jax.lax.cond(params.fully_obs, flatten_state, flatten_obs)
        action_type = jax.nn.one_hot(0, num_classes=len(ActionType), dtype=jnp.uint8)
        aux_info = jnp.array([0, 0, 0, 0], dtype=jnp.uint8)
        obs = jnp.append(initial_obs, action_type)
        obs = jnp.append(obs, aux_info)

        max_possible_reward = self._compute_max_possible_reward(hosts_init, params)

        # Create initial environment state
        env_state = NASimJaxEnvState(
            time=0,
            hosts=hosts_init,
            last_obs=obs,
            steps=0,
            done=False,
            traffic_rules=self.traffic_rules,
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
        """Perform one environment step."""
        # Convert action index to action
        action_obj = get_action_from_arrays(self.action_array, action)

        # Use target host index directly from action
        target_host_idx = action_obj.target_host_idx

        # Perform action on the target host
        next_hosts, action_result = perform_action_on_host(
            state.hosts,
            target_host_idx,
            self.num_hosts,
            action_obj,
            state.traffic_rules,
            key,
            params,
        )

        def flatten_state():
            return get_host_vector_flat(next_hosts)

        def flatten_obs():
            return get_host_vector_flat(
                get_observation(next_hosts, action_obj, action_result)
            )

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


if __name__ == "__main__":
    from nasimjax.scenarios import make_benchmark_scenario

    medium = make_benchmark_scenario("medium")

    env = NASimEnvJAX(medium)
