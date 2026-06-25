# Copyright (c) 2026 The NASimJax Authors.
# Licensed under the MIT License. See LICENSE in the project root.
"""
This module provides functional implementations for observation operations
that are compatible with JAX transformations.
"""

import jax
import jax.numpy as jnp

from nasimjax.envs.action import ActionData, ActionResult
from nasimjax.envs.host_vector_batched import HostVectorBatched


def get_observation(
    state: HostVectorBatched, action: ActionData, action_result: ActionResult
):
    """
    Create the observation for the environment based on the previous state, action taken, and action result.

    Each type of action, and wheter it was successful, results in a different observation for the agent.

    Parameters
    ----------
    state : HostVectorBatched
        The previous state of the environment, containing information about all hosts.
    action : ActionData
        The action taken by the agent, containing the action type and target host index.
    action_result : ActionResult
        The result of the action taken, containing information about whether the action was successful and any newly discovered hosts.

    Returns
    -------
    HostVectorBatched
        The observation for the current state of the environment, containing information about the targeted host and any newly discovered hosts based on the action taken."""

    def empty_obs():
        return HostVectorBatched(
            subnet_address=jnp.zeros_like(state.subnet_address),
            reachable=jnp.zeros_like(state.reachable),
            discovered=jnp.zeros_like(state.discovered),
            sensitive=jnp.zeros_like(state.sensitive),
            access_level=jnp.zeros_like(state.access_level),
            os=jnp.zeros_like(state.os),
            services=jnp.zeros_like(state.services),
            processes=jnp.zeros_like(state.processes),
        )

    def construct_obs():
        mask_1d = jnp.zeros_like(state.reachable, dtype=state.reachable.dtype)
        mask_1d = mask_1d.at[action.target_host_idx].set(1)

        # Helper function to apply mask based on array dimensionality
        def apply_mask(arr, preserve=True):
            if not preserve:
                return jnp.zeros_like(arr)
            if arr.ndim == 2:
                return arr * mask_1d.reshape(-1, 1)
            else:
                return arr * mask_1d

        # Define handlers for each action type
        def handle_noop():
            return empty_obs()  # NOOP returns basic obs

        def handle_os_scan():
            return HostVectorBatched(
                subnet_address=apply_mask(state.subnet_address),
                reachable=apply_mask(state.reachable),
                discovered=apply_mask(state.discovered),
                sensitive=apply_mask(state.sensitive, preserve=False),
                access_level=apply_mask(state.access_level, preserve=False),
                os=apply_mask(state.os),
                services=apply_mask(state.services, preserve=False),
                processes=apply_mask(state.processes, preserve=False),
            )

        def handle_service_scan():
            return HostVectorBatched(
                subnet_address=apply_mask(state.subnet_address),
                reachable=apply_mask(state.reachable),
                discovered=apply_mask(state.discovered),
                sensitive=apply_mask(state.sensitive, preserve=False),
                access_level=apply_mask(state.access_level, preserve=False),
                os=apply_mask(state.os, preserve=False),
                services=apply_mask(state.services),
                processes=apply_mask(state.processes, preserve=False),
            )

        def handle_process_scan():
            return HostVectorBatched(
                subnet_address=apply_mask(state.subnet_address),
                reachable=apply_mask(state.reachable),
                discovered=apply_mask(state.discovered),
                sensitive=apply_mask(state.sensitive, preserve=False),
                access_level=apply_mask(state.access_level),
                os=apply_mask(state.os, preserve=False),
                services=apply_mask(state.services, preserve=False),
                processes=apply_mask(state.processes),
            )

        def handle_subnet_scan():
            # Since this action reveals more information than just about the
            # targeted host, we need to use another mask here.
            return HostVectorBatched(
                subnet_address=action_result.discovered[:, jnp.newaxis]
                * state.subnet_address,
                reachable=state.reachable * action_result.discovered,
                discovered=state.discovered
                * action_result.discovered,  # Reveal discovered for all discovered hosts
                sensitive=apply_mask(state.sensitive, preserve=False),
                access_level=apply_mask(state.access_level, preserve=False),
                os=apply_mask(state.os, preserve=False),
                services=apply_mask(state.services, preserve=False),
                processes=apply_mask(state.processes, preserve=False),
            )

        def handle_exploit():
            return HostVectorBatched(
                subnet_address=apply_mask(state.subnet_address),
                reachable=apply_mask(state.reachable),
                discovered=apply_mask(state.discovered),
                sensitive=apply_mask(state.sensitive),
                access_level=apply_mask(state.access_level),
                os=apply_mask(state.os),
                services=apply_mask(state.services),
                processes=apply_mask(state.processes, preserve=False),
            )

        def handle_privesc():
            return HostVectorBatched(
                subnet_address=apply_mask(state.subnet_address),
                reachable=apply_mask(state.reachable),
                discovered=apply_mask(state.discovered),
                sensitive=apply_mask(state.sensitive),
                access_level=apply_mask(state.access_level),
                os=apply_mask(state.os),
                services=apply_mask(state.services, preserve=False),
                processes=apply_mask(state.processes),
            )

        # JAX switch - order MUST match ActionType enum values (0,1,2,3,4,5,6)
        return jax.lax.switch(
            action.action_type,  # Integer index from ActionData
            [
                lambda: handle_noop(),  # ActionType.NOOP = 0
                lambda: handle_os_scan(),  # ActionType.OS_SCAN = 1
                lambda: handle_service_scan(),  # ActionType.SERVICE_SCAN = 2
                lambda: handle_process_scan(),  # ActionType.PROCESS_SCAN = 3
                lambda: handle_subnet_scan(),  # ActionType.SUBNET_SCAN = 4
                lambda: handle_exploit(),  # ActionType.EXPLOIT = 5
                lambda: handle_privesc(),  # ActionType.PRIV_ESC = 6
            ],
        )

    return jax.lax.cond(action_result.success, construct_obs, empty_obs)


def get_initial_observation(state: HostVectorBatched):
    """Create the initial observation for the environment.

    The initial observation is based on which hosts are discovered at t_0.
    If a host is discovered, return it's address, the reachable and discovered
    status. The remaining information is zeroed out.

    Parameters
    ----------
    state : HostVectorBatched
        The initial state of the environment, containing information about all hosts.

    Returns
    -------
    HostVectorBatched
        The initial observation, containing information only about discovered hosts.
    """
    mask_1d = state.discovered

    # Helper function to apply mask based on array dimensionality
    def apply_mask(arr, preserve=True):
        if not preserve:
            return jnp.zeros_like(arr)
        if arr.ndim == 2:
            return arr * mask_1d.reshape(-1, 1)
        else:
            return arr * mask_1d

    return HostVectorBatched(
        subnet_address=apply_mask(state.subnet_address),
        reachable=apply_mask(state.reachable),
        discovered=apply_mask(state.discovered),
        sensitive=apply_mask(state.sensitive, preserve=False),
        access_level=apply_mask(state.access_level),
        os=apply_mask(state.os, preserve=False),
        services=apply_mask(state.services, preserve=False),
        processes=apply_mask(state.processes, preserve=False),
    )


if __name__ == "__main__":
    from nasimjax.scenarios import make_benchmark_scenario
    from nasimjax.envs import NASimEnvJAX

    get_obs_jit = jax.jit(get_observation)

    scenario = make_benchmark_scenario("small")
    env = NASimEnvJAX(scenario)

    key = jax.random.key(17)
    obs, state = env.reset(key)

    action = env.action_space().sample(key)

    obs, state, reward, done, info = env.step(key, state, action)
