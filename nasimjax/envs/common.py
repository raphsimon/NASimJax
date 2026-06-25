# Copyright (c) 2026 The NASimJax Authors.
# Licensed under the MIT License. See LICENSE in the project root.
import jax.numpy as jnp
from flax import struct

from nasimjax.envs.environment_base import EnvParams, EnvState
from nasimjax.envs.host_vector_batched import HostVectorBatched


@struct.dataclass
class NASimJaxEnvState(EnvState):
    """JAX-compatible environment state."""

    time: int
    hosts: HostVectorBatched  # Batched host vector state
    last_obs: jnp.ndarray  # Last observation
    steps: int
    done: bool
    traffic_rules: jnp.ndarray  # Network traffic rules for this environment
    max_possible_reward: int  # Maximum possible reward in this environment


@struct.dataclass
class NASimJaxEnvParams(EnvParams):
    """JAX-compatible environment parameters."""

    max_steps_in_episode: int = 300
    step_limit: int = 300
    fully_obs: bool = False
    flat_actions: bool = True
    flat_obs: bool = True
    discovery_value: int = 1
    host_value: int = 0
    sensitive_host_value: int = 50
    scan_cost: int = 1
    service_scan_cost: int = 1
    process_scan_cost: int = 1
    os_scan_cost: int = 1
    subnet_scan_cost: int = 1
    exploit_cost: int = 3
    privesc_cost: int = 3
    num_hosts: int = 10
    num_subnets: int = 5
    num_services: int = 2
    num_os: int = 2
    num_processes: int = 2
    distribute_homogeneous: bool = False
    topology_density: float = 0.5
    service_density: float = 0.2
    process_density: float = 0.2
    sensitive_density: float = 0.1
