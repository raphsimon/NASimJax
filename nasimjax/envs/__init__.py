"""JAX-compatible NASim environment implementations.

This module provides JAX-based implementations of the NASim environment
components that are compatible with JAX transformations like JIT compilation
and vectorization.
"""

from nasimjax.envs.environment_base import Environment, EnvState, EnvParams
from nasimjax.envs.common import NASimJaxEnvState, NASimJaxEnvParams
from nasimjax.envs.environment import NASimEnvJAX
from nasimjax.envs.generated_environment import ProcGenNASimJaxEnv
from nasimjax.envs.wrappers import AugmentedObservationsWrapper
from nasimjax.envs.network_generator import (
    Level,
    get_connected_subnets,
    ensure_attack_path,
    generate,
    make_eval_levels_and_names,
    make_level_generator,
)

from nasimjax.envs.action import (
    ActionJAX,
    FlatActionSpaceJAX,
    ActionData,
    ActionResult,
)
from nasimjax.envs.host_vector_batched import (
    HostVectorBatched,
    create_host_vector_batched,
    set_reachable,
    set_discovered,
    set_access,
)
from nasimjax.envs.transition_logic import perform_action_on_host
from nasimjax.envs.utils import (
    AccessLevel,
    ActionType,
)

__all__ = [
    # Base classes
    "Environment",
    "EnvState",
    "EnvParams",
    # Main environment classes
    "NASimEnvJAX",
    "NASimJaxEnvState",
    "NASimJaxEnvParams",
    "ProcGenNASimJaxEnv",
    # Network generation
    "Level",
    "get_connected_subnets",
    "ensure_attack_path",
    "generate",
    "make_eval_levels_and_names",
    "make_level_generator",
    # Wrappers
    "AugmentedObservationsWrapper",
    # Action classes and functions
    "ActionJAX",
    "FlatActionSpaceJAX",
    "ActionData",
    "ActionResult",
    # Host vector functions
    "HostVectorBatched",
    "create_host_vector_batched",
    "set_reachable",
    "set_discovered",
    "set_access",
    # Transition logic
    "perform_action_on_host",
    # Utility classes and functions
    "AccessLevel",
    "ActionType",
]
