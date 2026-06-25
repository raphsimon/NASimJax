# `utils.py` — Enums and graph helpers

Small building blocks used throughout `nasimjax/envs`.

## Enums

- `OneHotBool` — 3-valued boolean (`NONE`, `TRUE`, `FALSE`) used when the
  agent's knowledge of a fact is itself uncertain.
- `ServiceState` — `UNKNOWN` / `PRESENT` / `ABSENT` for a service on a host.
- `AccessLevel` — `NONE` / `USER` / `ROOT`.
- `ActionType` — `NOOP`, `OS_SCAN`, `SERVICE_SCAN`, `PROCESS_SCAN`,
  `SUBNET_SCAN`, `EXPLOIT`, `PRIV_ESC`. The integer values match the
  branches of the `jax.lax.switch` used in
  [`observation_function`](observation_function.md) and
  [`transition_logic`](transition_logic.md).

## Graph utilities

JIT-friendly helpers that operate on the subnet adjacency matrix.

- `floyd_warshall` — all-pairs shortest path distances.
- `get_minimal_hops_to_goal_jax` — minimum number of subnet hops needed to
  reach every sensitive subnet. Used for reward shaping.
- `min_subnet_depth_jax` — per-subnet distance from the internet.
- `subnet_connected_to_internet`, `subnets_connected`,
  `get_reachable_subnets` — connectivity queries.

`get_minimal_hops_to_goal` and `min_subnet_depth` are plain Python
counterparts used at environment-construction time.

## API reference

::: nasimjax.envs.utils
    options:
      members:
        - OneHotBool
        - ServiceState
        - AccessLevel
        - ActionType
        - floyd_warshall
        - get_minimal_hops_to_goal_jax
        - min_subnet_depth_jax
        - subnet_connected_to_internet
        - subnets_connected
        - get_reachable_subnets
        - get_minimal_hops_to_goal
        - min_subnet_depth
