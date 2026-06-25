# `host_vector_batched.py` — Network state

`HostVectorBatched` is the core data structure that holds the network state.
Because JAX cannot operate on ragged Python lists of host objects, every host
attribute is stored as a batched array — one row per host.

| Field | Shape | Meaning |
|-------|-------|---------|
| `subnet_address` | `(num_hosts, num_subnets)` | One-hot subnet membership. |
| `reachable`      | `(num_hosts,)`             | Whether the attacker can currently talk to this host. |
| `discovered`     | `(num_hosts,)`             | Whether the attacker has seen this host at all. |
| `sensitive`      | `(num_hosts,)`             | Whether compromising this host yields reward. |
| `access_level`   | `(num_hosts, 3)`           | One-hot `NONE`/`USER`/`ROOT`. |
| `os`             | `(num_hosts, num_os)`      | OS running on each host. |
| `services`       | `(num_hosts, num_services)`| Services running on each host. |
| `processes`      | `(num_hosts, num_processes)`| Processes running on each host. |

All arrays are `uint8` so the structure is compact and easy to compare.

!!! note "Fixed sizes"
    Every shape is fixed at construction time. The environment reserves room
    for the **maximum** number of hosts; unused slots are zero and ignored by
    the transition and observation logic. This is what allows procedurally
    generated networks of different sizes to share the same JIT-compiled
    graph.

Update helpers (`set_reachable`, `set_discovered`, `set_access`) are all pure
— they return a new `HostVectorBatched` via `replace`, never mutating in
place. `get_host_vector_flat` / `reconstruct_host_vector_from_flat` convert
between the dataclass and the flat `uint8` vector consumed by the agent.

## API reference

::: nasimjax.envs.host_vector_batched
    options:
      members:
        - HostVectorBatched
        - get_empty_hosts
        - create_host_vector_batched
        - set_reachable
        - set_discovered
        - set_access
        - get_host_vector_batched_2D
        - get_host_vector_flat
        - reconstruct_host_vector_from_flat
