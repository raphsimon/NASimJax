# `mutators.py` — Level mutation operators (ACCEL)

Atomic, JIT-friendly edits to a generated
[`Level`](network_generator.md), used to drive the ACCEL curriculum from
[JaxUED](https://github.com/DramaCow/jaxued). Each mutation is a pure
function of `(PRNGKey, Level)` and is designed to:

- preserve **solvability** of the scenario after each edit,
- vary **smoothly** in regret (small edits → levels near the agent's
  current frontier),
- compose under `jax.lax.scan` so a sequence of edits stays inside a
  single `jit`.

## Mutation types

The atomic operators are dispatched via `jax.lax.switch` and identified
by the `MutationType` enum:

| ID | Name             | Target             | Solvability risk                     |
|----|------------------|--------------------|--------------------------------------|
| 0  | `NO_OP`          | —                  | None (used for masking unused edits) |
| 1  | `TOGGLE_SERVICE` | Host services      | Low (repaired post-hoc)              |
| 2  | `TOGGLE_PROCESS` | Host processes     | Low (repaired post-hoc)              |
| 3  | `CHANGE_OS`      | Host OS assignment | Low (repaired post-hoc)              |
| 4  | `SWAP_EDGES`     | Adjacency matrix   | Medium (may disconnect subnets)      |
| 5  | `MOVE_SENSITIVE` | Sensitivity flags  | Low (constrained to reachable hosts) |

`NO_OP` is reserved for masking edits beyond the requested count and must
not be passed in the `mutation_types` argument of
[`make_level_mutator`](#api-reference).

## Pipeline

Each call to the mutator built by `make_level_mutator` runs:

1. Sample `max_num_edits` mutation types uniformly from the enabled set.
2. Mask everything past `num_edits` to `NO_OP`.
3. Apply edits sequentially via `jax.lax.scan`.
4. Ensure at least one sensitive host is still reachable from the
   attacker subnet.
5. Re-run `ensure_attack_path` (from
   [`network_generator`](network_generator.md)) to repair
   service/process/OS invariants broken by the edits.
6. Recompute derived metadata (`num_active_hosts`, `is_flat_topo`,
   `num_sensitive_hosts`).

The returned callable has signature
`(PRNGKey, Level, int) -> Level`, which is the interface
[ACCEL][accel-paper] expects.

## Usage

```python
from nasimjax.envs.network_generator import make_level_generator
from nasimjax.envs.mutators import make_level_mutator

sample_level = make_level_generator(env_params)
mutate_level = make_level_mutator(
    max_num_edits=10,
    num_hosts=env_params.num_hosts,
    num_subnets=env_params.num_subnets,
    num_services=env_params.num_services,
    num_os=env_params.num_os,
    num_processes=env_params.num_processes,
)

key = jax.random.PRNGKey(0)
parent = sample_level(key)
child = mutate_level(key, parent, num_edits=5)
```

`max_num_edits` is the unrolled scan length and is therefore a
compile-time constant; `num_edits` is dynamic and may vary between
calls without retracing.

[accel-paper]: https://arxiv.org/abs/2203.01302

## API reference

::: nasimjax.envs.mutators
    options:
      members:
        - MutationType
        - make_level_mutator