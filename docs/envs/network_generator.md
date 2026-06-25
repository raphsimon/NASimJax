# `network_generator.py` — Procedural networks

`generate` builds a random network topology and populates a fresh
[`HostVectorBatched`](host_vector_batched.md) with random services,
processes, operating systems and sensitive markers. It is JIT-compiled with
every shape parameter (`num_hosts`, `num_subnets`, `num_services`, `num_os`,
`num_processes`) marked as static so that the resulting arrays have
compile-time shapes.

Densities control how sparse the generated networks are:

- `topology_density` — probability of an inter-subnet link. Lower values
  produce more disconnected networks (which reduces the number of reachable
  "active" hosts).
- `service_density`, `process_density` — probability that any given
  service/process is running on a host.
- `sensitive_density` — probability that a host is flagged sensitive.
- `distribute_homogeneous` — if `True`, draws services/processes from a
  per-subnet beta distribution so that hosts within the same subnet tend to
  look alike.

The module also exposes helpers used by UED / curriculum-learning training
loops:

- `Level` — the frozen description of one generated network.
- `make_level_generator(params)` — returns a function
  `key -> Level` that can be `vmap`-ped to batch-generate levels.
- `make_eval_levels_and_names(...)` — builds a fixed set of evaluation
  levels with human-readable names.
- `make_level_mutator`, `make_level_mutator_minimax` — mutators used by
  minimax-style UED algorithms.

## API reference

::: nasimjax.envs.network_generator.Level

::: nasimjax.envs.network_generator.generate

::: nasimjax.envs.network_generator.ensure_attack_path

::: nasimjax.envs.network_generator.make_level_generator

::: nasimjax.envs.network_generator.make_eval_levels_and_names

::: nasimjax.envs.network_generator.make_level_mutator

::: nasimjax.envs.network_generator.make_level_mutator_minimax
