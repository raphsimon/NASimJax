# `gen_env_pool.py` — Environment pool

`GenEnvPool` is a simple fixed-size buffer of generated networks. On
construction it calls [`generate`](network_generator.md) inside a
`jax.vmap` to build `env_pool_size` networks in parallel, and stores the
stacked `HostVectorBatched` and `traffic_rules` as pytrees of batched
arrays.

At reset time, [`GeneratedNASimEnvJAX`](generated_environment.md) calls
`sample(key)` to draw one random index from the buffer and return the
corresponding `(hosts, traffic_rules)` tuple. Because the whole buffer is
materialised up front, every reset runs in constant time and without
recompilation — the JIT graph only depends on the fixed shapes.

This trades memory for sample efficiency: the pool acts as an
approximation of the initial-state distribution the policy will see at
training time, and more elaborate sampling strategies (e.g. UED) can be
layered on top by picking indices in a non-uniform way.

## API reference

::: nasimjax.envs.gen_env_pool.GenEnvPool
