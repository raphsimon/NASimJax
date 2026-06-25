# NASimJAX

NASimJAX is a JAX port of the [Network Attack Simulator
(NASim)](https://networkattacksimulator.readthedocs.io/en/latest/) designed to
train Reinforcement Learning agents on network penetration testing tasks. It
exposes a [Gymnax](https://github.com/RobertTLange/gymnax)-compatible interface
and is written in a purely functional style so that `jax.jit`, `jax.vmap` and
`jax.lax.scan` can be applied end-to-end.

This documentation focuses on the code living under `nasimjax/envs/`, which
implements the environment, the action and observation systems, network
generation, and the wrappers used during training. The API reference pages are
auto-generated from the source docstrings via
[mkdocstrings](https://mkdocstrings.github.io/).

## Quick start

### Benchmark scenario

```python
import jax
from nasimjax.scenarios import make_benchmark_scenario
from nasimjax.envs import NASimEnvJAX

scenario = make_benchmark_scenario("tiny")   # tiny / small / medium
env = NASimEnvJAX(scenario, fully_obs=False)
params = env.default_params

key = jax.random.key(42)
obs, state = env.reset(key, params)
action = 0
obs, state, reward, done, info = env.step(key, state, action, params)
```

### Procedurally generated network

```python
import jax
from nasimjax.envs import GeneratedNASimEnvJAX

env = GeneratedNASimEnvJAX(
    key=jax.random.key(0),
    num_hosts=10,
    num_subnets=5,
    num_services=3,
    num_processes=2,
    num_os=2,
    env_pool_size=1000,
)
```

## Layout

- [Overview](envs/overview.md) — how the pieces in `nasimjax/envs` fit together.
- [Base interface](envs/environment_base.md), [state & params](envs/common.md).
- [`NASimEnvJAX`](envs/environment.md), [`GeneratedNASimEnvJAX`](envs/generated_environment.md).
- [Host vector](envs/host_vector_batched.md), [actions](envs/action.md),
  [transition logic](envs/transition_logic.md), [observations](envs/observation_function.md).
- [Network generator](envs/network_generator.md),
  [env pool](envs/gen_env_pool.md), [wrappers](envs/wrappers.md),
  [renderer](envs/renderer.md), [utilities](envs/utils.md).
