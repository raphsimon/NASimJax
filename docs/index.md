# NASimJax

NASimJax is a pure-JAX port and extension of the [Network Attack Simulator
(NASim)](https://networkattacksimulator.readthedocs.io/en/latest/) designed to
train Reinforcement Learning agents on network penetration testing tasks. It
exposes a [Gymnax](https://github.com/RobertTLange/gymnax)-compatible interface
and is written in a purely functional style so that `jax.jit`, `jax.vmap` and
`jax.lax.scan` can be applied end-to-end. Notable changes:

* State-space representation optimized for JAX.
* Environment and task is framed as a Contextual POMDP.
* Parameterized and procedurally generated networks to make training distributional, helping with generalization.
* Networks are guaranteed solvable to not impact training performance with dead-ends.
* Through GPU-accelerated training, learning policies on larger networsk becomes feasible.

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
from nasimjax.envs import ProcGenNASimJaxEnv, NASimJaxEnvParams

# Topology and density parameters define the distribution that networks
# are sampled from. The 16-host reference config from the paper is a
# good starting point — see agents/config/envs/16-hosts-gen.yaml.
params = NASimJaxEnvParams(
    num_hosts=16,
    num_subnets=7,
    num_services=3,
    num_processes=3,
    num_os=2,
    topology_density=0.15,
    service_density=0.7,
    process_density=0.7,
    sensitive_density=0.2,
    distribute_homogeneous=True,
    step_limit=300,
)

# The PRNGKey only seeds the static scaffold (action-space layout,
# observation shape). A fresh network is sampled from the distribution
# on every reset.
env = ProcGenNASimJaxEnv(key=jax.random.key(0), params=params)
env_params = env.default_params

key = jax.random.key(42)
obs, state = env.reset(key, env_params)
action = env.action_space(env_params).sample(key)
obs, state, reward, done, info = env.step(key, state, action, env_params)
```

For training- and evaluation-time wrappers (`AugmentedObservationsWrapper`,
`NormalizeRewardWrapper`, `LogWrapper`, `AutoResetWrapper`) see
[`wrappers.py`](envs/wrappers.md). A self-contained PPO loop with action
masking that wires all of this up end-to-end lives in `agents/ppo_masked.py`.

## Layout

- [Overview](envs/overview.md) — how the pieces in `nasimjax/envs` fit together.
- [Base interface](envs/environment_base.md), [state & params](envs/common.md).
- [`NASimEnvJAX`](envs/environment.md), [`ProcGenNASimJaxEnv`](envs/generated_environment.md).
- [Host vector](envs/host_vector_batched.md), [actions](envs/action.md),
  [transition logic](envs/transition_logic.md), [observations](envs/observation_function.md).
- [Network generator](envs/network_generator.md),
  [wrappers](envs/wrappers.md), [renderer](envs/renderer.md),
  [utilities](envs/utils.md).
