# `environment.py` — `NASimEnvJAX`

`NASimEnvJAX` is the scenario-backed environment. It is instantiated from a
NASim `Scenario` object (same format as the upstream NASim repository) and is
compatible with the Gymnax API.

```python
from nasimjax.scenarios import make_benchmark_scenario
from nasimjax.envs import NASimEnvJAX

scenario = make_benchmark_scenario("tiny")
env = NASimEnvJAX(scenario, fully_obs=False)
```

On construction the environment builds a
[`FlatActionSpaceJAX`](action.md) from the scenario, derives a
`(num_subnets, num_subnets, 1 + num_services)` `traffic_rules` tensor from
the scenario topology and firewall, and pre-computes an `_initial_hosts`
[`HostVectorBatched`](host_vector_batched.md) with host 0 flagged as the
attacker (reachable, discovered, root access).

!!! warning "Hard-coded attacker host"
    The current implementation always assumes that the first host is the
    attacker's starting machine and that the second host is reachable from
    it. A TODO in the source notes that this should eventually be read from
    the scenario definition.

The step logic decodes the integer action into an
[`ActionData`](action.md), runs
[`perform_action_on_host`](transition_logic.md) to get the next hosts and the
[`ActionResult`](action.md#nasimjax.envs.action.ActionResult), builds the
observation (fully or partially observable) and appends a one-hot
`ActionType` plus four auxiliary bits
`(success, connection_error, permission_error, undefined_error)`. The reward
is `action_result.value - action_result.cost`, and `done` becomes `True` once
every sensitive host has been compromised with root access or the step limit
is reached.

## API reference

::: nasimjax.envs.environment.NASimEnvJAX
