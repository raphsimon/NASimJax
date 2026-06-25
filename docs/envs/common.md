# `common.py` — State and parameters

Defines the two Flax dataclasses carried through every JIT boundary in
NASimJAX.

`NASimJaxEnvState` holds the mutable state of an episode: the batched host
vector, the last observation, the step counter, the `done` flag, the
per-environment `traffic_rules` tensor, and the precomputed
`max_possible_reward` (used for reward normalisation).

`NASimJaxEnvParams` holds static configuration: step limits, observation mode,
reward/cost values, and the network shape (`num_hosts`, `num_subnets`,
`num_services`, ...). Concrete environments override `default_params` to fill
these in from the scenario.

!!! warning "Shape-related fields are immutable after construction"
    Changing `num_hosts`, `num_subnets`, `num_services`, `num_processes` or
    `num_os` after the environment has been instantiated will break JIT
    compilation because it alters array shapes.

## API reference

::: nasimjax.envs.common
    options:
      members:
        - NASimJaxEnvState
        - NASimJaxEnvParams
