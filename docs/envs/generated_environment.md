# `generated_environment.py` — `GeneratedNASimEnvJAX`

`GeneratedNASimEnvJAX` is the procedurally generated counterpart of
[`NASimEnvJAX`](environment.md). Instead of loading a fixed scenario, it
samples networks from a [`GenEnvPool`](gen_env_pool.md) buffer, so every
`reset` yields a different topology drawn from the same distribution. This
makes the environment directly usable for training with unsupervised
environment design (UED) libraries such as JaxUED.

A `Level` is a frozen, JIT-friendly description of a generated network —
accepted by `reset_env_to_level` to deterministically restart an episode
from a specific network. This is what UED loops and curriculum-learning
code use to replay or rank levels.

!!! note "Fixed shapes under JIT"
    `num_hosts`, `num_services`, `num_processes`, `num_os` and `num_subnets`
    determine the shapes of every JAX array in the environment and cannot be
    changed after construction. Procedurally generated networks smaller than
    `num_hosts` simply leave the tail slots as padding.

The per-step interface is identical to
[`NASimEnvJAX`](environment.md): action decoding,
[`perform_action_on_host`](transition_logic.md), observation masking, reward
computation and termination check are all shared. In addition,
`get_valid_actions(state, host_idx)` returns a 0/1 mask over the action space
given the services, processes and OS of one host — useful for action masking
during training.

## API reference

::: nasimjax.envs.generated_environment.Level

::: nasimjax.envs.generated_environment.GeneratedNASimEnvJAX
