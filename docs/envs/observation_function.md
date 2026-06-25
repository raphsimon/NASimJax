# `observation_function.py` — Partial observability

`get_observation` builds the partial observation the agent receives after a
step. Starting from the full post-step `HostVectorBatched`, it masks every
field that the current action did **not** actually reveal, so that the agent
only sees what penetration-testing semantics would expose.

Dispatch is implemented with `jax.lax.switch` over `ActionType`, so the
masking rule is picked without introducing Python-level branches:

| Action type | What the observation reveals |
|-------------|------------------------------|
| `NOOP` | Nothing (empty observation). |
| `OS_SCAN` | Target's OS only. |
| `SERVICE_SCAN` | Target's services only. |
| `PROCESS_SCAN` | Target's processes only. |
| `SUBNET_SCAN` | Subnet / reachable / discovered bits for every newly discovered host in the subnet. |
| `EXPLOIT` | Full host info except processes. |
| `PRIV_ESC` | Full host info except services. |

If `action_result.success` is `False`, the observation is returned as an
all-zero `HostVectorBatched`. `get_initial_observation` is used on reset: it
simply exposes the already-discovered hosts and their subnet/reachable bits.

## API reference

::: nasimjax.envs.observation_function.get_observation

::: nasimjax.envs.observation_function.get_initial_observation
