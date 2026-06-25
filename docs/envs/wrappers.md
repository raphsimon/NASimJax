# `wrappers.py` — Environment wrappers

Gymnax-style wrappers that sit between the agent and the base environment.
They follow the same convention as upstream Gymnax: they subclass
`GymnaxWrapper`, forward attribute access via `__getattr__`, and override
`step` / `reset` / `reset_to_level` to inject their own logic while keeping
everything JIT-compatible.

| Wrapper | Purpose |
|---------|---------|
| `AugmentedObservationsWrapper` | Concatenates the current raw observation with an *aggregated* observation that keeps track of everything the agent has ever seen (OR over discrete fields, max over access levels). Useful for partially observable training. |
| `NormalizeRewardWrapper` | Rescales rewards to a bounded range, typically using `state.max_possible_reward`. |
| `LogWrapper` | Adds per-episode logging statistics (return, length, ...) to `info`. |
| `AutoReplayWrapper` | Stores the initial `key`/`level` so an episode can be replayed deterministically. |
| `AutoResetWrapper` | Standalone auto-reset, used when the base environment's built-in auto-reset is disabled. |

## API reference

::: nasimjax.envs.wrappers.GymnaxWrapper

::: nasimjax.envs.wrappers.AugmentedObservationsWrapper

::: nasimjax.envs.wrappers.NormalizeRewardWrapper

::: nasimjax.envs.wrappers.LogEnvState

::: nasimjax.envs.wrappers.LogWrapper

::: nasimjax.envs.wrappers.AutoReplayState

::: nasimjax.envs.wrappers.AutoReplayWrapper

::: nasimjax.envs.wrappers.AutoResetState

::: nasimjax.envs.wrappers.AutoResetWrapper
