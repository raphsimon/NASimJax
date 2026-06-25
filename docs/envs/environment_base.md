# `environment_base.py` — Gymnax interface

Defines the abstract `Environment` class that all NASimJAX environments
inherit from. It mirrors the [Gymnax](https://github.com/RobertTLange/gymnax)
interface so that any code accepting a Gymnax environment can be reused.

Subclasses override the `*_env` hooks (`reset_env`, `step_env`,
`reset_env_to_level`, `get_obs`, `is_terminal`). The public `step` / `reset`
methods are JIT-compiled and apply the standard Gymnax auto-reset trick: both
the step and the reset branches are always computed and merged with
`jax.lax.select` on `done`.

## API reference

::: nasimjax.envs.environment_base
    options:
      members:
        - EnvState
        - EnvParams
        - Level
        - Environment
