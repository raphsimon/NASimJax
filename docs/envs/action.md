# `action.py` — Action system

NASimJAX encodes every possible action as a row in a pre-computed set of JAX
arrays. A discrete agent action is just an integer index into those arrays,
which is decoded at JIT-time into an `ActionData` struct. This keeps the hot
path side-effect-free and branch-friendly.

## Action types

The action types are:

- **Scans** — `SERVICE_SCAN`, `OS_SCAN`, `PROCESS_SCAN`, `SUBNET_SCAN`.
  They reveal information about a host or a subnet.
- **Exploit** — remote attack that yields user access to a target host by
  targeting a running service.
- **Privilege escalation** — local attack that raises access from USER to
  ROOT by targeting a running process.
- **NOOP** — used as a safe default when an action is invalid.

Each action carries a `cost`, a success `prob` and a required access level
(`NONE`/`USER`/`ROOT`). Factory helpers on `ActionJAX` build the right
`ActionData` for each type.

## Flat action spaces

- `FlatActionSpaceJAX` builds the enumeration from a NASim `Scenario`
  (used by [`NASimEnvJAX`](environment.md)).
- `GeneratedFlatActionSpaceJAX` builds the enumeration from raw shape
  parameters (`num_hosts`, `num_services`, `num_processes`, `num_os`) for
  [`GeneratedNASimEnvJAX`](generated_environment.md).

`get_action_from_arrays` is the JIT-compatible lookup used in `step_env` to
turn an integer action into an `ActionData`.

## API reference

::: nasimjax.envs.action.ActionData

::: nasimjax.envs.action.ActionResult

::: nasimjax.envs.action.ActionJAX

::: nasimjax.envs.action.FlatActionSpaceJAX

::: nasimjax.envs.action.GeneratedFlatActionSpaceJAX

::: nasimjax.envs.action.get_action_from_arrays
