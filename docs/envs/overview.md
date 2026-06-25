# Environment module overview

All the code described in this section lives under `nasimjax/envs/`. The
modules are organised into a small number of layers:

| Layer | Module | Purpose |
|-------|--------|---------|
| Base interface | [`environment_base.py`](environment_base.md) | Abstract `Environment`, `EnvState`, `EnvParams` (Gymnax-style). |
| State & params | [`common.py`](common.md) | `NASimJaxEnvState`, `NASimJaxEnvParams` dataclasses. |
| Environments | [`environment.py`](environment.md), [`generated_environment.py`](generated_environment.md) | Concrete JIT-compatible environments. |
| Network model | [`host_vector_batched.py`](host_vector_batched.md) | Batched, immutable host representation. |
| Action system | [`action.py`](action.md) | `ActionData`, `ActionResult`, flat action spaces. |
| Transition | [`transition_logic.py`](transition_logic.md) | Pure function that applies one action to the hosts. |
| Observation | [`observation_function.py`](observation_function.md) | Partial-observability masking rules. |
| Generation | [`network_generator.py`](network_generator.md) | Procedural network generation. |
| Wrappers | [`wrappers.py`](wrappers.md) | Gymnax-style wrappers (e.g. observation augmentation). |
| Rendering | [`renderer.py`](renderer.md) | Pretty-printing of state/observation for debugging. |
| Utilities | [`utils.py`](utils.md) | Enums (`ActionType`, `AccessLevel`) and UED rollout-stat helpers. |

## Functional programming contract

NASimJax follows the rules imposed by `jax.jit`:

1. Every "hot" function is a **pure function** of its inputs — no Python side
   effects, no in-place mutation.
2. State is stored in **Flax dataclasses** (`@struct.dataclass`) so they can be
   treated as pytrees and passed through `jit`/`vmap`/`scan`.
3. Array shapes are **fixed at construction time**. In particular, the number
   of hosts, subnets, services, processes and OS is baked into the shapes of
   `HostVectorBatched`. Variable-size networks are emulated via bounds
   checking: unused slots in the arrays are simply zero and ignored.
4. Control flow that depends on runtime values uses `jax.lax.cond`,
   `jax.lax.switch` and `jax.lax.fori_loop` instead of Python `if`/`for`.

## Lifecycle of one step

```
action (int)
    │
    ▼
FlatActionSpaceJAX.action_arrays ──► ActionData     (action.py)
    │
    ▼
perform_action_on_host(hosts, action, traffic_rules, ...)      (transition_logic.py)
    │  returns (new_hosts, ActionResult)
    ▼
get_observation(new_hosts, action, action_result)              (observation_function.py)
    │  returns a masked HostVectorBatched
    ▼
flatten + append (action_type one-hot, aux info)               (environment.py)
    │
    ▼
(obs, next_state, reward, done, info)
```

The reward is simply `action_result.value - action_result.cost`, and `done` is
true as soon as every sensitive host has been compromised with root access or
the step limit is reached.
