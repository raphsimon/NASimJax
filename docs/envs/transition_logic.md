# `transition_logic.py` — Applying an action

`perform_action_on_host` is the core pure function that takes the current
host state, an `ActionData`, the network `traffic_rules` and the environment
parameters, and returns `(next_hosts, ActionResult)`. It is dispatched on the
action type via `jax.lax.switch` so that JIT compilation produces a single
compact graph for every possible action.

The function handles all of NASim's penetration-testing mechanics:

- **Firewall and reachability checks.** An action can fail with a
  `connection_error` if the subnet topology or the per-service firewall
  blocks the required traffic.
- **Required access.** An action can fail with a `permission_error` if the
  attacker does not already hold the required access level on the target.
- **Stochastic success.** Exploits and privilege escalations succeed with
  probability `prob`, sampled from `rng_key`.
- **Information disclosure.** Successful scans populate
  `services_discovered`, `os_discovered`, `processes_discovered` and/or
  `hosts_discovered` fields of the returned `ActionResult`.
- **Reward shaping.** The `value` and `cost` fields of the returned
  `ActionResult` feed directly into the environment reward
  (`reward = value - cost`).

## API reference

::: nasimjax.envs.transition_logic.perform_action_on_host
