# `utils.py` — Enums and UED rollout helpers

Small building blocks used throughout `nasimjax/envs`.

## Enums

- `AccessLevel` — `NONE` / `USER` / `ROOT`. Tracks the level of access the
  attacker has on a given host.
- `ActionType` — `NOOP`, `OS_SCAN`, `SERVICE_SCAN`, `PROCESS_SCAN`,
  `SUBNET_SCAN`, `EXPLOIT`, `PRIV_ESC`. The integer values match the
  branches of the `jax.lax.switch` used in
  [`transition_logic`](transition_logic.md) and the action-type one-hot
  appended to observations in [`environment.py`](environment.md).

Both enums subclass `enum.IntEnum`, so they compare and arithmetic exactly
like the underlying integers — convenient for indexing JAX arrays without
casting.

## UED rollout helpers

A small set of JIT-compatible rollout statistics used to score levels in
Unsupervised Environment Design (UED) loops such as PLR / ACCEL. They
operate on per-step arrays produced by a rolled-out trajectory and are
adapted from
[minimax](https://github.com/facebookresearch/minimax/blob/2ae9e04d37f97d7c14308f5a26237dcfca63470f/src/minimax/util/rl/ued_scores.py).

- `accumulate_rollout_stats(dones, metrics, *, time_average)` — the core
  primitive. Walks a `(T, B)` trajectory with `jax.lax.scan`, accumulating
  a per-episode value of `metrics` (either the running sum, or the running
  mean when `time_average=True`), and returns its per-environment **mean
  over completed episodes**, its per-environment **max over completed
  episodes**, and the **episode count** for each environment in the batch.
- `compute_max_returns(dones, rewards)` — convenience wrapper returning the
  maximum episodic return seen per environment. Useful as the "best
  achievable return" baseline in regret-style UED scores.
- `compute_max_mean_returns_epcount(dones, rewards)` — like
  `compute_max_returns` but also returns the mean episodic return and
  episode counts.
- `max_mc(dones, values, max_returns, incomplete_value=-jnp.inf)` —
  **maximum Monte-Carlo regret** score: the per-episode mean of
  `max_return - value` averaged over completed episodes. Used as a PLR
  scoring function. Environments that did not complete any episode are
  filled with `incomplete_value`.
- `positive_value_loss(dones, advantages, incomplete_value=-jnp.inf)` —
  **positive value loss** score: the per-episode mean of
  `max(advantage, 0)` averaged over completed episodes. Also used as a
  PLR scoring function.

## API reference

::: nasimjax.envs.utils
    options:
      members:
        - AccessLevel
        - ActionType
        - accumulate_rollout_stats
        - compute_max_returns
        - compute_max_mean_returns_epcount
        - max_mc
        - positive_value_loss