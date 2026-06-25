# Copyright 2024 Matteo Gallici, Mattie Fellows, Benjamin Ellis, Bartomeu Pou,
# Ivan Masmitja, Jakob Nicolaus Foerster, Mario Martin (PureJaxQL authors).
# Modifications Copyright (c) 2026 The NASimJax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Derived from PureJaxQL (https://github.com/mttga/purejaxql) at commit bd3f7e2,
# file purejaxql/pqn_gymnax.py.
#
# Notable modifications:
#   - Added NASimJax environment setup with generated levels and benchmark evaluation buckets
#   - Added action masking for invalid host/exploit combinations
#   - Modified rollout/update for NASimJax observation structure
#   - Adapted evaluation to run on NASimJax benchmark levels with density buckets
#   - Restructured logic to match JaxUED's training loop and logging conventions

"""
Usage:
    python -m agents.pqn_dr_masked_epsgreedy_multi_eval alg=pqn_dr_masked_epsgreedy +envs=16-hosts-gen
"""

import csv
import json
import os
import time
from datetime import datetime
from functools import partial
from typing import Any, Tuple

import chex
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState as BaseTrainState
from omegaconf import OmegaConf
import wandb

from nasimjax.envs.environment_base import EnvState, EnvParams, Environment
from nasimjax.envs.generated_environment import GeneratedNASimEnvJAX
from nasimjax.envs.common import NASimJaxEnvParams
from nasimjax.envs.network_generator import (
    make_eval_levels_and_names,
    make_level_generator,
)
from nasimjax.envs.wrappers import (
    AugmentedObservationsWrapper,
    NormalizeRewardWrapper,
    AutoResetWrapper,
    LogWrapper,
)


class MaskEncoder:
    """Encoder/decoder for boolean action masks using bit packing."""

    def __init__(self, total_actions: int, chunk_size: int = 31):
        self.total_actions = total_actions
        self.chunk_size = chunk_size
        self.n_chunks = (total_actions + chunk_size - 1) // chunk_size
        self.padded_size = self.n_chunks * chunk_size
        self.powers = 2 ** jnp.arange(chunk_size)

    @jax.named_scope("encode_mask")
    @partial(jax.jit, static_argnames=("self",))
    def encode(self, mask: jnp.ndarray) -> jnp.ndarray:
        padded = jnp.zeros(self.padded_size, dtype=jnp.bool_)
        padded = padded.at[: self.total_actions].set(mask.astype(jnp.bool_))
        chunks = padded.reshape(self.n_chunks, self.chunk_size)
        encoded = jnp.sum(chunks * self.powers[None, :], axis=1)
        return encoded.astype(jnp.uint32)

    @jax.named_scope("decode_mask")
    @partial(jax.jit, static_argnames=("self",))
    def decode(self, encoded: jnp.ndarray) -> jnp.ndarray:
        bit_indices = jnp.arange(self.chunk_size)[None, :]
        chunks = (encoded[:, None] >> bit_indices) & 1
        flattened = chunks.flatten().astype(bool)
        return flattened[: self.total_actions]


# -----------------------------------------------------------------------------
# Q-network
# -----------------------------------------------------------------------------
class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 128
    num_layers: int = 2
    norm_type: str = "layer_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # Keep a BatchNorm variable around so batch_stats is always present
            # in the param tree (simplifies pytree-shape matching for orbax).
            _ = nn.BatchNorm(use_running_average=not train)(x)

        if self.norm_type == "layer_norm":
            normalize = lambda y: nn.LayerNorm()(y)
        elif self.norm_type == "batch_norm":
            normalize = lambda y: nn.BatchNorm(use_running_average=not train)(y)
        else:
            normalize = lambda y: y

        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        return nn.Dense(self.action_dim)(x)


# -----------------------------------------------------------------------------
# Containers
# -----------------------------------------------------------------------------
@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    q_val: chex.Array
    encoded_action_mask: chex.Array


class TrainState(BaseTrainState):
    """PQN train state. Mirrors PPO's TrainState but adds PQN-specific fields
    (batch_stats for BatchNorm, timesteps/grad_steps counters for schedules)."""

    batch_stats: Any
    update_count: int
    timesteps: int
    grad_steps: int
    last_obs: Any
    last_env_state: Any


def setup_checkpointing(
    config, train_state: TrainState, env: Environment, env_params: EnvParams
) -> ocp.CheckpointManager:
    overall_save_dir = os.path.join(
        os.getcwd(), "checkpoints", f"{config['run_name']}", str(config["seed"])
    )
    os.makedirs(overall_save_dir, exist_ok=True)

    with open(os.path.join(overall_save_dir, "config.json"), "w+") as f:
        as_dict = config.as_dict() if hasattr(config, "as_dict") else dict(config)
        f.write(json.dumps(as_dict, indent=True, default=str))

    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, "models"),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config["checkpoint_save_interval"],
            max_to_keep=config["max_number_of_checkpoints"],
        ),
    )
    return checkpoint_manager


def single_run(config=None, project="NASimJax"):
    # Flatten alg/env sub-dicts into top-level (matches PPO's pattern).
    config = {**config, **config["alg"], **config["envs"]}

    # 1. Initialize W&B FIRST so the sweep can inject its parameters
    wandb.init(
        config=config,
        project=project,
        tags=[
            "DR",
            config["alg_name"].upper(),
            config["env_name"].upper(),
            f"jax_{jax.__version__}",
        ],
        name=config.get(
            "name",
            f"DR_{config['alg_name']}_{config['env_name']}",
        ),
        group=config.get("group", None),
        mode=config.get("wandb_mode", "online"),
    )

    # 2. Grab the updated config that now contains the swept hyperparameters
    config = wandb.config

    # 3. NOW calculate num_updates and num_updates_decay using the swept parameters
    updates_dict = {}
    if config.get("total_timesteps") is not None:
        updates_dict["num_updates"] = int(
            config["total_timesteps"] // (config["num_envs"] * config["num_steps"])
        )

    if config.get("total_timesteps_decay") is not None:
        updates_dict["num_updates_decay"] = int(
            config["total_timesteps_decay"]
            // (config["num_envs"] * config["num_steps"])
        )
    elif "num_updates" in updates_dict:
        updates_dict["num_updates_decay"] = updates_dict["num_updates"]

    # 4. Push these calculated updates back into wandb config
    wandb.config.update(updates_dict, allow_val_change=True)

    # Validate minibatches using the newly injected sweep params
    assert (config["num_envs"] * config["num_steps"]) % config[
        "num_minibatches"
    ] == 0, "num_minibatches must divide num_envs * num_steps"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    scratch = os.environ.get("VSC_SCRATCH")
    base_dir = scratch if scratch else os.getcwd()

    log_dir = os.path.join(
        base_dir,
        "logs",
        f"DR_{config['alg_name']}",
        config["env_name"],
        f"{timestamp}_s{config['seed']}",
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "metrics.csv")

    wandb.define_metric("num_env_steps")
    for prefix in ["stoch_", "greedy_"]:
        wandb.define_metric(f"{prefix}solve_rate/*", step_metric="num_env_steps")
        wandb.define_metric(f"{prefix}return/*", step_metric="num_env_steps")
        wandb.define_metric(f"{prefix}eval_ep_lengths/*", step_metric="num_env_steps")

    wandb.define_metric("agent/*", step_metric="num_env_steps")
    wandb.define_metric("train/*", step_metric="num_env_steps")

    def log_eval(stats, log_file_path=None):
        print(f"Logging update: {stats['update_count']}")

        env_steps = (
            int(stats["update_count"]) * config["num_envs"] * config["num_steps"]
        )
        log_dict = {
            "num_updates": int(stats["update_count"]),
            "num_env_steps": env_steps,
            "total_timesteps": env_steps,
            "sps": env_steps / max(stats["time_delta"], 1e-8),
        }

        # Track per-mode aggregations
        all_metrics = {
            "stoch": {"solve": [], "ret": []},
            "greedy": {"solve": [], "ret": []},
        }

        # Per-bucket metrics
        for density_tag, bucket_stats in stats["eval_buckets"].items():
            for mode in ["stoch", "greedy"]:
                solve_rates = bucket_stats[f"{mode}_eval_solve_rates"]
                returns = bucket_stats[f"{mode}_eval_returns"]
                ep_lengths = bucket_stats[f"{mode}_eval_ep_lengths"]

                log_dict[f"{mode}_solve_rate/{density_tag}/mean"] = float(
                    solve_rates.mean()
                )
                log_dict[f"{mode}_return/{density_tag}/mean"] = float(returns.mean())
                log_dict[f"{mode}_eval_ep_lengths/{density_tag}/mean"] = float(
                    ep_lengths.mean()
                )

                all_metrics[mode]["solve"].append(solve_rates)
                all_metrics[mode]["ret"].append(returns)

        # Overall aggregates
        for mode in ["stoch", "greedy"]:
            cat_solve = np.concatenate(
                [np.asarray(s) for s in all_metrics[mode]["solve"]]
            )
            cat_return = np.concatenate(
                [np.asarray(r) for r in all_metrics[mode]["ret"]]
            )
            log_dict[f"{mode}_solve_rate/mean"] = float(cat_solve.mean())
            log_dict[f"{mode}_return/mean"] = float(cat_return.mean())

        # Agent metrics (PQN-specific inside the "agent/*" namespace)
        log_dict.update(
            {
                "agent/loss": float(stats["losses"]["td_loss"]),
                "agent/td_loss": float(stats["losses"]["td_loss"]),
                "agent/qvals": float(stats["losses"]["qvals"]),
                "agent/epsilon": float(stats["losses"]["epsilon"]),
                "train/mean_return": float(stats["train_metrics"]["mean_return"]),
                "train/mean_length": float(stats["train_metrics"]["mean_length"]),
            }
        )

        wandb.log(log_dict)

        if log_file_path:
            csv_row = {
                k: float(v)
                if hasattr(v, "__float__") and not isinstance(v, (str, bool))
                else v
                for k, v in log_dict.items()
            }
            file_exists = os.path.isfile(log_file_path)
            with open(log_file_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(csv_row)

    train_env_params = NASimJaxEnvParams(
        num_hosts=config["env_kwargs"]["num_hosts"],
        num_subnets=config["env_kwargs"]["num_subnets"],
        num_services=config["env_kwargs"]["num_services"],
        num_os=config["env_kwargs"]["num_os"],
        num_processes=config["env_kwargs"]["num_processes"],
        distribute_homogeneous=config["env_kwargs"]["distribute_homogeneous"],
        topology_density=config["env_kwargs"]["topology_density"],
        service_density=config["env_kwargs"]["service_density"],
        process_density=config["env_kwargs"]["process_density"],
        sensitive_density=float(config["env_kwargs"]["sensitive_density"]),
        service_scan_cost=config["env_kwargs"].get("scan_cost", 1),
        process_scan_cost=config["env_kwargs"].get("scan_cost", 1),
        os_scan_cost=config["env_kwargs"].get("scan_cost", 1),
        subnet_scan_cost=config["env_kwargs"].get("scan_cost", 1),
        exploit_cost=config["env_kwargs"]["exploit_cost"],
        privesc_cost=config["env_kwargs"]["privesc_cost"],
        fully_obs=config["fully_obs"],
        step_limit=config["env_kwargs"]["step_limit"],
    )
    env = GeneratedNASimEnvJAX(key=jax.random.PRNGKey(0), params=train_env_params)
    if config["normalize_reward"]:
        env = NormalizeRewardWrapper(env)
    if config["aug_obs"]:
        env = AugmentedObservationsWrapper(env)
    sample_random_level = make_level_generator(train_env_params)

    base_eval_env_kwargs = config["eval_env_kwargs"]
    eval_topology_densities = config.get(
        "eval_topology_densities",
        [base_eval_env_kwargs["topology_density"]],
    )
    EVAL_BUCKETS: dict = {}
    for density in eval_topology_densities:
        density = float(density)
        tag = f"topo_{density:.2f}"
        bucket_eval_params = train_env_params.replace(
            topology_density=density,
            service_density=float(base_eval_env_kwargs["service_density"]),
            process_density=float(base_eval_env_kwargs["process_density"]),
            sensitive_density=float(base_eval_env_kwargs["sensitive_density"]),
        )
        levels, level_names = make_eval_levels_and_names(
            bucket_eval_params, size=config["num_eval_levels"]
        )
        EVAL_BUCKETS[tag] = (levels, level_names)

    eval_env = env
    env = AutoResetWrapper(env, sample_random_level)
    env = LogWrapper(env)
    env_params = env.default_params

    action_dim = env.action_space(env_params).n
    mask_encoder = MaskEncoder(total_actions=action_dim)

    def get_inner_state(state):
        s = state
        while not hasattr(s, "hosts"):
            s = s.env_state
        return s

    def get_action_mask(state):
        inner = get_inner_state(state)
        host_os = inner.hosts.os
        host_services = inner.hosts.services
        host_processes = inner.hosts.processes
        host_mask = jnp.logical_and(inner.hosts.reachable, inner.hosts.discovered)
        H = host_os.shape[0]
        exploit = (host_services[:, :, None] * host_os[:, None, :]).reshape(H, -1)
        privesc = (host_processes[:, :, None] * host_os[:, None, :]).reshape(H, -1)
        global_actions = jnp.ones((H, 4), dtype=exploit.dtype)
        action_mask = jnp.concatenate([global_actions, exploit, privesc], axis=1)
        action_mask = action_mask * host_mask[:, None]
        action_mask = action_mask.flatten()
        return action_mask, mask_encoder.encode(action_mask)

    def eps_greedy_exploration(rng, q_vals, eps, mask):
        """Masked epsilon-greedy: greedy over masked Q-values, random over
        *valid* actions only."""
        rng_a, rng_e = jax.random.split(rng)
        masked_q_vals = jnp.where(mask, q_vals, -1e9)
        greedy_actions = jnp.argmax(masked_q_vals, axis=-1)
        random_logits = jnp.where(mask, 0.0, -1e9)
        random_actions = jax.random.categorical(rng_a, random_logits, axis=-1)
        chosen_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            random_actions,
            greedy_actions,
        )
        return chosen_actions

    # -------------------------------------------------------------------------
    # Schedules (defined once; closures capture them)
    # -------------------------------------------------------------------------
    eps_scheduler = optax.linear_schedule(
        config["eps_start"],
        config["eps_finish"],
        int(config["eps_decay"] * config["num_updates_decay"]),
    )

    lr_scheduler = optax.linear_schedule(
        init_value=config["lr"],
        end_value=1e-20,
        transition_steps=config["num_updates_decay"]
        * config["num_minibatches"]
        * config["num_epochs"],
    )
    lr = lr_scheduler if config.get("lr_linear_decay", False) else config["lr"]

    # -------------------------------------------------------------------------
    # Network instance shared by train_step, evals, and checkpoint loading.
    # -------------------------------------------------------------------------
    network = QNetwork(
        action_dim=action_dim,
        hidden_size=config.get("hidden_size", 128),
        num_layers=config.get("num_layers", 2),
        norm_type=config["norm_type"],
        norm_input=config.get("norm_input", False),
    )

    def create_train_state(rng) -> TrainState:
        rng_init_net, rng_levels, rng_reset = jax.random.split(rng, 3)

        # BatchNorm needs a batch dim of at least 1
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
        network_variables = network.init(rng_init_net, init_x, train=False)

        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.radam(learning_rate=lr),
        )

        new_levels = jax.vmap(sample_random_level)(
            jax.random.split(rng_levels, config["num_envs"])
        )
        init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
            jax.random.split(rng_reset, config["num_envs"]),
            new_levels,
            env_params,
        )

        return TrainState.create(
            apply_fn=network.apply,
            params=network_variables["params"],
            batch_stats=network_variables["batch_stats"],
            tx=tx,
            update_count=0,
            timesteps=0,
            grad_steps=0,
            last_obs=init_obs,
            last_env_state=init_env_state,
        )

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        rng, train_state = carry

        # ---- Rollout ----
        def _step_env(carry_inner, _):
            obs, env_state, rng_in = carry_inner
            rng_in, rng_a, rng_s = jax.random.split(rng_in, 3)

            q_vals = network.apply(
                {"params": train_state.params, "batch_stats": train_state.batch_stats},
                obs,
                train=False,
            )
            action_masks, encoded_masks = jax.vmap(get_action_mask)(env_state)

            eps = jnp.full(config["num_envs"], eps_scheduler(train_state.update_count))
            action = jax.vmap(eps_greedy_exploration)(
                jax.random.split(rng_a, config["num_envs"]),
                q_vals,
                eps,
                action_masks,
            )

            next_obs, next_env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(
                jax.random.split(rng_s, config["num_envs"]),
                env_state,
                action,
                env_params,
            )
            # AutoResetWrapper injects a raw PRNGKey under info["rng"]; drop it
            # so it doesn't pollute scalar-metric reductions.
            info.pop("rng", None)

            transition = Transition(
                obs=obs,
                action=action,
                reward=config.get("rew_scale", 1.0) * reward,
                done=done,
                next_obs=next_obs,
                q_val=q_vals,
                encoded_action_mask=encoded_masks,
            )
            return (next_obs, next_env_state, rng_in), (transition, info)

        rng, _rng = jax.random.split(rng)
        (last_obs, last_env_state, _), (transitions, infos) = jax.lax.scan(
            _step_env,
            (train_state.last_obs, train_state.last_env_state, _rng),
            None,
            config["num_steps"],
        )

        # ---- Training metrics from LogWrapper ----
        completed_episodes_mask = infos["returned_episode"]
        num_completed = jnp.maximum(completed_episodes_mask.sum(), 1)
        mean_train_return = (
            infos["returned_episode_returns"] * completed_episodes_mask
        ).sum() / num_completed
        mean_train_length = (
            infos["returned_episode_lengths"] * completed_episodes_mask
        ).sum() / num_completed

        # ---- Bootstrap value (max Q over valid actions at final state) ----
        last_q_full = network.apply(
            {"params": train_state.params, "batch_stats": train_state.batch_stats},
            transitions.next_obs[-1],
            train=False,
        )
        last_action_mask, _ = jax.vmap(get_action_mask)(last_env_state)
        last_q = jnp.max(jnp.where(last_action_mask, last_q_full, -1e9), axis=-1)

        decoded_masks = jax.vmap(jax.vmap(mask_encoder.decode))(
            transitions.encoded_action_mask
        )  # (NUM_STEPS, NUM_ENVS, action_dim)

        # ---- Peng's Q(λ) lambda-return targets ----
        def _get_target(carry_tgt, inp):
            lambda_returns, next_q = carry_tgt
            transition, mask = inp
            target_bootstrap = (
                transition.reward + config["gamma"] * (1 - transition.done) * next_q
            )
            delta = lambda_returns - next_q
            lambda_returns = (
                target_bootstrap + config["gamma"] * config["lambda"] * delta
            )
            lambda_returns = (
                1 - transition.done
            ) * lambda_returns + transition.done * transition.reward
            next_q = jnp.max(jnp.where(mask, transition.q_val, -1e9), axis=-1)
            return (lambda_returns, next_q), lambda_returns

        last_q = last_q * (1 - transitions.done[-1])
        lambda_returns = transitions.reward[-1] + config["gamma"] * last_q
        _, targets = jax.lax.scan(
            _get_target,
            (lambda_returns, last_q),
            (
                jax.tree_util.tree_map(lambda x: x[:-1], transitions),
                decoded_masks[:-1],
            ),
            reverse=True,
        )
        lambda_targets = jnp.concatenate((targets, lambda_returns[jnp.newaxis]))

        # ---- Epoch × minibatch SGD ----
        def _learn_epoch(carry_e, _):
            train_state, rng_e = carry_e

            def _learn_phase(carry_p, minibatch_and_target):
                train_state, rng_p = carry_p
                minibatch, target = minibatch_and_target

                def _loss_fn(params):
                    q_vals, updates = network.apply(
                        {"params": params, "batch_stats": train_state.batch_stats},
                        minibatch.obs,
                        train=True,
                        mutable=["batch_stats"],
                    )
                    chosen = jnp.take_along_axis(
                        q_vals,
                        jnp.expand_dims(minibatch.action, axis=-1),
                        axis=-1,
                    ).squeeze(axis=-1)
                    loss = 0.5 * jnp.square(chosen - target).mean()
                    return loss, (updates, chosen)

                (loss, (updates, qvals)), grads = jax.value_and_grad(
                    _loss_fn, has_aux=True
                )(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(
                    grad_steps=train_state.grad_steps + 1,
                    batch_stats=updates["batch_stats"],
                )
                return (train_state, rng_p), (loss, qvals)

            def preprocess(x, rng_pp):
                x = x.reshape(-1, *x.shape[2:])
                x = jax.random.permutation(rng_pp, x)
                return x.reshape(config["num_minibatches"], -1, *x.shape[1:])

            rng_e, _rng_pp = jax.random.split(rng_e)
            minibatches = jax.tree_util.tree_map(
                lambda x: preprocess(x, _rng_pp), transitions
            )
            targets_mb = jax.tree_util.tree_map(
                lambda x: preprocess(x, _rng_pp), lambda_targets
            )

            (train_state, rng_e), (loss, qvals) = jax.lax.scan(
                _learn_phase, (train_state, rng_e), (minibatches, targets_mb)
            )
            return (train_state, rng_e), (loss, qvals)

        rng, _rng = jax.random.split(rng)
        (train_state, _), (loss, qvals) = jax.lax.scan(
            _learn_epoch, (train_state, _rng), None, config["num_epochs"]
        )

        metrics = {
            "losses": {
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
                "epsilon": eps_scheduler(train_state.update_count),
            },
            "mean_return": mean_train_return,
            "mean_length": mean_train_length,
        }

        train_state = train_state.replace(
            update_count=train_state.update_count + 1,
            timesteps=train_state.timesteps + config["num_steps"] * config["num_envs"],
            last_obs=last_obs,
            last_env_state=last_env_state,
        )
        return (rng, train_state), metrics

    @jax.jit
    def train_step_batch(runner_state):
        """Run eval_freq consecutive train_step calls; return aggregated metrics.
        Signature identical to PPO's train_step_batch."""
        (rng, train_state), metrics = jax.lax.scan(
            train_step, runner_state, None, config["eval_freq"]
        )
        agg_train = {
            "mean_return": metrics["mean_return"].mean(),
            "mean_length": metrics["mean_length"].mean(),
        }
        result = {
            "update_count": train_state.update_count,
            "losses": jax.tree_util.tree_map(lambda x: x.mean(), metrics["losses"]),
            "train_metrics": agg_train,
        }
        return (rng, train_state), result

    def make_evaluate_policy_fn(eval_levels):
        num_levels = jax.tree_util.tree_flatten(eval_levels)[0][0].shape[0]

        def _single_attempt(rng, train_state, eps_val):
            rng, rng_reset = jax.random.split(rng)
            init_obs, init_state = jax.vmap(
                eval_env.reset_to_level, in_axes=(0, 0, None)
            )(jax.random.split(rng_reset, num_levels), eval_levels, env_params)

            def _step(carry, _):
                rng, obs, state, mask, ep_len = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    obs,
                    train=False,
                )
                action_masks, _ = jax.vmap(get_action_mask)(state)

                # Dynamic epsilon applied here
                eps = jnp.full(num_levels, eps_val)
                action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(rng_a, num_levels),
                    q_vals,
                    eps,
                    action_masks,
                )
                next_obs, next_state, reward, done, _ = jax.vmap(
                    eval_env.step, in_axes=(0, 0, 0, None)
                )(jax.random.split(rng_s, num_levels), state, action, env_params)
                next_mask = mask & ~done
                ep_len = ep_len + mask
                return (rng, next_obs, next_state, next_mask, ep_len), reward

            (_, _, _, _, episode_lengths), rewards = jax.lax.scan(
                _step,
                (
                    rng,
                    init_obs,
                    init_state,
                    jnp.ones(num_levels, dtype=bool),
                    jnp.zeros(num_levels, dtype=jnp.int32),
                ),
                None,
                length=env_params.max_steps_in_episode,
            )
            step_mask = (
                jnp.arange(env_params.max_steps_in_episode)[:, None] < episode_lengths
            )
            cum_rewards = (rewards * step_mask).sum(axis=0)
            return cum_rewards, episode_lengths

        @jax.jit
        def evaluate_policy(rng, train_state):
            def run_eval_batch(eval_rng, eps_val):
                # mapped across eval_num_attempts; eps_val is scalar (in_axes=None)
                cum_rewards, episode_lengths = jax.vmap(
                    lambda r: _single_attempt(r, train_state, eps_val)
                )(jax.random.split(eval_rng, config["eval_num_attempts"]))

                eval_solve_rates = jnp.where(cum_rewards > 0, 1.0, 0.0).mean(axis=0)
                eval_returns = cum_rewards.mean(axis=0)
                return {
                    "eval_returns": eval_returns,
                    "eval_solve_rates": eval_solve_rates,
                    "eval_ep_lengths": episode_lengths[0],
                }

            rng_stoch, rng_greedy = jax.random.split(rng)

            # Stochastic Pass
            stoch_eps = config.get("eps_test", 0.05)
            stoch_out = run_eval_batch(rng_stoch, stoch_eps)

            # Greedy Pass
            greedy_out = run_eval_batch(rng_greedy, 0.0)

            # Zip them together with prefixes
            return {
                **{f"stoch_{k}": v for k, v in stoch_out.items()},
                **{f"greedy_{k}": v for k, v in greedy_out.items()},
            }

        return evaluate_policy

    bucket_eval_fns = {
        tag: make_evaluate_policy_fn(levels)
        for tag, (levels, _) in EVAL_BUCKETS.items()
    }

    def eval_checkpoint(og_config):
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))

        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, "config.json")) as f:
                loaded_config = json.load(f)
            checkpoint_manager = ocp.CheckpointManager(
                os.path.join(os.getcwd(), checkpoint_directory, "models"),
                ocp.PyTreeCheckpointer(),
            )
            train_state_og = create_train_state(rng_init)
            step = (
                checkpoint_manager.latest_step()
                if og_config["checkpoint_to_eval"] == -1
                else og_config["checkpoint_to_eval"]
            )
            loaded = checkpoint_manager.restore(step)
            train_state = train_state_og.replace(
                params=loaded["params"],
                batch_stats=loaded.get("batch_stats", train_state_og.batch_stats),
            )
            return train_state, loaded_config

        train_state, _ = load(rng_init, og_config["checkpoint_directory"])

        all_results = {}
        for tag, (levels, level_names) in EVAL_BUCKETS.items():
            fn = make_evaluate_policy_fn(levels)
            out = fn(rng_eval, train_state)

            # Unpacks the dual outputs dynamically
            all_results[tag] = {
                **{k: np.asarray(v) for k, v in out.items()},
                "levels": np.asarray(levels),
                "level_names": level_names,
            }

        save_loc = og_config["checkpoint_directory"].replace("checkpoints", "results")
        os.makedirs(save_loc, exist_ok=True)
        print("Saving evaluation results to:", save_loc)
        np.savez_compressed(
            os.path.join(save_loc, "results.npz"),
            **{
                f"{tag}_{k}": v
                for tag, bucket in all_results.items()
                for k, v in bucket.items()
                if isinstance(v, np.ndarray)
            },
        )
        return all_results

    if config.get("mode") == "eval":
        return eval_checkpoint(config)

    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)
    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    if config.get("checkpoint_save_interval", 0) > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)

    for eval_step in range(int(config["num_updates"] // config["eval_freq"])):
        start_time = time.time()

        # Train
        runner_state, train_metrics = train_step_batch(runner_state)

        # Eval — run each density bucket independently
        rng_eval = jax.random.fold_in(runner_state[0], eval_step)
        eval_bucket_stats = {}
        for tag, eval_fn in bucket_eval_fns.items():
            bucket_metrics = eval_fn(rng_eval, runner_state[1])
            eval_bucket_stats[tag] = {
                **bucket_metrics,
                "level_names": EVAL_BUCKETS[tag][1],
            }

        metrics = {
            **train_metrics,
            "eval_buckets": eval_bucket_stats,
            "time_delta": time.time() - start_time,
        }
        log_eval(metrics, log_file_path=log_file_path)

        if config.get("checkpoint_save_interval", 0) > 0:
            checkpoint_manager.save(
                eval_step,
                runner_state[1],
                args=ocp.args.StandardSave(runner_state[1]),
            )
            checkpoint_manager.wait_until_finished()

    return runner_state[1]


def tune(config=None, project="NASimJax"):
    if config.get("sweep_id") is not None:
        sweep_id = config["sweep_id"]
        print(f"Using existing Sweep ID: {sweep_id}")
    else:
        sweep_configuration = {
            "method": "bayes",
            "name": f"PQN_DR_epsgreedy_MultiEval_Sweep_w_Minibatches_{config['envs']['env_name']}",
            "metric": {
                "name": "solve_rate/mean",
                "goal": "maximize",
            },
            "parameters": {
                "lr": {"values": [1e-4, 2e-4, 3e-4]},
                "num_envs": {"values": [1024]},
                "num_steps": {"values": [32, 64, 128]},
                "hidden_size": {"values": [256, 512]},
                "gamma": {"values": [0.975, 0.99, 0.995]},
                "lambda": {"values": [0.3, 0.5, 0.65, 0.8, 0.9]},
                "eps_start": {"values": [1.0]},
                "eps_finish": {"values": [0.01, 0.03, 0.05, 0.08, 0.1, 0.12]},
                "num_minibatches": {"values": [4, 8, 16]},
                "eps_decay": {"values": [0.8, 0.5, 0.3]},
            },
        }
        sweep_id = wandb.sweep(sweep_configuration, project=project)
        print(f"Generated new Sweep ID: {sweep_id}")

    def sweep_agent():
        try:
            single_run(config, project=project)
        except Exception as e:
            print(f"Run failed with error: {e}")
            if wandb.run is not None:
                wandb.finish(exit_code=1)
            raise
        finally:
            if wandb.run is not None:
                wandb.finish()

    wandb.agent(
        sweep_id,
        function=sweep_agent,
        entity=config["entity"],
        project=config["project"],
        count=config["num_trials"],
    )
    return sweep_id


@hydra.main(version_base=None, config_path="config", config_name="config_ued")
def main(config):
    def lowercase_keys(obj):
        """Recursively lowercase every dict key (matches PPO's convention)."""
        if isinstance(obj, dict):
            return {k.lower(): lowercase_keys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [lowercase_keys(item) for item in obj]
        return obj

    config = OmegaConf.to_container(config)
    config = lowercase_keys(config)

    print("Config:\n", OmegaConf.to_yaml(config))

    if config.get("hyp_tune"):
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
