# Copyright 2024 Samuel Coward, Michael Beukman, Jakob Foerster (JaxUED authors).
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
# Derived from JaxUED (https://github.com/DramaCow/jaxued) at commit 0f8f128,
# file examples/maze_plr.py.
#
# Notable modifications:
#   - Replaced Maze environment with NASimJax
#   - Added action masking for invalid host/exploit combinations
#   - Modified rollout/update for NASimJax observation structure
#   - Adapted evaluation to run on NASimJax benchmark levels with density buckets

import csv
import json
import os
import time
from datetime import datetime
from enum import IntEnum
from functools import partial
from typing import NamedTuple, Tuple

import chex
import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import core, struct
from flax.linen.initializers import constant, orthogonal
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
from nasimjax.envs.level_sampler import LevelSampler
from nasimjax.envs.mutators import make_level_mutator, MutationType
from nasimjax.envs.utils import compute_max_returns, max_mc, positive_value_loss
from nasimjax.envs.wrappers import (
    NormalizeRewardWrapper,
    AugmentedObservationsWrapper,
    AutoReplayWrapper,
    LogWrapper,
)


class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1
    MUTATE = 2


class MaskEncoder:
    def __init__(self, total_actions: int, chunk_size: int = 31):
        """
        Encoder/decoder for boolean action masks using bit packing.

        Parameters:
            total_actions: Total number of actions in the flat action space
            chunk_size: Number of boolean values packed into each uint32 chunk
        """
        self.total_actions = total_actions
        self.chunk_size = chunk_size
        # Pre-calculate at init time (compile-time constants)
        self.n_chunks = (total_actions + chunk_size - 1) // chunk_size
        self.padded_size = self.n_chunks * chunk_size
        self.powers = 2 ** jnp.arange(chunk_size)

    @jax.named_scope("encode_mask")
    @partial(jax.jit, static_argnames=("self",))
    def encode(self, mask: jnp.ndarray) -> jnp.ndarray:
        # Pad to fixed size (padded_size is compile-time constant)
        padded = jnp.zeros(self.padded_size, dtype=jnp.bool_)
        padded = padded.at[: self.total_actions].set(mask)

        # Reshape and pack
        chunks = padded.reshape(self.n_chunks, self.chunk_size)
        encoded = jnp.sum(chunks * self.powers[None, :], axis=1)
        return encoded.astype(jnp.uint32)

    @jax.named_scope("decode_mask")
    @partial(jax.jit, static_argnames=("self",))
    def decode(self, encoded: jnp.ndarray) -> jnp.ndarray:
        # Vectorized bit extraction
        bit_indices = jnp.arange(self.chunk_size)[None, :]
        chunks = (encoded[:, None] >> bit_indices) & 1
        # Flatten and trim to original size
        flattened = chunks.flatten().astype(bool)
        return flattened[: self.total_actions]


class MaskedCategorical(distrax.Categorical):
    """Categorical distribution with action masking support."""

    def __init__(self, logits=None, probs=None, mask=None):
        if mask is not None:
            # Apply mask: set invalid actions to very negative logits
            if logits is not None:
                masked_logits = jnp.where(mask, logits, -1e8)
                super().__init__(logits=masked_logits)
            elif probs is not None:
                masked_probs = jnp.where(mask, probs, 0.0)
                masked_probs = masked_probs / jnp.sum(
                    masked_probs, axis=-1, keepdims=True
                )
                super().__init__(probs=masked_probs)
        else:
            super().__init__(logits=logits, probs=probs)

        self.mask = mask


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    encoded_action_mask: jnp.ndarray


class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    # === Below is used for logging ===
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)


# region PPO helper functions
def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """This takes in arrays of shape (NUM_STEPS, NUM_ENVS) and returns the advantages and targets.

    Args:
        gamma (float):
        lambd (float):
        last_value (chex.Array):  Shape (NUM_ENVS)
        values (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        rewards (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        dones (chex.Array): Shape (NUM_STEPS, NUM_ENVS)

    Returns:
        Tuple[chex.Array, chex.Array]: advantages, targets; each of shape (NUM_STEPS, NUM_ENVS)
    """

    def compute_gae_at_timestep(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values


def sample_trajectories(
    rng: chex.PRNGKey,
    env: Environment,
    env_params: EnvParams,
    train_state: TrainState,
    init_obs: jnp.ndarray,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
    get_action_mask_fn: callable,
) -> Tuple[
    Tuple[chex.PRNGKey, TrainState, jnp.ndarray, EnvState, chex.Array],
    Transition,
]:
    """This samples trajectories from the environment using the agent specified by the `train_state`.

    Args:

        rng (chex.PRNGKey): Singleton
        env (UnderspecifiedEnv):
        env_params (EnvParams):
        train_state (TrainState): Singleton
        init_obs (jnp.ndarray): The initial observation, shape (NUM_ENVS, ...)
        init_env_state (EnvState): The initial env state (NUM_ENVS, ...)
        num_envs (int): The number of envs that are vmapped over.
        max_episode_length (int): The maximum episode length, i.e., the number of steps to do the rollouts for.
        get_action_mask_fn (callable): A function that takes in an env_state and returns the action mask.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, jnp.ndarray, EnvState, chex.Array],
        Tuple[jnp.ndarray, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]:
        (rng, train_state, hstate, last_obs, last_env_state, last_value), traj, where traj is
        (obs, action, reward, done, log_prob, value, info).
        The first element in the tuple consists of arrays that have shapes (NUM_ENVS, ...)
        (except `rng` and and `train_state` which are singleton).
        The second element in the tuple is of shape (NUM_STEPS, NUM_ENVS, ...), and it contains the trajectory.
    """

    def sample_step(carry, _):
        rng, train_state, obs, env_state = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        action_masks, encoded_mask = jax.vmap(get_action_mask_fn)(env_state)

        pi, value = train_state.apply_fn(train_state.params, obs, action_masks)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        carry = (rng, train_state, next_obs, env_state)
        transition = Transition(
            done, action, value, reward, log_prob, obs, info, encoded_mask
        )
        return carry, transition

    (rng, train_state, last_obs, last_env_state), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            train_state,
            init_obs,
            init_env_state,
        ),
        None,
        length=max_episode_length,
    )

    last_action_masks, _ = jax.vmap(get_action_mask_fn)(last_env_state)
    _, last_value = train_state.apply_fn(
        train_state.params, last_obs, last_action_masks
    )

    return (
        rng,
        train_state,
        last_obs,
        last_env_state,
        last_value,
    ), traj


def evaluate(
    rng: chex.PRNGKey,
    env: Environment,
    env_params: EnvParams,
    train_state: TrainState,
    init_obs: jnp.ndarray,
    init_env_state: EnvState,
    max_episode_length: int,
    get_action_mask_fn: callable,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """This runs the agent on the environment, given an initial state and observation,
    and returns (states, rewards, episode_lengths)

    Args:
        rng (chex.PRNGKey):
        env (UnderspecifiedEnv):
        env_params (EnvParams):
        train_state (TrainState):
        init_obs (jnp.ndarray): Shape (num_levels, )
        init_env_state (EnvState): Shape (num_levels, )
        max_episode_length (int):
        get_action_mask_fn: callable,

    Returns:
        Tuple[chex.Array, chex.Array, chex.Array]: (States, rewards, episode lengths) ((NUM_STEPS, NUM_LEVELS), (NUM_STEPS, NUM_LEVELS), (NUM_LEVELS,)
    """
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]

    def step(carry, _):
        rng, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        action_masks, _ = jax.vmap(get_action_mask_fn)(state)
        pi, _ = train_state.apply_fn(train_state.params, obs, action_masks)
        action = pi.sample(seed=rng_action)

        obs, next_state, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(rng_step, num_levels), state, action, env_params
        )

        next_mask = mask & ~done
        episode_length += mask

        return (rng, obs, next_state, done, next_mask, episode_length), (
            state,
            reward,
        )

    (_, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (
            rng,
            init_obs,
            init_env_state,
            jnp.zeros(num_levels, dtype=bool),
            jnp.ones(num_levels, dtype=bool),
            jnp.zeros(num_levels, dtype=jnp.int32),
        ),
        None,
        length=max_episode_length,
    )

    return states, rewards, episode_lengths


def update_actor_critic(
    rng: chex.PRNGKey,
    train_state: TrainState,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    ent_coef: float,
    vf_coef: float,
    mask_decoder: MaskEncoder,
    update_grad: bool = True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state."""
    traj_batch, advantages, targets = batch

    # FIX: Global Advantage Normalization
    # We do this once over the entire (NUM_STEPS, NUM_ENVS) array
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            mb_traj, mb_advantages, mb_targets = minibatch

            def loss_fn(params, traj, gae, target_vals):
                action_masks = jax.vmap(mask_decoder.decode)(traj.encoded_action_mask)

                # RERUN NETWORK
                pi, value = train_state.apply_fn(params, traj.obs, action_masks)
                log_prob = pi.log_prob(traj.action)

                # CALCULATE VALUE LOSS
                value_pred_clipped = mb_traj.value + (value - mb_traj.value).clip(
                    -clip_eps, clip_eps
                )
                value_losses = jnp.square(value - target_vals)
                value_losses_clipped = jnp.square(value_pred_clipped - target_vals)
                value_loss = (
                    0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                )

                # CALCULATE ACTOR LOSS
                ratio = jnp.exp(log_prob - mb_traj.log_prob)
                loss_actor1 = ratio * gae
                loss_actor2 = (
                    jnp.clip(
                        ratio,
                        1.0 - clip_eps,
                        1.0 + clip_eps,
                    )
                    * gae
                )
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean()
                entropy = pi.entropy().mean()

                total_loss = loss_actor + vf_coef * value_loss - ent_coef * entropy
                return total_loss, (value_loss, loss_actor, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(
                train_state.params, mb_traj, mb_advantages, mb_targets
            )
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)

        batch_size = num_envs * n_steps
        permutation = jax.random.permutation(rng_perm, batch_size)

        _batch = (traj_batch, advantages, targets)
        _batch = jax.tree_util.tree_map(
            lambda x: x.reshape((batch_size,) + x.shape[2:]), _batch
        )
        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0), _batch
        )

        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, [n_minibatch, -1] + list(x.shape[1:])),
            shuffled_batch,
        )

        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)


class ActorCritic(nn.Module):
    action_dim: int
    layer_size: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, action_mask=None):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Shared layers
        shared = nn.Dense(
            self.layer_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        shared = activation(shared)
        shared = nn.Dense(
            self.layer_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(shared)
        shared = activation(shared)

        # Actor head
        actor_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(shared)

        # Create masked categorical distribution
        pi = MaskedCategorical(logits=actor_logits, mask=action_mask)

        # Critic head
        critic = nn.Dense(
            self.layer_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(shared)
        critic = activation(critic)
        critic = nn.Dense(
            self.layer_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


# endregion


# region checkpointing
def setup_checkpointing(
    config: dict, train_state: TrainState, env: Environment, env_params: EnvParams
) -> ocp.CheckpointManager:
    """This takes in the train state and config, and returns an orbax checkpoint manager."""
    overall_save_dir = os.path.join(
        os.getcwd(), "checkpoints", f"{config['run_name']}", str(config["seed"])
    )
    os.makedirs(overall_save_dir, exist_ok=True)

    # save the config
    with open(os.path.join(overall_save_dir, "config.json"), "w+") as f:
        f.write(json.dumps(config.as_dict(), indent=True))

    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, "models"),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config["checkpoint_save_interval"],
            max_to_keep=config["max_number_of_checkpoints"],
        ),
    )
    return checkpoint_manager


# endregion


def train_state_to_log_dict(
    train_state: TrainState, level_sampler: LevelSampler
) -> dict:
    """To prevent the entire (large) train_state to be copied to the CPU when
    doing logging, this function returns all of the important information in
    a dictionary format."""
    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)
    return {
        "log": {
            "level_sampler/size": sampler["size"],
            "level_sampler/episode_count": sampler["episode_count"],
            "level_sampler/max_score": sampler["scores"].max(),
            "level_sampler/weighted_score": (
                sampler["scores"] * level_sampler.level_weights(sampler)
            ).sum(),
            "level_sampler/mean_score": (sampler["scores"] * idx).sum() / s,
        },
        "info": {
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        },
    }


def compute_score(config, dones, values, max_returns, advantages):
    if config["score_function"] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config["score_function"] == "pvl":
        return positive_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")


def single_run(config=None, project="NASimJax"):
    config = {**config, **config["alg"], **config["envs"], **config["ued"]}

    if config["total_timesteps"] is not None:
        config["num_updates"] = config["total_timesteps"] // (
            config["num_envs"] * config["num_steps"]
        )
    # Hard code minibatch size to be 4096
    config["num_minibatches"] = config["num_envs"] * config["num_steps"] // 4096

    tags = [
        config["alg_name"].upper(),
        config["env_name"].upper(),
        f"jax_{jax.__version__}",
        "Masked",
        config["prioritization"],
    ]
    if not config["exploratory_grad_updates"]:
        tags.append("robust")
    if config["use_accel"]:
        tags.append("ACCEL")
    else:
        tags.append("PLR")

    wandb.init(
        config=config,
        project=project,
        tags=tags,
        name=config.get(
            "NAME",
            f"{'ACCEL' if config['use_accel'] else 'PLR'}_{config['alg_name']}_{config['env_name']}",
        ),
        group=config.get("group", None),
    )
    config = wandb.config

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    scratch = os.environ.get("VSC_SCRATCH")
    base_dir = scratch if scratch else os.getcwd()

    # Create results directory
    log_dir = os.path.join(
        base_dir,
        "logs",
        f"{'ACCEL' if config['use_accel'] else 'PLR'}_{config['alg_name']}",
        config["env_name"],
        f"{timestamp}_s{config['seed']}",  # Putting timestamp before seed
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "metrics.csv")

    # wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_env_steps")
    wandb.define_metric("level_sampler/*", step_metric="num_env_steps")
    wandb.define_metric("agent/*", step_metric="num_env_steps")
    wandb.define_metric("return/*", step_metric="num_env_steps")
    wandb.define_metric("eval_ep_lengths/*", step_metric="num_env_steps")
    wandb.define_metric("train/*", step_metric="num_env_steps")

    def log_eval(stats, sampler_log=None, log_file_path=None):
        print(f"Logging update: {stats['update_count']}")

        env_steps = stats["update_count"] * config["num_envs"] * config["num_steps"]
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "total_timesteps": env_steps,
            "sps": env_steps / max(stats["time_delta"], 1e-8),
        }

        # --- per-density-bucket eval metrics ---
        all_solve_rates, all_returns = [], []
        for density_tag, bucket_stats in stats["eval_buckets"].items():
            solve_rates = bucket_stats["eval_solve_rates"]
            returns = bucket_stats["eval_returns"]
            ep_lengths = bucket_stats["eval_ep_lengths"]

            log_dict[f"solve_rate/{density_tag}/mean"] = float(solve_rates.mean())
            log_dict[f"return/{density_tag}/mean"] = float(returns.mean())
            log_dict[f"eval_ep_lengths/{density_tag}/mean"] = float(ep_lengths.mean())

            all_solve_rates.append(solve_rates)
            all_returns.append(returns)

        all_solve_rates_cat = np.concatenate([np.asarray(s) for s in all_solve_rates])
        all_returns_cat = np.concatenate([np.asarray(r) for r in all_returns])
        log_dict["solve_rate/mean"] = float(all_solve_rates_cat.mean())
        log_dict["return/mean"] = float(all_returns_cat.mean())

        # --- training losses ---
        loss, (critic_loss, actor_loss, entropy) = stats["losses"]
        log_dict.update(
            {
                "agent/loss": float(loss),
                "agent/critic_loss": float(critic_loss),
                "agent/actor_loss": float(actor_loss),
                "agent/entropy": float(entropy),
            }
        )

        # --- training returns / lengths (aggregate + per-branch) ---
        tm = stats["train_metrics"]
        log_dict.update(
            {
                "train/mean_return": float(tm["mean_return"]),
                "train/mean_length": float(tm["mean_length"]),
                "train/dr_mean_return": float(tm["dr_mean_return"]),
                "train/replay_mean_return": float(tm["replay_mean_return"]),
                "train/mutation_mean_return": float(tm["mutation_mean_return"]),
            }
        )

        # Per-branch lengths and fractions (emitted only if train_step_batch provides them)
        for branch in ("dr", "replay", "mutation"):
            for suffix in ("mean_length", "fraction"):
                key = f"{branch}_{suffix}"
                if key in tm:
                    log_dict[f"train/{key}"] = float(tm[key])

        # --- level sampler diagnostics ---
        if sampler_log is not None:
            # level_sampler/* scalars (size, max/mean/weighted score, episode_count)
            for k, v in sampler_log.get("log", {}).items():
                log_dict[k] = float(v) if hasattr(v, "__float__") else v

            # cumulative branch update counts -> under level_sampler/ namespace so they
            # already line up with the step-metric definition you declared
            for k, v in sampler_log.get("info", {}).items():
                log_dict[f"level_sampler/{k}"] = int(v) if hasattr(v, "__int__") else v

            # derived fill ratio — uses level_sampler.capacity from the enclosing scope
            size = sampler_log.get("log", {}).get("level_sampler/size")
            if size is not None:
                log_dict["level_sampler/fill_ratio"] = (
                    float(size) / level_sampler.capacity
                )

        wandb.log(log_dict)

        # --- CSV mirror ---
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

    # Setup the environment
    # Train env params are used for the level generator
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
        sensitive_density=float(
            config["env_kwargs"]["sensitive_density"]
        ),  # Using float() to avoid possible int/float issues
        scan_cost=config["env_kwargs"]["scan_cost"],
        exploit_cost=config["env_kwargs"]["exploit_cost"],
        privesc_cost=config["env_kwargs"]["privesc_cost"],
        fully_obs=config["fully_obs"],
        step_limit=config["env_kwargs"]["step_limit"],
    )
    env = GeneratedNASimEnvJAX(
        key=jax.random.PRNGKey(0),
        params=train_env_params,
    )
    if config["normalize_reward"]:
        env = NormalizeRewardWrapper(env)
    if config["aug_obs"]:
        env = AugmentedObservationsWrapper(env)
    sample_random_level = make_level_generator(train_env_params)

    # -------------------------------------------------------------------------
    # Build one eval-level set per topology density.
    #
    # Config example (yaml):
    #   eval_topology_densities: [0.06, 0.115, 0.15, 0.195, 0.24]
    #
    # If the key is absent we fall back to the single density already present in
    # eval_env_kwargs so the original behaviour is preserved.
    # -------------------------------------------------------------------------
    base_eval_env_kwargs = config["eval_env_kwargs"]
    eval_topology_densities: list = config.get(
        "eval_topology_densities",
        [base_eval_env_kwargs["topology_density"]],  # backward-compatible default
    )

    # Build a dict: density_tag -> (EVAL_LEVELS, EVAL_LEVEL_NAMES)
    # The tag is used as a namespace in W&B, e.g. "topo_0.06"
    EVAL_BUCKETS: dict[str, tuple] = {}
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

    env = AutoReplayWrapper(env)
    env = LogWrapper(env)
    env_params = env.default_params

    # Use the max number of edits we allow in the sweep. This value is used as a compile-time constant. If num_edits < max_num_edits, we just fill the scan with NO_OP
    MUTATION_SETS = {
        "toggles": (
            MutationType.TOGGLE_SERVICE,
            MutationType.TOGGLE_PROCESS,
            MutationType.CHANGE_OS,
        ),
        "swap_edges": (MutationType.SWAP_EDGES,),
        "move_sensitive": (MutationType.MOVE_SENSITIVE,),
        "combined": (
            MutationType.TOGGLE_SERVICE,
            MutationType.TOGGLE_PROCESS,
            MutationType.CHANGE_OS,
            MutationType.SWAP_EDGES,
            MutationType.MOVE_SENSITIVE,
        ),
    }

    try:
        mutation_set = MUTATION_SETS[config["accel_mutation"]]
    except KeyError:
        raise ValueError(
            f"Unknown mutation type: {config['accel_mutation']}. "
            f"Valid options: {list(MUTATION_SETS)}"
        )

    mutate_level = make_level_mutator(
        max_num_edits=10,
        num_hosts=train_env_params.num_hosts,
        num_subnets=train_env_params.num_subnets,
        num_services=train_env_params.num_services,
        num_os=train_env_params.num_os,
        num_processes=train_env_params.num_processes,
        mutation_types=mutation_set,
    )

    mask_encoder = MaskEncoder(total_actions=env.action_space(env_params).n)

    def get_inner_state(state):
        """Recursively unwrap the environment state until we hit the base NASimJax state."""
        s = state
        while not hasattr(s, "hosts"):
            s = s.env_state
        return s

    def get_action_mask(state):
        # Use the helper to safely unwrap regardless of which env passed the state
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

        encoded_action_mask = mask_encoder.encode(action_mask)
        return action_mask, encoded_action_mask

    # And the level sampler
    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={
            "temperature": config["temperature"],
            "k": config["topk_k"],
        },
        duplicate_check=config["buffer_duplicate_check"],
    )

    @jax.jit
    @jax.named_scope("creaete_train_state")
    def create_train_state(rng) -> TrainState:
        # Creates the train state
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["num_minibatches"] * config["update_epochs"]))
                / config["num_updates"]
            )
            return config["lr"] * frac

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        init_mask = jnp.ones(env.action_space(env_params).n, dtype=bool)

        network = ActorCritic(
            env.action_space(env_params).n,
            layer_size=config["layer_size"],
            activation="tanh",
        )
        network_params = network.init(_rng, init_x, init_mask)

        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )

        pholder_level = sample_random_level(jax.random.PRNGKey(0))
        sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
        pholder_level_batch = jax.tree_util.tree_map(
            lambda x: jnp.array([x]).repeat(config["num_envs"], axis=0),
            pholder_level,
        )
        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            update_state=0,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            mutation_last_level_batch=pholder_level_batch,
        )

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        """
        This is the main training loop. It basically calls either `on_new_levels`, `on_replay_levels`, or `on_mutate_levels` at every step.
        """

        def on_new_levels(rng: chex.PRNGKey, train_state: TrainState):
            """Samples new levels and updates the agent."""
            sampler = train_state.sampler

            # Reset
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            new_levels = jax.vmap(sample_random_level)(
                jax.random.split(rng_levels, config["num_envs"])
            )
            init_obs, init_env_state = jax.vmap(
                env.reset_to_level, in_axes=(0, 0, None)
            )(
                jax.random.split(rng_reset, config["num_envs"]),
                new_levels,
                env_params,
            )
            # Rollout
            (
                (rng, train_state, last_obs, last_env_state, last_value),
                traj_batch,
            ) = sample_trajectories(
                rng,
                env,
                env_params,
                train_state,
                init_obs,
                init_env_state,
                config["num_envs"],
                config["num_steps"],
                get_action_mask,
            )

            # Process training metrics from LogWrapper
            completed_episodes_mask = traj_batch.info["returned_episode"]
            num_completed_episodes = jnp.maximum(completed_episodes_mask.sum(), 1)
            mean_train_return = (
                traj_batch.info["returned_episode_returns"] * completed_episodes_mask
            ).sum() / num_completed_episodes
            mean_train_length = (
                traj_batch.info["returned_episode_lengths"] * completed_episodes_mask
            ).sum() / num_completed_episodes

            advantages, targets = compute_gae(
                config["gamma"],
                config["gae_lambda"],
                last_value,
                traj_batch.value,
                traj_batch.reward,
                traj_batch.done,
            )
            max_returns = compute_max_returns(traj_batch.done, traj_batch.reward)
            scores = compute_score(
                config, traj_batch.done, traj_batch.value, max_returns, advantages
            )
            sampler, _ = level_sampler.insert_batch(
                sampler, new_levels, scores, {"max_return": max_returns}
            )

            # Update
            (rng, train_state), losses = update_actor_critic(
                rng,
                train_state,
                (traj_batch, advantages, targets),
                config["num_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["update_epochs"],
                config["clip_eps"],
                config["ent_coef"],
                config["vf_coef"],
                mask_encoder,
                update_grad=config["exploratory_grad_updates"],
            )

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_return": mean_train_return,
                "mean_length": mean_train_length,
                "update_state": UpdateState.DR,
            }

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_dr_updates=train_state.num_dr_updates + 1,
                dr_last_level_batch=new_levels,
            )
            return (rng, train_state), metrics

        def on_replay_levels(rng: chex.PRNGKey, train_state: TrainState):
            """Samples levels from the buffer and updates the policy."""
            sampler = train_state.sampler

            # Collect trajectories on replay levels
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(
                sampler, rng_levels, config["num_envs"]
            )
            init_obs, init_env_state = jax.vmap(
                env.reset_to_level, in_axes=(0, 0, None)
            )(jax.random.split(rng_reset, config["num_envs"]), levels, env_params)

            (
                (rng, train_state, last_obs, last_env_state, last_value),
                traj_batch,
            ) = sample_trajectories(
                rng,
                env,
                env_params,
                train_state,
                init_obs,
                init_env_state,
                config["num_envs"],
                config["num_steps"],
                get_action_mask,
            )

            # Process training metrics from LogWrapper
            completed_episodes_mask = traj_batch.info["returned_episode"]
            num_completed_episodes = jnp.maximum(completed_episodes_mask.sum(), 1)
            mean_train_return = (
                traj_batch.info["returned_episode_returns"] * completed_episodes_mask
            ).sum() / num_completed_episodes
            mean_train_length = (
                traj_batch.info["returned_episode_lengths"] * completed_episodes_mask
            ).sum() / num_completed_episodes

            advantages, targets = compute_gae(
                config["gamma"],
                config["gae_lambda"],
                last_value,
                traj_batch.value,
                traj_batch.reward,
                traj_batch.done,
            )
            max_returns = jnp.maximum(
                level_sampler.get_levels_extra(sampler, level_inds)["max_return"],
                compute_max_returns(traj_batch.done, traj_batch.reward),
            )
            scores = compute_score(
                config, traj_batch.done, traj_batch.value, max_returns, advantages
            )
            sampler = level_sampler.update_batch(
                sampler, level_inds, scores, {"max_return": max_returns}
            )

            # Update the policy
            (rng, train_state), losses = update_actor_critic(
                rng,
                train_state,
                (traj_batch, advantages, targets),
                config["num_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["update_epochs"],
                config["clip_eps"],
                config["ent_coef"],
                config["vf_coef"],
                mask_encoder,
                update_grad=True,
            )

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_return": mean_train_return,
                "mean_length": mean_train_length,
                "update_state": UpdateState.REPLAY,
            }

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.REPLAY,
                num_replay_updates=train_state.num_replay_updates + 1,
                replay_last_level_batch=levels,
            )
            return (rng, train_state), metrics

        def on_mutate_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
            This mutates the previous batch of replay levels and potentially adds them to the level buffer.
            This also updates the policy iff `config["exploratory_grad_updates"]` is True.
            """
            sampler = train_state.sampler
            rng, rng_mutate, rng_reset = jax.random.split(rng, 3)

            # mutate
            parent_levels = train_state.replay_last_level_batch
            child_levels = jax.vmap(mutate_level, (0, 0, None))(
                jax.random.split(rng_mutate, config["num_envs"]),
                parent_levels,
                config["num_edits"],
            )
            init_obs, init_env_state = jax.vmap(
                env.reset_to_level, in_axes=(0, 0, None)
            )(
                jax.random.split(rng_reset, config["num_envs"]),
                child_levels,
                env_params,
            )

            # rollout
            (
                (rng, train_state, last_obs, last_env_state, last_value),
                traj_batch,
            ) = sample_trajectories(
                rng,
                env,
                env_params,
                train_state,
                init_obs,
                init_env_state,
                config["num_envs"],
                config["num_steps"],
                get_action_mask,
            )

            # Process training metrics from LogWrapper
            completed_episodes_mask = traj_batch.info["returned_episode"]
            num_completed_episodes = jnp.maximum(completed_episodes_mask.sum(), 1)
            mean_train_return = (
                traj_batch.info["returned_episode_returns"] * completed_episodes_mask
            ).sum() / num_completed_episodes
            mean_train_length = (
                traj_batch.info["returned_episode_lengths"] * completed_episodes_mask
            ).sum() / num_completed_episodes

            advantages, targets = compute_gae(
                config["gamma"],
                config["gae_lambda"],
                last_value,
                traj_batch.value,
                traj_batch.reward,
                traj_batch.done,
            )
            max_returns = compute_max_returns(traj_batch.done, traj_batch.reward)
            scores = compute_score(
                config, traj_batch.done, traj_batch.value, max_returns, advantages
            )
            sampler, _ = level_sampler.insert_batch(
                sampler, child_levels, scores, {"max_return": max_returns}
            )

            # Update: train_state only modified if exploratory_grad_updates is on
            (rng, train_state), losses = update_actor_critic(
                rng,
                train_state,
                (traj_batch, advantages, targets),
                config["num_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["update_epochs"],
                config["clip_eps"],
                config["ent_coef"],
                config["vf_coef"],
                mask_encoder,
                update_grad=config["exploratory_grad_updates"],
            )

            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_return": mean_train_return,
                "mean_length": mean_train_length,
                "update_state": UpdateState.MUTATE,
            }

            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,  # Don't use MUTATE here, because it breaks the branching logic.
                num_mutation_updates=train_state.num_mutation_updates + 1,
                mutation_last_level_batch=child_levels,
            )
            return (rng, train_state), metrics

        rng, train_state = carry
        rng, rng_replay = jax.random.split(rng)

        # The train step makes a decision on which branch to take, either on_new, on_replay or on_mutate.
        # on_mutate is only called if the replay branch has been taken before (as it uses `train_state.update_state`).
        if config["use_accel"]:
            s = train_state.update_state
            branch = (1 - s) * level_sampler.sample_replay_decision(
                train_state.sampler, rng_replay
            ) + 2 * s
        else:
            branch = level_sampler.sample_replay_decision(
                train_state.sampler, rng_replay
            ).astype(int)

        return jax.lax.switch(
            branch,
            [
                on_new_levels,
                on_replay_levels,
                on_mutate_levels,
            ],
            rng,
            train_state,
        )

    @jax.jit
    def train_step_batch(runner_state):
        """Runs training and periodic evaluation."""
        # Train
        (rng, train_state), metrics = jax.lax.scan(
            train_step, runner_state, None, config["eval_freq"]
        )

        is_dr = metrics["update_state"] == UpdateState.DR
        is_replay = metrics["update_state"] == UpdateState.REPLAY
        is_mutate = metrics["update_state"] == UpdateState.MUTATE
        dr_returns = metrics["mean_return"] * is_dr
        replay_returns = metrics["mean_return"] * is_replay
        mutate_returns = metrics["mean_return"] * is_mutate

        aggregated_train_metrics = {
            "mean_return": metrics["mean_return"].mean(),
            "mean_length": metrics["mean_length"].mean(),
            "dr_mean_return": dr_returns.sum() / jnp.maximum(is_dr.sum(), 1),
            "replay_mean_return": replay_returns.sum()
            / jnp.maximum(is_replay.sum(), 1),
            "mutation_mean_return": mutate_returns.sum()
            / jnp.maximum(is_mutate.sum(), 1),
        }

        update_count = (
            train_state.num_dr_updates
            + train_state.num_replay_updates
            + train_state.num_mutation_updates
        )

        return (rng, train_state), {
            "update_count": update_count,
            "losses": jax.tree_util.tree_map(lambda x: x.mean(), metrics["losses"]),
            "train_metrics": aggregated_train_metrics,
        }

    def eval_fn(rng: chex.PRNGKey, train_state: TrainState, eval_levels):
        rng, rng_reset = jax.random.split(rng)
        num_levels = len(jax.tree_util.tree_flatten(eval_levels)[0][0])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(
            jax.random.split(rng_reset, num_levels), eval_levels, env_params
        )
        states, rewards, episode_lengths = evaluate(
            rng,
            eval_env,
            env_params,
            train_state,
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
            get_action_mask,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths

    def make_evaluate_policy_fn(eval_levels):
        @jax.jit
        def evaluate_policy(rng, train_state):
            states, cum_rewards, episode_lengths = jax.vmap(
                lambda r: eval_fn(r, train_state, eval_levels)
            )(jax.random.split(rng, config["eval_num_attempts"]))
            eval_solve_rates = jnp.where(cum_rewards > 0, 1.0, 0.0).mean(axis=0)
            eval_returns = cum_rewards.mean(axis=0)
            _, episode_lengths = jax.tree_util.tree_map(
                lambda x: x[0], (states, episode_lengths)
            )
            return {
                "eval_returns": eval_returns,
                "eval_solve_rates": eval_solve_rates,
                "eval_ep_lengths": episode_lengths,
            }

        return evaluate_policy

    bucket_eval_fns = {
        tag: make_evaluate_policy_fn(levels)
        for tag, (levels, _) in EVAL_BUCKETS.items()
    }

    # Set up the train states
    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)

    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)

    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        start_time = time.time()

        runner_state, train_metrics = train_step_batch(runner_state)

        rng_eval = jax.random.fold_in(runner_state[0], eval_step)
        eval_bucket_stats = {}
        for tag, eval_fn_jit in bucket_eval_fns.items():
            eval_bucket_stats[tag] = eval_fn_jit(rng_eval, runner_state[1])

        metrics = {
            **train_metrics,
            "eval_buckets": eval_bucket_stats,
            "time_delta": time.time() - start_time,
        }
        log_eval(
            metrics,
            train_state_to_log_dict(runner_state[1], level_sampler),
            log_file_path=log_file_path,
        )

        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(
                eval_step, args=ocp.args.StandardSave(runner_state[1])
            )
            checkpoint_manager.wait_until_finished()
    return runner_state[1]


def tune(config=None, project="NASimJax"):
    """
    Executes a Hyperparameter Sweep using WandB for PLR parameters.
    """

    entity = config["entity"]

    if config.get("sweep_id") is not None:
        sweep_id = config["sweep_id"]
        print(f"Using existing Sweep ID: {sweep_id}")
    else:
        # 1. Define the Sweep Configuration
        # We use flat keys here because single_run flattens the config dictionaries
        # (config['ued'] -> config) before usage.
        sweep_configuration = {
            "method": "bayes",  # Bayesian optimization is efficient for numeric params
            "name": f"ACCEL_tuning_full_mutations_{config['envs']['env_name']}",
            "metric": {
                "name": "solve_rate/mean",  # The metric to optimize
                "goal": "maximize",
            },
            "parameters": {
                # Student PPO (keep similar to DR/PLR-tuned values to isolate UED effect)
                "lr": {"values": [2e-4, 3e-4, 5e-4]},
                "num_steps": {"values": [64, 128]},
                "layer_size": {"values": [512]},
                "vf_coef": {"values": [0.5]},
                "ent_coef": {"values": [0.005, 0.01]},
                "clip_eps": {"values": [0.1, 0.2, 0.3]},
                "gae_lambda": {"values": [0.8, 0.95]},
                "gamma": {"values": [0.99]},
                "max_grad_norm": {"values": [0.5]},
                # ACCEL-specific
                "prioritization": {"values": ["rank"]},
                "replay_prob": {"values": [0.5, 0.7, 0.8, 0.9]},
                "staleness_coeff": {"values": [0.1, 0.3, 0.5, 0.7]},
                "temperature": {"values": [0.1, 0.3, 1.0, 2.0]},
                "level_buffer_capacity": {"values": [1000, 4000, 10000]},
                "minimum_fill_ratio": {"values": [0.3, 0.5]},
                "score_function": {"values": ["MaxMC", "pvl"]},
                "num_edits": {"values": [3, 5, 10]},  # increased
                "exploratory_grad_updates": {"values": [True, False]},  # PLR vs PLR⊥
            },
        }

        # 2. Initialize the Sweep
        # This registers the sweep with the WandB server and gets an ID
        sweep_id = wandb.sweep(sweep_configuration, project=project, entity=entity)
        print(f"Generated new Sweep ID: {sweep_id}")

    def sweep_agent():
        try:
            single_run(config)
        except Exception as e:
            print(f"Run failed with error: {e}")
            if wandb.run is not None:
                wandb.finish(exit_code=1)
            raise  # let the agent see the failure
        finally:
            wandb.finish()

    wandb.agent(
        sweep_id,
        function=sweep_agent,
        entity=entity,
        project=project,
        count=config.get("NUM_TRIALS", 150),
    )
    return sweep_id


@hydra.main(version_base=None, config_path="config", config_name="config_ued")
def main(config):
    def lowercase_keys(obj):
        """Recursively convert all dictionary keys to lowercase"""
        if isinstance(obj, dict):
            return {k.lower(): lowercase_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [lowercase_keys(item) for item in obj]
        else:
            return obj

    config = OmegaConf.to_container(config)
    config = lowercase_keys(config)

    print("Config:\n", OmegaConf.to_yaml(config))

    if config["hyp_tune"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
