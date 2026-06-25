# Copyright 2023 Chris Lu (PureJaxRL).
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
# Derived from PureJaxRL (https://github.com/luchris429/purejaxrl) at commit
# 5343613, file purejaxrl/ppo.py.
#
# Notable modifications:
#   - Added hyperparameter sweep functionality with W&B
#   - NASimJax environment setup
#   - Parameterized network architecture and training hyperparameters


import os
import time
import copy
import jax
import jax.numpy as jnp
import numpy as np

import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict, Union
from flax.training.train_state import TrainState
import distrax
import gymnax
import wandb
import hydra
from omegaconf import OmegaConf

# Local imports
from agents.wrappers import LogWrapper
from nasimjax.envs import NASimEnvJAX, ProcGenNASimJaxEnv
from nasimjax.envs.wrappers import AugmentedObservationsWrapper, NormalizeRewardWrapper
from nasimjax.scenarios import make_benchmark_scenario
from nasimjax.envs.common import NASimJaxEnvParams


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    jax.debug.print("Number of updates to perform {x}", x=config["NUM_UPDATES"])
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # SETUP ENVIRONMENT
    if "gen" in config["ENV_NAME"].lower():
        train_env_params = NASimJaxEnvParams(
            fully_obs=config["FULLY_OBS"],
            num_hosts=config["ENV_KWARGS"]["num_hosts"],
            num_subnets=config["ENV_KWARGS"]["num_subnets"],
            num_services=config["ENV_KWARGS"]["num_services"],
            num_os=config["ENV_KWARGS"]["num_os"],
            num_processes=config["ENV_KWARGS"]["num_processes"],
            distribute_homogeneous=config["ENV_KWARGS"]["distribute_homogeneous"],
            topology_density=config["ENV_KWARGS"]["topology_density"],
            service_density=config["ENV_KWARGS"]["service_density"],
            process_density=config["ENV_KWARGS"]["process_density"],
            sensitive_density=config["ENV_KWARGS"]["sensitive_density"],
            step_limit=config["ENV_KWARGS"]["step_limit"],
        )
        basic_env = ProcGenNASimJaxEnv(
            key=jax.random.key(config["SEED"]),
            params=train_env_params,
        )
        env_params = basic_env.default_params
        if config["AUG_OBS"]:
            basic_env = AugmentedObservationsWrapper(basic_env)
        if config["NORMALIZE_REWARD"]:
            basic_env = NormalizeRewardWrapper(basic_env)
    elif "nasimjax" in config["ENV_NAME"].lower():
        scenario_name = "-".join(config["ENV_NAME"].split("-")[1:])
        scenario = make_benchmark_scenario(scenario_name.lower())

        # Create pure JAX environment
        basic_env = NASimEnvJAX(scenario, fully_obs=config["FULLY_OBS"])
        env_params = basic_env.default_params
        if config["AUG_OBS"]:
            basic_env = AugmentedObservationsWrapper(basic_env)
        if config["NORMALIZE_REWARD"]:
            basic_env = NormalizeRewardWrapper(basic_env)
    else:
        basic_env, env_params = gymnax.make(config["ENV_NAME"])

    env = LogWrapper(basic_env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).n,
            activation=config["ACTIVATION"],
            config=config,
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ENVS"], (
                    "batch size must be equal to number of steps * number of envs"
                )
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            # Debugging mode
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    episode_lengths = info["returned_episode_lengths"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )  # I think here we multiply the array by a scalar
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}, episode length={episode_lengths[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def single_run(config):
    config = {**config, **config["alg"], **config["envs"]}

    alg_name = config.get("ALG_NAME", "ppo")
    env_name = config["ENV_NAME"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=config.get(
            "NAME",
            f"{config['ALG_NAME']}_{config['ENV_NAME']}_steps_{config['TOTAL_TIMESTEPS']:.0e}",
        ),
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))

    print(f"Took {time.time() - t0:.02f} seconds to complete.")

    # Log metrics to W&B
    metrics = outs["metrics"]

    # Convert JAX arrays to numpy
    import numpy as np

    returns = np.array(metrics["returned_episode_returns"])

    # Log all data points - W&B will create the learning curve automatically
    num_updates = returns.shape[1]
    for step in range(num_updates):
        log_dict = {}

        if config["NUM_SEEDS"] == 1:
            # Single seed: log mean across environments
            step_returns = returns[0, step]  # (num_envs,)
            log_dict["charts/episodic_return"] = step_returns.mean()
        else:
            # Multiple seeds: log each seed separately
            for seed_idx in range(config["NUM_SEEDS"]):
                step_returns = returns[seed_idx, step]  # (num_envs,)
                log_dict[f"charts/episodic_return_seed_{seed_idx}"] = (
                    step_returns.mean()
                )

            # Also log aggregate statistics
            mean_per_seed = returns[:, step].mean(axis=1)  # (num_seeds,)
            log_dict["charts/episodic_return_mean"] = mean_per_seed.mean()
            log_dict["charts/episodic_return_std"] = mean_per_seed.std()

        wandb.log(log_dict, step=step)

    # Log summary statistics
    wandb.run.summary["final_mean_return"] = returns.mean()
    wandb.run.summary["training_time"] = time.time() - t0

    import matplotlib.pyplot as plt

    #    if config["NUM_SEEDS"] == 1:
    #        plt.plot(outs["metrics"]["returned_episode_returns"].mean(-1).reshape(-1))
    #        plt.xlabel("Update Step")
    #        plt.ylabel("Return")
    #        plt.show()
    #    else:
    #        for i in range(config["NUM_SEEDS"]):
    #            plt.plot(
    #                outs["metrics"]["returned_episode_returns"][i].mean(-1).reshape(-1)
    #            )
    #        plt.xlabel("Update Step")
    #        plt.ylabel("Return")
    #        plt.show()

    if config.get("SAVE_PATH", None) is not None:
        from safetensors.flax import save_file
        from flax.traverse_util import flatten_dict
        from datetime import datetime

        def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
            flattened_dict = flatten_dict(params, sep=",")
            save_file(flattened_dict, filename)

        model_state = outs["runner_state"][0]
        metrics = outs["metrics"]  # Extract metrics

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(
            config["SAVE_PATH"],
            env_name,
            alg_name,
            timestamp,
        )
        os.makedirs(save_dir, exist_ok=True)

        # Save config
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f"{alg_name}_{env_name}_seed{config['SEED']}_config.yaml"
            ),
        )

        # Save model parameters for each seed
        for i, rng in enumerate(rngs):
            params = jax.tree_util.tree_map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f"{alg_name}_{env_name}_seed{config['SEED']}_vmap{i}.safetensors",
            )
            save_params(params, save_path)

        # Save metrics using numpy
        if config.get("SAVE_METRICS", False) is not False:
            import numpy as np

            metrics_numpy = jax.tree_util.tree_map(lambda x: np.array(x), metrics)
            metrics_np_path = os.path.join(
                save_dir, f"{alg_name}_{env_name}_seed{config['SEED']}_metrics.npz"
            )
            np.savez(metrics_np_path, **metrics_numpy)

            print(f"Saved model and metrics to {save_dir}")


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {
        **default_config,
        **default_config["alg"],
        **default_config["envs"],
    }
    alg_name = default_config.get("ALG_NAME", "ppo")
    env_name = default_config["ENV_NAME"]

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))

        # Pure JAX extraction
        mask = outs["metrics"]["returned_episode"]
        returns = outs["metrics"]["returned_episode_returns"]

        # Get all valid returns in one operation
        valid_returns = returns[mask]

        # Calculate final performance (last 100 episodes or last half)
        n_episodes = mask.sum()
        window = jnp.minimum(100, n_episodes // 2)
        final_performance = jnp.mean(valid_returns[-window:]) if n_episodes > 0 else 0.0

        # Log all data points - W&B will create the learning curve automatically
        num_updates = returns.shape[1]
        for update in range(0, num_updates, 20):  # Use step to consume less resources
            log_dict = {}

            if config["NUM_SEEDS"] == 1:
                # Single seed: log mean across environments
                step_returns = returns[0, update]  # (num_envs,)
                log_dict["Charts/episodic_return"] = step_returns.mean()
            else:
                # Multiple seeds: log each seed separately
                for seed_idx in range(config["NUM_SEEDS"]):
                    step_returns = returns[seed_idx, update]  # (num_envs,)
                    log_dict[f"Charts/episodic_return_seed_{seed_idx}"] = (
                        step_returns.mean()
                    )

                # Also log aggregate statistics
                mean_per_seed = returns[:, update].mean(axis=1)  # (num_seeds,)
                log_dict["Charts/episodic_return_mean"] = mean_per_seed.mean()
                log_dict["Charts/episodic_return_std"] = mean_per_seed.std()

            wandb.log(log_dict, step=update)

        # Log metrics
        wandb.log(
            {
                "final_performance": float(final_performance),
                "final_std": float(jnp.std(valid_returns[-window:]))
                if n_episodes > 0
                else 0.0,
                "total_episodes": int(n_episodes),
                "total_mean_return": float(jnp.mean(valid_returns))
                if n_episodes > 0
                else 0.0,
                "best_episode": float(jnp.max(valid_returns))
                if n_episodes > 0
                else 0.0,
            }
        )

        return float(final_performance)

    if default_config.get("SWEEP_ID") is None:
        sweep_config = {
            "name": f"{alg_name}_{env_name}",
            "method": "bayes",
            "metric": {
                "name": "final_performance",
                "goal": "maximize",
            },
            "parameters": {
                "LR": {"values": [1e-5, 3e-5, 5e-5, 1e-4, 2e-4]},
            },
        }

        sweep_id = wandb.sweep(
            sweep_config,
            entity=default_config["ENTITY"],
            project=default_config["PROJECT"],
        )
        print(f"Created new sweep: {sweep_id}")
    else:
        sweep_id = default_config["SWEEP_ID"]
        print(f"Continuing existing sweep: {sweep_id}")

    wandb.agent(
        sweep_id,
        wrapped_make_train,
        entity=default_config["ENTITY"],
        project=default_config["PROJECT"],
        count=default_config["NUM_TRIALS"],
    )
    return sweep_id


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
