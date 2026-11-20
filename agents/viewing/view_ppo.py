import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

import gymnax

from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

# Local imports
from agents.ppo_purejaxrl import ActorCritic
from nasimjax.envs import NASimEnvJAX, ProcGenNASimJaxEnv
from nasimjax.envs.action import get_action_from_arrays
from nasimjax.envs.utils import ActionType
from nasimjax.envs.wrappers import AugmentedObservationsWrapper
from nasimjax.scenarios import make_benchmark_scenario
from nasimjax.envs.renderer import Renderer


def main(args):
    # TODO: Provide config in the saved model folder and load it here.
    # with open(os.path.join(args.path, "config.yaml")) as f:
    #    raw_config = yaml.load(f, Loader=yaml.Loader)

    #    config = {}
    #    for key, value in raw_config.items():
    #        if isinstance(value, dict) and "value" in value:
    #            config[key] = value["value"]

    config["NUM_ENVS"] = 1

    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = CheckpointManager(args.path, orbax_checkpointer, options)

    # SETUP ENVIRONMENT
    if "nasimjax-gen" in config["ENV_NAME"].lower():
        print("Creating generated environment")
        # Create pure JAX environment
        basic_env = ProcGenNASimJaxEnv(
            fully_obs=config["FULLY_OBS"],
            num_hosts=config["GEN_PARAMS"]["num_hosts"],
            num_os=config["GEN_PARAMS"]["num_os"],
            num_services=config["GEN_PARAMS"]["num_services"],
            num_processes=config["GEN_PARAMS"]["num_processes"],
        )
        env_params = basic_env.default_params
        if config["AUG_OBS"]:
            print("Wrapping with AggregatedObservationsWrapper")
            basic_env = AugmentedObservationsWrapper(basic_env)
        renderer = Renderer()
    elif "nasimjax" in config["ENV_NAME"].lower():
        scenario_name = config["ENV_NAME"].split("-")[-1]
        print(f"Creating environment from scenario definition: {scenario_name}")
        scenario = make_benchmark_scenario(scenario_name.lower())

        # Create pure JAX environment
        basic_env = NASimEnvJAX(scenario, fully_obs=config["FULLY_OBS"])
        env_params = basic_env.default_params
        if config["AUG_OBS"]:
            print("Wrapping with AggregatedObservationsWrapper")
            basic_env = AugmentedObservationsWrapper(basic_env)
        renderer = Renderer(scenario)
    else:
        basic_env, env_params = gymnax.make(config["ENV_NAME"])

    env = basic_env

    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])

    init_x = jnp.zeros((config["NUM_ENVS"], *env.observation_space(env_params).shape))

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng, __rng = jax.random.split(rng, 3)
    network_params = network.init(_rng, init_x)

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    train_state = checkpoint_manager.restore(
        int(config["TOTAL_TIMESTEPS"]), items=train_state
    )

    obs, env_state = env.reset(key=__rng)
    done = 0
    cum_reward = 0

    renderer.render_state(state=env_state)

    while not done:
        done = np.array([done], dtype=bool)
        obs = jnp.expand_dims(obs, axis=0)

        pi, value = network.apply(train_state.params, obs)
        rng, _rng = jax.random.split(rng)
        action = pi.sample(seed=_rng)[0]

        if action is not None:
            # Get env for action decoding (unwrap from wrappers)
            action_obj = get_action_from_arrays(env.action_array, action)

            # Get values for pretty print.
            os_idx = action_obj.os_id
            os = (
                env.scenario.os[os_idx]
                if os_idx < len(env.scenario.os)
                else f"OS_{os_idx}"
            )
            svc_idx = action_obj.service_id
            svc_name = (
                env.scenario.services[svc_idx]
                if svc_idx < len(env.scenario.services)
                else f"SVC_{svc_idx}"
            )

            print(
                f"Action {action}: target_host_idx={action_obj.target_host_idx}, "
                f"type={ActionType(action_obj.action_type)}, cost={action_obj.cost}, "
                f"os={os}, srv={svc_name}, proc={action_obj.process_id}"
            )

            rng, _rng = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(
                _rng, env_state, action, env_params
            )
            cum_reward += reward
            print(f"Cumulative reward: {cum_reward}")

            if done:
                print("Success!")
            else:
                renderer.render_state(state=env_state, obs_array=obs)

        if not done and args.debug:
            input("Press ENTER to continue...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--debug", action="store_true")

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 128,
        "NUM_STEPS": 64,  # 128 is a bit overkill with 1024 // envs
        "TOTAL_TIMESTEPS": 5e6,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "NASimJax-Small",
        "AUG_OBS": True,
        "GEN_PARAMS": {
            "num_hosts": 30,
            "num_os": 2,
            "num_services": 4,
            "num_processes": 4,
        },
        "FULLY_OBS": False,
        "SEED": 30,
        "NUM_SEEDS": 1,
        "ANNEAL_LR": True,
        "DEBUG": False,
        "SAVE_POLICY": False,
        "USE_OPTIMISTIC_RESETS": False,
        "OPTIMISTIC_RESET_RATIO": 32,
    }

    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)
