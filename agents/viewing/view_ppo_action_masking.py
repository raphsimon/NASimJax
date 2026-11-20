import os
import sys
import glob
import argparse

import jax
import jax.numpy as jnp
import optax

import gymnax

from flax.training.train_state import TrainState
from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict
from typing import Dict, Union
from omegaconf import OmegaConf

# Local imports - IMPORT ALL NECESSARY COMPONENTS
from agents.ppo_action_masking import ActorCritic, HostMaskEncoder
from nasimjax.envs import NASimEnvJAX, ProcGenNASimJaxEnv
from nasimjax.envs.wrappers import AugmentedObservationsWrapper
from nasimjax.envs.action import get_action_from_arrays
from nasimjax.envs.utils import ActionType
from nasimjax.scenarios import make_benchmark_scenario
from nasimjax.envs.renderer import Renderer


def test_action_probabilities(
    train_state, network, obs, env_state, get_action_mask, env
):
    """Test whether the policy learned to avoid invalid actions."""

    # Get the true action mask
    action_mask, _ = get_action_mask(env_state, env.action_space().n)

    # Get policy predictions WITHOUT masking
    pi_unmasked, _ = network.apply(train_state.params, obs, None)  # No mask
    probs_unmasked = pi_unmasked.probs

    # Get policy predictions WITH masking
    pi_masked, _ = network.apply(train_state.params, obs, action_mask)
    probs_masked = pi_masked.probs

    # print("=== ACTION PROBABILITY ANALYSIS ===")
    # print(f"Valid actions (mask=True): {jnp.sum(action_mask)}")
    # print(f"Invalid actions (mask=False): {jnp.sum(~action_mask)}")

    # Analyze probabilities for invalid actions
    invalid_action_probs = probs_unmasked[~action_mask]
    valid_action_probs = probs_unmasked[action_mask]

    # print("\nWithout masking:")
    # print(f"  Max prob on invalid action: {jnp.max(invalid_action_probs):.6f}")
    # print(f"  Mean prob on invalid actions: {jnp.mean(invalid_action_probs):.6f}")
    # print(f"  Sum of invalid action probs: {jnp.sum(invalid_action_probs):.6f}")
    # print(f"  Max prob on valid action: {jnp.max(valid_action_probs):.6f}")

    # print("\nWith masking:")
    # print(f"  Max prob on valid action: {jnp.max(probs_masked[action_mask]):.6f}")

    # Check if policy would select invalid action
    selected_action_unmasked = jnp.argmax(probs_unmasked)
    is_invalid_selected = ~action_mask[selected_action_unmasked]

    # print("\nGreedy action selection:")
    # print(f"  Without mask - selected action: {selected_action_unmasked}")
    # print(f"  Is selected action invalid? {is_invalid_selected}")

    return is_invalid_selected


def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    """Save parameters to a safetensors file."""
    flattened_dict = flatten_dict(params, sep=",")
    save_file(flattened_dict, filename)


def load_params(filename: Union[str, os.PathLike]) -> Dict:
    """Load parameters from a safetensors file."""
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


def main(config, params, args):
    """Main evaluation loop.

    Args:
        config: Configuration dictionary from YAML
        params: Loaded model parameters from safetensors
        args: Command line arguments
    """

    # SETUP ENVIRONMENT
    if "gen" in config["ENV_NAME"].lower():
        # Create pure JAX environment
        basic_env = ProcGenNASimJaxEnv(
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
        env_params = basic_env.default_params
        if config["AUG_OBS"]:
            basic_env = AugmentedObservationsWrapper(basic_env)
    elif "nasimjax" in config["ENV_NAME"].lower():
        scenario_name = config["ENV_NAME"].split("-")[-1]
        scenario = make_benchmark_scenario(scenario_name.lower())
        renderer = Renderer(scenario=scenario)

        # Create pure JAX environment
        basic_env = NASimEnvJAX(scenario, fully_obs=config["FULLY_OBS"])
        env_params = basic_env.default_params
        if config["AUG_OBS"]:
            basic_env = AugmentedObservationsWrapper(basic_env)
    else:
        basic_env, env_params = gymnax.make(config["ENV_NAME"])

    env = basic_env

    # CREATE MASK ENCODER (same as training)
    mask_encoder_decoder = HostMaskEncoder(n_hosts=env_params.num_hosts)
    actions_per_host = int(env.action_space().n / env_params.num_hosts)

    def get_action_mask(env_state, num_actions: int):
        """
        Get action mask for the current state. For NASimJax, we use the
        simple condition that a hosts needs to be discoverd and reachable.

        Parameters:
            env_state: type: LogWrapper: Contains NASimEnvState by accessing
                env_state.env_state
            num_actions: Number of actions in the environment
        """
        # We use env_state.env_state because the state we care about is
        # wrapped into the LogWrapper
        mask = jnp.logical_and(env_state.hosts.reachable, env_state.hosts.discovered)

        # Encode the host mask to store it in the transition
        encoded_host_mask = mask_encoder_decoder.encode(mask)

        # Expand every value of the vector to be equal to the number of
        # actions per host
        return jnp.repeat(
            mask, actions_per_host, total_repeat_length=num_actions
        ), encoded_host_mask

    # INITIALIZE NETWORK WITH ACTION MASK
    network = ActorCritic(
        env.action_space().n, activation=config["ACTIVATION"], config=config
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rng, _rng, __rng = jax.random.split(rng, 3)

    # Create optimizer (even though we won't train, TrainState needs it)
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )

    # Create TrainState with LOADED params instead of initialized ones
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,  # Use the loaded params here!
        tx=tx,
    )

    # RESET ENVIRONMENT
    obs, env_state = env.reset(key=__rng, params=env_params)
    done = False

    episode_reward = 0
    episode_length = 0

    if renderer:
        renderer.render_state(state=env_state)

    print("\n=== Starting Episode Evaluation ===")

    while not done:
        # GET ACTION MASK FOR CURRENT STATE
        action_mask, _ = get_action_mask(env_state, env.action_space().n)

        if args.debug:
            # Show which actions would have been picked without masking
            is_invalid = test_action_probabilities(
                train_state, network, obs, env_state, get_action_mask, env
            )
            if is_invalid:
                print("WARNING: Policy would select invalid action without masking!")

        # APPLY NETWORK WITH ACTION MASK
        pi, value = network.apply(train_state.params, obs, action_mask)

        rng, _rng = jax.random.split(rng)
        action = pi.sample(seed=_rng)

        if action is not None:
            # Get the basic_env for action decoding (unwrap from wrappers)
            action_obj = get_action_from_arrays(env.action_array, action)

            # Get values for pretty print
            os_idx = action_obj.os_id
            os = (
                env.scenario.os[os_idx]
                if hasattr(env, "scenario") and os_idx < len(env.scenario.os)
                else f"OS_{os_idx}"
            )
            svc_idx = action_obj.service_id
            svc_name = (
                env.scenario.services[svc_idx]
                if hasattr(env, "scenario") and svc_idx < len(env.scenario.services)
                else f"SVC_{svc_idx}"
            )

            print(
                f"Step {episode_length}: Action {action} - "
                f"target_host={action_obj.target_host_idx}, "
                f"type={ActionType(action_obj.action_type).name}, "
                f"cost={action_obj.cost}, os={os}, srv={svc_name}, "
                f"proc={action_obj.process_id}, value={value:.3f}"
            )

            rng, _rng = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(
                _rng, env_state, action, env_params
            )

            episode_reward += reward
            episode_length += 1

            print(f"  Reward: {reward:.3f}, Cumulative: {episode_reward:.3f}")

            if done:
                print("\n=== Episode Complete ===")
                print(f"Total Reward: {episode_reward:.3f}")
                print(f"Episode Length: {episode_length}")
                if "episode_return" in info:
                    print(f"Episode Return (from info): {info['episode_return']}")
                if "episode_length" in info:
                    print(f"Episode Length (from info): {info['episode_length']}")
                print("Success!" if reward > 0 else "Episode terminated")
            elif renderer:
                renderer.render_state(state=env_state, obs_array=obs)
                renderer.render_observation_from_array(state=env_state, obs_array=obs)

        if not done and args.debug:
            input("Press ENTER to continue...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL policy using safetensors format"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the folder containing the run config (.yaml) and the safetensors file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with step-by-step execution",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to evaluate (default: 1)",
    )

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    # Find files
    config_files = glob.glob(os.path.join(args.path, "*.yaml"))
    safetensor_files = glob.glob(os.path.join(args.path, "*.safetensors"))

    if not config_files:
        raise FileNotFoundError(f"No .yaml config found in {args.path}")
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors file found in {args.path}")

    # If multiple files, use the first or let user know
    if len(config_files) > 1:
        print(f"Multiple config files found, using: {config_files[0]}")
    if len(safetensor_files) > 1:
        print(f"Multiple safetensor files found, using: {safetensor_files[0]}")

    config_path = config_files[0]
    safetensor_path = safetensor_files[0]

    # Load files
    print(f"\nLoading config from: {config_path}")
    config = OmegaConf.load(config_path)

    print(f"Loading parameters from: {safetensor_path}")
    params = load_params(safetensor_path)

    print("\nConfig:")
    print(OmegaConf.to_yaml(config))

    print("\nParameter structure:")

    def print_param_structure(params, indent=0):
        for key, value in params.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_param_structure(value, indent + 1)
            else:
                shape = value.shape if hasattr(value, "shape") else type(value)
                print("  " * indent + f"{key}: {shape}")

    print_param_structure(params)

    # Run evaluation
    print("\n" + "=" * 50)
    for episode_num in range(args.episodes):
        if args.episodes > 1:
            print(f"\nEpisode {episode_num + 1}/{args.episodes}")
            print("-" * 30)

        if args.debug:
            with jax.disable_jit():
                main(config, params, args)
        else:
            main(config, params, args)
