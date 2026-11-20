# Copyright (c) 2026 The NASimJax Authors.
# Licensed under the MIT License. See LICENSE in the project root.
"""Helpers shared by the policy viewing scripts in this folder.

A "run directory" is what `agents/ppo.py` / `agents/ppo_masked.py` write to
`SAVE_PATH/<env>/<alg>/<timestamp>/`: a `*_config.yaml` and one
`*_vmap{i}.safetensors` per vmapped seed.
"""

import glob
import os

import jax
import jax.numpy as jnp
from flax.traverse_util import unflatten_dict
from omegaconf import OmegaConf
from safetensors.flax import load_file

from agents.viewing.renderer import Renderer
from nasimjax.envs.action import get_action_from_arrays
from nasimjax.envs.utils import ActionType


def load_run(path, vmap=0):
    """Load the config and the params of seed `vmap` from a run directory."""
    config_files = sorted(glob.glob(os.path.join(path, "*_config.yaml")))
    if not config_files:
        raise FileNotFoundError(f"No *_config.yaml found in {path}")
    param_files = glob.glob(os.path.join(path, f"*_vmap{vmap}.safetensors"))
    if not param_files:
        raise FileNotFoundError(f"No *_vmap{vmap}.safetensors found in {path}")

    config = OmegaConf.to_container(OmegaConf.load(config_files[0]), resolve=True)
    params = unflatten_dict(load_file(param_files[0]), sep=",")
    return config, params


def make_renderer(env, config):
    """Renderer matching the env: scenario names for legacy envs, plain ids otherwise."""
    scenario = getattr(env, "scenario", None)
    return Renderer(scenario=scenario, augmented=bool(config["AUG_OBS"]))


def describe_action(env, action):
    """One-line, human readable description of a flat action index."""
    a = get_action_from_arrays(env.action_array, action)
    scenario = getattr(env, "scenario", None)

    def name(names_attr, idx, prefix):
        names = getattr(scenario, names_attr, None)
        idx = int(idx)
        if names is not None and idx < len(names):
            return names[idx]
        return f"{prefix}_{idx}"

    return (
        f"target_host={int(a.target_host_idx)}, "
        f"type={ActionType(int(a.action_type)).name}, cost={float(a.cost)}, "
        f"os={name('os', a.os_id, 'OS')}, "
        f"srv={name('services', a.service_id, 'SVC')}, "
        f"proc={name('processes', a.process_id, 'PROC')}"
    )


def rollout(
    env,
    env_params,
    network,
    params,
    renderer,
    key,
    mask_fn=None,
    debug=False,
):
    """Roll out one episode, printing and rendering every step.

    Parameters
    ----------
    mask_fn : callable, optional
        ``state -> bool[num_actions]``. If given, the policy is evaluated with
        this action mask (as in `agents/ppo_masked.py`) and the sampled action is
        checked against it.
    debug : bool
        Pause after every step and, with a mask, warn when the unmasked policy
        would have picked an invalid action.

    Returns
    -------
    (episode_return, episode_length, goal_reached)
    """
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key, env_params)
    renderer.render_state(state=state)

    episode_return, steps, done = 0.0, 0, False
    while not done:
        if mask_fn is not None:
            mask = mask_fn(state)
            pi, value = network.apply(params, obs, mask)
            if debug:
                pi_unmasked, _ = network.apply(params, obs, None)
                if not bool(mask[jnp.argmax(pi_unmasked.probs)]):
                    print("WARNING: unmasked policy would pick an invalid action")
        else:
            pi, value = network.apply(params, obs)

        key, action_key, step_key = jax.random.split(key, 3)
        action = pi.sample(seed=action_key)
        if mask_fn is not None:
            assert bool(mask[action]), f"Sampled masked-out action {int(action)}"

        print(
            f"Step {steps}: action {int(action)} - {describe_action(env, action)}, "
            f"value={float(value):.3f}"
        )

        # NB: `step` auto-resets, so when `done` the returned state/obs already
        # belong to the next episode and must not be rendered.
        obs, state, reward, done, _ = env.step(step_key, state, action, env_params)
        episode_return += float(reward)
        steps += 1
        print(f"  Reward: {float(reward):.3f}, cumulative: {episode_return:.3f}")

        if not done:
            renderer.render_observation_from_array(obs_array=obs, state=state)
            renderer.render_state(state=state)
            if debug:
                input("Press ENTER to continue...")

    # The info dict is empty, so tell goal and step limit apart by episode length.
    goal_reached = steps < env_params.step_limit
    print(
        f"\n=== Episode complete: return={episode_return:.3f}, length={steps}, "
        f"{'goal reached' if goal_reached else 'step limit reached'} ==="
    )
    return episode_return, steps, goal_reached
