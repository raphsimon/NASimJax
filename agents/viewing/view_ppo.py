"""Roll out a policy trained with `agents/ppo.py` and render every step.

Example:
    python -m agents.viewing.view_ppo --path policies/NASimJax-small/ppo/<timestamp>
"""

import argparse

import jax

from agents.ppo import ActorCritic, make_env
from agents.viewing.common import load_run, make_renderer, rollout


def main(args):
    config, params = load_run(args.path, args.vmap)
    env, env_params = make_env(config)
    renderer = make_renderer(env, config)
    network = ActorCritic(
        env.action_space(env_params).n,
        activation=config["ACTIVATION"],
        config=config,
    )

    key = jax.random.PRNGKey(args.seed)
    for episode in range(args.episodes):
        if args.episodes > 1:
            print(f"\n######## Episode {episode + 1}/{args.episodes} ########")
        key, episode_key = jax.random.split(key)
        rollout(
            env,
            env_params,
            network,
            params,
            renderer,
            episode_key,
            debug=args.debug,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", required=True, help="Run dir with config + params.")
    parser.add_argument("--vmap", type=int, default=0, help="Which seed to load.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--debug", action="store_true", help="Disable JIT and pause after each step."
    )
    args = parser.parse_args()

    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)
