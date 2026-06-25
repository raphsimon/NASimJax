# Third-Party Notices

This repository contains code derived from external open-source projects.
This file documents the original sources, their licenses, and which files
in this repository derive from them.

## Code derived from gymnax

Source: https://github.com/RobertTLange/gymnax;
License: Apache License 2.0 (see LICENSE-APACHE);
Citation: Lange, R. T. (2022). gymnax: A JAX-based Reinforcement Learning Environment
Library.

Files in this repository derived from gymnax (commit a93274c):

| File in this repo                  | Derived from upstream file          |
|------------------------------------|-------------------------------------|
| nasimjax/envs/environment_base.py  | gymnax/environments/environment.py  |

## Code derived from JaxUED

Source: https://github.com/DramaCow/jaxued;
License: Apache License 2.0 (see LICENSE-APACHE);
Citation: Coward, S., Beukman, M., and Foerster, J. (2024). 
          JaxUED: A simple and useable UED library in Jax. arXiv:2403.13091.

Files in this repository derived from JaxUED (commit 0f8f128):

| File in this repo            | Derived from upstream file     |
|------------------------------|--------------------------------|
| nasimjax/envs/level_sampler.py | src/jaxued/level_sampler.py      |
| nasimjax/envs/wrappers.py | src/jaxued/wrappers.py (AutoReset, AutoReplay) |

## Code derived from PureJaxRL

Source: https://github.com/luchris429/purejaxrl;
License: Apache License 2.0 (see LICENSE-APACHE);
Citation: Lu, C. (2023). PureJaxRL: Really Fast End-to-End Jax RL Implementations.
          Companion paper: Lu et al. (2022), "Discovered Policy Optimisation,"
          NeurIPS.

Files in this repository derived from PureJaxRL (commit 5343613):

| File in this repo            | Derived from upstream file     |
|------------------------------|--------------------------------|
| agents/ppo.py      | purejaxrl/ppo.py               |
| agents/wrappers.py      | purejaxrl/wrappers.py               |
| nasimjax/envs/wrappers.py | purejaxrl/wrappers.py (GymnaxWrapper, LogWrapper) |


## Code derived from NASim

  Source: https://github.com/Jjschwartz/NetworkAttackSimulator
  License: MIT (see LICENSE-NASIM)

  Files in this repository derived from NASim (commit 4f26de3):

| File in this repo                              | Derived from upstream file       |
|------------------------------------------------|----------------------------------|
| nasimjax/scenarios/host.py                     | nasim/scenarios/host.py          |
| nasimjax/scenarios/loader.py                   | nasim/scenarios/loader.py        |
| nasimjax/scenarios/scenario.py                 | nasim/scenarios/scenario.py      |
| nasimjax/scenarios/utils.py                    | nasim/scenarios/utils.py         |
| nasimjax/scenarios/__init__.py                 | nasim/scenarios/__init__.py      |
| nasimjax/scenarios/benchmark/__init__.py       | nasim/scenarios/benchmark/__init__.py |
| nasimjax/scenarios/benchmark/*.yaml            | nasim/scenarios/benchmark/*.yaml |


## Runtime dependencies (no code redistributed)

The following packages are imported but no source code is redistributed:

| Package      | License     | Citation                                  |
|--------------|-------------|-------------------------------------------|
| JAX          | Apache-2.0  | Bradbury et al. (2018)                    |
| Flax         | Apache-2.0  | Heek et al. (2024)                        |