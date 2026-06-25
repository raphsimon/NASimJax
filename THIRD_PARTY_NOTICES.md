# Third-Party Notices

This repository contains code derived from external open-source projects.
This file documents the original sources, their licenses, and which files
in this repository derive from them.

## Code derived from JaxUED

Source: https://github.com/DramaCow/jaxued
License: Apache License 2.0 (see LICENSE-APACHE)
Citation: Coward, S., Beukman, M., and Foerster, J. (2024). 
          JaxUED: A simple and useable UED library in Jax. arXiv:2403.13091.

Files in this repository derived from JaxUED (commit 0f8f128):

| File in this repo            | Derived from upstream file     |
|------------------------------|--------------------------------|
| agents/ppo_dr_masked_multi_eval.py | examples/maze_dr.py      |
| agents/ppo_plr_accel_masked_multi_eval.py | examples/maze_plr.py |


(Add or remove rows as appropriate.)

## Code derived from PureJaxQL

Source: https://github.com/mttga/purejaxql
License: Apache License 2.0 (see LICENSE-APACHE)
Citation: Gallici, M., Fellows, M., Ellis, B., Pou, B., Masmitja, I.,
          Foerster, J. N., and Martin, M. (2025). Simplifying Deep Temporal
          Difference Learning. ICLR 2025.

Files in this repository derived from PureJaxQL (commit bd3f7e2):

| File in this repo                | Derived from upstream file       |
|----------------------------------|----------------------------------|
| agents/pqn_dr_masked_epsgreedy_multi_eval.py | purejaxql/pqn_gymnax.py |

## Code derived from PureJaxRL

Source: https://github.com/luchris429/purejaxrl
License: Apache License 2.0 (see LICENSE-APACHE)
Citation: Lu, C. (2023). PureJaxRL: Really Fast End-to-End Jax RL Implementations.
          Companion paper: Lu et al. (2022), "Discovered Policy Optimisation,"
          NeurIPS.

Files in this repository derived from PureJaxRL (commit 5343613):

| File in this repo            | Derived from upstream file     |
|------------------------------|--------------------------------|
| algorithms/ppo_wandb.py      | purejaxrl/ppo.py               |

## Runtime dependencies (no code redistributed)

The following packages are imported but no source code is redistributed:

| Package      | License     | Citation                                  |
|--------------|-------------|-------------------------------------------|
| NASim        | MIT         | Schwartz & Kurniawatti (2019)             |
| PureJaxRL    | Apache-2.0  | Lu (2023)                                 |
| Gymnax       | Apache-2.0  | Lange (2022)                              |
| JAX          | Apache-2.0  | Bradbury et al. (2018)                    |
| Flax         | Apache-2.0  | Heek et al.                               |