# 🛡️ NASimJax

NASimJax is a [JAX](https://github.com/google/jax)-native research framework for studying generalization in reinforcement-learning–based **penetration testing**. It is a full JAX reimplementation of the Network Attack Simulator (NASim) that formulates pentesting as a Contextual POMDP, ships a parametric network generator producing structurally diverse and **guaranteed-solvable** scenarios, and reaches up to **100× higher environment throughput** than the original CPU-bound simulator.

NASimJax conforms to the [gymnax](https://github.com/RobertTLange/gymnax) interface and is built to drop into the JAX RL ecosystem, including [PureJaxRL](https://github.com/luchris429/purejaxrl) and [JaxUED](https://github.com/DramaCow/jaxued).

> **Scope of this repository.** This repo provides the **NASimJax environment**, the network generation pipeline, and its documentation, together with a **minimal PPO example** so you can confirm the environment is wired up and start building on it. It is *not* the full experiment-reproduction pipeline. Algorithm implementations (Domain Randomization, PLR, PLR⊥) and the code to reproduce the paper's figures will accompany a **later release**. See [Roadmap](#-roadmap).

This code accompanies the paper *NASimJax: GPU-Accelerated Policy Learning Framework for Penetration Testing* (see [Citation](#-citation)).

---

## ✨ Highlights

- **GPU-native pentesting environment.** End-to-end JIT compilation and `vmap`-based vectorization across thousands of parallel networks. Reaches ~1.6M steps/s at 4096 workers on a single entry-level GPU (NVIDIA RTX A4000), a ~100× speed-up over the original NASim.
- **Contextual POMDP formulation.** Each episode is conditioned on a network instance sampled from a configurable distribution, turning what was historically a fixed-scenario problem into a distributional one suitable for studying zero-shot transfer.
- **Configurable network generator.** Topology density, host count, service/process density, and sensitive-host density are exposed as parameters. Feasibility constraints guarantee every generated network is solvable.
- **Compatible with the JAX RL ecosystem.** Drop-in `gymnax` API, ready to use with PureJaxRL, JaxUED, and other JAX-native frameworks.
- **Designed as a research abstraction, not a fixed benchmark.** The generator can scale complexity and difficulty so the environment stays challenging as algorithms improve, and discourages overfitting to narrow scenarios.

---

## 📜 Basic Usage

NASimJax conforms to the gymnax interface. An environment is built from a
`NASimJaxEnvParams` dataclass and the `ProcGenNASimJaxEnv` class:

```python
import jax
from nasimjax.envs.common import NASimJaxEnvParams
from nasimjax.envs.generated_environment import ProcGenNASimJaxEnv

# 26-host scenario from Table 1 (cf. agents/config/envs/26-hosts-gen.yaml)
params = NASimJaxEnvParams(
    num_hosts=26,
    num_subnets=10,
    num_services=3,
    num_processes=3,
    num_os=2,
    topology_density=0.12,
    service_density=0.7,
    process_density=0.7,
    sensitive_density=0.15,
    distribute_homogeneous=True,
    step_limit=300,
)

# The PRNGKey passed to the constructor only seeds the static scaffold
# (action-space layout, observation shape). A fresh network is generated
# on every reset by the built-in level sampler.
env = ProcGenNASimJaxEnv(key=jax.random.PRNGKey(0), params=params)
env_params = env.default_params

rng = jax.random.PRNGKey(0)
rng, rng_reset, rng_action, rng_step = jax.random.split(rng, 4)

# Reset samples a fresh network from the configured distribution
obs, state = env.reset(rng_reset, env_params)

# Sample a random (unmasked) action. The environment exposes the
# information needed to construct an action mask; see the minimal
# example for how masking is applied in practice.
action = env.action_space(env_params).sample(rng_action)

# Step. env.step auto-resets the per-env state when `done` is True, sampling
# a new network from the same distribution.
obs, state, reward, done, info = env.step(rng_step, state, action, env_params)
```

For training- and evaluation-time wrappers (`AutoResetWrapper` for explicit
level control, `LogWrapper`, `NormalizeRewardWrapper`,
`AugmentedObservationsWrapper`) see `nasimjax/envs/wrappers.py`.

A self-contained PPO training loop with action masking — enough to verify the
environment end to end and to serve as a starting point for your own agents —
lives in `agents/ppo_masked.py`. An unmasked variant for the legacy NASim
benchmark scenarios lives in `agents/ppo.py`.

---

## ⬇️ Installation

NASimJax is distributed via PyPI, but the `agents/` examples are not part of 
the distribution. Cloning the repository is required to successfully run them.
It is a GPU-accelerated framework, so a CUDA-capable NVIDIA GPU is assumed. 

Install from PyPI:
```bash
pip install nasimjax
```

Install from source:
```bash
# We recommend Python 3.12.4
git clone https://github.com/raphsimon/NASimJax.git
cd NASimJax
python3 -m venv .venv && source .venv/bin/activate  # optional, but highly recommended
pip install -e .
```

### Verify the install

The minimal PPO example doubles as a smoke test. Run it for a short budget to
confirm the environment, generator, and an end-to-end training loop are wired
up correctly before building on top of NASimJax:

```bash
# Short PPO with Action Masking run on the 16-host reference config (~1e7 steps)
python -m agents.ppo_masked +envs=16-hosts-gen +alg=ppo_masked alg.TOTAL_TIMESTEPS=1e7
```
For scenarios ported over from NASim, run PPO without action masking:
```bash
# Short PPO run on the backward-compatible "small" benchmark from NASim (~1e7 steps)
python -m agents.ppo +envs=small +alg=ppo alg.TOTAL_TIMESTEPS=1e7
```
**Attention:** *Masking is currently not compatible with these old scenarios, as the action space is specific to every defined scenario.*

---

## ⚙️ Configurable Generator

The network generation pipeline exposes the following parameters (see Section 4.5 of the paper):

| Parameter | Symbol | Description |
|---|---|---|
| `num_hosts` | $N_h$ | Total host count (fixed for static memory layout) |
| `num_subnets` | $N_s$ | Number of subnets, including Internet and DMZ |
| `topology_density` | $t_d$ | Probability that two subnets are connected |
| `service_density` | $\text{svc}_d$ | Per-host probability of a vulnerable service |
| `process_density` | $\text{proc}_d$ | Per-host probability of a vulnerable process |
| `sensitive_density` | $s_d$ | Per-host probability of being a sensitive target |

Three reference configurations (16, 26, 40 hosts) used throughout the paper are provided in `agents/config/envs/`. These are illustrative rather than normative — researchers are encouraged to define their own configurations rather than treating these as canonical benchmarks.

---

## 📚 Documentation

The documentation source lives in `docs/`. API reference pages are
auto-generated from the source docstrings via
[mkdocstrings](https://mkdocstrings.github.io/). The doc tooling is an
optional extra — install it with:

```bash
pip install -e ".[docs]"
```

Build the static site into `site/`:

```bash
mkdocs build         # writes the rendered HTML to ./site/
```

Then serve it as static files:

```bash
python -m http.server -d site 8000
# open http://localhost:8000
```

Or skip the build step and use the live-reloading dev server while editing:

```bash
mkdocs serve         # live preview at http://localhost:8000
```

---

## 🔪 Gotchas

### Static host count

`num_hosts` is fixed at environment creation to allow JIT-compiled static memory layouts. Networks of *different* sizes cannot be batched in a single `vmap`. To vary effective problem size during training, vary `topology_density` instead — unreachable subnets are deactivated for the episode, producing a natural distribution over active host counts at fixed `num_hosts` (cf. Figure 3).

### Reward scaling

Because procedural generation produces a variable number of sensitive hosts, raw episodic returns vary widely across contexts and are biased toward larger networks, which can destabilize value estimation. NASimJax scales rewards by the theoretical maximum (`Ns * Vh`) by default so the learning signal reflects structural difficulty rather than network size. This also matters for any downstream regret-based method (e.g. PLR): without scaling, regret estimates conflate network size with learning potential. See Section 5.1.3 of the paper.

### Action masking

The environment exposes the information needed to mask actions targeting undiscovered or unreachable hosts, and exploits/privilege escalations incompatible with a host's OS–service or OS–process combination. Actions invalid only due to missing privilege levels are intentionally *left unmasked* — this signal is recoverable from observations, and leaving it in lets the agent learn to chain privilege escalations. The minimal example shows how the mask is constructed and applied.

---

## 🗺️ Roadmap

This release focuses on the environment as a reusable artifact. Planned additions:

- Full algorithm implementations used in the paper (Domain Randomization, PLR, PLR⊥), adapted from JaxUED and PureJaxRL.
- Experiment runners and per-density evaluation harnesses to reproduce the paper's figures and tables.

---

## 📄 License

NASimJax (the environment and generation pipeline) is released under the MIT License — see [LICENSE](LICENSE).

The minimal PPO example in `agents/` is a derivative work of PureJaxRL and retains its original Apache-2.0 license — see [LICENSE-APACHE](LICENSE-APACHE). Per-file attribution is documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The `nasimjax/scenarios/` subpackage is an MIT-licensed port of the upstream NASim scenarios module; its original copyright notice is preserved in [LICENSE-NASIM](LICENSE-NASIM). Per-file attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## 📚 Citation

If you use NASimJax in your research, please cite:

```bibtex
@article{simon2026nasimjax,
      title={NASimJax: A GPU-Accelerated Policy Learning Framework for Penetration Testing}, 
      author={Raphael Simon and José Carrasquel and Wim Mees and Pieter Libin},
      year={2026},
      eprint={2603.19864},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.19864}, 
}
```
