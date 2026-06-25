# Copyright (c) 2026 The NASimJax Authors.
# Licensed under the MIT License. See LICENSE in the project root.
"""Level mutation operators for ACCEL-based curriculum learning in NASimJax.

This module implements level mutation functions that produce small, composable
edits to generated network scenarios. These mutations are designed for use with
the ACCEL (Adversarially Compounding Complexity by Editing Levels) algorithm
from [Parker-Holder et al., 2023](https://arxiv.org/abs/2203.01302), integrated
via the [JaxUED](https://github.com/DramaCow/jaxued) library.

The mutations operate on the [`Level`][nasimjax.envs.network_generator.Level]
dataclass and are designed to:

- Preserve **solvability** of the network scenario after each edit.
- Vary **smoothly** in regret, so that small edits produce levels near the
  frontier of the agent's current capabilities.
- Be **composable**: multiple mutations can be chained via `jax.lax.scan`.

## Mutation Types

The following atomic mutations are supported:

| ID | Name                | Target             | Solvability Risk                  |
|----|---------------------|--------------------|-----------------------------------|
| 0  | `NO_OP`             | —                  | None                              |
| 1  | `SWAP_EDGES`     | Adjacency matrix   | Medium (may disconnect subnets)   |
| 2  | `TOGGLE_SERVICE`    | Host services      | Low (repaired post-hoc)           |
| 3  | `TOGGLE_PROCESS`    | Host processes     | Low (repaired post-hoc)           |
| 4  | `MOVE_SENSITIVE`    | Sensitivity flags  | Low (constrained to reachable hosts) |
| 5  | `CHANGE_OS`         | Host OS assignment | Low (repaired post-hoc)           |

## Usage

```python
from nasimjax.envs.network_generator import Level, make_level_generator
from nasimjax.envs.mutators import make_level_mutator

env_params = ...  # NASimJaxEnvParams instance
sample_level = make_level_generator(env_params)
mutate_level = make_level_mutator(
    max_num_edits=10,
    num_hosts=env_params.num_hosts,
    num_subnets=env_params.num_subnets,
    num_services=env_params.num_services,
    num_os=env_params.num_os,
    num_processes=env_params.num_processes,
)

key = jax.random.PRNGKey(0)
level = sample_level(key)
mutated_level = mutate_level(key, level, num_edits=5)
```

## Integration with JaxUED / ACCEL

The returned mutator callable has the signature
`(PRNGKey, Level, int) -> Level`, which is the interface expected by
[`jaxued`'s ACCEL implementation](https://github.com/DramaCow/jaxued).
The ACCEL training loop calls this function after each replay step to
produce candidate child levels, which are then scored by estimated regret
(positive value loss or MaxMAC) and inserted into the level replay buffer
if they meet the curation threshold.

References:
    Parker-Holder, J., Jiang, M., Dennis, M., et al. (2023).
    "Evolving Curricula with Regret-Based Environment Design."
    *arXiv preprint arXiv:2203.01302*.
"""

from enum import IntEnum
from typing import Callable, Iterable

import chex
import jax
import jax.numpy as jnp

from nasimjax.envs.host_vector_batched import HostVectorBatched
from nasimjax.envs.network_generator import (
    Level,
    get_connected_subnets,
    ensure_attack_path,
)


class MutationType(IntEnum):
    """Enumeration of atomic mutation operators.

    Each value corresponds to an index used in `jax.lax.switch` to
    dispatch the appropriate mutation function during the scan over
    edit steps.

    Attributes:
        NO_OP: Identity mutation; leaves the level unchanged.
        SWAP_EDGES: Swaps a single directed edge between two
            internal subnets in the adjacency matrix.
        TOGGLE_SERVICE: Toggles a single service bit on a random host.
        TOGGLE_PROCESS: Toggles a single process bit on a random host.
        MOVE_SENSITIVE: Moves the sensitive designation from one host
            to another reachable, non-attacker host.
        CHANGE_OS: Re-assigns the operating system of a random host.
    """

    NO_OP = 0
    TOGGLE_SERVICE = 1
    TOGGLE_PROCESS = 2
    CHANGE_OS = 3
    SWAP_EDGES = 4
    MOVE_SENSITIVE = 5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# This function is only useful after swapping/flipping some edges or moving sensitive hosts.
# Therefore we comment it out, as we currently just look at simple toggles.
def _recompute_level_metadata(
    hosts: HostVectorBatched,
    topology: jnp.ndarray,
) -> tuple:
    """Recompute derived level fields after mutations.

    After one or more mutations have been applied to hosts or topology,
    this function recalculates:

    - `num_active_hosts`: based on subnet reachability from subnet 0.
    - `is_flat_topo`: whether the full adjacency matrix is all-ones.
    - `num_sensitive_hosts`: sum of the sensitivity vector.

    Args:
        hosts: The (potentially mutated) host-vector batch.
        topology: The subnet adjacency matrix of shape
            `(num_subnets, num_subnets)`.

    Returns:
        A tuple `(num_active_hosts, is_flat_topo, num_sensitive_hosts)`.
    """
    adj = topology[:, :, 0] if topology.ndim == 3 else topology
    connected = get_connected_subnets(adj)

    # Map subnet reachability to per-host reachability.
    host_subnet_idx = jnp.argmax(hosts.subnet_address, axis=1)
    host_reachable = connected[host_subnet_idx]

    # Internet (host 0) and DMZ hosts are always active.
    dmz_mask = hosts.subnet_address[:, 1].astype(jnp.uint8)
    internet_mask = jnp.zeros_like(dmz_mask).at[0].set(1)
    active_mask = host_reachable | dmz_mask | internet_mask

    num_active_hosts = jnp.sum(active_mask)
    is_flat_topo = jnp.all(adj == 1)
    num_sensitive_hosts = jnp.sum(hosts.sensitive)

    return num_active_hosts, is_flat_topo, num_sensitive_hosts


# ---------------------------------------------------------------------------
# Atomic mutation operators
# ---------------------------------------------------------------------------


def _swap_edges(
    rng: chex.PRNGKey,
    level: Level,
    num_subnets: int,
) -> Level:
    """Swap/flip a single directed edge between two internal subnets.

    Selects two subnets uniformly at random from the internal subnets
    (indices ≥ 2, excluding Internet and DMZ) and toggles the directed
    edge between them in the adjacency matrix.  The Internet ↔ DMZ
    connectivity and self-loops on the diagonal are never modified.

    After the flip, subnet reachability is **not** recomputed here;
    that is deferred to the post-edit repair pass in
    [`_repair_and_finalize`][nasimjax.envs.level_mutator._repair_and_finalize].

    Args:
        rng: JAX PRNG key.
        level: The current level to mutate.
        num_subnets: Total number of subnets (compile-time constant).

    Returns:
        A new `Level` with one topology edge flipped.
    """
    rng_i, rng_j = jax.random.split(rng)
    topology = level.topology

    # Extract adjacency slice.
    adj = topology[:, :, 0]

    # Sample two distinct internal subnet indices (≥ 2).
    i = jax.random.randint(rng_i, (), minval=2, maxval=num_subnets)
    j = jax.random.randint(rng_j, (), minval=2, maxval=num_subnets)

    # Flip the edge (only when i != j to avoid touching the diagonal).
    flipped_val = 1 - adj[i, j]
    new_adj = jnp.where(i != j, adj.at[i, j].set(flipped_val), adj)

    new_topology = topology.at[:, :, 0].set(new_adj)
    return level.replace(topology=new_topology)


def _toggle_service(
    rng: chex.PRNGKey,
    level: Level,
    num_hosts: int,
    num_services: int,
) -> Level:
    """Toggle a single service bit on a random non-attacker host.

    Selects a host index uniformly from `[1, num_hosts)` (excluding the
    attacker at index 0) and a service index uniformly from
    `[0, num_services)`, then flips the corresponding bit.

    If this removes the last service from a sensitive host, the
    post-edit repair pass ([`_ensure_attack_path`]) will restore it.

    Args:
        rng: JAX PRNG key.
        level: The current level to mutate.
        num_hosts: Total number of hosts (compile-time constant).
        num_services: Number of available services (compile-time
            constant).

    Returns:
        A new `Level` with one service bit toggled.
    """
    rng_h, rng_s = jax.random.split(rng)
    host_idx = jax.random.randint(rng_h, (), minval=1, maxval=num_hosts)
    svc_idx = jax.random.randint(rng_s, (), minval=0, maxval=num_services)

    current = level.hosts.services[host_idx, svc_idx]
    new_services = level.hosts.services.at[host_idx, svc_idx].set(1 - current)
    new_hosts = level.hosts.replace(services=new_services)
    return level.replace(hosts=new_hosts)


def _toggle_process(
    rng: chex.PRNGKey,
    level: Level,
    num_hosts: int,
    num_processes: int,
) -> Level:
    """Toggle a single process bit on a random non-attacker host.

    Analogous to [`_toggle_service`][nasimjax.envs.level_mutator._toggle_service]
    but operates on the process vector.

    Args:
        rng: JAX PRNG key.
        level: The current level to mutate.
        num_hosts: Total number of hosts (compile-time constant).
        num_processes: Number of available processes (compile-time
            constant).

    Returns:
        A new `Level` with one process bit toggled.
    """
    rng_h, rng_p = jax.random.split(rng)
    host_idx = jax.random.randint(rng_h, (), minval=1, maxval=num_hosts)
    proc_idx = jax.random.randint(rng_p, (), minval=0, maxval=num_processes)

    current = level.hosts.processes[host_idx, proc_idx]
    new_processes = level.hosts.processes.at[host_idx, proc_idx].set(1 - current)
    new_hosts = level.hosts.replace(processes=new_processes)
    return level.replace(hosts=new_hosts)


def _move_sensitive(
    rng: chex.PRNGKey,
    level: Level,
    num_hosts: int,
    num_subnets: int,
) -> Level:
    """Move a sensitive designation from one host to another.

    This mutation:

    1. Selects a currently-sensitive host and removes its sensitivity.
    2. Selects a candidate host that is (a) not the attacker (index 0),
       (b) not in the Internet or DMZ subnet (subnet index ≥ 2), and
       (c) in a subnet reachable from the attacker, and marks it
       sensitive.

    If there are no currently-sensitive hosts, the mutation degenerates
    to adding a new sensitive host.  The post-edit repair pass ensures
    that the new sensitive host has the required services and processes.

    Args:
        rng: JAX PRNG key.
        level: The current level to mutate.
        num_hosts: Total number of hosts (compile-time constant).
        num_subnets: Total number of subnets (compile-time constant).

    Returns:
        A new `Level` with one sensitive host moved.
    """
    rng_remove, rng_add = jax.random.split(rng)
    sensitive = level.hosts.sensitive
    adj = level.topology[:, :, 0]

    # --- Select a sensitive host to un-mark ---
    # Weight by current sensitivity; tie-break with uniform noise.
    remove_weights = sensitive.astype(jnp.float32)
    # If no sensitive hosts exist, fallback: uniform over non-attacker.
    remove_weights = jnp.where(
        remove_weights.sum() > 0,
        remove_weights,
        jnp.ones(num_hosts).at[0].set(0.0),
    )
    remove_weights = remove_weights * jax.random.uniform(rng_remove, (num_hosts,))
    remove_idx = jnp.argmax(remove_weights)

    # Remove sensitivity.
    new_sensitive = sensitive.at[remove_idx].set(0)

    # --- Select a reachable, internal host to mark sensitive ---
    connected = get_connected_subnets(adj)
    # Exclude Internet (subnet 0) and DMZ (subnet 1).
    internal_connected = connected.at[:2].set(0)
    host_subnet_idx = jnp.argmax(level.hosts.subnet_address, axis=1)
    host_reachable = internal_connected[host_subnet_idx]

    # Candidate mask: reachable, not the attacker, not already sensitive.
    candidate_mask = host_reachable.astype(jnp.float32) * (
        1.0 - new_sensitive.astype(jnp.float32)
    )
    candidate_mask = candidate_mask.at[0].set(0.0)  # Exclude attacker.

    # Weighted random selection via Gumbel-max trick.
    add_weights = candidate_mask * jax.random.uniform(rng_add, (num_hosts,))
    add_idx = jnp.argmax(add_weights)

    # Only apply the addition if at least one candidate exists.
    has_candidate = candidate_mask.sum() > 0
    new_sensitive = jnp.where(
        has_candidate,
        new_sensitive.at[add_idx].set(1),
        new_sensitive,
    )

    new_hosts = level.hosts.replace(sensitive=new_sensitive)
    return level.replace(hosts=new_hosts)


def _change_os(
    rng: chex.PRNGKey,
    level: Level,
    num_hosts: int,
    num_os: int,
) -> Level:
    """Change the operating system of a random non-attacker host.

    Selects a host uniformly from `[1, num_hosts)` and assigns a new
    OS sampled uniformly from `[0, num_os)`.  The OS is stored as a
    one-hot vector.

    Changing the OS alters which exploit and privilege-escalation
    actions are valid for the target host.  The post-edit repair pass
    will ensure that sensitive hosts retain at least one compatible
    service–OS and process–OS pair.

    Args:
        rng: JAX PRNG key.
        level: The current level to mutate.
        num_hosts: Total number of hosts (compile-time constant).
        num_os: Number of available operating systems (compile-time
            constant).

    Returns:
        A new `Level` with one host's OS changed.
    """
    rng_h, rng_os = jax.random.split(rng)
    host_idx = jax.random.randint(rng_h, (), minval=1, maxval=num_hosts)
    new_os_idx = jax.random.randint(rng_os, (), minval=0, maxval=num_os)

    new_os_vec = jax.nn.one_hot(new_os_idx, num_classes=num_os, dtype=jnp.uint8)
    new_os = level.hosts.os.at[host_idx].set(new_os_vec)
    new_hosts = level.hosts.replace(os=new_os)
    return level.replace(hosts=new_hosts)


def _prune_and_rescue_sensitives(
    rng: chex.PRNGKey,
    hosts: HostVectorBatched,
    topology: jnp.ndarray,
    num_hosts: int,
) -> HostVectorBatched:
    """Repair the sensitive-host set after topology edits.

    Two-stage procedure:

    1. **Prune**: remove the sensitive flag from hosts whose subnet is no
       longer reachable from the attacker (subnet 0), *provided at least
       one sensitive host remains reachable after pruning*.
    2. **Rescue**: if pruning would leave zero reachable sensitive hosts
       (or there were none to begin with), pick a random reachable
       internal host and mark it sensitive, clearing all others.

    This preserves the solvability invariant that the goal
    (root on every sensitive host) is satisfiable.

    Args:
        rng: JAX PRNG key.
        hosts: Host batch after mutations.
        topology: 3D traffic-rules tensor; slice ``[:, :, 0]`` is adjacency.
        num_hosts: Compile-time constant.

    Returns:
        Host batch with a sensitivity pattern guaranteed to be satisfiable.
    """
    adj = topology[:, :, 0]
    connected = get_connected_subnets(adj)
    # Internal subnets only; attacker (subnet 0) and DMZ (subnet 1) never
    # host sensitive machines under the secure-topology generator.
    internal_connected = connected.at[:2].set(0)
    host_subnet_idx = jnp.argmax(hosts.subnet_address, axis=1)
    host_reachable = internal_connected[host_subnet_idx].astype(jnp.bool_)

    sensitive_bool = hosts.sensitive.astype(jnp.bool_)
    pruned_sensitive = sensitive_bool & host_reachable  # keep only reachable
    any_reachable_sensitive = jnp.any(pruned_sensitive)

    # Rescue candidates: reachable, internal, not the attacker.
    candidate_mask = host_reachable.astype(jnp.float32)
    candidate_mask = candidate_mask.at[0].set(0.0)
    has_candidate = candidate_mask.sum() > 0

    # Gumbel-max-style random pick among candidates.
    noise = jax.random.uniform(rng, (num_hosts,))
    rescue_idx = jnp.argmax(candidate_mask * noise)

    # Build the rescue sensitivity vector: a single 1 at rescue_idx,
    # only if we actually have a candidate.
    rescue_sensitive = jnp.zeros(num_hosts, dtype=jnp.bool_)
    rescue_sensitive = jnp.where(
        has_candidate,
        rescue_sensitive.at[rescue_idx].set(True),
        sensitive_bool,  # pathological fallback: keep original (no-op)
    )

    new_sensitive = jnp.where(
        any_reachable_sensitive,
        pruned_sensitive,  # normal case: pruning sufficed
        rescue_sensitive,  # rescue case: nothing reachable, promote one
    ).astype(hosts.sensitive.dtype)

    return hosts.replace(sensitive=new_sensitive)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_level_mutator(
    max_num_edits: int,
    num_hosts: int,
    num_subnets: int,
    num_services: int,
    num_os: int,
    num_processes: int,
    mutation_types: Iterable[MutationType] = (
        MutationType.TOGGLE_SERVICE,
        MutationType.TOGGLE_PROCESS,
        MutationType.CHANGE_OS,
        MutationType.SWAP_EDGES,
        MutationType.MOVE_SENSITIVE,
    ),
) -> Callable[[chex.PRNGKey, Level, int], Level]:
    """Create a level mutator function for use with ACCEL.

    Returns a pure function that applies a sequence of random atomic
    mutations to a [`Level`][nasimjax.envs.network_generator.Level],
    followed by a solvability repair pass.  The function is compatible
    with `jax.jit` and `jax.vmap`.

    The mutation pipeline proceeds as follows:

    1. Sample `max_num_edits` mutation types uniformly from
       [`MutationType`][nasimjax.envs.mutators.MutationType].
    2. Mask out mutations beyond the requested `num_edits`.
    3. Apply each mutation sequentially via `jax.lax.scan`.
    4. Ensure at least one sensitive host is reachable
       (internal `_ensure_reachable_sensitive` pass).
    5. Repair service/process invariants via
       [`ensure_attack_path`][nasimjax.envs.network_generator.ensure_attack_path].
    6. Recompute derived metadata (`num_active_hosts`, etc.).

    Args:
        max_num_edits: Maximum number of edit steps per mutation call.
            This is a compile-time constant that determines the
            unrolled scan length.  The actual number of applied edits
            is controlled by the `num_edits` argument at call time.
        num_hosts: Total number of hosts in the network.
        num_subnets: Total number of subnets.
        num_services: Number of available services.
        num_os: Number of available operating systems.
        num_processes: Number of available processes.

    Returns:
        A callable with signature
        `(rng: PRNGKey, level: Level, num_edits: int) -> Level`
        that applies `num_edits` random mutations (up to
        `max_num_edits`) and returns a solvable mutated level.

    Example:
        ```python
        mutate = make_level_mutator(
            max_num_edits=10,
            num_hosts=26,
            num_subnets=10,
            num_services=3,
            num_os=2,
            num_processes=3,
        )
        key = jax.random.PRNGKey(42)
        child_level = mutate(key, parent_level, num_edits=5)
        ```
    """
    mutation_types = tuple(mutation_types)
    assert MutationType.NO_OP not in mutation_types, (
        "NO_OP is reserved for masking; do not include it explicitly."
    )
    assert len(mutation_types) > 0, "Need at least one mutation type."

    # Build the dispatch table. Index 0 is always NO_OP (for masking);
    # indices 1..N are the enabled mutations in the order given.
    def _no_op(r, lvl):
        return lvl

    def _swap(r, lvl):
        return _swap_edges(r, lvl, num_subnets)

    def _toggle_svc(r, lvl):
        return _toggle_service(r, lvl, num_hosts, num_services)

    def _toggle_proc(r, lvl):
        return _toggle_process(r, lvl, num_hosts, num_processes)

    def _move_sens(r, lvl):
        return _move_sensitive(r, lvl, num_hosts, num_subnets)

    def _change_os_fn(r, lvl):
        return _change_os(r, lvl, num_hosts, num_os)

    op_map = {
        MutationType.SWAP_EDGES: _swap,
        MutationType.TOGGLE_SERVICE: _toggle_svc,
        MutationType.TOGGLE_PROCESS: _toggle_proc,
        MutationType.MOVE_SENSITIVE: _move_sens,
        MutationType.CHANGE_OS: _change_os_fn,
    }

    branches = [_no_op] + [op_map[m] for m in mutation_types]
    n_enabled = len(mutation_types)

    def mutate_level(rng, level, num_edits=1):
        rng, rng_types, rng_repair_sens, rng_repair_path = jax.random.split(rng, 4)
        step_keys = jax.random.split(rng, max_num_edits)

        # Sample branch indices in [1, n_enabled] -- never NO_OP -- then
        # mask out edits beyond num_edits with branch 0 (NO_OP).
        mutations = jax.random.randint(
            rng_types, shape=(max_num_edits,), minval=1, maxval=n_enabled + 1
        )
        mutations = jnp.where(
            jnp.arange(max_num_edits) < num_edits,
            mutations,
            0,  # NO_OP branch
        )

        def mutation_step(carry, step_input):
            step_rng, branch_idx = step_input
            new_level = jax.lax.switch(branch_idx, branches, step_rng, carry)
            return new_level, None

        mutated_level, _ = jax.lax.scan(mutation_step, level, (step_keys, mutations))

        # Repair pipeline
        repaired_hosts = _prune_and_rescue_sensitives(
            rng_repair_sens,
            mutated_level.hosts,
            mutated_level.topology,
            num_hosts,
        )

        # Step 2: Ensure service/process invariants on sensitive hosts
        #         and per-subnet pivotability.
        repaired_hosts = ensure_attack_path(
            rng_repair_path,
            repaired_hosts,
            num_hosts,
            num_services,
            num_processes,
        )

        # Step 3: Recompute derived metadata.
        num_active, is_flat, num_sens = _recompute_level_metadata(
            repaired_hosts,
            mutated_level.topology,
        )

        return Level(
            hosts=repaired_hosts,
            topology=mutated_level.topology,
            num_active_hosts=num_active,
            is_flat_topo=is_flat,
            num_sensitive_hosts=num_sens,
        )

    return mutate_level


"""Smoke test for the level mutator.

Run with: `python -m nasimjax.envs.mutators` (adjust import path as needed).

Verifies that after applying random mutations (including SWAP_EDGES and
MOVE_SENSITIVE):
  1. Every *reachable* sensitive host has >=1 service and >=1 process.
  2. Every *reachable* non-Internet subnet has >=1 host running a service
     (pivotability).
  3. All sensitive hosts are in internal subnets reachable from subnet 0
     (goal is achievable).
  4. The OS field remains a valid one-hot vector.
  5. At least one sensitive host exists (goal is non-trivial).
  6. Level metadata (num_active_hosts, is_flat_topo, num_sensitive_hosts) is
     consistent with the actual mutated level (not stale from the parent).
  7. Drift sanity: topology/sensitivity mutations actually change the level
     structure on a non-trivial fraction of samples.
"""

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    from nasimjax.envs.network_generator import (
        make_level_generator,
        get_connected_subnets,
    )
    from nasimjax.envs.common import NASimJaxEnvParams

    # ---- Test configuration ------------------------------------------------
    NUM_LEVELS = 200
    NUM_EDITS = 10
    MAX_NUM_EDITS = 20

    params = NASimJaxEnvParams(
        num_hosts=26,
        num_subnets=10,
        num_services=3,
        num_os=2,
        num_processes=3,
        topology_density=0.12,
        service_density=0.7,
        process_density=0.7,
        sensitive_density=0.15,
        distribute_homogeneous=True,
    )

    sample_level = make_level_generator(params)
    mutate_level = make_level_mutator(
        max_num_edits=MAX_NUM_EDITS,
        num_hosts=params.num_hosts,
        num_subnets=params.num_subnets,
        num_services=params.num_services,
        num_os=params.num_os,
        num_processes=params.num_processes,
    )

    # ---- Generate and mutate a batch of levels ----------------------------
    master_key = jax.random.PRNGKey(0)
    gen_keys = jax.random.split(master_key, NUM_LEVELS)
    mut_keys = jax.random.split(jax.random.PRNGKey(1), NUM_LEVELS)

    levels = jax.vmap(sample_level)(gen_keys)
    mutated = jax.vmap(mutate_level, in_axes=(0, 0, None))(mut_keys, levels, NUM_EDITS)

    print(f"Generated and mutated {NUM_LEVELS} levels with {NUM_EDITS} edits each.\n")

    # ---- Precompute per-level reachability for all checks -----------------
    # adj: (L, S, S) from traffic_rules[:, :, :, 0]
    adj = mutated.topology[:, :, :, 0]
    # connected: (L, S) -- subnets reachable from subnet 0
    connected = jax.vmap(get_connected_subnets)(adj)  # uint8

    # Per-host reachability: lookup subnet for each host, then mask by `connected`.
    host_subnet_idx = jnp.argmax(mutated.hosts.subnet_address, axis=-1)  # (L, H)
    # Gather: connected[l, host_subnet_idx[l, h]] -> (L, H)
    host_connected = jnp.take_along_axis(connected, host_subnet_idx, axis=-1).astype(
        jnp.bool_
    )

    # Internal-only reachability (excludes subnets 0 and 1).
    internal_connected = connected.at[:, :2].set(0)
    host_internal_reachable = jnp.take_along_axis(
        internal_connected, host_subnet_idx, axis=-1
    ).astype(jnp.bool_)

    services_per_host = mutated.hosts.services.sum(axis=-1)  # (L, H)
    processes_per_host = mutated.hosts.processes.sum(axis=-1)  # (L, H)
    sensitive = mutated.hosts.sensitive.astype(jnp.bool_)  # (L, H)

    # ---- (1) Reachable sensitive hosts have >=1 service and >=1 process ----
    reachable_sensitive = sensitive & host_internal_reachable
    svc_ok = jnp.where(reachable_sensitive, services_per_host >= 1, True).all(axis=-1)
    proc_ok = jnp.where(reachable_sensitive, processes_per_host >= 1, True).all(axis=-1)

    n_ok_svc = int(svc_ok.sum())
    n_ok_proc = int(proc_ok.sum())
    print(
        f"[1a] Reachable sensitive hosts with >=1 service: "
        f"{n_ok_svc}/{NUM_LEVELS} levels OK"
    )
    print(
        f"[1b] Reachable sensitive hosts with >=1 process: "
        f"{n_ok_proc}/{NUM_LEVELS} levels OK"
    )

    # ---- (2) Every reachable non-Internet subnet has a pivotable host ------
    host_has_service = (services_per_host >= 1).astype(jnp.uint8)  # (L, H)
    # (L, S): count of service-running hosts per subnet.
    subnet_service_count = jnp.einsum(
        "lhs,lh->ls",
        mutated.hosts.subnet_address.astype(jnp.uint8),
        host_has_service,
    )
    subnet_has_host = mutated.hosts.subnet_address.sum(axis=1) > 0  # (L, S)

    # Reachable non-Internet subnets: connected[:, 1:] (exclude subnet 0).
    reachable_non_internet = connected.astype(jnp.bool_).at[:, 0].set(False)
    subnet_must_pivot = reachable_non_internet & subnet_has_host
    pivot_ok = jnp.where(subnet_must_pivot, subnet_service_count >= 1, True).all(
        axis=-1
    )
    n_ok_pivot = int(pivot_ok.sum())
    print(
        f"[2]  Reachable non-Internet subnets with >=1 service: "
        f"{n_ok_pivot}/{NUM_LEVELS} levels OK"
    )

    # ---- (3) All sensitive hosts are reachable AND in internal subnets -----
    all_sensitive_reachable = jnp.where(sensitive, host_internal_reachable, True).all(
        axis=-1
    )
    n_ok_goal = int(all_sensitive_reachable.sum())
    print(
        f"[3]  All sensitive hosts reachable + internal (goal achievable): "
        f"{n_ok_goal}/{NUM_LEVELS} levels OK"
    )

    # ---- (4) OS one-hot ----------------------------------------------------
    os_sums = mutated.hosts.os.sum(axis=-1)  # (L, H)
    os_one_hot = (os_sums == 1).all(axis=-1)
    n_ok_os = int(os_one_hot.sum())
    print(f"[4]  OS is one-hot on all hosts: {n_ok_os}/{NUM_LEVELS}")

    # ---- (5) At least one sensitive host exists ----------------------------
    has_sensitive = sensitive.any(axis=-1)
    n_ok_nontrivial = int(has_sensitive.sum())
    print(f"[5]  Non-trivial goal (>=1 sensitive host): {n_ok_nontrivial}/{NUM_LEVELS}")

    # ---- (6) Level metadata consistency ------------------------------------
    # num_sensitive_hosts field should equal actual count.
    actual_num_sens = sensitive.sum(axis=-1)
    sens_meta_ok = mutated.num_sensitive_hosts == actual_num_sens
    n_ok_sens_meta = int(sens_meta_ok.sum())
    print(
        f"[6a] num_sensitive_hosts metadata matches actual: "
        f"{n_ok_sens_meta}/{NUM_LEVELS}"
    )

    # num_active_hosts should equal count of hosts in reachable subnets
    # (including Internet host 0 and DMZ hosts, per _recompute_level_metadata).
    dmz_mask = mutated.hosts.subnet_address[:, :, 1].astype(jnp.bool_)
    internet_mask = jnp.zeros_like(dmz_mask).at[:, 0].set(True)
    active_mask = host_connected | dmz_mask | internet_mask
    actual_num_active = active_mask.sum(axis=-1)
    active_meta_ok = mutated.num_active_hosts == actual_num_active
    n_ok_active_meta = int(active_meta_ok.sum())
    print(
        f"[6b] num_active_hosts metadata matches actual: "
        f"{n_ok_active_meta}/{NUM_LEVELS}"
    )

    # is_flat_topo should equal (all adjacency entries == 1).
    actual_is_flat = jnp.all(adj == 1, axis=(-1, -2))
    flat_meta_ok = mutated.is_flat_topo == actual_is_flat
    n_ok_flat_meta = int(flat_meta_ok.sum())
    print(f"[6c] is_flat_topo metadata matches actual: {n_ok_flat_meta}/{NUM_LEVELS}")

    # ---- (7) Drift sanity: mutations actually change structure -------------
    services_changed = (mutated.hosts.services != levels.hosts.services).any(
        axis=(-1, -2)
    )
    processes_changed = (mutated.hosts.processes != levels.hosts.processes).any(
        axis=(-1, -2)
    )
    os_changed = (mutated.hosts.os != levels.hosts.os).any(axis=(-1, -2))
    sensitive_changed = (mutated.hosts.sensitive != levels.hosts.sensitive).any(axis=-1)
    topology_changed = (mutated.topology != levels.topology).any(axis=(-1, -2, -3))
    num_active_changed = mutated.num_active_hosts != levels.num_active_hosts

    any_changed = (
        services_changed
        | processes_changed
        | os_changed
        | sensitive_changed
        | topology_changed
    )
    print(f"\n[7] Drift summary (fraction of levels that changed along each axis):")
    print(f"     any change:         {int(any_changed.sum())}/{NUM_LEVELS}")
    print(f"     services changed:   {int(services_changed.sum())}/{NUM_LEVELS}")
    print(f"     processes changed:  {int(processes_changed.sum())}/{NUM_LEVELS}")
    print(f"     os changed:         {int(os_changed.sum())}/{NUM_LEVELS}")
    print(f"     sensitive changed:  {int(sensitive_changed.sum())}/{NUM_LEVELS}")
    print(f"     topology changed:   {int(topology_changed.sum())}/{NUM_LEVELS}")
    print(
        f"     num_active changed: {int(num_active_changed.sum())}/{NUM_LEVELS}  "
        f"(expect >0 if SWAP_EDGES is enabled)"
    )

    # ---- Overall verdict --------------------------------------------------
    all_ok = (
        n_ok_svc == NUM_LEVELS
        and n_ok_proc == NUM_LEVELS
        and n_ok_pivot == NUM_LEVELS
        and n_ok_goal == NUM_LEVELS
        and n_ok_os == NUM_LEVELS
        and n_ok_nontrivial == NUM_LEVELS
        and n_ok_sens_meta == NUM_LEVELS
        and n_ok_active_meta == NUM_LEVELS
        and n_ok_flat_meta == NUM_LEVELS
    )
    # Minimum drift expectation: with 10 edits over 5 mutation types,
    # virtually every level should differ from its parent.
    drift_ok = int(any_changed.sum()) >= int(0.95 * NUM_LEVELS)
    if not drift_ok:
        print(
            f"\nWARNING: only {int(any_changed.sum())}/{NUM_LEVELS} levels "
            f"show any change — mutations may not be firing."
        )

    print("\n" + ("ALL CORRECTNESS CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
