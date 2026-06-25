# Copyright (c) 2026 The NASimJax Authors.
# Licensed under the MIT License. See LICENSE in the project root.
from functools import partial

import chex
from flax import struct
import jax
import jax.numpy as jnp
from typing import Callable, Tuple

from nasimjax.envs.environment_base import EnvParams
from nasimjax.envs.host_vector_batched import HostVectorBatched
from nasimjax.envs.utils import AccessLevel


@struct.dataclass
class Level:
    """A single network scenario used as a training level.

    Attributes:
        hosts: Batched host-vector representation containing per-host
            properties (subnet address, access level, OS, services,
            processes, reachable, discovered, sensitive).
        topology: Traffic-rules tensor of shape
            `(num_subnets, num_subnets, num_services + 1)`.
            Slice `[:, :, 0]` is the subnet adjacency matrix.
        num_active_hosts: Number of hosts reachable from the attacker's
            starting position.
        is_flat_topo: Whether every pair of subnets is connected.
        num_sensitive_hosts: Total number of sensitive hosts in the
            network.
    """

    hosts: HostVectorBatched
    topology: jnp.ndarray
    num_active_hosts: int
    is_flat_topo: bool
    num_sensitive_hosts: int


def get_connected_subnets(topology: jnp.ndarray) -> jnp.ndarray:
    """Compute reachable subnets from subnet 0 via transitive closure.

    Uses repeated boolean matrix–vector multiplication to propagate
    reachability from the attacker's starting subnet (index 0) through
    the directed adjacency matrix.

    Args:
        topology: Boolean adjacency matrix of shape
            `(num_subnets, num_subnets)`.

    Returns:
        A `uint8` vector of shape `(num_subnets,)` where entry `i` is
        1 if subnet `i` is reachable from subnet 0, and 0 otherwise.
    """
    num_subnets = topology.shape[0]

    def mat_mult(_, v):
        reachable, adj = v
        return (reachable | (reachable @ adj)), adj

    reachable_init = jnp.zeros(num_subnets, dtype=jnp.bool_).at[0].set(True)
    result = jax.lax.fori_loop(
        0,
        num_subnets,
        mat_mult,
        (reachable_init, topology.astype(jnp.bool_)),
    )[0]
    return result.astype(jnp.uint8)


def ensure_attack_path(
    rng: jax.random.key,
    hosts: HostVectorBatched,
    num_hosts: int,
    num_services: int,
    num_processes: int,
) -> HostVectorBatched:
    """
    This functions makes sure that:
        - Every sensitive host has a running service and process (so it can be owned)
        - Every subnet has at least one host running a service (so that it can be used to pivot)

    Args:
        rng: JAX PRNG key.
        hosts: The host-vector batch to repair.
        num_hosts: Total number of hosts.
        num_services: Number of available services.
        num_processes: Number of available processes.

    Returns:
        A repaired `HostVectorBatched` satisfying the solvability
        invariants.
    """

    k1, k2 = jax.random.split(rng, 2)

    # Find sensitive hosts with no services. To these hosts we will add a random service.
    needs_service_mask = (
        jnp.sum(hosts.services, axis=1, dtype=jnp.uint8) < hosts.sensitive
    )

    subnet_needs_service = (
        jnp.logical_not(
            jnp.any(
                jnp.any(hosts.services, axis=1).astype(jnp.uint8)[:, jnp.newaxis]
                * hosts.subnet_address,
                axis=0,
            )
        )
        .at[0]
        .set(False)
    )

    # Boolean vector saying wether each host is the first host of its subnet
    first_host_of_subnet = (
        jnp.zeros(shape=(num_hosts,), dtype=jnp.bool)
        .at[jnp.argmax(hosts.subnet_address, axis=0).astype(int)]
        .set(True)
    )

    # Add to the mask the first host of the subnets that have no services.
    needs_service_mask = jnp.logical_or(
        jnp.logical_and(
            subnet_needs_service[jnp.argmax(hosts.subnet_address, axis=1).astype(int)],
            first_host_of_subnet,
        ),
        needs_service_mask,
    )

    # Add random services to sensitive hosts with no services
    service_assignment = jax.random.randint(k1, (num_hosts,), 0, num_services)
    services_updated = hosts.services.copy()
    services_updated = jnp.where(
        needs_service_mask[:, None],
        jnp.eye(num_services, dtype=jnp.uint8)[service_assignment],
        hosts.services,
    )

    # Find sensitive hosts with no processes
    needs_process_mask = (
        jnp.sum(hosts.processes[:num_hosts], axis=1, dtype=jnp.uint8)
        < hosts.sensitive[:num_hosts]
    )

    # Add random process to sensitive hosts with no process
    process_assignment = jax.random.randint(k2, (num_hosts,), 0, num_processes)
    processes_updated = hosts.processes.copy()
    processes_updated = processes_updated.at[:num_hosts, :].set(
        jnp.where(
            needs_process_mask[:, None],
            jnp.eye(num_processes, dtype=jnp.uint8)[process_assignment],
            hosts.processes[:num_hosts],
        )
    )

    return hosts.replace(services=services_updated, processes=processes_updated)


@partial(
    jax.jit,
    static_argnames=[
        "num_hosts",
        "num_subnets",
        "num_services",
        "num_os",
        "num_processes",
    ],
)
@jax.named_scope("generate_network")
def generate(
    key: jax.random.key,
    num_hosts: int = 10,
    num_subnets: int = 5,
    num_services: int = 2,
    num_os: int = 2,
    num_processes: int = 2,
    distribute_homogeneous: bool = False,
    topology_density: float = 0.5,
    service_density: float = 0.2,
    process_density: float = 0.2,
    sensitive_density: float = 0.1,
    secure_topology: bool = True,
):
    """Generate the network configuration.

    Parameters
    ----------
    num_hosts : int
        number of hosts to include in network (minimum is 3)
    num_subnets : int
        number of subnets in network
    num_services : int
        number of services running on network (minimum is 1)
    num_os : int, optional
        number of OS running on network (minimum is 1) (default=2)
    num_processes : int, optional
        number of processes running on hosts on network (minimum is 1)
        (default=2)
    distribute_homogeneous : bool, optional
        whether to distribute services, os and processes homogeneously among
        subnets (default=False). If True, the services and processes will be
        drawn from a beta distribution per subnet.
    topology_density : float, optional
        density of the network topology (default=0.5). The lower the density,
        the less connected the network will be, resulting in more disjoint
        subnets, therefore less active hosts (reachable from the attacker start
        point).
    service_density : float, optional
        density of services running on hosts (default=0.2). The lower the
        density, the less services will be running on each host.
    process_density : float, optional
        density of processes running on hosts (default=0.2). The lower the
        density, the less processes will be running on each host.
    sensitive_density : float, optional
        Hosts can be marked as sensitive, according to Pr(host is sensitive) =
        sensitive_density (default=0.1)

    Returns
    -------
    HostVectorBatched
        Batched host vector representation. cf. HostVectorBatched definition
    jnp.ndarray
        Traffic rules. Extended topology tensor that also contains the allowed
        services between subnets.
    int
        Number of active hosts (reachable from attacker start point)
    bool
        Whether the generated topology is flat, i.e., all subnets are connected
    """

    def get_connected_subnets(topology: jnp.ndarray) -> jnp.ndarray:
        """
        Given a topology, compute which subnets are reachable if you start from the first subnet.
        """

        def mat_mult(i, v):
            return ((v[0] + (v[0] @ v[1])), v[1])

        num_subnets = topology.shape[0]
        return jax.lax.fori_loop(
            0,
            num_subnets,
            mat_mult,
            (
                jnp.zeros(shape=(num_subnets,), dtype=jnp.bool).at[0].set(True),
                topology.astype(jnp.bool),
            ),
        )[0].astype(jnp.uint8)

    def generate_generic(key: jax.random.key):
        def generate_random_network(key_local: jax.random.key):
            k1, k2, k3, k4, k5, k6 = jax.random.split(key_local, 6)

            # Random connectivity among subnets. We add at least a one in the first row to make
            # sure that the attacker can move to at least a second subnet.
            # For this to work, the attacker should always start from the first host and the
            # first host should be in the first subnet.
            random_connectity = (
                jax.random.bernoulli(
                    k1, shape=(num_subnets, num_subnets), p=topology_density
                )
                .astype(jnp.uint8)
                .at[
                    0,
                    jax.random.randint(
                        k2, shape=(1,), minval=1, maxval=num_subnets - 1
                    ),
                ]
                .set(1)
            )

            # Hosts in the same subnet can communicate (ones in the diagonal)
            # This can be removed if we want to model private VLANS
            topology = jnp.fill_diagonal(random_connectity, 1, inplace=False)

            # We start the arange from zero to allow perhaps more than one host in the first subnet.
            host_subnets_int = jnp.concatenate(
                [
                    jnp.zeros(1, dtype=jnp.uint8),
                    jax.random.permutation(k3, jnp.arange(1, num_subnets)).astype(
                        jnp.uint8
                    ),
                    jax.random.choice(
                        k4,
                        jnp.arange(0, num_subnets),
                        shape=(num_hosts - num_subnets,),
                    ).astype(jnp.uint8),
                ]
            )

            # We start with the identity matrix on the first num_subnet hosts. This ensures
            # at least one host per subnet.
            subnet_address = jax.nn.one_hot(
                host_subnets_int, num_classes=num_subnets, dtype=jnp.uint8
            )

            connected_subnets = get_connected_subnets(topology)

            connected_hosts = connected_subnets[jnp.argmax(subnet_address, axis=1)]

            num_active_hosts = jnp.sum(connected_hosts)

            random_connected_host = jnp.argmax(
                connected_hosts[1:]
                * jax.random.uniform(k5, shape=(num_hosts - 1,)).astype(jnp.uint8)
            )
            sensitive = (
                (
                    connected_hosts.astype(jnp.uint8)
                    * jax.random.bernoulli(
                        key=k6, p=sensitive_density, shape=(num_hosts,)
                    ).astype(jnp.uint8)
                )
                .at[0]
                .set(0)
                .at[random_connected_host + 1]
                .set(1)
            )

            return topology, subnet_address, sensitive, num_active_hosts

        def generate_secure_random_network(key_local: jax.random.key):
            k1, k2, k3, k4, k5, k6 = jax.random.split(key_local, 6)

            # zeros of maximal size
            topology = jnp.zeros((num_subnets, num_subnets), dtype=jnp.uint8)

            # Random connectivity among subnets. We add at least a one in the first row to make
            # sure that the dmz is connected to something internally
            random_connectity = (
                jax.random.bernoulli(
                    k1,
                    shape=(num_subnets - 1, num_subnets - 1),
                    p=topology_density,
                )
                .astype(jnp.uint8)
                .at[
                    0,
                    jax.random.randint(
                        k2, shape=(1,), minval=1, maxval=num_subnets - 1
                    ),
                ]
                .set(1)
            )

            # Apply random connectivity
            topology = topology.at[1:num_subnets, 1:num_subnets].set(random_connectity)

            # NOTE: we are droppig the concept of sensitive subnet. Instead we will consider a
            # subnet sensitive if there is at least one sensitive host in it.
            # Internet can only communicate to and from DMZ
            topology = topology.at[0:2, 0:2].set(
                jnp.ones(shape=(2, 2), dtype=jnp.uint8)
            )

            # Hosts in the same subnet can communicate.
            # This can be removed if we want to model private VLANS
            topology = jnp.fill_diagonal(topology, 1, inplace=False)

            # We start the arange from one to make sure the internet only has one host
            host_subnets_int = jnp.concatenate(
                [
                    jnp.zeros(1, dtype=jnp.uint8),
                    jax.random.permutation(k3, jnp.arange(1, num_subnets)).astype(
                        jnp.uint8
                    ),
                    jax.random.choice(
                        k4,
                        jnp.arange(1, num_subnets),
                        shape=(num_hosts - num_subnets,),
                    ).astype(jnp.uint8),
                ]
            )

            subnet_address = jax.nn.one_hot(
                host_subnets_int, num_classes=num_subnets, dtype=jnp.uint8
            )

            # We don't allow sensitive hosts to be in the internet or the dmz
            connected_subnets = get_connected_subnets(topology).at[:2].set(0)

            connected_hosts = connected_subnets[jnp.argmax(subnet_address, axis=1)]

            # since we removed the hosts in the internet or dmz, we add their count here
            num_active_hosts = (
                jnp.sum(connected_hosts) + jnp.sum(subnet_address[:, 1]) + 1
            )

            random_connected_host = jnp.argmax(
                connected_hosts[1:].astype(jnp.float32)
                * jax.random.uniform(k5, shape=(num_hosts - 1,))
            )

            sensitive = (
                (
                    connected_hosts
                    * jax.random.bernoulli(
                        key=k6, p=sensitive_density, shape=(num_hosts,)
                    ).astype(jnp.uint8)
                )
                .at[random_connected_host + 1]
                .set(1)
            )

            return topology, subnet_address, sensitive, num_active_hosts

        def generate_host(
            key_local: jax.random.key,
        ) -> Tuple[HostVectorBatched, jnp.ndarray, jnp.ndarray]:
            """Generate host configurations using uniform distribution.

            Creates HostVectorBatched structure with randomly assigned OS, services,
            and processes for each host using uniform sampling.

            Parameters
            ----------
            key : jax.random.key
                Random key for generation
            subnets : jnp.ndarray
                Array of subnet sizes
            num_services : int
                Number of available services
            num_os : int
                Number of available operating systems
            num_processes : int
                Number of available processes

            Returns
            -------
            HostVectorBatched
                Initialized host vector structure with random configurations
            """

            (
                gen_nw_key,
                gen_host_cfg_key,
            ) = jax.random.split(key_local, 2)

            reachable = jnp.zeros(num_hosts, dtype=jnp.uint8).at[0].set(1)
            discovered = jnp.zeros(num_hosts, dtype=jnp.uint8).at[0].set(1)

            topology, subnet_address, sensitive, num_active_hosts = jax.lax.cond(
                secure_topology,
                generate_secure_random_network,
                generate_random_network,
                gen_nw_key,
            )

            # Get Addess Space bounds
            num_access_levels = len(AccessLevel)

            # Initial access level (NONE = 0) (one-hot)
            access_level = jnp.zeros((num_hosts, num_access_levels), dtype=jnp.uint8)
            access_level = access_level.at[1:, 0].set(1)
            # internet hosts have maximal access_level
            access_level = access_level.at[0, -1].set(1)

            def generate_host_config_uniform():
                os_key, svc_key, proc_key = jax.random.split(gen_host_cfg_key, 3)
                # Generate random OS assignments (one-hot)
                os_choices = jax.random.randint(
                    os_key, (num_hosts,), 0, num_os, dtype=jnp.uint8
                )

                os = jax.nn.one_hot(os_choices, num_classes=num_os, dtype=jnp.uint8)

                # Apply service configurations
                services = jax.random.bernoulli(
                    svc_key, shape=(num_hosts, num_services), p=service_density
                ).astype(jnp.uint8)

                processes = jax.random.bernoulli(
                    proc_key, shape=(num_hosts, num_processes), p=process_density
                ).astype(jnp.uint8)

                return (os, services, processes)

            def generate_host_config_homogeneous():
                os_key1, os_key2, scv_key1, scv_key2, proc_key1, proc_key2 = (
                    jax.random.split(gen_host_cfg_key, 6)
                )

                # What is the probability of a host in subnet i to have os j.
                os_distribution = jax.random.beta(
                    key=os_key1,
                    shape=(num_subnets, num_os),
                    a=0.5,
                    b=0.5,
                )

                # The matrix product is of size (num_hosts, num_os) saying what is the probability of
                # host i to run os j.
                os_choices = jnp.argmax(
                    (subnet_address.astype(jnp.float32) @ os_distribution)
                    * jax.random.uniform(os_key2, shape=(num_hosts, num_os)),
                    axis=1,
                )

                os = jax.nn.one_hot(os_choices, num_classes=num_os, dtype=jnp.uint8)

                # What is the probability of a host in subnet i to have service j.
                service_distribution = jax.random.beta(
                    key=scv_key1,
                    shape=(num_subnets, num_services),
                    a=service_density,
                    b=1 - service_density,
                )

                # The matrix product is of size (num_hosts, num_services) saying what is the probability of
                # host i to run service j.
                services = (
                    subnet_address.astype(jnp.float32) @ service_distribution
                    > jax.random.uniform(scv_key2, shape=(num_hosts, num_services))
                ).astype(jnp.uint8)

                # What is the probability of a host in subnet i to have process j.
                process_distribution = jax.random.beta(
                    key=proc_key1,
                    shape=(num_subnets, num_processes),
                    a=process_density,
                    b=1 - process_density,
                )

                # The matrix product is of size (num_hosts, num_processes) saying what is the probability of
                # host i to run service j.
                processes = (
                    subnet_address.astype(jnp.float32) @ process_distribution
                    > jax.random.uniform(proc_key2, shape=(num_hosts, num_processes))
                ).astype(jnp.uint8)

                return (os, services, processes)

            (os, services, processes) = jax.lax.cond(
                distribute_homogeneous,
                generate_host_config_homogeneous,
                generate_host_config_uniform,
            )

            return (
                HostVectorBatched(
                    subnet_address=subnet_address,
                    reachable=reachable,
                    discovered=discovered,
                    sensitive=sensitive,
                    access_level=access_level,
                    os=os,
                    services=services,
                    processes=processes,
                ),
                topology,
                num_active_hosts,
            )

        hosts, topology, num_active_hosts = generate_host(key)

        compromised_subnets = jnp.sum(
            hosts.subnet_address
            * (jnp.argmax(hosts.access_level, axis=1) > AccessLevel.NONE).astype(
                jnp.uint8
            )[:, jnp.newaxis],
            axis=0,
        ).astype(jnp.uint8)

        # Mask saying whether a host is in a subnet accessible from the compromised subnet.
        connected_subnet_mask = compromised_subnets @ topology
        connected_mask = connected_subnet_mask[jnp.argmax(hosts.subnet_address, axis=1)]

        # Update reachability using vectorized operation
        # Note that the or opertion only updates the hosts that were
        # not already reachable
        new_reachable = hosts.reachable | connected_mask
        hosts = hosts.replace(reachable=new_reachable.astype(jnp.uint8))

        # Create "Allow All" policy, for now.
        traffic_rules = (
            jnp.ones(
                (
                    topology.shape[0],
                    topology.shape[1],
                    num_services + 1,
                ),
                dtype=jnp.uint8,
            )
            .at[:, :, 0]
            .set(topology)
        )
        return hosts, traffic_rules, num_active_hosts, jnp.all(topology == 1)

    key_gen_generic, key_ensure_attack_path = jax.random.split(key)

    hosts, traffic_rules, num_active_hosts, is_flat_topology = generate_generic(
        key_gen_generic
    )
    hosts = ensure_attack_path(
        key_ensure_attack_path,
        hosts,
        num_hosts,
        num_services,
        num_processes,
    )

    return hosts, traffic_rules, num_active_hosts, is_flat_topology


def make_level_generator(params: EnvParams) -> Callable[[jax.Array], Level]:
    def sample(rng: jax.Array) -> Level:
        hosts, topology, num_active_hosts, is_flat_topo = generate(
            key=rng,
            num_hosts=params.num_hosts,
            num_subnets=params.num_subnets,
            num_services=params.num_services,
            num_os=params.num_os,
            num_processes=params.num_processes,
            distribute_homogeneous=params.distribute_homogeneous,
            topology_density=params.topology_density,
            service_density=params.service_density,
            process_density=params.process_density,
            sensitive_density=params.sensitive_density,
        )

        return Level(
            hosts=hosts,
            topology=topology,
            num_active_hosts=num_active_hosts,
            is_flat_topo=is_flat_topo,
            num_sensitive_hosts=hosts.sensitive.sum(),
        )

    return sample


def make_eval_levels_and_names(
    params: EnvParams, size: int = 10
) -> tuple[jax.Array, list[str]]:
    keys = jax.random.split(jax.random.key(3), size)

    def make_level(key):
        hosts, topology, num_active_hosts, is_flat_topo = generate(
            key=key,
            num_hosts=params.num_hosts,
            num_subnets=params.num_subnets,
            num_services=params.num_services,
            num_os=params.num_os,
            num_processes=params.num_processes,
            distribute_homogeneous=params.distribute_homogeneous,
            topology_density=params.topology_density,
            service_density=params.service_density,
            process_density=params.process_density,
            sensitive_density=params.sensitive_density,
        )

        return Level(
            hosts=hosts,
            topology=topology,
            num_active_hosts=num_active_hosts,
            is_flat_topo=is_flat_topo,
            num_sensitive_hosts=hosts.sensitive.sum(),
        )

    levels = jax.vmap(make_level)(keys)
    # extract the batched fields produced by vmap and build human-readable names
    num_active_hosts = levels.num_active_hosts
    is_flat_topo = levels.is_flat_topo
    names = [
        f"L{i}_AH-{int(ah)}_F-{bool(ft)}_NS-{int(ns)}"
        for i, (ah, ft, ns) in enumerate(
            zip(num_active_hosts, is_flat_topo, levels.num_sensitive_hosts), start=1
        )
    ]
    return levels, names


def make_level_mutator(
    max_num_edits: int,
) -> Callable[[chex.PRNGKey, Level, int], Level]:
    raise NotImplementedError


def make_level_mutator_minimax(
    max_num_edits: int,
) -> Callable[[chex.PRNGKey, Level, int], Level]:
    raise NotImplementedError
