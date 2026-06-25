# Copyright (c) 2026 The NASimJax Authors.
# Licensed under the MIT License. See LICENSE in the project root.
"""JAX-compatible HostVector implementation.

This module provides functional implementations for host vector operations
that are compatible with JAX transformations like JIT compilation.
"""

import jax.numpy as jnp
from flax import struct

from nasimjax.envs.utils import AccessLevel


@struct.dataclass
class HostVectorBatched:
    """This dataclass represents the batched hosts in the network.
    The state of the network basically consists of the state of each host.
    We batch everything together here because of the constraints imposed by
    JAX, where we can't simply define an array of HostVectors. Therefore we
    batch them together into one.

    The features of a host, listed in order, are:
    1. subnet address - one-hot encoding with length equal to num_subnets
                        of subnets in the network
    2. reachable - boolean
    3. discovered - boolean
    4. sensitive - boolean
    5. access - one-hot encoding representing the current access on the machine
    6. OS - boolean for each OS in scenario (only one OS has value of true)
    7. services running - boolean for each service in scenario
    8. processes running - boolean for each process in scenario

    The original NASim environment also lists the discovery value, but this
    shouldn't be part of a host. It's rather part of the reward function.

    Notes
    -----
    - The size of the vector is equal to:

        #subnets + max #hosts in any subnet + 4 + #OS + #services + #processes.

    - Where the + 4 is for  reachable, discovered, sensitive, and
        access level
    - The vector is a uint8 vector so True/False is represented as 1 or 0
    """

    # num_hosts is the the maximum number of hosts allowed in the scenarios.
    subnet_address: jnp.ndarray  # shape is (num_hosts,num_subnets)
    reachable: jnp.ndarray  # shape is (num_hosts,)
    discovered: jnp.ndarray  # shape is (num_hosts,)
    sensitive: jnp.ndarray  # shape is (num_hosts,)
    access_level: jnp.ndarray  # shape is (num_hosts,num_access_levels)
    os: jnp.ndarray  # shape is (num_hosts,num_os)
    services: jnp.ndarray  # shape is (num_hosts,num_services)
    processes: jnp.ndarray  # shape is (num_hosts,num_processes)


def get_empty_hosts(
    num_hosts: int,
    num_subnets: int,
    num_access_levels: int,
    num_os: int,
    num_services: int,
    num_processes: int,
):
    """Create an empty host vector with the dimensions according to the passed parameters.
    Helper function to get the dimensions right when creating the environment.
    """
    empty_hosts = HostVectorBatched(
        subnet_address=jnp.zeros((num_hosts, num_subnets)),
        reachable=jnp.zeros(num_hosts),
        discovered=jnp.zeros(num_hosts),
        sensitive=jnp.zeros(num_hosts),
        access_level=jnp.zeros((num_hosts, num_access_levels)),
        os=jnp.zeros((num_hosts, num_os)),
        services=jnp.zeros((num_hosts, num_services)),
        processes=jnp.zeros((num_hosts, num_processes)),
    )
    return empty_hosts


def set_reachable(
    hosts: HostVectorBatched, host_idx: int, value: bool
) -> HostVectorBatched:
    """Set reachable status for host at given index."""
    return hosts.replace(reachable=hosts.reachable.at[host_idx].set(jnp.uint8(value)))


def set_discovered(
    hosts: HostVectorBatched, host_idx: int, value: bool
) -> HostVectorBatched:
    """Set discovered status for host at given index."""
    return hosts.replace(discovered=hosts.discovered.at[host_idx].set(jnp.uint8(value)))


def set_access(
    hosts: HostVectorBatched, host_idx: int, access_level: int
) -> HostVectorBatched:
    """Set access level for host at given index."""
    # Create new access level one-hot encoding
    # TODO: Add bound checking
    new_access = jnp.zeros_like(hosts.access_level[host_idx])
    new_access = new_access.at[access_level].set(1)
    hosts = hosts.replace(access_level=hosts.access_level.at[host_idx].set(new_access))
    return hosts


def get_host_vector_batched_2D(hosts: HostVectorBatched) -> jnp.ndarray:
    """
    Flattens the HostVectorBatched into shape (N, total_feature_size),
    by concatenating all feature columns for each host.
    """
    # Expand rank-1 arrays to rank-2 for consistent concatenation
    reachable = hosts.reachable[:, None]
    discovered = hosts.discovered[:, None]
    sensitive = hosts.sensitive[:, None]

    # Concatenate all features along the last axis (feature axis)
    return jnp.concatenate(
        [
            hosts.subnet_address,
            reachable,
            discovered,
            sensitive,
            hosts.access_level,
            hosts.os,
            hosts.services,
            hosts.processes,
        ],
        axis=-1,
    )


def get_host_vector_flat(hosts: HostVectorBatched) -> jnp.ndarray:
    return jnp.ravel(get_host_vector_batched_2D(hosts))


def reconstruct_host_vector_from_flat(
    obs_flat: jnp.ndarray, template_hosts: HostVectorBatched
) -> HostVectorBatched:
    """Reconstruct HostVectorBatched from flattened observation array.

    This is the inverse function of get_host_vector_flat(). It takes a flattened
    observation array and reconstructs it back into a HostVectorBatched structure
    using the template to determine dimensions and shapes.

    Args:
        obs_flat: Flattened observation array (excluding aux info)
        template_hosts: Template HostVectorBatched to get dimensions from

    Returns:
        Reconstructed HostVectorBatched structure
    """
    num_hosts = template_hosts.subnet_address.shape[0]
    num_subnets = template_hosts.subnet_address.shape[1]
    num_access_levels = template_hosts.access_level.shape[1]
    num_os = template_hosts.os.shape[1]
    num_services = template_hosts.services.shape[1]
    num_processes = template_hosts.processes.shape[1]

    # Calculate total features per host
    features_per_host = (
        num_subnets  # subnet_address
        + 1  # reachable
        + 1  # discovered
        + 1  # sensitive
        + num_access_levels  # access_level
        + num_os  # os
        + num_services  # services
        + num_processes  # processes
    )

    # Reshape flat array to 2D [num_hosts, features_per_host]
    obs_2d = obs_flat.reshape(num_hosts, features_per_host)

    # Split back into individual feature arrays
    idx = 0

    # Extract subnet_address
    subnet_address = obs_2d[:, idx : idx + num_subnets]
    idx += num_subnets

    reachable = obs_2d[:, idx]
    idx += 1

    discovered = obs_2d[:, idx]
    idx += 1

    sensitive = obs_2d[:, idx]
    idx += 1

    # Extract access_level
    access_level = obs_2d[:, idx : idx + num_access_levels]
    idx += num_access_levels

    # Extract os
    os = obs_2d[:, idx : idx + num_os]
    idx += num_os

    # Extract services
    services = obs_2d[:, idx : idx + num_services]
    idx += num_services

    # Extract processes
    processes = obs_2d[:, idx : idx + num_processes]
    idx += num_processes

    return HostVectorBatched(
        subnet_address=subnet_address,
        reachable=reachable,
        discovered=discovered,
        sensitive=sensitive,
        access_level=access_level,
        os=os,
        services=services,
        processes=processes,
    )


def create_host_vector_batched(scenario) -> HostVectorBatched:
    """Create a HostVectorBatched structure from scenario.

    Args:
        scenario: NASim scenario object

    Returns:
        Initialized HostVectorBatched structure
    """

    num_subnets = len(scenario.subnets)
    num_os = scenario.num_os
    num_services = scenario.num_services
    num_processes = scenario.num_processes
    num_access_levels = len(AccessLevel)
    num_hosts = len(scenario.address_space)

    # Initialize arrays
    subnet_address = jnp.zeros((num_hosts, num_subnets), dtype=jnp.uint8)
    reachable = jnp.zeros(num_hosts, dtype=jnp.uint8)
    discovered = jnp.zeros(num_hosts, dtype=jnp.uint8)
    sensitive = jnp.zeros(num_hosts, dtype=jnp.uint8)
    access_level = jnp.zeros((num_hosts, num_access_levels), dtype=jnp.uint8)
    os = jnp.zeros((num_hosts, num_os), dtype=jnp.uint8)
    services = jnp.zeros((num_hosts, num_services), dtype=jnp.uint8)
    processes = jnp.zeros((num_hosts, num_processes), dtype=jnp.uint8)

    # Fill in data from scenario
    for i, addr in enumerate(scenario.address_space):
        # Set subnet address (one-hot)
        subnet_address = subnet_address.at[i, addr[0]].set(1)

        # Set sensitive status
        if addr in scenario.sensitive_hosts:
            sensitive = sensitive.at[i].set(1)

        # Get host data from scenario
        host_data = scenario.hosts[addr]

        # Set OS (one-hot)
        for os_idx, os_name in enumerate(scenario.os):
            if host_data.os.get(os_name, False):
                os = os.at[i, os_idx].set(1)

        # Set services
        for service_idx, service_name in enumerate(scenario.services):
            if host_data.services.get(service_name, False):
                services = services.at[i, service_idx].set(1)

        # Set processes
        for proc_idx, proc_name in enumerate(scenario.processes):
            if host_data.processes.get(proc_name, False):
                processes = processes.at[i, proc_idx].set(1)

        # Set initial access level (NONE = 0)
        access_level = access_level.at[i, 0].set(1)

    return HostVectorBatched(
        subnet_address=subnet_address,
        reachable=reachable,
        discovered=discovered,
        sensitive=sensitive,
        access_level=access_level,
        os=os,
        services=services,
        processes=processes,
    )
