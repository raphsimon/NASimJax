# Copyright (c) 2026 The NASimJax Authors.
# Licensed under the MIT License. See LICENSE in the project root.
import jax
import jax.numpy as jnp

from typing import Tuple

from nasimjax.envs.host_vector_batched import (
    HostVectorBatched,
    set_access,
)
from nasimjax.envs.action import ActionData, ActionResult
from nasimjax.envs.common import NASimJaxEnvParams
from nasimjax.envs.utils import AccessLevel


def perform_action_on_host(
    hosts: HostVectorBatched,
    target_idx: int,
    num_hosts: int,
    action: ActionData,
    traffic_rules: jnp.ndarray,
    rng_key: jax.random.PRNGKey,
    params: NASimJaxEnvParams,
) -> Tuple[HostVectorBatched, ActionResult]:
    """Perform an action on a specific host in the batched structure.

    Args:
        hosts: Batched host vector structure
        target_idx: Index of the target host
        num_hosts: Total number of hosts in the current network
        action: ActionData object containing action information
        traffic_rules: Network topology adjacency matrix with information on allowed services
        rng_key: Key for random number generation
        params: Environment parameters to look-up the reward to distribute
    Returns:
        Tuple of (new_hosts, action_result)
    """
    ## Commonly used vectors
    # Get subnet for each host from batched representation
    # hosts.subnet_address is one-hot encoded, so argmax gives subnet ID
    host_subnets = jnp.argmax(hosts.subnet_address, axis=1)
    host_access_levels = jnp.argmax(hosts.access_level, axis=1)

    # Pre-checks for action validity
    target_subnet = host_subnets[target_idx]
    target_reachable = hosts.reachable[target_idx]
    target_compromised = host_access_levels[target_idx] > AccessLevel.NONE
    target_discovered = hosts.discovered[target_idx]
    target_services = hosts.services[target_idx, :]
    target_processes = hosts.processes[target_idx, :]
    target_access_level = jnp.argmax(hosts.access_level[target_idx, :]).astype(
        jnp.uint8
    )
    target_os = hosts.os[target_idx]

    # Check if target is reachable and discovered (required for most actions)
    def check_basic_requirements():
        return jnp.logical_and(target_reachable, target_discovered)

    # Check remote action permissions (need compromised host with required access in connected subnet)
    def check_remote_permissions():
        def has_permission():

            def check_compromised_hosts():
                def check_connectivity():
                    # For exploits: check if service is allowed through firewall
                    # We use service_id + 1 because index 0 is the topology information.
                    def exploit_case():
                        return (
                            traffic_rules[
                                host_subnets, target_subnet, action.service_id + 1
                            ]
                            == 1
                        )

                    # I'm not sure yet, but it seems like this is not necessary.
                    # If a host is attackable, it means it is visible to the attacker.
                    def scan_case():
                        return traffic_rules[host_subnets, target_subnet, 0] == 1

                    return jax.lax.cond(
                        action.action_type == 5, exploit_case, scan_case
                    )

                connectivity_ok = check_connectivity()

                # Check if hosts have required access level
                has_required_access = host_access_levels >= action.req_access

                # Combine all requirements
                qualified_hosts = (
                    (host_access_levels > AccessLevel.NONE)  # Host must be compromised
                    & connectivity_ok  # Must have connectivity/firewall permission
                    & has_required_access  # Must have required access level
                )

                # Return True if ANY host meets all requirements
                return jnp.any(qualified_hosts)

            return check_compromised_hosts()

        def no_permission_needed():
            return True

        return jax.lax.cond(action.is_remote, has_permission, no_permission_needed)

    basic_ok = check_basic_requirements()
    permissions_ok = check_remote_permissions()
    action_valid = jnp.logical_and(basic_ok, permissions_ok)

    # Return failure result if action is invalid
    def action_failed():
        result = ActionResult(
            success=False,
            value=jnp.int16(0),
            cost=action.cost,
            connection_error=jnp.logical_not(basic_ok),
            permission_error=jnp.logical_not(permissions_ok),
            services_discovered=jnp.zeros_like(hosts.services[target_idx, :]),
            os_discovered=jnp.zeros_like(hosts.os[target_idx, :]),
            processes_discovered=jnp.zeros_like(hosts.processes[target_idx, :]),
            access_gained=jnp.uint8(AccessLevel.NONE),
            hosts_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
            newly_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
            discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
        )
        return hosts, result

    def execute_valid_action():
        # JAX-compatible action dispatch using switch
        return jax.lax.switch(
            action.action_type,
            [
                noop_action,  # NOOP = 0
                os_scan,  # OS_SCAN = 1
                service_scan,  # SERVICE_SCAN = 2
                process_scan,  # PROCESS_SCAN = 3
                subnet_scan,  # SUBNET_SCAN = 4
                exploit,  # EXPLOIT = 5
                privilege_escalation,  # PRIV_ESC = 6
            ],
        )

    # Service scan
    def service_scan():
        result = ActionResult(
            success=True,
            value=jnp.int16(0),
            cost=action.cost,
            services_discovered=target_services,
            os_discovered=jnp.zeros_like(target_os),
            processes_discovered=jnp.zeros_like(target_processes),
            access_gained=jnp.uint8(AccessLevel.NONE),
            hosts_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
            newly_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
            discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
        )
        return hosts, result

    # OS scan
    def os_scan():
        result = ActionResult(
            success=True,
            value=jnp.int16(0),
            cost=action.cost,
            services_discovered=jnp.zeros_like(target_services),
            os_discovered=target_os,
            processes_discovered=jnp.zeros_like(target_processes),
            access_gained=jnp.uint8(AccessLevel.NONE),
            hosts_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
            newly_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
            discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
        )
        return hosts, result

    # Exploit
    def exploit():
        service_running = hosts.services[target_idx, action.service_id]
        # I assume this is because os_id==0 on an exploit actions means that
        # the exploit works on any os. I don't think this abstraction is necessary and
        # it collides with the fact that os_id==0 is a valid Os for a host.
        os_compatible = hosts.os[target_idx, action.os_id]

        # Check if exploit requirements are met
        requirements_met = jnp.logical_and(service_running, os_compatible)

        def attempt_exploit():
            # Already compromised hosts don't fail due to randomness
            def guaranteed_success():
                return True

            def probabilistic_success():
                # Use random key for probabilistic success
                random_val = jax.random.uniform(rng_key)
                return random_val < action.prob

            # If we want to simulate the fact that different services can give
            # different access levels, which is very realistic, then this should be removed.
            success = jax.lax.cond(
                target_compromised, guaranteed_success, probabilistic_success
            )

            def exploit_success():
                current_access = target_access_level

                def update_access():
                    new_hosts_with_access = set_access(hosts, target_idx, action.access)
                    # If we got root access, then give value depending on host sensitivity,
                    # otherwise return 0 value.
                    temp_value = jax.lax.select(
                        hosts.sensitive[target_idx],
                        params.sensitive_host_value,
                        params.host_value,
                    )
                    value = jnp.where(action.access == AccessLevel.ROOT, temp_value, 0)
                    return new_hosts_with_access, value

                def keep_access():
                    return hosts, 0

                new_hosts_final, reward = jax.lax.cond(
                    current_access != AccessLevel.ROOT, update_access, keep_access
                )

                def update_reachable_hosts(
                    hosts: HostVectorBatched,
                    newly_compromised_subnet: int,
                    topology: jnp.ndarray,
                ) -> HostVectorBatched:
                    """Update reachability of hosts after a new host is compromised.
                    The rule is that, if a host is compromised, then all the hosts in subnets connected the
                    compromised host's subnet become reachable.
                    """
                    # Mask saying whether a host is in a subnet accessible from the compromised subnet.
                    connected_mask = topology[newly_compromised_subnet, host_subnets, 0]

                    # Update reachability using vectorized operation
                    # Note that the or opertion only updates the hosts that were
                    # not alreadye reachable
                    new_reachable = hosts.reachable | connected_mask

                    return hosts.replace(reachable=new_reachable)

                # Update reachability if this is a new compromise
                def update_reachability():
                    return update_reachable_hosts(
                        new_hosts_final, target_subnet, traffic_rules
                    )

                def no_reachability_update():
                    return new_hosts_final

                hosts_with_reachability = jax.lax.cond(
                    jnp.logical_not(target_compromised),
                    update_reachability,
                    no_reachability_update,
                )

                result = ActionResult(
                    success=True,
                    value=jnp.int16(reward),
                    cost=action.cost,
                    services_discovered=target_services,
                    os_discovered=target_os,
                    processes_discovered=jnp.zeros_like(target_processes),
                    access_gained=action.access,
                    hosts_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                    newly_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                    discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                )
                return hosts_with_reachability, result

            def exploit_fail():
                result = ActionResult(
                    success=False,
                    value=jnp.int16(0),
                    cost=action.cost,
                    undefined_error=True,
                    services_discovered=jnp.zeros_like(target_services),
                    os_discovered=jnp.zeros_like(target_os),
                    processes_discovered=jnp.zeros_like(target_processes),
                    access_gained=jnp.uint8(AccessLevel.NONE),
                    hosts_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                    newly_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                    discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                )
                return hosts, result

            return jax.lax.cond(success, exploit_success, exploit_fail)

        def exploit_requirements_not_met():
            result = ActionResult(
                success=False,
                value=jnp.int16(0),
                cost=action.cost,
                undefined_error=True,
                services_discovered=jnp.zeros_like(target_services),
                os_discovered=jnp.zeros_like(target_os),
                processes_discovered=jnp.zeros_like(target_processes),
                access_gained=jnp.uint8(AccessLevel.NONE),
                hosts_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                newly_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
            )
            return hosts, result

        return jax.lax.cond(
            requirements_met, attempt_exploit, exploit_requirements_not_met
        )

    # Process scan
    def process_scan():
        has_permission = jnp.logical_and(
            target_compromised, action.req_access <= target_access_level
        )

        def scan_success():
            result = ActionResult(
                success=True,
                value=jnp.int16(0),
                cost=action.cost,
                services_discovered=jnp.zeros_like(target_services),
                os_discovered=jnp.zeros_like(target_os),
                processes_discovered=target_processes,
                access_gained=target_access_level,
                hosts_discovered=jnp.zeros_like(hosts.discovered),
                newly_discovered=jnp.zeros_like(hosts.discovered),
                discovered=jnp.zeros_like(hosts.discovered),
            )
            return hosts, result

        def scan_fail():
            # Create empty arrays to match the structure of scan_success
            result = ActionResult(
                success=False,
                value=jnp.int16(0),
                cost=action.cost,
                permission_error=True,
                services_discovered=jnp.zeros_like(target_services),
                os_discovered=jnp.zeros_like(target_os),
                processes_discovered=jnp.zeros_like(target_processes),
                access_gained=jnp.uint8(AccessLevel.NONE),
                hosts_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                newly_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
            )
            return hosts, result

        return jax.lax.cond(has_permission, scan_success, scan_fail)

    def privilege_escalation():
        # Prerequisites: host must be compromised AND have required access level
        has_permission = action.req_access <= target_access_level

        def privesc_with_permission():
            # Check whether we have an intersection between process running on host
            # and targeted process by the action. Same for OS.
            requirements_met = jnp.logical_and(
                target_processes[action.process_id],
                target_os[action.os_id],
            )

            def attempt_privesc():
                random_val = jax.random.uniform(rng_key)
                success = random_val < action.prob

                def privesc_success():
                    def update_access():
                        new_hosts_with_access = set_access(
                            hosts, target_idx, action.access
                        )
                        temp_value = jax.lax.select(
                            hosts.sensitive[target_idx],
                            params.sensitive_host_value,
                            params.host_value,
                        )
                        value = jnp.where(
                            action.access == AccessLevel.ROOT, temp_value, 0
                        )
                        return new_hosts_with_access, value

                    def keep_access():
                        return hosts, 0

                    new_hosts_final, reward = jax.lax.cond(
                        target_access_level != AccessLevel.ROOT,
                        update_access,
                        keep_access,
                    )

                    result = ActionResult(
                        success=True,
                        value=jnp.int16(reward),
                        cost=action.cost,
                        services_discovered=jnp.zeros_like(target_services),
                        os_discovered=target_os,
                        processes_discovered=target_processes,
                        access_gained=action.access,
                        hosts_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                        newly_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                        discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                    )
                    return new_hosts_final, result

                def privesc_fail():
                    result = ActionResult(
                        success=False,
                        value=jnp.int16(0),
                        cost=action.cost,
                        undefined_error=True,
                        services_discovered=jnp.zeros_like(target_services),
                        os_discovered=jnp.zeros_like(target_os),
                        processes_discovered=jnp.zeros_like(target_processes),
                        access_gained=jnp.uint8(AccessLevel.NONE),
                        hosts_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                        newly_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                        discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                    )
                    return hosts, result

                return jax.lax.cond(success, privesc_success, privesc_fail)

            def privesc_requirements_not_met():
                result = ActionResult(
                    success=False,
                    value=jnp.int16(0),
                    cost=action.cost,
                    undefined_error=True,
                    services_discovered=jnp.zeros_like(target_services),
                    os_discovered=jnp.zeros_like(target_os),
                    processes_discovered=jnp.zeros_like(target_processes),
                    access_gained=jnp.uint8(AccessLevel.NONE),
                    hosts_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                    newly_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                    discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                )
                return hosts, result

            return jax.lax.cond(
                requirements_met, attempt_privesc, privesc_requirements_not_met
            )

        def privesc_no_permission():
            result = ActionResult(
                success=False,
                value=jnp.int16(0),
                cost=action.cost,
                connection_error=False,
                permission_error=True,
                services_discovered=jnp.zeros_like(target_services),
                os_discovered=jnp.zeros_like(target_os),
                processes_discovered=jnp.zeros_like(target_processes),
                access_gained=jnp.uint8(AccessLevel.NONE),
                hosts_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                newly_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
                discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
            )
            return hosts, result

        return jax.lax.cond(
            has_permission, privesc_with_permission, privesc_no_permission
        )

    def subnet_scan():
        # Check if host is compromised and has required access
        has_permission = action.req_access <= target_access_level

        def scan_success():
            # Check which hosts are in subnets connected to target subnet
            subnets_connected = traffic_rules[target_subnet, host_subnets, 0]

            # Check which hosts are not already discovered
            not_already_discovered = ~hosts.discovered.astype(bool)

            # Hosts to be newly discovered: connected AND not already discovered
            newly_discovered_mask = subnets_connected & not_already_discovered

            # All hosts that are discovered after scan: already discovered OR newly discovered
            final_discovered_mask = hosts.discovered.astype(bool) | subnets_connected

            # Calculate discovery reward (only for newly discovered hosts)
            num_newly_discovered = jnp.sum(newly_discovered_mask)
            discovery_reward = num_newly_discovered * params.discovery_value

            # Update hosts with new discovery status
            updated_hosts = hosts.replace(
                discovered=final_discovered_mask.astype(jnp.uint8)
            )

            result = ActionResult(
                success=True,
                value=discovery_reward.astype(jnp.int16),
                cost=action.cost,
                services_discovered=jnp.zeros_like(target_services),
                os_discovered=jnp.zeros_like(target_os),
                processes_discovered=jnp.zeros_like(target_processes),
                access_gained=jnp.uint8(AccessLevel.NONE),
                hosts_discovered=final_discovered_mask.astype(jnp.uint8),
                newly_discovered=newly_discovered_mask.astype(jnp.uint8),
                discovered=final_discovered_mask.astype(jnp.uint8),  # For compatibility
            )
            return updated_hosts, result

        def scan_fail():
            # Return empty discovery arrays of correct size
            empty_discovered = jnp.zeros(num_hosts, dtype=jnp.uint8)

            result = ActionResult(
                success=False,
                value=jnp.int16(0),
                cost=action.cost,
                connection_error=False,
                permission_error=True,
                services_discovered=jnp.zeros_like(target_services),
                os_discovered=jnp.zeros_like(target_os),
                processes_discovered=jnp.zeros_like(target_processes),
                access_gained=jnp.uint8(AccessLevel.NONE),
                hosts_discovered=empty_discovered,
                newly_discovered=empty_discovered,
                discovered=empty_discovered,
            )
            return hosts, result

        return jax.lax.cond(has_permission, scan_success, scan_fail)

    def noop_action():
        # This action always succeeds.
        result = ActionResult(
            success=True,
            value=jnp.int16(0),
            cost=action.cost,
            services_discovered=jnp.zeros_like(target_services),
            os_discovered=jnp.zeros_like(target_os),
            processes_discovered=jnp.zeros_like(target_processes),
            access_gained=jnp.uint8(AccessLevel.NONE),
            hosts_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
            newly_discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
            discovered=jnp.zeros(num_hosts, dtype=jnp.uint8),
        )
        return hosts, result

    # Main conditional: check if action is valid, then execute
    return jax.lax.cond(action_valid, execute_valid_action, action_failed)
