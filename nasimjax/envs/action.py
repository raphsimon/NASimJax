# Copyright (c) 2026 The NASimJax Authors.
# Licensed under the MIT License. See LICENSE in the project root.
"""JAX-compatible action implementation for NASim.

This module provides functional implementations for action operations
that are compatible with JAX transformations.
"""

from typing import List

import jax.numpy as jnp
from flax import struct

from nasimjax.envs.utils import ActionType, AccessLevel


@struct.dataclass
class ActionData:
    """JAX-compatible action representation."""

    target_host_idx: int  # Use host index instead of address tuple
    cost: int
    prob: float
    req_access: int
    action_type: int = ActionType.NOOP
    is_remote: bool = True
    service_id: int = 0  # Use service ID instead of string
    os_id: int = 0  # Use OS ID instead of string
    process_id: int = 0  # Use process ID instead of string
    access: int = AccessLevel.NONE
    value: int = 0

    # Action type flags for efficient checking
    is_noop: bool = False
    is_exploit: bool = False
    is_privilege_escalation: bool = False
    is_service_scan: bool = False
    is_os_scan: bool = False
    is_subnet_scan: bool = False
    is_process_scan: bool = False
    is_scan: bool = False


@struct.dataclass
class ActionResult:
    """JAX-compatible action result."""

    success: bool
    value: int = 0
    cost: int = 0
    connection_error: bool = False
    permission_error: bool = False
    undefined_error: bool = False
    # Service/OS/process discovery (use arrays instead of dicts)
    services_discovered: jnp.ndarray = None
    os_discovered: jnp.ndarray = None
    processes_discovered: jnp.ndarray = None
    access_gained: int = AccessLevel.NONE
    # Host discovery arrays (required for observation generation) # TODO check whether we *really* need them.
    hosts_discovered: jnp.ndarray = None  # All hosts discovered in this action
    newly_discovered: jnp.ndarray = None  # Hosts discovered for the first time
    discovered: jnp.ndarray = (
        None  # Alternative name for hosts_discovered (compatibility)
    )
    reachable: jnp.ndarray = None


class ActionJAX:
    """JAX-compatible action utilities."""

    @staticmethod
    def create_noop_action() -> ActionData:
        """Create a no-operation action."""
        return ActionData(
            target_host_idx=jnp.int32(0),  # Use first host as default target
            cost=jnp.int16(0),
            prob=jnp.float16(1),
            req_access=AccessLevel.NONE,
            action_type=ActionType.NOOP,
            is_noop=True,
        )

    @staticmethod
    def create_exploit_action(
        target_host_idx: int,
        cost: int,
        prob: float,
        service_id: int,
        os_id: int = jnp.uint8(0),
        access: int = AccessLevel.USER,
        value: int = jnp.int16(0),
    ) -> ActionData:
        """Create an exploit action."""
        return ActionData(
            target_host_idx=jnp.uint16(target_host_idx),
            cost=jnp.int16(cost),
            prob=jnp.float16(prob),
            req_access=AccessLevel.NONE,
            action_type=ActionType.EXPLOIT,
            is_remote=True,
            service_id=jnp.uint8(service_id),
            os_id=jnp.uint8(os_id),
            access=jnp.uint8(access),
            value=jnp.int16(value),
            is_exploit=True,
            is_scan=False,
        )

    @staticmethod
    def create_privesc_action(
        target_host_idx: int,
        cost: int,
        prob: float,
        process_id: int = jnp.uint8(0),
        os_id: int = jnp.uint8(0),
        access: int = AccessLevel.ROOT,
    ) -> ActionData:
        """Create a privilege escalation action."""
        return ActionData(
            target_host_idx=jnp.uint16(target_host_idx),
            cost=jnp.int16(cost),
            prob=jnp.float16(prob),
            req_access=AccessLevel.USER,
            action_type=ActionType.PRIV_ESC,
            is_remote=False,
            process_id=jnp.uint8(process_id),
            os_id=jnp.uint8(os_id),
            access=jnp.uint8(access),
            is_privilege_escalation=True,
            is_scan=False,
        )

    @staticmethod
    def create_service_scan_action(target_host_idx: int, cost: int) -> ActionData:
        """Create a service scan action."""
        return ActionData(
            target_host_idx=jnp.uint16(target_host_idx),
            cost=jnp.int16(cost),
            prob=jnp.float16(1),
            req_access=AccessLevel.NONE,
            action_type=ActionType.SERVICE_SCAN,
            is_remote=True,
            is_service_scan=True,
            is_scan=True,
        )

    @staticmethod
    def create_os_scan_action(
        target_host_idx: int,
        cost: int,
        prob: float = jnp.float16(1),
        req_access: int = AccessLevel.NONE,
    ) -> ActionData:
        """Create an OS scan action."""
        return ActionData(
            target_host_idx=jnp.uint16(target_host_idx),
            cost=jnp.int16(cost),
            prob=jnp.float16(prob),
            req_access=req_access,
            action_type=ActionType.OS_SCAN,
            is_remote=True,
            is_os_scan=True,
            is_scan=True,
        )

    @staticmethod
    def create_subnet_scan_action(target_host_idx: int, cost: int) -> ActionData:
        """Create a subnet scan action."""
        return ActionData(
            target_host_idx=jnp.uint16(target_host_idx),
            cost=jnp.int16(cost),
            prob=jnp.float16(1),
            req_access=AccessLevel.USER,
            action_type=ActionType.SUBNET_SCAN,
            is_remote=False,
            is_subnet_scan=True,
            is_scan=True,
        )

    @staticmethod
    def create_process_scan_action(target_host_idx: int, cost: int) -> ActionData:
        """Create a process scan action."""
        return ActionData(
            target_host_idx=jnp.uint16(target_host_idx),
            cost=jnp.int16(cost),
            prob=jnp.float16(1),
            req_access=AccessLevel.USER,
            action_type=ActionType.PROCESS_SCAN,
            is_remote=False,
            is_process_scan=True,
            is_scan=True,
        )


class FlatActionSpaceJAX:
    """JAX-compatible flat action space implementation."""

    def __init__(self, scenario):
        """Initialize action space from scenario.

        Parameters
        ----------
        scenario : Scenario
            Environment scenario
        """
        self.scenario = scenario

        # Create address-to-index mapping for host indices
        self.addr_to_idx = {addr: i for i, addr in enumerate(scenario.address_space)}
        self.idx_to_addr = {i: addr for i, addr in enumerate(scenario.address_space)}

        self.actions = self._load_action_list(scenario)
        self.n = len(self.actions)

        # Create action lookup arrays for JAX
        self._create_action_arrays()

    def _load_action_list(self, scenario) -> List[ActionData]:
        """Load list of actions for scenario."""
        action_list = []

        # Add actions for each host
        for address in scenario.address_space:
            host_idx = self.addr_to_idx[address]

            # Scan actions
            action_list.append(
                ActionJAX.create_service_scan_action(
                    host_idx, scenario.service_scan_cost
                )
            )
            action_list.append(
                ActionJAX.create_os_scan_action(host_idx, scenario.os_scan_cost)
            )
            action_list.append(
                ActionJAX.create_subnet_scan_action(host_idx, scenario.subnet_scan_cost)
            )
            action_list.append(
                ActionJAX.create_process_scan_action(
                    host_idx, scenario.process_scan_cost
                )
            )

            # Exploit actions
            for e_name, e_def in scenario.exploits.items():
                # Get service ID (simplified - use index in services list)
                service_id = 0
                if "service" in e_def and e_def["service"] in scenario.services:
                    service_id = scenario.services.index(e_def["service"])

                os_id = 0
                if "os" in e_def and e_def["os"] in scenario.os:
                    os_id = scenario.os.index(e_def["os"])

                action_list.append(
                    ActionJAX.create_exploit_action(
                        host_idx,
                        e_def.get("cost", 1),
                        e_def.get("prob", 1.0),
                        service_id,
                        os_id,
                        e_def.get("access", AccessLevel.USER),
                        e_def.get("value", 0),
                    )
                )

            # Privilege escalation actions
            for pe_name, pe_def in scenario.privescs.items():
                process_id = 0
                if "process" in pe_def and pe_def["process"] in scenario.processes:
                    process_id = scenario.processes.index(pe_def["process"])

                os_id = 0
                if "os" in pe_def and pe_def["os"] in scenario.os:
                    os_id = scenario.os.index(pe_def["os"])

                action_list.append(
                    ActionJAX.create_privesc_action(
                        host_idx,
                        pe_def.get("cost", 1),
                        pe_def.get("prob", 1.0),
                        process_id,
                        os_id,
                        pe_def.get("access", AccessLevel.ROOT),
                    )
                )

        return action_list

    def _create_action_arrays(self):
        """Create JAX arrays for action data."""
        # Extract action properties into arrays
        targets = jnp.array(
            [action.target_host_idx for action in self.actions], dtype=jnp.int16
        )
        costs = jnp.array([action.cost for action in self.actions], dtype=jnp.int16)
        probs = jnp.array([action.prob for action in self.actions], dtype=jnp.float16)
        req_access = jnp.array(
            [action.req_access for action in self.actions], dtype=jnp.int16
        )

        # Boolean flags
        is_noop = jnp.array(
            [action.is_noop for action in self.actions], dtype=jnp.bool_
        )
        is_exploit = jnp.array(
            [action.is_exploit for action in self.actions], dtype=jnp.bool_
        )
        is_privesc = jnp.array(
            [action.is_privilege_escalation for action in self.actions], dtype=jnp.bool_
        )
        is_service_scan = jnp.array(
            [action.is_service_scan for action in self.actions], dtype=jnp.bool_
        )
        is_os_scan = jnp.array(
            [action.is_os_scan for action in self.actions], dtype=jnp.bool_
        )
        is_subnet_scan = jnp.array(
            [action.is_subnet_scan for action in self.actions], dtype=jnp.bool_
        )
        is_process_scan = jnp.array(
            [action.is_process_scan for action in self.actions], dtype=jnp.bool_
        )
        is_remote = jnp.array(
            [action.is_remote for action in self.actions], dtype=jnp.bool_
        )

        # Access levels and values
        access_levels = jnp.array(
            [action.access for action in self.actions], dtype=jnp.uint8
        )
        values = jnp.array([action.value for action in self.actions], dtype=jnp.int16)

        # Extract type information and other flags
        action_types = jnp.array(
            [action.action_type for action in self.actions], dtype=jnp.int16
        )
        service_ids = jnp.array(
            [action.service_id for action in self.actions], dtype=jnp.int16
        )
        os_ids = jnp.array([action.os_id for action in self.actions], dtype=jnp.int16)
        process_ids = jnp.array(
            [action.process_id for action in self.actions], dtype=jnp.int16
        )

        # Store arrays
        self.action_arrays = {
            "targets": targets,
            "costs": costs,
            "probs": probs,
            "req_access": req_access,
            "action_types": action_types,
            "service_ids": service_ids,
            "os_ids": os_ids,
            "process_ids": process_ids,
            "is_noop": is_noop,
            "is_exploit": is_exploit,
            "is_privilege_escalation": is_privesc,
            "is_service_scan": is_service_scan,
            "is_os_scan": is_os_scan,
            "is_subnet_scan": is_subnet_scan,
            "is_process_scan": is_process_scan,
            "is_remote": is_remote,
            "access": access_levels,
            "values": values,
        }


def get_action_from_arrays(arrays, idx):
    return ActionData(
        target_host_idx=arrays["targets"][idx],
        cost=arrays["costs"][idx],
        prob=arrays["probs"][idx],
        req_access=arrays["req_access"][idx],
        action_type=arrays["action_types"][idx],
        is_remote=arrays["is_remote"][idx],
        service_id=arrays["service_ids"][idx],
        os_id=arrays["os_ids"][idx],
        process_id=arrays["process_ids"][idx],
        access=arrays["access"][idx],
        value=arrays["values"][idx],
        is_noop=arrays["is_noop"][idx],
        is_exploit=arrays["is_exploit"][idx],
        is_privilege_escalation=arrays["is_privilege_escalation"][idx],
        is_service_scan=arrays["is_service_scan"][idx],
        is_os_scan=arrays["is_os_scan"][idx],
        is_subnet_scan=arrays["is_subnet_scan"][idx],
        is_process_scan=arrays["is_process_scan"][idx],
        is_scan=(
            arrays["is_service_scan"][idx]
            | arrays["is_os_scan"][idx]
            | arrays["is_subnet_scan"][idx]
            | arrays["is_process_scan"][idx]
        ),
    )


class GeneratedFlatActionSpaceJAX:
    """Class that represents a generated, flat, action space."""

    def __init__(
        self,
        num_hosts: int,
        num_services: int,
        num_processes: int,
        num_os: int,
        exploit_success_prob: float,
        privesc_success_prob: float,
        service_scan_cost: int,
        process_scan_cost: int,
        os_scan_cost: int,
        subnet_scan_cost: int,
        exploit_cost: int,
        privesc_cost: int,
    ):
        """Generate action space given the number of services, processes, and OSes.

        Every host will have four scan actions associated to it (Service, Process, OS, Subnet),
        and then 'num_services * num_os' exploits, and 'num_processes * num_os' privilege
        escalation actions will be generated.

        The goal is to create a 'tool box'. Such that every host has every possible action
        associated to it, but whether the action is valid depends on the services, processes
        and the OS it is running.

        Parameters
        ----------
        num_hosts : int
            Number of hosts in the network
        num_services : int
            Number of services available
        num_processes : int
            Number of processes available
        num_os : int
            Number of operating systems available
        service_scan_cost : int
            Cost for service scan actions
        process_scan_cost : int
            Cost for process scan actions
        os_scan_cost : int
            Cost for OS scan actions
        subnet_scan_cost : int
            Cost for subnet scan actions
        exploit_cost : int
            Cost for exploit actions
        privesc_cost : int
            Cost for privilege escalation actions
        """
        self.num_hosts = num_hosts
        self.num_services = num_services
        self.num_processes = num_processes
        self.num_os = num_os

        # Action successs probabilitites
        self.exploit_success_prob = exploit_success_prob
        self.privesc_success_prob = privesc_success_prob

        # Store costs
        self.service_scan_cost = service_scan_cost
        self.process_scan_cost = process_scan_cost
        self.os_scan_cost = os_scan_cost
        self.subnet_scan_cost = subnet_scan_cost
        self.exploit_cost = exploit_cost
        self.privesc_cost = privesc_cost

        # Generate all possible actions
        self.actions = self._generate_all_actions()
        self.n = len(self.actions)

        # Create action lookup arrays for JAX
        self._create_action_arrays()

    def _generate_all_actions(self) -> List[ActionData]:
        """Generate all possible actions for all hosts."""
        action_list = []

        for host_idx in range(self.num_hosts):
            # Add scan actions for each host
            action_list.extend(self._generate_scan_actions(host_idx))

            # Add exploit actions for each host (all service/OS combinations)
            action_list.extend(
                self._generate_exploit_actions(host_idx, self.exploit_success_prob)
            )

            # Add privilege escalation actions for each host (all process/OS combinations)
            action_list.extend(
                self._generate_privesc_actions(host_idx, self.privesc_success_prob)
            )

        return action_list

    def _generate_scan_actions(self, host_idx: int) -> List[ActionData]:
        """Generate scan actions for a specific host."""
        scan_actions = []

        # Service scan
        scan_actions.append(
            ActionJAX.create_service_scan_action(host_idx, self.service_scan_cost)
        )

        # OS scan
        scan_actions.append(
            ActionJAX.create_os_scan_action(host_idx, self.os_scan_cost)
        )

        # Subnet scan
        scan_actions.append(
            ActionJAX.create_subnet_scan_action(host_idx, self.subnet_scan_cost)
        )

        # Process scan
        scan_actions.append(
            ActionJAX.create_process_scan_action(host_idx, self.process_scan_cost)
        )

        return scan_actions

    def _generate_exploit_actions(
        self,
        host_idx: int,
        exploit_success_prob: float,
    ) -> List[ActionData]:
        """Generate exploit actions for a specific host."""
        exploit_actions = []

        # Generate exploits for each service/OS combination
        for service_id in range(self.num_services):
            for os_id in range(self.num_os):
                exploit_actions.append(
                    ActionJAX.create_exploit_action(
                        target_host_idx=host_idx,
                        cost=self.exploit_cost,
                        prob=exploit_success_prob,
                        service_id=service_id,
                        os_id=os_id,
                        access=AccessLevel.USER,  # Default access level
                        value=0,
                    )
                )

        return exploit_actions

    def _generate_privesc_actions(
        self,
        host_idx: int,
        privesc_success_prob: float,
    ) -> List[ActionData]:
        """Generate privilege escalation actions for a specific host."""
        privesc_actions = []

        # Generate privesc for each process/OS combination
        for process_id in range(self.num_processes):
            for os_id in range(self.num_os):
                privesc_actions.append(
                    ActionJAX.create_privesc_action(
                        target_host_idx=host_idx,
                        cost=self.privesc_cost,
                        prob=privesc_success_prob,
                        process_id=process_id,
                        os_id=os_id,
                        access=AccessLevel.ROOT,
                    )
                )

        return privesc_actions

    def _create_action_arrays(self):
        """Create JAX arrays for action data."""
        # Extract action properties into arrays
        targets = jnp.array(
            [action.target_host_idx for action in self.actions], dtype=jnp.uint16
        )
        costs = jnp.array([action.cost for action in self.actions], dtype=jnp.int16)
        probs = jnp.array([action.prob for action in self.actions], dtype=jnp.float16)
        req_access = jnp.array(
            [action.req_access for action in self.actions], dtype=jnp.uint8
        )

        # Boolean flags
        is_noop = jnp.array(
            [action.is_noop for action in self.actions], dtype=jnp.bool_
        )
        is_exploit = jnp.array(
            [action.is_exploit for action in self.actions], dtype=jnp.bool_
        )
        is_privesc = jnp.array(
            [action.is_privilege_escalation for action in self.actions], dtype=jnp.bool_
        )
        is_service_scan = jnp.array(
            [action.is_service_scan for action in self.actions], dtype=jnp.bool_
        )
        is_os_scan = jnp.array(
            [action.is_os_scan for action in self.actions], dtype=jnp.bool_
        )
        is_subnet_scan = jnp.array(
            [action.is_subnet_scan for action in self.actions], dtype=jnp.bool_
        )
        is_process_scan = jnp.array(
            [action.is_process_scan for action in self.actions], dtype=jnp.bool_
        )
        is_remote = jnp.array(
            [action.is_remote for action in self.actions], dtype=jnp.bool_
        )

        # Access levels and values
        access_levels = jnp.array(
            [action.access for action in self.actions], dtype=jnp.uint8
        )
        values = jnp.array([action.value for action in self.actions], dtype=jnp.int16)

        # Extract type information and other flags
        action_types = jnp.array(
            [action.action_type for action in self.actions], dtype=jnp.uint8
        )
        service_ids = jnp.array(
            [action.service_id for action in self.actions], dtype=jnp.uint8
        )
        os_ids = jnp.array([action.os_id for action in self.actions], dtype=jnp.uint8)
        process_ids = jnp.array(
            [action.process_id for action in self.actions], dtype=jnp.uint8
        )

        # Store arrays
        self.action_arrays = {
            "targets": targets,
            "costs": costs,
            "probs": probs,
            "req_access": req_access,
            "action_types": action_types,
            "service_ids": service_ids,
            "os_ids": os_ids,
            "process_ids": process_ids,
            "is_noop": is_noop,
            "is_exploit": is_exploit,
            "is_privilege_escalation": is_privesc,
            "is_service_scan": is_service_scan,
            "is_os_scan": is_os_scan,
            "is_subnet_scan": is_subnet_scan,
            "is_process_scan": is_process_scan,
            "is_remote": is_remote,
            "access": access_levels,
            "values": values,
        }

    def get_action(self, idx: int) -> ActionData:
        """Get action by index."""
        if idx >= self.n:
            raise IndexError(f"Action index {idx} out of range (max: {self.n - 1})")
        return self.actions[idx]

    def get_action_from_arrays(self, idx: int) -> ActionData:
        """Get action by index using JAX arrays."""
        return get_action_from_arrays(self.action_arrays, idx)

    def get_actions_for_host(self, host_idx: int) -> List[ActionData]:
        """Get all actions targeting a specific host."""
        return [action for action in self.actions if action.target_host_idx == host_idx]

    def get_scan_actions_for_host(self, host_idx: int) -> List[ActionData]:
        """Get all scan actions for a specific host."""
        return [
            action
            for action in self.actions
            if action.target_host_idx == host_idx and action.is_scan
        ]

    def get_exploit_actions_for_host(self, host_idx: int) -> List[ActionData]:
        """Get all exploit actions for a specific host."""
        return [
            action
            for action in self.actions
            if action.target_host_idx == host_idx and action.is_exploit
        ]

    def get_privesc_actions_for_host(self, host_idx: int) -> List[ActionData]:
        """Get all privilege escalation actions for a specific host."""
        return [
            action
            for action in self.actions
            if action.target_host_idx == host_idx and action.is_privilege_escalation
        ]

    @property
    def num_actions(self) -> int:
        """Total number of actions in the action space."""
        return self.n

    def __len__(self) -> int:
        """Return the size of the action space."""
        return self.n
