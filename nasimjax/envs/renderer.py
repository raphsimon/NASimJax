# Copyright (c) 2026 The NASimJax Authors.
# Licensed under the MIT License. See LICENSE in the project root.
"""
This file should contain all the rendering functionality. It's a class
that shall be given a state or an observation, and render it accordingly.
The goal is to limit code duplication between the environment created
through scenario descriptions, and the generated environment.
"""

import jax.numpy as jnp
from prettytable import PrettyTable

from nasimjax.envs.common import NASimJaxEnvState
from nasimjax.envs.utils import ActionType, AccessLevel
from nasimjax.envs.host_vector_batched import (
    HostVectorBatched,
    reconstruct_host_vector_from_flat,
)


class Renderer:
    def __init__(self, scenario=None):
        self.scenario = scenario

    def render_state(self, mode="human", state=None, obs_array=None):
        """Render state for debugging purposes.

        Provides a readable tabular representation of the current network state.

        Parameters
        ----------
        mode : str
            rendering mode ("human" or "ansi" supported)
        state : NASimEnvState, optional
            the state to render, if None will render current state
        """
        if mode is None:
            return

        if state is None:
            # For JAX environments, we don't maintain a current_state
            # User should pass the state explicitly
            raise ValueError("State must be provided for JAX environment rendering")

        if not isinstance(state, NASimJaxEnvState):
            raise ValueError("State must be a NASimEnvState instance")

        if mode in ("human", "ansi"):
            if obs_array is not None:
                # Extract auxiliary information (last 4 elements)
                aux_info = obs_array[-4:]
                success = bool(aux_info[0] > 0.5)
                connection_error = bool(aux_info[1] > 0.5)
                permission_error = bool(aux_info[2] > 0.5)
                undefined_error = bool(aux_info[3] > 0.5)

                # Render auxiliary info table first
                self._render_auxiliary_info(
                    success, connection_error, permission_error, undefined_error
                )

            return self._render_readable_state(state.hosts)
        else:
            raise NotImplementedError(
                f"Render mode '{mode}' not supported. Use 'human' or 'ansi'."
            )

    def _render_readable_state(self, hosts: HostVectorBatched, title: str = "State"):
        """Print a readable tabular version of state to stdout.

        Args:
            hosts: HostVectorBatched containing current host states
            title: Title for the output table
        """
        from prettytable import PrettyTable

        # Create host data for table
        host_data = []
        num_hosts = hosts.subnet_address.shape[0]

        for i in range(num_hosts):
            subnet_addr = int(jnp.argmax(hosts.subnet_address[i]))

            # Get access level as string
            access_val = int(jnp.argmax(hosts.access_level[i, :]))
            access_map = {0: "None", 1: "User", 2: "Root"}
            access = access_map.get(access_val, "Unknown")

            # Boolean flags
            compromised = bool(access_val > AccessLevel.NONE)
            reachable = bool(hosts.reachable[i] > 0.5)
            discovered = bool(hosts.discovered[i] > 0.5)
            sensitive = bool(hosts.sensitive[i] > 0.5)

            # OS services (get indices of active ones)
            os_list = []
            for os_idx in range(hosts.os.shape[1]):
                if hosts.os[i, os_idx] > 0.5:
                    if self.scenario:
                        os_name = (
                            self.scenario.os[os_idx]
                            if os_idx < len(self.scenario.os)
                            else f"OS_{os_idx}"
                        )
                    else:
                        os_name = f"OS_{os_idx}"
                    os_list.append(os_name)
            os_str = ", ".join(os_list) if os_list else "None"

            # Services
            service_list = []
            for svc_idx in range(hosts.services.shape[1]):
                if hosts.services[i, svc_idx] > 0.5:
                    if self.scenario:
                        svc_name = (
                            self.scenario.services[svc_idx]
                            if svc_idx < len(self.scenario.services)
                            else f"SVC_{svc_idx}"
                        )
                    else:
                        svc_name = f"SVC_{svc_idx}"
                    service_list.append(svc_name)
            services_str = ", ".join(service_list) if service_list else "None"

            # Processes
            process_list = []
            for proc_idx in range(hosts.processes.shape[1]):
                if hosts.processes[i, proc_idx] > 0.5:
                    if self.scenario:
                        proc_name = (
                            self.scenario.processes[proc_idx]
                            if proc_idx < len(self.scenario.processes)
                            else f"PROC_{proc_idx}"
                        )
                    else:
                        proc_name = f"PROC_{proc_idx}"
                    process_list.append(proc_name)
            process_str = ", ".join(process_list) if process_list else "None"

            host_data.append(
                {
                    "Host": i,
                    "Subnet": subnet_addr,
                    "Compromised": compromised,
                    "Reachable": reachable,
                    "Discovered": discovered,
                    "Sensitive": sensitive,
                    "Access": access,
                    "OS": os_str,
                    "Services": services_str,
                    "Processes": process_str,
                }
            )

        # Create and populate table
        if host_data:
            headers = list(host_data[0].keys())
            table = PrettyTable(headers)

            for host in host_data:
                row = [str(host[k]) for k in headers]
                table.add_row(row)

            print(f"{title}:")
            print(table)

    def _render_auxiliary_info(
        self,
        success: bool,
        connection_error: bool,
        permission_error: bool,
        undefined_error: bool,
    ):
        """Render auxiliary information table.

        Args:
            success: Whether the last action succeeded
            connection_error: Whether there was a connection error
            permission_error: Whether there was a permission error
            undefined_error: Whether there was an undefined error
        """
        aux_table = PrettyTable(
            ["Success", "Connection Error", "Permission Error", "Undefined Error"]
        )
        aux_table.add_row(
            [success, connection_error, permission_error, undefined_error]
        )
        print(aux_table)

    def render_observation(self, mode="human", obs=None, state=None):
        """Render observation for debugging partially observable environments.

        Reconstructs the host structure from the observation array and displays
        what the agent observes. This works without requiring action context.

        Parameters
        ----------
        mode : str
            rendering mode ("human" or "ansi" supported)
        state : NASimEnvState, optional
            the state to render observation from
        """
        if mode is None:
            return

        if obs is None:
            raise ValueError(
                "Observation must be provided for JAX environment rendering"
            )

        if not isinstance(state, NASimJaxEnvState):
            raise ValueError("State must be a NASimEnvState instance")

        if not isinstance(obs, jnp.ndarray):
            raise ValueError("Observation must be a jnp.ndarray instance")

        if mode in ("human", "ansi"):
            # Use the reconstructed observation rendering
            return self.render_observation_from_array(
                obs, state, title="Agent Observation"
            )
        else:
            raise NotImplementedError(
                f"Render mode '{mode}' not supported. Use 'human' or 'ansi'."
            )

    def _render_obs_from_array(self, obs_array: jnp.ndarray):
        """Render observation from raw array when no action context is available."""
        print("Raw Observation Array:")
        print(f"Shape: {obs_array.shape}")
        print(f"Values: {obs_array}")
        print("(Cannot provide detailed host information without action context)")

    def render_observation_from_array(
        self,
        obs_array: jnp.ndarray,
        state: NASimJaxEnvState,
        title: str = "Reconstructed Observation",
    ):
        """Render observation from raw array by reconstructing hosts structure.

        Args:
            obs_array: Complete observation array from environment
            title: Title for the rendered output
        """
        # Extract auxiliary information (last 4 elements)
        aux_info = obs_array[-4:]
        success = bool(aux_info[0] > 0.5)
        connection_error = bool(aux_info[1] > 0.5)
        permission_error = bool(aux_info[2] > 0.5)
        undefined_error = bool(aux_info[3] > 0.5)

        # Render auxiliary info table first
        self._render_auxiliary_info(
            success, connection_error, permission_error, undefined_error
        )

        host_data = obs_array[: -(len(ActionType) + 4)]

        # Then render the host state
        reconstructed_hosts = reconstruct_host_vector_from_flat(host_data, state.hosts)
        self._render_readable_state(reconstructed_hosts, title=title)
