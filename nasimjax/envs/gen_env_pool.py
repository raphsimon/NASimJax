# Copyright (c) 2026 The NASimJax Authors.
# Licensed under the MIT License. See LICENSE in the project root.
"""
This class represents a buffer of environments that we can sample from during
training. This is important in conjuction with generated environments, as we
aim to cover a significant part of the initial state distribution to be able
to learn a more robust policy. More elaborate methods of sampling environments
may also help with sample efficiency.
"""

import jax
import jax.numpy as jnp
from functools import partial

from nasimjax.envs.common import NASimJaxEnvParams
from nasimjax.envs.host_vector_batched import HostVectorBatched
from nasimjax.envs.network_generator import generate


class GenEnvPool:
    """
    The buffer saves tuples of (hosts: HostVectorBatched, traffic_rules: jnp.array)
    """

    def __init__(
        self,
        key: jax.random.key,
        env_pool_size: int,
        params: NASimJaxEnvParams,
    ):
        self.env_pool_size = env_pool_size
        self.params = params

        gen_keys = jax.random.split(key, env_pool_size)

        # # Stack the pytree (dataclass with jnp.ndarray properties)
        # # This will stack each array property in the dataclass
        # self.host_vectors_batched = jax.tree_util.tree_map(
        #     lambda *args: jnp.stack(args), *host_vectors
        # )
        # # Stack the separate arrays
        # self.traffic_rules_batched = jnp.stack(traffic_rules)

        def wrapped_gen(
            key
        ):
            return generate(
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

        self.hosts_batched, self.traffic_rules_batched, _, _ = jax.vmap(wrapped_gen)(gen_keys)

    def _index_env(self, i: int) -> HostVectorBatched:
        """Extract the i-th environment out of the batched HostVectorBatched."""
        return jax.tree_util.tree_map(lambda x: x[i], self.hosts_batched)

    @partial(jax.jit, static_argnames=("self",))
    def sample(self, key: jax.random.PRNGKey):
        """Sample a single environment by index"""
        idx = jax.random.randint(key, shape=(), minval=0, maxval=self.env_pool_size)
        hosts = self._index_env(idx)
        traffic_rules = self.traffic_rules_batched[idx]
        return hosts, traffic_rules


if __name__ == "__main__":
    # Test memory consumption of the buffer

    import time
    import numpy as np
    from typing import Any

    t0 = time.time()

    env_pool = GenEnvPool(
        key=jax.random.PRNGKey(0),
        env_pool_size=1000,
        params=NASimJaxEnvParams(
            num_hosts=100,
            num_processes=6,
            num_services=6,
            num_os=4,
            num_subnets=20,
            topology_density=0.1,
            service_density=0.2,
            process_density=0.2,
            sensitive_density=0.05,
            distribute_homogeneous=True,
        )
    )

    print(f"Total generation time: {time.time() - t0:.2f} s")

    def sizeof_jax(obj: Any) -> int:
        """Recursively compute memory usage of JAX arrays inside an object."""
        if isinstance(obj, jnp.ndarray):
            return obj.nbytes
        else:
            return sum(
                sizeof_jax(getattr(obj, f.name))
                for f in obj.__dataclass_fields__.values()
            )

    total_bytes = 0

    total_bytes += sizeof_jax(env_pool.hosts_batched)
    total_bytes += sizeof_jax(env_pool.traffic_rules_batched)

    print(f"Env pool memory: {total_bytes / 1024**2:.2f} MB")

    # Even better: use JAX's tree utilities
    def sizeof_jax_robust(pytree: Any) -> int:
        """Use JAX's tree utilities to compute memory usage."""
        leaves = jax.tree_util.tree_leaves(pytree)
        return sum(
            leaf.nbytes
            for leaf in leaves
            if isinstance(leaf, (jnp.ndarray, np.ndarray))
        )

    total_bytes = 0

    total_bytes += sizeof_jax_robust(env_pool.hosts_batched)
    total_bytes += sizeof_jax_robust(env_pool.traffic_rules_batched)

    print(f"Env pool memory (robust calc): {total_bytes / 1024**2:.2f} MB")

    # key = jax.random.key(7)
    # host, traffic_rules = env_pool.sample(key)

    # print(host)
    # print(traffic_rules)
