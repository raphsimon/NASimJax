# `renderer.py` — Debug rendering

The `Renderer` class prints human-readable tables of a network state or an
agent observation. It is not used on the hot path — rendering is impure and
will trigger device-to-host transfers — but it is invaluable when debugging
scenarios, transitions and partial observations.

The renderer understands both the full [`HostVectorBatched`](host_vector_batched.md)
and the flat observation vector produced by the environment, and relies on
`reconstruct_host_vector_from_flat` when given the latter.

```python
env.render_state(state=state)          # full network state
env.render_observation(state=state)    # agent's partial view
```

## API reference

::: nasimjax.envs.renderer.Renderer
