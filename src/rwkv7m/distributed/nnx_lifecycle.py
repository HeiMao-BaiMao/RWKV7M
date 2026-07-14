from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp


class NNXShardingProbe(nnx.Module):
    """Small tensor-parallel module used to validate the NNX lifecycle."""

    def __init__(self, width: int, *, rngs: nnx.Rngs):
        column_kernel = nnx.with_partitioning(
            nnx.initializers.lecun_normal(),
            (None, "model"),
        )
        column_bias = nnx.with_partitioning(nnx.initializers.zeros_init(), ("model",))
        row_kernel = nnx.with_partitioning(
            nnx.initializers.lecun_normal(),
            ("model", None),
        )
        row_bias = nnx.with_partitioning(nnx.initializers.zeros_init(), (None,))
        self.in_proj = nnx.Linear(
            width,
            width,
            kernel_init=column_kernel,
            bias_init=column_bias,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(
            width,
            width,
            kernel_init=row_kernel,
            bias_init=row_bias,
            rngs=rngs,
        )

    def __call__(self, x):
        x = jax.lax.with_sharding_constraint(x, jax.P("data", None))
        x = self.in_proj(x)
        x = jax.lax.with_sharding_constraint(x, jax.P("data", "model"))
        x = jax.nn.relu(x)
        x = self.out_proj(x)
        return jax.lax.with_sharding_constraint(x, jax.P("data", None))


def _make_probe_bundle(width, learning_rate, rng_key):
    model = NNXShardingProbe(width, rngs=nnx.Rngs(rng_key))
    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate),
        wrt=nnx.Param,
    )
    return model, optimizer


def initialize_nnx_probe(mesh, *, width=8, learning_rate=1e-3, seed=0):
    """Initialize model and Adam state directly under the active mesh."""

    @jax.jit
    def initialize(rng_key):
        return _make_probe_bundle(width, learning_rate, rng_key)

    with jax.set_mesh(mesh):
        return initialize(jax.random.key(seed))


def train_nnx_probe_step(model, optimizer, inputs, targets, mesh):
    """Run one functional NNX step while preserving every state sharding."""
    graphdef, state = nnx.split((model, optimizer))
    state_shardings = nnx.get_named_sharding(state, mesh)
    batch_sharding = jax.NamedSharding(mesh, jax.P("data", None))
    scalar_sharding = jax.NamedSharding(mesh, jax.P())

    @jax.jit(
        in_shardings=(state_shardings, batch_sharding, batch_sharding),
        out_shardings=(scalar_sharding, state_shardings),
    )
    def train_step(state, inputs, targets):
        model, optimizer = nnx.merge(graphdef, state)

        def loss_fn(model):
            predictions = model(inputs)
            return jnp.mean(jnp.square(predictions - targets))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss, nnx.state((model, optimizer))

    loss, new_state = train_step(state, inputs, targets)
    nnx.update((model, optimizer), new_state)
    return loss


def nnx_state_sharding_summary(model, optimizer):
    summary = {}
    for path, variable_state in nnx.to_flat_state(nnx.state((model, optimizer))):
        value = variable_state.get_value()
        sharding = getattr(value, "sharding", None)
        metadata = variable_state.get_metadata()
        logical_axes = metadata.get("out_sharding")
        summary["/".join(str(part) for part in path)] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sharding": None if sharding is None else str(sharding.spec),
            "logical_axes": None if logical_axes is None else list(logical_axes),
        }
    return summary


def save_nnx_probe_checkpoint(checkpoint_dir, model, optimizer):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpointer = ocp.StandardCheckpointer()
    try:
        checkpointer.save(checkpoint_dir, nnx.state((model, optimizer)))
        checkpointer.wait_until_finished()
    finally:
        checkpointer.close()
    return checkpoint_dir


def restore_nnx_probe_checkpoint(
    checkpoint_dir,
    mesh,
    *,
    width=8,
    learning_rate=1e-3,
    seed=0,
):
    """Restore into an abstract sharded model/optimizer state."""
    graphdef, abstract_state = nnx.get_abstract_model(
        lambda: _make_probe_bundle(width, learning_rate, jax.random.key(seed)),
        mesh,
    )
    checkpointer = ocp.StandardCheckpointer()
    try:
        restored_state = checkpointer.restore(
            Path(checkpoint_dir).resolve(),
            target=abstract_state,
        )
    finally:
        checkpointer.close()
    return nnx.merge(graphdef, restored_state)
