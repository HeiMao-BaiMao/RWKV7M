# TRC Scaling Design

This document defines the scaling contract for Google TPU Research Cloud. It
separates accelerator topology, JAX process ownership, logical model/data axes,
input ownership, and checkpoint storage. Treating any two of these as the same
boundary produced real failures on a four-worker v5e-16 slice.

## Topology hierarchy

```text
training job
  TPU slices connected by DCN
    TPU chips connected by ICI
      JAX processes / worker VMs
        process-addressable devices
```

A JAX process boundary inside one TPU slice is not a slow-network boundary.
Conversely, a TPU slice boundary is significant even when process counts and
logical mesh factors happen to divide evenly.

The production mesh policy is therefore:

- one identified TPU slice: call topology-aware `jax.make_mesh()` over all
  devices;
- multiple TPU slices: group devices by `slice_index`, then call
  `create_hybrid_device_mesh(ici_mesh_shape, dcn_mesh_shape, devices)`;
- devices without slice metadata: use process granules as a compatibility
  fallback only when TPU physical coordinates are globally unique;
- repeated TPU coordinates without `slice_index`: reject the mesh because a
  DCN-aware slice grouping cannot be proven;
- allocate fast inner factors from the trailing logical axis first, keeping the
  `model` axis on ICI where the requested factors allow it;
- reject a requested logical shape that cannot be factored exactly into the
  observed inner and outer topology.

This follows the [JAX multi-process](https://docs.jax.dev/en/latest/multi_process.html)
and [hybrid device mesh](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.mesh_utils.create_hybrid_device_mesh.html)
contracts. A real multi-slice run is still required to validate the current
factor policy against an allocated TRC pod topology.

## Input ownership contract

The input pipeline does not use `mesh.local_mesh`. It enumerates the global
`mesh.devices` array and derives the unique logical `data` coordinates touched
by each process.

For logical data-axis size `D` and global microbatch `B`:

```text
B % D == 0
rows per logical data shard = B / D
process rows = rows per shard * data coordinates addressed by that process
```

Each logical data coordinate owns a deterministic binidx sampler with
`rank=data_coordinate` and `world_size=D`. If a model shard spans two or more
processes, those processes address the same data coordinate and must construct
identical local rows. Different values for a replicated shard are invalid even
if JAX does not report an error. This rule is also called out by the
[JAX distributed data loading guide](https://docs.jax.dev/en/latest/distributed_data_loading.html).

`jax.make_array_from_process_local_data` receives the explicit global shape.
This supports process layouts in which a process owns several logical data
shards or only a replicated view of one shard.

## State and parameter contract

The Explicit NNX scale path keeps these placements:

```text
hidden/logits        P("data", None, "model")
WKV state            P("data", "model", None, None)
screening slots      P("data", None, "model")
token/age state      P("data", None)
```

Row- and column-parallel parameters are initialized directly under the target
mesh. Model dimensions selected for the model axis must divide `d_model`,
`n_heads`, `d_slot` when screening is active, and `vocab_size` when vocabulary
parallelism is active.

For models that fit per device, data parallelism can be faster than model
parallelism. The model axis exists to satisfy HBM and scale requirements, not
as an automatic throughput improvement. Mesh selection must be based on the
largest production shape, HBM preflight, and an end-to-end throughput gate.

## Checkpoint storage contract

All processes participate in an Orbax save. Therefore they must see the same
checkpoint namespace and the same files. A path such as `/tmp/run` is private
on ordinary TPU worker VMs even if every process receives the same string.

TRC jobs use two roots:

```text
--output-dir      process-0 local logs and run artifacts
--checkpoint-dir shared GCS prefix or verified shared filesystem
```

Multi-process checkpointing requires Orbax and an explicit
`--checkpoint-dir`. `gs://` URIs are preserved rather than passed through
`pathlib.Path.resolve()`. The TPU dependency extra installs `gcsfs` so the
project-owned `checkpoint.json` metadata can use the same GCS namespace as the
Orbax state. Orbax itself requires every host to call save and stores a
per-process OCDBT fragment before finalizing the global view; see the
[Orbax format guide](https://orbax.readthedocs.io/en/latest/guides/checkpoint/checkpoint_format.html)
and [Checkpointer save contract](https://orbax.readthedocs.io/en/stable/_modules/orbax/checkpoint/_src/checkpointers/checkpointer.html).

The shared-GCS save, restore under the same topology, restore under a different
topology, and preemption drill are required production gates. They have not yet
been completed on TPU.

## Validation ladder

Do not jump directly from a local unit test to a long pod run. Each larger tier
must pass the previous gates:

1. Unit gate: mesh factoring, replica-safe sampler streams, checkpoint policy,
   config divisibility, and local Orbax round trip.
2. Multi-process array gate: exact process-local assembly, elementwise output,
   and global reduction for data-only and multiple data/model factorizations.
3. Complete-model gate: screening read/write forward, backward, optimizer,
   state shardings, parameter shardings, and compiled collective audit.
4. Input gate: actual binidx CLI update through prefetch and global-array
   construction.
5. Checkpoint gate: shared-GCS periodic save, final save deduplication, restore,
   next optimizer update, carried runtime state, and checkpoint rotation.
6. Failure gate: terminate one worker after a committed checkpoint, recreate
   the job, restore dataset/RNG/runtime state, and verify the next update.
7. Multi-slice gate: repeat the array and complete-model gates with a model axis
   both confined to ICI and intentionally crossing DCN.
8. Performance gate: steady-state windows with compile, checkpoint, validation,
   and host logging excluded or reported separately; capture XProf collective,
   input, and checkpoint traces.

## Current evidence

On 2026-07-16, a single v5e-16 slice exposed 16 devices through four JAX
processes. These JAX 0.10.0 devices did not expose `slice_index`, so this run
validated the process-granule compatibility path. Each process corresponded to
a contiguous `2x2` quadrant of the physical `4x4` topology. The array gate
passed exactly for `data=16`, `data=4/model=4`,
`data=2/model=8`, and `data=1/model=16`. The last two layouts replicate a data
shard across two and four processes respectively.

The complete-model `data=1/model=16` gate passed with read/write screening,
vocabulary parallelism, finite loss, and optimizer step 1. The actual binidx
CLI also completed a training update. The checkpoint gate found that both the
Flax global gather and worker-private local Orbax paths are invalid for this
topology. The CLI/storage changes resulting from that finding are locally
tested but not yet revalidated against shared GCS on TPU.

The v5e-16 result proves multi-process execution within one slice. It does not
prove multi-slice DCN scaling, 7B execution, GCS checkpoint recovery, or linear
throughput scaling.
