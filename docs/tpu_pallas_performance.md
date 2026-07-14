# TPU Pallas WKV Performance Validation

## Summary

On 2026-07-14, the new TPU Pallas WKV implementation was compared with the
portable `lax.scan` recurrence on a Google Cloud TPU v5e. For the tested
`T=128, B=1, H=12, N=64` BF16 WKV shape, Pallas reduced forward latency by
10.0% and forward-plus-backward latency by 29.8%.

| Operation | Pallas TPU | `lax.scan` reference | Speedup | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Forward | 0.276 ms | 0.307 ms | 1.11x | 10.0% |
| Forward + backward | 0.538 ms | 0.766 ms | 1.42x | 29.8% |

These are WKV-only kernel measurements on one TPU device. They are not
complete-model tokens-per-second results and must not be combined with the
older end-to-end preset measurements as if they used the same workload.

## Implementation change

The previous accelerator path expressed the time recurrence as `lax.scan` over
general JAX operations. The new path uses:

- a TPU-specific persistent Pallas forward kernel;
- a TPU-specific Pallas reverse recurrence connected through `custom_vjp`;
- BF16 vector storage with FP32 state and accumulation;
- FP32 `sa` values and interval state checkpoints for backward reconstruction;
- a local `jax.shard_map` boundary for sharded Pallas execution;
- `pl.dslice`-based, slice-only `Ref` access required by sharded JAX 0.10
  Pallas references.

Automatic dispatch now selects `pallas_tpu` on TPU. CPU continues to use the
reference recurrence. FFI is not selected automatically.

## Test environment

| Item | Value |
| --- | --- |
| Zone | `us-west4-a` |
| TPU | single-host `v5litepod-4` slice, four TPU v5e devices |
| Microbenchmark placement | one TPU device from the slice |
| Python | 3.13.14 |
| JAX / jaxlib | 0.10.0 / 0.10.0 |
| libtpu | 0.0.40 |
| Vector dtype | BF16 |
| State and accumulation dtype | FP32 |
| Transparent hugepages | disabled |

The temporary VM name was `rwkv7m-pallas-260714`.

## Microbenchmark method

The benchmark used six random BF16 vector inputs and one random FP32 initial
state with the following WKV shape:

```text
time       T = 128
batch      B = 1
heads      H = 12
head size  N = 64
vectors        [T, B, H, N]
state          [B, H, N, N]
```

Pallas and reference functions were compiled independently with `jax.jit`.
Compilation and one warmup execution were excluded. Each reported latency is
the elapsed time for 100 sequential calls divided by 100, with
`jax.block_until_ready` applied to every result.

The backward measurement used `jax.value_and_grad` over all six vector inputs
and the initial state. Its scalar objective was the sum of squared FP32-cast WKV
activations. The final recurrent state did not contribute directly to this
microbenchmark loss.

Raw final measurements were:

```text
Pallas forward:              0.27634697 ms
reference forward:           0.30719018 ms
Pallas forward + backward:   0.53761736 ms
reference forward + backward: 0.76567165 ms
```

The Pallas and reference final FP32 states had a measured maximum absolute
difference of `0.0` in this run. Separate real-accelerator parity tests also
checked BF16 outputs and gradients for all seven differentiable inputs.

## Integration validation

The timing result was interpreted only after the following real-TPU checks
passed with `interpret=False`:

1. BF16 forward output and FP32 final-state parity against the reference.
2. Gradient parity for all six vector inputs and the initial FP32 state.
3. A complete one-layer BF16 NNX model loss and parameter backward pass.
4. A four-device `data=1, model=4` forward and optimizer step.

The four-device audit completed with finite outputs, loss `3.9594927`, and
optimizer step 1. Its post-SPMD executable retained the previous collective
counts: 13 all-reduces, 3 all-to-alls, and no all-gathers.

The real-accelerator parity test is invoked with:

```bash
uv run pytest tests/test_wkv_pallas_accelerator.py -q
```

The four-way TPU model-sharding audit used:

```bash
uv run rwkv7m-audit-nnx-model-parallel \
  --model-axis-size 4 \
  --d-model 32 \
  --n-heads 4 \
  --head-size 8 \
  --screening \
  --write-screening
```

## Interpretation and limitations

The measured result supports the narrower conclusion that the current Pallas
WKV recurrence is faster than the current `lax.scan` WKV reference for this one
v5e shape, especially in backward. It does not yet establish:

- complete-model or optimizer-step throughput improvement;
- performance for other batch, sequence, head, or checkpoint shapes;
- multi-host TPU scaling;
- XProf-confirmed bottleneck attribution;
- results with transparent hugepages enabled;
- production stability across future experimental Pallas API changes.

The next performance gate should compare complete steady-state training steps
for tracked presets, then use XProf to tune checkpoint intervals, VMEM usage,
and program layout. Kernel-level numbers should remain diagnostic evidence, not
the final dispatch criterion.

## Resource cleanup

After all correctness, sharding, and timing checks completed, the temporary TPU
VM `rwkv7m-pallas-260714` was deleted. The Google Cloud delete operation
completed successfully, and a subsequent TPU VM listing for `us-west4-a`
returned no remaining TPU VMs.
