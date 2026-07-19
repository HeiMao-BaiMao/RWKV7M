"""Measure local JAX/NNX train compute with a fixed device-resident batch."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from benchmark_common import (
    COMPUTE_BENCHMARK_SCHEMA_VERSION,
    fixed_batch_content_sha256,
    gc_policy,
    measurement_contract,
    timing_summary,
)
from cuda_profiler_range import cuda_profiler_range
from rwkv7m.api import create_train_runtime
from rwkv7m.cli.config import (
    add_optimizer_backend_arg,
    add_training_vocab_tiling_args,
    apply_execution_overrides,
)
from rwkv7m.io import load_model_config
from rwkv7m.kernels import (
    resolve_optimizer_backend,
    resolve_screening_backend,
    resolve_training_loss_backend,
    resolve_wkv_backend,
)
from rwkv7m.model import MODEL_PRESET_NAMES, model_preset
from rwkv7m.train.nnx_train import nnx_model_loss


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-batch", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-config")
    source.add_argument("--model-preset", choices=MODEL_PRESET_NAMES)
    parser.add_argument(
        "--variant",
        choices=("baseline", "screening", "read_write"),
        default="baseline",
    )
    parser.add_argument("--ctx-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-iterations", type=int, default=50)
    parser.add_argument("--disable-python-gc", action="store_true")
    parser.add_argument(
        "--profile-mode",
        choices=("none", "xprof", "cuda_profiler_api"),
        default="none",
    )
    parser.add_argument(
        "--profile-target",
        choices=("forward", "backward", "optimizer", "full_step"),
        default="full_step",
    )
    parser.add_argument("--profile-iterations", type=int, default=3)
    parser.add_argument("--profile-output", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    remat = parser.add_mutually_exclusive_group()
    remat.add_argument("--remat-blocks", dest="remat_blocks", action="store_true")
    remat.add_argument("--no-remat-blocks", dest="remat_blocks", action="store_false")
    parser.set_defaults(remat_blocks=None)
    sequence = parser.add_mutually_exclusive_group()
    sequence.add_argument("--sequence-chunk-size", type=int, default=None)
    sequence.add_argument("--no-sequence-chunking", action="store_true")
    head = parser.add_mutually_exclusive_group()
    head.add_argument("--head-chunk-size", type=int, default=None)
    head.add_argument("--no-head-chunking", action="store_true")
    add_training_vocab_tiling_args(parser)
    add_optimizer_backend_arg(parser)
    args = parser.parse_args(argv)
    if args.ctx_len <= 0 or args.batch_size <= 0:
        parser.error("--ctx-len and --batch-size must be positive")
    if args.benchmark_warmup < 0 or args.benchmark_iterations <= 0:
        parser.error("benchmark warmup must be non-negative and iterations positive")
    if args.profile_iterations <= 0:
        parser.error("--profile-iterations must be positive")
    if args.profile_mode == "xprof" and args.profile_output is None:
        parser.error("--profile-output is required for --profile-mode=xprof")
    return args


def _git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _measure(function, arguments, *, warmup, iterations):
    for _ in range(warmup):
        jax.block_until_ready(function(*arguments))
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        result = function(*arguments)
        jax.block_until_ready(result)
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return timing_summary(samples)


def _measure_full_step(
    function,
    train_state,
    *,
    warmup,
    iterations,
):
    for _ in range(warmup):
        train_state, loss = function(train_state)
        jax.block_until_ready((train_state, loss))
    samples = []
    last_loss = None
    for _ in range(iterations):
        started = time.perf_counter_ns()
        train_state, last_loss = function(train_state)
        jax.block_until_ready((train_state, last_loss))
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return timing_summary(samples), last_loss


def _measure_optimizer(
    function,
    train_state,
    gradients,
    *,
    warmup,
    iterations,
):
    for _ in range(warmup):
        train_state = function(train_state, gradients)
        jax.block_until_ready(train_state)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        train_state = function(train_state, gradients)
        jax.block_until_ready(train_state)
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return timing_summary(samples)


def _run_profile_iterations(
    *,
    target,
    iterations,
    forward,
    params,
    pullback,
    loss_cotangent,
    optimizer_step,
    bundle_state,
    gradients,
    full_step,
):
    active_state = bundle_state
    for iteration in range(iterations):
        with jax.profiler.TraceAnnotation(
            f"rwkv7m_{target}", iteration=iteration
        ):
            if target == "forward":
                result = forward(params)
            elif target == "backward":
                result = pullback(loss_cotangent)
            elif target == "optimizer":
                active_state = optimizer_step(active_state, gradients)
                result = active_state
            else:
                active_state, loss = full_step(active_state)
                result = (active_state, loss)
            jax.block_until_ready(result)


def main(argv=None):
    args = parse_args(argv)
    config = (
        load_model_config(args.model_config)
        if args.model_config
        else model_preset(args.model_preset)
    )
    config.use_screening = args.variant != "baseline"
    if config.use_screening:
        config.screening.use_write_screening = args.variant == "read_write"
    config = apply_execution_overrides(config, args)
    if args.ctx_len > config.max_seq_len:
        raise SystemExit("--ctx-len exceeds the model maximum sequence length")
    phase = "read_write" if args.variant == "read_write" else "read_screening_only"

    with np.load(args.fixed_batch, allow_pickle=False) as archive:
        fixed_batch = {
            "input_ids": archive["input_ids"],
            "target_ids": archive["target_ids"],
        }
        if "mask" in archive:
            fixed_batch["mask"] = archive["mask"]
    if fixed_batch["input_ids"].shape != (args.batch_size, args.ctx_len):
        raise SystemExit("fixed batch shape does not match --batch-size/--ctx-len")
    if fixed_batch["input_ids"].min() < 0 or fixed_batch["input_ids"].max() >= config.vocab_size:
        raise SystemExit("fixed batch contains token ids outside the model vocabulary")
    fixed_batch_file_sha256 = hashlib.sha256(
        args.fixed_batch.read_bytes()
    ).hexdigest()
    fixed_batch_content_fingerprint = fixed_batch_content_sha256(fixed_batch)
    fixed_batch = jax.device_put(fixed_batch)
    runtime, train_state = create_train_runtime(
        jax.random.key(args.seed),
        config,
        batch_size=args.batch_size,
        total_steps=args.benchmark_warmup + args.benchmark_iterations + 1,
    )
    fixed_rwkv = jax.device_put(runtime.initial_rwkv_state)
    fixed_screen = jax.device_put(runtime.initial_screen_state)
    graphdef, params = nnx.split(train_state.model, nnx.Param)
    parameter_count = sum(value.size for value in jax.tree.leaves(params))
    bundle_graphdef, bundle_state = nnx.split(
        (train_state.model, train_state.optimizer)
    )
    phase_training_step = jnp.zeros((), dtype=jnp.uint32)

    def loss_function(active_params):
        model = nnx.merge(graphdef, active_params)
        loss, _ = nnx_model_loss(
            model,
            fixed_batch,
            fixed_rwkv,
            fixed_screen,
            phase=phase,
            deterministic=False,
            include_l2wrap=True,
            training_step=phase_training_step,
        )
        return loss

    forward = jax.jit(loss_function)
    prepared_loss, pullback = jax.vjp(forward, params)
    jax.block_until_ready(prepared_loss)
    loss_cotangent = jnp.ones_like(prepared_loss)
    gradients = pullback(loss_cotangent)[0]
    jax.block_until_ready(gradients)

    @jax.jit
    def optimizer_step(active_state, active_gradients):
        model, optimizer = nnx.merge(bundle_graphdef, active_state)
        optimizer.update(model, active_gradients)
        return nnx.state((model, optimizer))

    @jax.jit
    def full_step(active_state):
        model, optimizer = nnx.merge(bundle_graphdef, active_state)

        def active_loss(active_model):
            loss, _ = nnx_model_loss(
                active_model,
                fixed_batch,
                fixed_rwkv,
                fixed_screen,
                phase=phase,
                deterministic=False,
                include_l2wrap=True,
                training_step=optimizer.step[...],
            )
            return loss

        loss, active_gradients = nnx.value_and_grad(active_loss)(model)
        optimizer.update(model, active_gradients)
        return nnx.state((model, optimizer)), loss

    # Ensure all setup, placement, and the VJP residual construction complete
    # before any measurement window opens.
    jax.block_until_ready((fixed_batch, fixed_rwkv, fixed_screen, gradients))
    with gc_policy(args.disable_python_gc):
        timings = {
            "forward": _measure(
                forward,
                (params,),
                warmup=args.benchmark_warmup,
                iterations=args.benchmark_iterations,
            ),
            "backward": _measure(
                pullback,
                (loss_cotangent,),
                warmup=args.benchmark_warmup,
                iterations=args.benchmark_iterations,
            ),
            "optimizer": _measure_optimizer(
                optimizer_step,
                bundle_state,
                gradients,
                warmup=args.benchmark_warmup,
                iterations=args.benchmark_iterations,
            ),
        }
        timings["full_step"], last_loss = _measure_full_step(
            full_step,
            bundle_state,
            warmup=args.benchmark_warmup,
            iterations=args.benchmark_iterations,
        )

    profile = {
        "mode": args.profile_mode,
        "target": args.profile_target,
        "iterations": args.profile_iterations,
    }
    if args.profile_mode == "xprof":
        args.profile_output.mkdir(parents=True, exist_ok=True)
        options = jax.profiler.ProfileOptions()
        options.python_tracer_level = 0
        options.host_tracer_level = 1
        jax.profiler.start_trace(
            str(args.profile_output),
            create_perfetto_trace=True,
            profiler_options=options,
        )
        try:
            _run_profile_iterations(
                target=args.profile_target,
                iterations=args.profile_iterations,
                forward=forward,
                params=params,
                pullback=pullback,
                loss_cotangent=loss_cotangent,
                optimizer_step=optimizer_step,
                bundle_state=bundle_state,
                gradients=gradients,
                full_step=full_step,
            )
        finally:
            jax.profiler.stop_trace()
        profile["output"] = str(args.profile_output.resolve())
    elif args.profile_mode == "cuda_profiler_api":
        with cuda_profiler_range():
            _run_profile_iterations(
                target=args.profile_target,
                iterations=args.profile_iterations,
                forward=forward,
                params=params,
                pullback=pullback,
                loss_cotangent=loss_cotangent,
                optimizer_step=optimizer_step,
                bundle_state=bundle_state,
                gradients=gradients,
                full_step=full_step,
            )

    tokens = args.batch_size * args.ctx_len
    timings["full_step"]["tokens_per_second_median"] = (
        tokens * 1000.0 / timings["full_step"]["median_ms"]
    )
    devices = jax.devices()
    payload = {
        "schema_version": COMPUTE_BENCHMARK_SCHEMA_VERSION,
        "benchmark_kind": "train_compute_only",
        "framework": "jax_nnx",
        "recorded_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_revision": _git_revision(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runtime": {"jax": jax.__version__, "jaxlib": jax.lib.__version__},
        "devices": [
            {
                "platform": device.platform,
                "device_kind": device.device_kind,
                "id": device.id,
            }
            for device in devices
        ],
        "shape": {
            "batch": args.batch_size,
            "tokens": args.ctx_len,
            "variant": args.variant,
            "dtype": config.dtype,
            "parameter_count": parameter_count,
            "n_layers": config.n_layers,
            "d_model": config.d_model,
            "d_ffn": config.d_ffn,
            "n_heads": config.n_heads,
            "head_size": config.head_size,
            "vocab_size": config.vocab_size,
        },
        "execution": {
            "remat_blocks": config.remat_blocks,
            "sequence_chunk_size": config.sequence_chunk_size,
            "head_chunk_size": config.head_chunk_size,
            "training_vocab_tile_size": config.training_vocab_tile_size,
            "wkv_backend": resolve_wkv_backend(),
            "screening_backend": resolve_screening_backend(),
            "training_loss_backend": (
                resolve_training_loss_backend()
                if config.training_vocab_tile_size is not None
                else "full_logits_xla"
            ),
            "optimizer_backend": resolve_optimizer_backend(
                config.optimizer_backend
            ),
        },
        "fixed_batch": {
            "path": str(args.fixed_batch.resolve()),
            "sha256": fixed_batch_file_sha256,
            "content_sha256": fixed_batch_content_fingerprint,
        },
        "method": measurement_contract(
            warmup=args.benchmark_warmup,
            iterations=args.benchmark_iterations,
            disable_python_gc=args.disable_python_gc,
        ),
        "profile": profile,
        "phase_details": {
            "forward": "loss forward from fixed params and fixed recurrent state",
            "backward": "VJP pullback from one precomputed forward residual",
            "optimizer": "NNX optimizer update from fixed gradients",
            "full_step": "value_and_grad plus optimizer with no intermediate host barrier",
            "training_step": (
                "forward/backward use step zero; full_step uses the bundled "
                "optimizer step so temporary curricula match real training"
            ),
        },
        "optimizer": {
            "implementation": resolve_optimizer_backend(
                config.optimizer_backend
            ),
            "lr_init": config.lr_init,
            "weight_decay": config.weight_decay,
            "grad_clip": config.max_grad_norm,
            "betas": [config.adam_beta1, config.adam_beta2],
            "eps": config.adam_eps,
        },
        "last_loss": float(last_loss),
        "timings": timings,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
