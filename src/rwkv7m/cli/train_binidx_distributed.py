import argparse
import gc
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import time
from pathlib import Path

import jax

from ..api import create_train_runtime
from ..distributed import (
    create_host_binidx_dataset,
    checkpoint_path,
    evaluate_batch_data_parallel,
    evaluate_batch_data_parallel_with_state,
    initialize_jax_distributed,
    iter_prefetched_global_batches,
    load_distributed_checkpoint_metadata,
    make_mesh,
    mean_metric_dict,
    metrics_to_host_dict,
    parameter_partition_summary,
    place_train_objects,
    restore_distributed_train_state,
    restore_distributed_runtime_state,
    rotate_checkpoints,
    runtime_version_manifest,
    save_data_parallel_checkpoint,
    train_global_batch_data_parallel,
    write_metric_record,
)
from ..io import model_config_to_dict
from ..model import MODEL_PRESET_NAMES
from ..model.nnx_model import NNXShardingConfig
from .config import add_training_vocab_tiling_args, parse_args_with_config
from .train_binidx import build_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Data-parallel rwkv7m binidx training skeleton for JAX distributed setups."
    )
    parser.add_argument("--data-file", required=True)
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument(
        "--model-config",
        default=None,
        help="Shared ModelConfig JSON used unchanged by small and large models",
    )
    model_source.add_argument(
        "--model-preset",
        choices=MODEL_PRESET_NAMES,
        default=None,
        help="Named shared NNX model configuration",
    )
    parser.add_argument("--ctx-len", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--magic-prime", type=int, default=None)
    parser.add_argument("--sampling-mode", choices=["magic", "sequential"], default="magic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", choices=["read_screening_only", "read_write"], default="read_screening_only")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--param-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--param-update-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--optimizer-state-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--gradient-accum-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--lm-head-init", choices=["orthogonal", "variance_scaled"], default="orthogonal")
    parser.add_argument("--vocab-parallel", action="store_true")
    remat_group = parser.add_mutually_exclusive_group()
    remat_group.add_argument(
        "--remat-blocks",
        dest="remat_blocks",
        action="store_true",
    )
    remat_group.add_argument(
        "--no-remat-blocks",
        dest="remat_blocks",
        action="store_false",
    )
    parser.set_defaults(remat_blocks=None)
    chunk_group = parser.add_mutually_exclusive_group()
    chunk_group.add_argument("--sequence-chunk-size", type=int, default=None)
    chunk_group.add_argument("--no-sequence-chunking", action="store_true")
    head_chunk_group = parser.add_mutually_exclusive_group()
    head_chunk_group.add_argument("--head-chunk-size", type=int, default=None)
    head_chunk_group.add_argument("--no-head-chunking", action="store_true")
    add_training_vocab_tiling_args(parser)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr-init", type=float, default=1e-3)
    parser.add_argument("--lr-final", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--lr-schedule", choices=["optax_cosine", "rwkv"], default="optax_cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ffn", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--head-size", type=int, default=32)
    parser.add_argument("--no-screening", dest="use_screening", action="store_false")
    parser.set_defaults(use_screening=True)
    parser.add_argument("--d-slot", type=int, default=64)
    parser.add_argument("--d-k", type=int, default=32)
    parser.add_argument("--d-v", type=int, default=32)
    parser.add_argument("--n-slots", type=int, default=4)
    parser.add_argument("--bank-ids", type=int, nargs="+", default=None)
    parser.add_argument("--screened-layers", type=int, nargs="*", default=[])
    parser.add_argument("--write-rel-floor", type=float, default=1e-3)
    parser.add_argument("--short-half-life-tokens", type=float, default=None)
    parser.add_argument("--mid-half-life-tokens", type=float, default=None)
    parser.add_argument("--long-half-life-tokens", type=float, default=None)
    parser.add_argument("--usage-ema-decay", type=float, default=0.99)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--keep-last-checkpoints", type=int, default=0)
    parser.add_argument("--checkpoint-backend", choices=["flax", "orbax"], default="flax")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--prefetch-size", type=int, default=2)
    parser.add_argument("--carry-state", action="store_true")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=1)
    parser.add_argument("--eval-data-file", default=None)
    parser.add_argument("--log-jsonl", default=None)
    parser.add_argument("--log-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-every", type=int, default=10)
    parser.add_argument(
        "--disable-python-gc",
        action="store_true",
        help=(
            "disable CPython cyclic GC during the training process; reference "
            "counting remains active and the prior GC state is restored on exit"
        ),
    )
    parser.add_argument("--save-best-checkpoint", action="store_true")
    parser.add_argument("--best-metric", default="loss")
    parser.add_argument("--best-mode", choices=["min", "max"], default="min")
    parser.add_argument("--mesh-axis-names", nargs="+", default=["data"])
    parser.add_argument("--mesh-axis-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--param-axis-name", default=None)
    return parse_args_with_config(parser, argv)


def _log_paths(args):
    if args.output_dir is None:
        return args.log_jsonl, args.log_csv
    output_dir = Path(args.output_dir)
    jsonl_path = args.log_jsonl if args.log_jsonl is not None else output_dir / "metrics.jsonl"
    csv_path = args.log_csv if args.log_csv is not None else output_dir / "metrics.csv"
    return jsonl_path, csv_path


def _summary_path(args):
    if args.summary_json is not None:
        return args.summary_json
    if args.output_dir is None:
        return None
    return Path(args.output_dir) / "run_summary.json"


def _now_utc():
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _print_once(info, message):
    if info["process_index"] == 0:
        print(message)


def _metric_record(args, split, step, metrics, *, elapsed=None):
    tokens = args.global_batch_size * args.ctx_len
    record = {
        "split": split,
        "step": int(step),
        "tokens": tokens,
        "tokens_per_sec": None,
        "loss": metrics.get("loss"),
        "total_loss": metrics.get("total_loss"),
        "perplexity": metrics.get("perplexity"),
        "rel_read_mean": metrics.get("rel_read_mean"),
        "active_slots_mean": metrics.get("active_slots_mean"),
        "u_norm_mean": metrics.get("u_norm_mean"),
        "rel_write_mean": metrics.get("rel_write_mean"),
        "rel_write_effective_mean": metrics.get("rel_write_effective_mean"),
    }
    if elapsed is not None and elapsed > 0:
        record["tokens_per_sec"] = tokens / elapsed
    return record


def _initial_run_summary(args, info, start_step):
    tokens_per_step = int(args.global_batch_size) * int(args.ctx_len)
    return {
        "status": "running",
        "started_at": _now_utc(),
        "updated_at": _now_utc(),
        "ended_at": None,
        "start_step": int(start_step),
        "current_step": int(start_step),
        "completed_steps": 0,
        "requested_steps": int(args.steps),
        "tokens_per_step": tokens_per_step,
        "tokens_seen": 0,
        "checkpoint_backend": args.checkpoint_backend,
        "sampling_mode": args.sampling_mode,
        "carry_state": args.carry_state,
        "latest_checkpoint": None,
        "best_eval": None,
        "last_train": None,
        "last_eval": None,
        "process_info": {
            key: value
            for key, value in info.items()
            if key != "devices"
        },
    }


def _write_run_summary(args, info, summary):
    path = _summary_path(args)
    if path is None or info["process_index"] != 0:
        return None
    summary["updated_at"] = _now_utc()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _write_best_eval(args, info, best_eval):
    if args.output_dir is None or info["process_index"] != 0 or best_eval is None:
        return None
    path = Path(args.output_dir) / "best_eval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(best_eval), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _protected_checkpoint_paths(summary):
    best_eval = summary.get("best_eval")
    if best_eval is None or best_eval.get("checkpoint") is None:
        return []
    return [best_eval["checkpoint"]]


def _is_better_eval(metric_value, best_eval, mode):
    if metric_value is None:
        return False
    metric_value = float(metric_value)
    if not math.isfinite(metric_value):
        return False
    if best_eval is None:
        return True
    best_value = float(best_eval["value"])
    if mode == "max":
        return metric_value > best_value
    return metric_value < best_value


def _write_run_config(args, config, info, *, params=None, mesh=None):
    if args.output_dir is None or info["process_index"] != 0:
        return None
    path = Path(args.output_dir) / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "model_config": model_config_to_dict(config),
        "process_info": {
            key: value
            for key, value in info.items()
            if key != "devices"
        },
        "devices": info.get("devices", []),
        "environment": runtime_version_manifest(),
    }
    if args.param_axis_name is not None and params is not None and mesh is not None:
        payload["parameter_partition_summary"] = parameter_partition_summary(
            params,
            mesh,
            axis_name=args.param_axis_name,
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _save_checkpoint(args, dist, config, step, info, *, protected_paths=None, metadata=None):
    checkpoint_metadata = {
        "data_file": args.data_file,
        "ctx_len": args.ctx_len,
        "global_batch_size": args.global_batch_size,
        "phase": args.phase,
        "carry_state": args.carry_state,
        "sampling_mode": args.sampling_mode,
        "eval_data_file": args.eval_data_file,
    }
    if metadata:
        checkpoint_metadata.update(metadata)
    checkpoint_dir = save_data_parallel_checkpoint(
        args.output_dir,
        step,
        dist,
        config,
        info,
        rng_key=jax.random.PRNGKey(args.seed),
        dataset_position={"step": int(step)},
        metadata=checkpoint_metadata,
        backend=args.checkpoint_backend,
        runtime_state=(
            _runtime_state_payload(dist)
            if args.carry_state
            else None
        ),
    )
    if checkpoint_dir is not None:
        _print_once(info, f"saved checkpoint {checkpoint_dir}")
        removed = rotate_checkpoints(
            args.output_dir,
            args.keep_last_checkpoints,
            info,
            protected_paths=protected_paths,
        )
        for path in removed:
            _print_once(info, f"removed old checkpoint {path}")
    return checkpoint_dir


def _run_validation(args, dist, dataset, completed_step):
    if args.eval_steps <= 0:
        return None
    metrics = []
    eval_rwkv_state = dist.initial_rwkv_state
    eval_screen_state = dist.initial_screen_state
    for eval_offset in range(args.eval_steps):
        eval_step = int(eval_offset)
        if args.carry_state and dataset.should_reset_state_before_step(eval_step):
            eval_rwkv_state = dist.initial_rwkv_state
            eval_screen_state = dist.initial_screen_state
        if args.carry_state:
            eval_metrics, eval_rwkv_state, eval_screen_state = (
                evaluate_batch_data_parallel_with_state(
                    dist,
                    dataset.get_batch(eval_step),
                    dataset.layout,
                    eval_rwkv_state,
                    eval_screen_state,
                    phase=args.phase,
                )
            )
        else:
            eval_metrics = evaluate_batch_data_parallel(
                dist,
                dataset.get_batch(eval_step),
                dataset.layout,
                phase=args.phase,
                carry_state=False,
            )
        metrics.append(metrics_to_host_dict(eval_metrics))
    mean_metrics = mean_metric_dict(metrics)
    if "loss" in mean_metrics:
        mean_metrics["perplexity"] = math.exp(min(mean_metrics["loss"], 20.0))
    return mean_metrics


def _runtime_state_payload(runtime_or_dist):
    return {
        "rwkv_state": runtime_or_dist.rwkv_state,
        "screen_state": runtime_or_dist.screen_state,
    }


def _validate_model_parallel_shapes(config, mesh, axis_name):
    axis_index = tuple(mesh.axis_names).index(axis_name)
    axis_size = int(mesh.devices.shape[axis_index])
    required = {
        "d_model": config.d_model,
        "n_heads": config.n_heads,
    }
    if config.use_screening and config.screening.screened_layers:
        required["d_slot"] = config.screening.d_slot
    if config.vocab_parallel:
        required["vocab_size"] = config.vocab_size
    invalid = {
        name: value for name, value in required.items() if value % axis_size != 0
    }
    if invalid:
        details = ", ".join(f"{name}={value}" for name, value in invalid.items())
        raise ValueError(
            f"model-parallel dimensions must be divisible by axis "
            f"{axis_name!r} size {axis_size}: {details}"
        )


def run_distributed_training(args):
    info = initialize_jax_distributed()
    mesh_axis_names = tuple(args.mesh_axis_names)
    if "data" not in mesh_axis_names:
        raise ValueError("mesh_axis_names must include 'data'")
    if args.param_axis_name is not None and args.param_axis_name not in mesh_axis_names:
        raise ValueError(
            f"param_axis_name {args.param_axis_name!r} is not present in "
            f"mesh_axis_names={mesh_axis_names!r}"
        )
    if args.carry_state and args.sampling_mode != "sequential":
        raise ValueError("--carry-state requires --sampling-mode sequential")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.global_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError(
            "global_batch_size must be divisible by gradient_accumulation_steps"
        )
    uses_explicit_model_parallel = args.param_axis_name not in (None, "data")
    axis_type = (
        jax.sharding.AxisType.Explicit
        if uses_explicit_model_parallel
        else jax.sharding.AxisType.Auto
    )
    mesh = make_mesh(
        mesh_axis_names,
        axis_sizes=args.mesh_axis_sizes,
        axis_types=(axis_type,) * len(mesh_axis_names),
    )
    data_axis_index = mesh_axis_names.index("data")
    data_axis_size = int(mesh.devices.shape[data_axis_index])
    microbatch_size = (
        args.global_batch_size // args.gradient_accumulation_steps
    )
    if microbatch_size % data_axis_size != 0:
        raise ValueError(
            "global microbatch size must be divisible by the data mesh axis size"
        )
    if args.resume:
        checkpoint_payload = load_distributed_checkpoint_metadata(args.resume)
        config = checkpoint_payload.config
        start_step = checkpoint_payload.start_step
        if args.model_config is not None or args.model_preset is not None:
            requested_config = build_config(args)
            if model_config_to_dict(requested_config) != model_config_to_dict(config):
                raise ValueError(
                    "--model-config does not match the model config stored in --resume"
                )
    else:
        config = build_config(args)
        start_step = 0
    if args.ctx_len > config.max_seq_len:
        raise ValueError(
            f"ctx_len={args.ctx_len} exceeds model max_seq_len={config.max_seq_len}"
        )
    if config.vocab_parallel and not uses_explicit_model_parallel:
        raise ValueError(
            "vocab_parallel requires a distinct explicit --param-axis-name"
        )
    if uses_explicit_model_parallel:
        _validate_model_parallel_shapes(config, mesh, args.param_axis_name)

    dataset = create_host_binidx_dataset(
        args.data_file,
        ctx_len=args.ctx_len,
        global_batch_size=args.global_batch_size,
        magic_prime=args.magic_prime,
        epoch_steps=max(args.steps, 1),
        process_index=info["process_index"],
        process_count=info["process_count"],
        local_device_count=info["local_device_count"],
        local_data_shard_count=int(mesh.local_mesh.shape["data"]),
        sampling_mode=args.sampling_mode,
    )
    eval_dataset = None
    if args.eval_every > 0 and args.eval_data_file is not None:
        eval_dataset = create_host_binidx_dataset(
            args.eval_data_file,
            ctx_len=args.ctx_len,
            global_batch_size=args.global_batch_size,
            magic_prime=args.magic_prime,
            epoch_steps=max(args.eval_steps, 1),
            process_index=info["process_index"],
            process_count=info["process_count"],
            local_device_count=info["local_device_count"],
            local_data_shard_count=int(mesh.local_mesh.shape["data"]),
            sampling_mode=args.sampling_mode,
        )
    if args.param_axis_name == "data":
        data_axis_index = tuple(mesh.axis_names).index("data")
        if int(mesh.devices.shape[data_axis_index]) != 1:
            raise ValueError(
                "NNX model parallelism requires a parameter axis distinct "
                "from the data axis"
            )
        # Preserve the historical one-device smoke invocation. There is no
        # physical model sharding to perform on an axis of size one.
        nnx_sharding = None
    else:
        nnx_sharding = (
            NNXShardingConfig(mesh, model_axis=args.param_axis_name)
            if args.param_axis_name is not None
            else None
        )
    runtime, train_state = create_train_runtime(
        jax.random.PRNGKey(args.seed),
        config,
        batch_size=args.global_batch_size,
        total_steps=max(start_step + args.steps, 1),
        sharding=nnx_sharding,
    )
    if args.resume:
        train_state, checkpoint_payload = restore_distributed_train_state(
            args.resume,
            train_state,
        )
        config = checkpoint_payload.config
        runtime.variables = {"params": train_state.nnx_params}
        if args.carry_state:
            runtime_state = restore_distributed_runtime_state(
                args.resume,
                _runtime_state_payload(runtime),
            )
            if runtime_state is not None:
                runtime.rwkv_state = runtime_state["rwkv_state"]
                runtime.screen_state = runtime_state["screen_state"]
            else:
                _print_once(
                    info,
                    "distributed runtime state not found; carry-state resume starts from zero runtime state",
                )
    dist = place_train_objects(
        runtime,
        train_state,
        mesh=mesh,
        axis_name="data",
        param_axis_name=args.param_axis_name,
    )
    last_checkpoint = None
    log_jsonl, log_csv = _log_paths(args)
    _write_run_config(args, config, info, params=train_state.params, mesh=mesh)
    summary = _initial_run_summary(args, info, start_step)
    best_eval = None
    _write_run_summary(args, info, summary)
    try:
        _print_once(
            info,
            f"process={info['process_index']}/{info['process_count']} "
            f"devices={info['local_device_count']} global_batch={args.global_batch_size} "
            f"process_batch={dataset.layout.process_batch_size} start_step={start_step}",
        )
        for global_step, global_batch in iter_prefetched_global_batches(
            dataset,
            dist.batch_sharding,
            start_step=start_step,
            steps=args.steps,
            prefetch_size=args.prefetch_size,
        ):
            local_step = global_step - start_step
            if args.carry_state and dataset.should_reset_state_before_step(global_step):
                dist = replace(
                    dist,
                    rwkv_state=dist.initial_rwkv_state,
                    screen_state=dist.initial_screen_state,
                )
            step_start = time.perf_counter()
            dist, metrics = train_global_batch_data_parallel(
                dist,
                global_batch,
                phase=args.phase,
                carry_state=args.carry_state,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
            # The loss output can become ready before every parameter and
            # optimizer-state leaf. This reference CLI reports per-step
            # throughput, so wait for the complete updated train state before
            # stopping the timer. The TPU-scale NNX path will measure larger
            # asynchronous windows instead of synchronizing every step.
            ready_state = (
                dist.train_state.ready_state()
                if hasattr(dist.train_state, "ready_state")
                else dist.train_state
            )
            jax.block_until_ready(ready_state)
            elapsed = time.perf_counter() - step_start
            host_metrics = metrics_to_host_dict(metrics)
            completed_step = int(dist.train_state.step)
            checkpoint_saved_this_step = False
            train_record = _metric_record(
                args,
                "train",
                completed_step,
                host_metrics,
                elapsed=elapsed,
            )
            write_metric_record(
                log_jsonl,
                log_csv,
                train_record,
                process_info=info,
            )
            summary["current_step"] = completed_step
            summary["completed_steps"] = max(0, completed_step - int(start_step))
            summary["tokens_seen"] = (
                summary["completed_steps"] * summary["tokens_per_step"]
            )
            summary["last_train"] = train_record
            if args.print_every and (
                local_step % args.print_every == 0 or local_step == args.steps - 1
            ):
                _print_once(
                    info,
                    f"step={completed_step} loss={host_metrics['loss']:.6f} "
                    f"tok/s={args.global_batch_size * args.ctx_len / max(elapsed, 1e-9):.2f}",
                )

            should_eval = (
                args.eval_every > 0
                and completed_step % args.eval_every == 0
            )
            if should_eval:
                eval_metrics = _run_validation(
                    args,
                    dist,
                    dataset if eval_dataset is None else eval_dataset,
                    completed_step,
                )
                if eval_metrics is not None:
                    eval_record = _metric_record(
                        args,
                        "eval",
                        completed_step,
                        eval_metrics,
                    )
                    write_metric_record(
                        log_jsonl,
                        log_csv,
                        eval_record,
                        process_info=info,
                    )
                    summary["last_eval"] = eval_record
                    _print_once(
                        info,
                        f"eval step={completed_step} loss={eval_metrics['loss']:.6f} "
                        f"perplexity={eval_metrics['perplexity']:.6f}",
                    )
                    metric_value = eval_metrics.get(args.best_metric)
                    if _is_better_eval(metric_value, best_eval, args.best_mode):
                        best_eval = {
                            "step": completed_step,
                            "metric": args.best_metric,
                            "mode": args.best_mode,
                            "value": float(metric_value),
                            "metrics": eval_metrics,
                            "checkpoint": None,
                        }
                        if args.save_best_checkpoint and args.output_dir is not None:
                            best_checkpoint = checkpoint_path(args.output_dir, completed_step)
                            last_checkpoint = _save_checkpoint(
                                args,
                                dist,
                                config,
                                completed_step,
                                info,
                                protected_paths=[best_checkpoint],
                                metadata={
                                    "checkpoint_reason": "best_eval",
                                    "best_metric": args.best_metric,
                                    "best_metric_value": float(metric_value),
                                },
                            )
                            checkpoint_saved_this_step = True
                            if last_checkpoint is not None:
                                best_eval["checkpoint"] = last_checkpoint
                                summary["latest_checkpoint"] = last_checkpoint
                        summary["best_eval"] = best_eval
                        _write_best_eval(args, info, best_eval)

            if (
                args.output_dir is not None
                and args.save_every > 0
                and completed_step % args.save_every == 0
                and not checkpoint_saved_this_step
            ):
                last_checkpoint = _save_checkpoint(
                    args,
                    dist,
                    config,
                    completed_step,
                    info,
                    protected_paths=_protected_checkpoint_paths(summary),
                )
                if last_checkpoint is not None:
                    summary["latest_checkpoint"] = last_checkpoint
                    if (
                        summary.get("best_eval") is not None
                        and summary["best_eval"].get("step") == completed_step
                        and summary["best_eval"].get("checkpoint") is None
                    ):
                        summary["best_eval"]["checkpoint"] = last_checkpoint
                        _write_best_eval(args, info, summary["best_eval"])

            should_write_summary = (
                args.summary_every > 0
                and completed_step % args.summary_every == 0
            ) or local_step == args.steps - 1
            if should_write_summary:
                _write_run_summary(args, info, summary)

        if args.output_dir is not None:
            final_step = int(dist.train_state.step)
            final_path = checkpoint_path(args.output_dir, final_step)
            if last_checkpoint != final_path:
                last_checkpoint = _save_checkpoint(
                    args,
                    dist,
                    config,
                    final_step,
                    info,
                    protected_paths=_protected_checkpoint_paths(summary),
                )
                if last_checkpoint is not None:
                    summary["latest_checkpoint"] = last_checkpoint
                    if (
                        summary.get("best_eval") is not None
                        and summary["best_eval"].get("step") == final_step
                        and summary["best_eval"].get("checkpoint") is None
                    ):
                        summary["best_eval"]["checkpoint"] = last_checkpoint
                        _write_best_eval(args, info, summary["best_eval"])
        summary["status"] = "completed"
        summary["ended_at"] = _now_utc()
        _write_run_summary(args, info, summary)
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_run_summary(args, info, summary)
        raise
    finally:
        if eval_dataset is not None:
            eval_dataset.close()
        dataset.close()
    return dist, last_checkpoint


def main(argv=None):
    args = parse_args(argv)
    gc_was_enabled = gc.isenabled()
    if args.disable_python_gc and gc_was_enabled:
        gc.disable()
    try:
        run_distributed_training(args)
    finally:
        if args.disable_python_gc and gc_was_enabled:
            gc.enable()


if __name__ == "__main__":
    main()
