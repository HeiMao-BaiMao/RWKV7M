import argparse
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
    initialize_jax_distributed,
    iter_prefetched_global_batches,
    load_distributed_checkpoint_metadata,
    make_mesh,
    mean_metric_dict,
    metrics_to_host_dict,
    place_train_objects,
    restore_distributed_train_state,
    rotate_checkpoints,
    save_data_parallel_checkpoint,
    train_global_batch_data_parallel,
    write_metric_record,
)
from ..io import model_config_to_dict
from .config import parse_args_with_config
from .train_binidx import build_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Data-parallel rwkv7m binidx training skeleton for JAX distributed setups."
    )
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--ctx-len", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--magic-prime", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", choices=["read_screening_only", "read_write"], default="read_screening_only")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
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


def _write_run_config(args, config, info):
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
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _save_checkpoint(args, dist, config, step, info):
    checkpoint_dir = save_data_parallel_checkpoint(
        args.output_dir,
        step,
        dist,
        config,
        info,
        rng_key=jax.random.PRNGKey(args.seed),
        dataset_position={"step": int(step)},
        metadata={
            "data_file": args.data_file,
            "ctx_len": args.ctx_len,
            "global_batch_size": args.global_batch_size,
            "phase": args.phase,
            "eval_data_file": args.eval_data_file,
        },
        backend=args.checkpoint_backend,
    )
    if checkpoint_dir is not None:
        _print_once(info, f"saved checkpoint {checkpoint_dir}")
        removed = rotate_checkpoints(args.output_dir, args.keep_last_checkpoints, info)
        for path in removed:
            _print_once(info, f"removed old checkpoint {path}")
    return checkpoint_dir


def _run_validation(args, dist, dataset, completed_step):
    if args.eval_steps <= 0:
        return None
    metrics = []
    for eval_offset in range(args.eval_steps):
        eval_step = int(completed_step) + eval_offset
        eval_metrics = evaluate_batch_data_parallel(
            dist,
            dataset.get_batch(eval_step),
            dataset.layout,
            phase=args.phase,
            carry_state=args.carry_state,
        )
        metrics.append(metrics_to_host_dict(eval_metrics))
    mean_metrics = mean_metric_dict(metrics)
    if "loss" in mean_metrics:
        mean_metrics["perplexity"] = math.exp(min(mean_metrics["loss"], 20.0))
    return mean_metrics


def run_distributed_training(args):
    info = initialize_jax_distributed()
    if "data" not in tuple(args.mesh_axis_names):
        raise ValueError("mesh_axis_names must include 'data'")
    mesh = make_mesh(tuple(args.mesh_axis_names), axis_sizes=args.mesh_axis_sizes)
    if args.resume:
        checkpoint_payload = load_distributed_checkpoint_metadata(args.resume)
        config = checkpoint_payload.config
        start_step = checkpoint_payload.start_step
    else:
        config = build_config(args)
        start_step = 0

    dataset = create_host_binidx_dataset(
        args.data_file,
        ctx_len=args.ctx_len,
        global_batch_size=args.global_batch_size,
        magic_prime=args.magic_prime,
        epoch_steps=max(args.steps, 1),
        process_index=info["process_index"],
        process_count=info["process_count"],
        local_device_count=info["local_device_count"],
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
        )
    runtime, train_state = create_train_runtime(
        jax.random.PRNGKey(args.seed),
        config,
        batch_size=args.global_batch_size,
        total_steps=max(start_step + args.steps, 1),
    )
    if args.resume:
        train_state, checkpoint_payload = restore_distributed_train_state(
            args.resume,
            train_state,
        )
        config = checkpoint_payload.config
        runtime.variables = {"params": train_state.params}
    dist = place_train_objects(
        runtime,
        train_state,
        mesh=mesh,
        axis_name="data",
        param_axis_name=args.param_axis_name,
    )
    last_checkpoint = None
    log_jsonl, log_csv = _log_paths(args)
    _write_run_config(args, config, info)
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
            step_start = time.perf_counter()
            dist, metrics = train_global_batch_data_parallel(
                dist,
                global_batch,
                phase=args.phase,
                carry_state=args.carry_state,
            )
            elapsed = time.perf_counter() - step_start
            host_metrics = metrics_to_host_dict(metrics)
            completed_step = int(dist.train_state.step)
            write_metric_record(
                log_jsonl,
                log_csv,
                _metric_record(args, "train", completed_step, host_metrics, elapsed=elapsed),
                process_info=info,
            )
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
                    write_metric_record(
                        log_jsonl,
                        log_csv,
                        _metric_record(args, "eval", completed_step, eval_metrics),
                        process_info=info,
                    )
                    _print_once(
                        info,
                        f"eval step={completed_step} loss={eval_metrics['loss']:.6f} "
                        f"perplexity={eval_metrics['perplexity']:.6f}",
                    )

            if (
                args.output_dir is not None
                and args.save_every > 0
                and completed_step % args.save_every == 0
            ):
                last_checkpoint = _save_checkpoint(args, dist, config, completed_step, info)

        if args.output_dir is not None:
            final_step = int(dist.train_state.step)
            final_path = checkpoint_path(args.output_dir, final_step)
            if last_checkpoint != final_path:
                last_checkpoint = _save_checkpoint(args, dist, config, final_step, info)
    finally:
        if eval_dataset is not None:
            eval_dataset.close()
        dataset.close()
    return dist, last_checkpoint


def main(argv=None):
    args = parse_args(argv)
    run_distributed_training(args)


if __name__ == "__main__":
    main()
