import argparse
from pathlib import Path

import jax

from ..api import create_train_runtime, train_batch
from ..data import create_binidx_dataset
from ..io import (
    load_train_checkpoint,
    load_train_checkpoint_metadata,
    load_train_runtime_state,
    save_train_checkpoint,
)
from ..model import ModelConfig, ScreeningConfig
from .config import parse_args_with_config
from .eval_binidx import evaluate_binidx, parse_args as parse_eval_args


def build_config(args):
    if args.use_screening:
        screened_layers = tuple(args.screened_layers)
        if not screened_layers and args.n_layers > 1:
            screened_layers = (args.n_layers // 2,)
        bank_ids = tuple(args.bank_ids) if args.bank_ids is not None else default_bank_ids(args.n_slots)
        screening = ScreeningConfig(
            d_model=args.d_model,
            d_slot=args.d_slot,
            d_k=args.d_k,
            d_v=args.d_v,
            n_slots=args.n_slots,
            screened_layers=screened_layers,
            bank_ids=bank_ids,
            use_write_screening=args.phase == "read_write",
            write_rel_floor=args.write_rel_floor,
            short_half_life_tokens=args.short_half_life_tokens,
            mid_half_life_tokens=args.mid_half_life_tokens,
            long_half_life_tokens=args.long_half_life_tokens,
            usage_ema_decay=args.usage_ema_decay,
        )
    else:
        screening = ScreeningConfig()
    return ModelConfig(
        d_model=args.d_model,
        d_ffn=args.d_ffn,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        head_size=args.head_size,
        vocab_size=args.vocab_size,
        max_seq_len=args.ctx_len,
        dtype=args.dtype,
        use_screening=args.use_screening,
        screening=screening,
        lr_init=args.lr_init,
        lr_final=args.lr_final,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
    )


def default_bank_ids(n_slots):
    if n_slots <= 0:
        raise ValueError("n_slots must be positive")
    if n_slots == 1:
        return (0,)
    if n_slots == 2:
        return (0, 2)
    short = max(1, n_slots // 2)
    mid = max(1, (n_slots - short) // 2)
    long = n_slots - short - mid
    return tuple([0] * short + [1] * mid + [2] * long)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train rwkv7m on RWKV-LM-V7 .bin/.idx data.")
    parser.add_argument("--data-file", required=True, help="Dataset prefix path without .bin/.idx")
    parser.add_argument("--ctx-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--magic-prime", type=int, default=None)
    parser.add_argument("--sampling-mode", choices=["magic", "sequential"], default="magic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", choices=["read_screening_only", "read_write"], default="read_screening_only")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
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
    parser.add_argument("--carry-state", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume", default=None, help="Checkpoint directory to resume from")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=1)
    return parse_args_with_config(parser, argv)


def _checkpoint_path(output_dir, step):
    return Path(output_dir) / f"ckpt-{int(step):08d}"


def _runtime_state_payload(runtime):
    return {
        "rwkv_state": runtime.rwkv_state,
        "screen_state": runtime.screen_state,
    }


def _save_checkpoint(args, train_state, cfg, step, runtime=None):
    if args.output_dir is None:
        return None
    checkpoint_dir = _checkpoint_path(args.output_dir, step)
    save_train_checkpoint(
        checkpoint_dir,
        train_state,
        cfg,
        rng_key=jax.random.PRNGKey(args.seed),
        dataset_position={"step": int(step)},
        metadata={
            "data_file": args.data_file,
            "ctx_len": args.ctx_len,
            "batch_size": args.batch_size,
            "phase": args.phase,
            "carry_state": args.carry_state,
            "sampling_mode": args.sampling_mode,
            "total_requested_steps": args.steps,
        },
        runtime_state=(
            _runtime_state_payload(runtime)
            if args.carry_state and runtime is not None
            else None
        ),
    )
    print(f"saved checkpoint {checkpoint_dir}")
    return checkpoint_dir


def _run_eval(args, checkpoint_dir, step):
    if checkpoint_dir is None:
        return None
    eval_argv = [
        "--data-file",
        args.data_file,
        "--checkpoint",
        str(checkpoint_dir),
        "--ctx-len",
        str(args.ctx_len),
        "--batch-size",
        str(args.batch_size),
        "--steps",
        str(args.eval_steps),
        "--seed",
        str(args.seed),
        "--phase",
        args.phase,
        "--sampling-mode",
        args.sampling_mode,
        "--print-every",
        "0",
    ]
    if args.carry_state:
        eval_argv.append("--carry-state")
    if args.magic_prime is not None:
        eval_argv.extend(["--magic-prime", str(args.magic_prime)])
    eval_args = parse_eval_args(eval_argv)
    metrics = evaluate_binidx(eval_args)
    print(
        f"eval step={step} loss={metrics['loss']:.6f} "
        f"perplexity={metrics['perplexity']:.6f}"
    )
    return metrics


def run_training(args):
    if args.carry_state and args.sampling_mode != "sequential":
        raise ValueError("--carry-state requires --sampling-mode sequential")
    if args.resume:
        cfg, payload = load_train_checkpoint_metadata(args.resume)
        start_step = int(payload.get("step", 0))
    else:
        cfg = build_config(args)
        start_step = 0

    dataset = create_binidx_dataset(
        args.data_file,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
        magic_prime=args.magic_prime,
        epoch_steps=max(args.steps, 1),
        sampling_mode=args.sampling_mode,
    )
    total_steps = max(start_step + args.steps, 1)
    runtime, train_state = create_train_runtime(
        jax.random.PRNGKey(args.seed),
        cfg,
        batch_size=args.batch_size,
        total_steps=total_steps,
    )
    if args.resume:
        train_state, cfg, _ = load_train_checkpoint(args.resume, train_state)
        runtime.variables = {"params": train_state.params}
        if args.carry_state:
            runtime_state = load_train_runtime_state(
                args.resume,
                _runtime_state_payload(runtime),
            )
            if runtime_state is not None:
                runtime.rwkv_state = runtime_state["rwkv_state"]
                runtime.screen_state = runtime_state["screen_state"]
            else:
                print("runtime_state.msgpack not found; carry-state resume starts from zero runtime state")

    last_checkpoint = None
    try:
        print(
            f"data_tokens={dataset.data_size} magic_prime={dataset.magic_prime} "
            f"ctx_len={args.ctx_len} batch_size={args.batch_size} "
            f"sampling_mode={args.sampling_mode} carry_state={args.carry_state} "
            f"start_step={start_step}"
        )
        for local_step in range(args.steps):
            global_step = start_step + local_step
            if args.carry_state and dataset.should_reset_state_before_step(global_step):
                runtime.rwkv_state = runtime.initial_rwkv_state
                runtime.screen_state = runtime.initial_screen_state
            batch = dataset.get_batch(global_step)
            train_state, metrics = train_batch(
                train_state,
                batch,
                runtime,
                phase=args.phase,
                carry_state=args.carry_state,
            )
            completed_step = int(train_state.step)
            if args.print_every and (
                local_step % args.print_every == 0 or local_step == args.steps - 1
            ):
                print(f"step={completed_step} loss={float(metrics['loss']):.6f}")

            should_save = (
                args.output_dir is not None
                and args.save_every > 0
                and completed_step % args.save_every == 0
            )
            if should_save:
                last_checkpoint = _save_checkpoint(args, train_state, cfg, completed_step, runtime)

            should_eval = (
                args.eval_every > 0
                and completed_step % args.eval_every == 0
                and args.output_dir is not None
            )
            if should_eval:
                if last_checkpoint is None or last_checkpoint.name != f"ckpt-{completed_step:08d}":
                    last_checkpoint = _save_checkpoint(args, train_state, cfg, completed_step, runtime)
                _run_eval(args, last_checkpoint, completed_step)

        if args.output_dir is not None:
            final_step = int(train_state.step)
            final_path = _checkpoint_path(args.output_dir, final_step)
            if last_checkpoint != final_path:
                last_checkpoint = _save_checkpoint(args, train_state, cfg, final_step, runtime)
    finally:
        dataset.close()
    return train_state, runtime, last_checkpoint


def main(argv=None):
    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
