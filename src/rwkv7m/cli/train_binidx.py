import argparse
import math
from pathlib import Path

import jax

from ..api import create_train_runtime, train_batch
from ..data import create_binidx_dataset
from ..io import (
    load_model_config,
    load_train_checkpoint,
    load_train_checkpoint_metadata,
    load_train_runtime_state,
    model_config_to_dict,
    save_train_checkpoint,
)
from ..model import MODEL_PRESET_NAMES, ModelConfig, ScreeningConfig, model_preset
from .config import (
    add_optimizer_backend_arg,
    add_screening_v2_args,
    add_training_vocab_tiling_args,
    apply_execution_overrides,
    parse_args_with_config,
    screening_v2_kwargs,
)
from .eval_binidx import evaluate_binidx, parse_args as parse_eval_args


def build_config(args):
    model_config_path = getattr(args, "model_config", None)
    preset_name = getattr(args, "model_preset", None)
    if model_config_path is not None and preset_name is not None:
        raise ValueError("--model-config and --model-preset are mutually exclusive")
    if model_config_path is not None or preset_name is not None:
        config = (
            load_model_config(model_config_path)
            if model_config_path is not None
            else model_preset(preset_name)
        )
        if args.ctx_len > config.max_seq_len:
            raise ValueError(
                f"ctx_len={args.ctx_len} exceeds model max_seq_len={config.max_seq_len}"
            )
        return apply_execution_overrides(config, args)
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
            **screening_v2_kwargs(args),
        )
    else:
        screening = ScreeningConfig()
    config = ModelConfig(
        d_model=args.d_model,
        d_ffn=args.d_ffn,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        head_size=args.head_size,
        vocab_size=args.vocab_size,
        max_seq_len=args.ctx_len,
        dtype=args.dtype,
        param_dtype=args.param_dtype,
        param_update_dtype=args.param_update_dtype,
        optimizer_state_dtype=args.optimizer_state_dtype,
        gradient_accum_dtype=args.gradient_accum_dtype,
        lm_head_init=args.lm_head_init,
        vocab_parallel=args.vocab_parallel,
        remat_blocks=bool(args.remat_blocks),
        sequence_chunk_size=args.sequence_chunk_size,
        head_chunk_size=args.head_chunk_size,
        use_screening=args.use_screening,
        screening=screening,
        lr_init=1e-3 if args.lr_init is None else args.lr_init,
        lr_final=1e-5 if args.lr_final is None else args.lr_final,
        warmup_steps=(
            10 if args.warmup_steps is None else args.warmup_steps
        ),
        lr_schedule=(
            "optax_cosine" if args.lr_schedule is None else args.lr_schedule
        ),
        max_grad_norm=(
            1.0 if args.max_grad_norm is None else args.max_grad_norm
        ),
        weight_decay=(
            0.001 if args.weight_decay is None else args.weight_decay
        ),
        adam_beta1=0.9 if args.adam_beta1 is None else args.adam_beta1,
        adam_beta2=0.999 if args.adam_beta2 is None else args.adam_beta2,
        adam_eps=1e-8 if args.adam_eps is None else args.adam_eps,
    )
    return apply_execution_overrides(config, args)


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
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument(
        "--model-config",
        default=None,
        help="Shared ModelConfig JSON; execution-only overrides may be applied",
    )
    model_source.add_argument(
        "--model-preset",
        choices=MODEL_PRESET_NAMES,
        default=None,
        help="Named shared NNX model configuration",
    )
    parser.add_argument("--ctx-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--magic-prime", type=int, default=None)
    parser.add_argument(
        "--loss-mask-after-token",
        type=int,
        default=None,
        help="train only targets immediately following this input token",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["magic", "sequential", "document", "document_sequential"],
        default="magic",
    )
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
    add_optimizer_backend_arg(parser)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr-init", type=float, default=None)
    parser.add_argument("--lr-final", type=float, default=None)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="optimizer warmup override; model-config value is kept when omitted",
    )
    parser.add_argument("--lr-schedule", choices=["optax_cosine", "rwkv"], default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--adam-beta1", type=float, default=None)
    parser.add_argument("--adam-beta2", type=float, default=None)
    parser.add_argument("--adam-eps", type=float, default=None)
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
    add_screening_v2_args(parser)
    parser.add_argument("--carry-state", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--require-finite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop immediately when the train loss becomes non-finite",
    )
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
    if args.loss_mask_after_token is not None:
        eval_argv.extend(
            ["--loss-mask-after-token", str(args.loss_mask_after_token)]
        )
    eval_args = parse_eval_args(eval_argv)
    metrics = evaluate_binidx(eval_args)
    print(
        f"eval step={step} loss={metrics['loss']:.6f} "
        f"perplexity={metrics['perplexity']:.6f}"
    )
    return metrics


def run_training(args):
    if args.carry_state and args.sampling_mode not in {
        "sequential",
        "document_sequential",
    }:
        raise ValueError(
            "--carry-state requires --sampling-mode sequential or "
            "document_sequential"
        )
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError(
            "batch_size must be divisible by gradient_accumulation_steps"
        )
    if args.resume:
        cfg, payload = load_train_checkpoint_metadata(args.resume)
        start_step = int(payload.get("step", 0))
        if args.model_config is not None or args.model_preset is not None:
            requested_config = build_config(args)
            if model_config_to_dict(requested_config) != model_config_to_dict(cfg):
                raise ValueError(
                    "--model-config does not match the model config stored in --resume"
                )
    else:
        cfg = build_config(args)
        start_step = 0
    if args.ctx_len > cfg.max_seq_len:
        raise ValueError(
            f"ctx_len={args.ctx_len} exceeds model max_seq_len={cfg.max_seq_len}"
        )
    if cfg.vocab_parallel:
        raise ValueError(
            "vocab_parallel model configs require rwkv7m-train-binidx-dp "
            "with a distinct model mesh axis"
        )

    dataset = create_binidx_dataset(
        args.data_file,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
        magic_prime=args.magic_prime,
        epoch_steps=max(args.steps, 1),
        sampling_mode=args.sampling_mode,
        loss_mask_after_token=args.loss_mask_after_token,
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
        runtime.variables = {"params": train_state.nnx_params}
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
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
            completed_step = int(train_state.step)
            nonfinite_losses = [
                name
                for name in ("loss", "total_loss")
                if name in metrics and not math.isfinite(float(metrics[name]))
            ]
            if args.require_finite and nonfinite_losses:
                raise FloatingPointError(
                    f"non-finite train metrics at step {completed_step}: "
                    + ", ".join(nonfinite_losses)
                )
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
