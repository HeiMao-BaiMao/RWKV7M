import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from ..api import create_runtime, load_runtime_params
from ..data import create_binidx_dataset
from ..io import load_model_config, load_model_safetensors
from ..model import MODEL_PRESET_NAMES, ModelConfig, ScreeningConfig, model_preset
from ..model.screened_rwkv import cross_entropy_loss
from ..model.state import reset_state_rows
from .config import parse_args_with_config


def default_bank_ids(n_slots):
    if n_slots == 1:
        return (0,)
    if n_slots == 2:
        return (0, 2)
    short = max(1, n_slots // 2)
    mid = max(1, (n_slots - short) // 2)
    return tuple([0] * short + [1] * mid + [2] * (n_slots - short - mid))


def build_config(args):
    if args.model_config is not None and args.model_preset is not None:
        raise ValueError("--model-config and --model-preset are mutually exclusive")
    if args.model_config is not None or args.model_preset is not None:
        config = (
            load_model_config(args.model_config)
            if args.model_config is not None
            else model_preset(args.model_preset)
        )
        if args.ctx_len > config.max_seq_len:
            raise ValueError(
                f"ctx_len={args.ctx_len} exceeds model max_seq_len={config.max_seq_len}"
            )
        return config
    if args.use_screening:
        screened_layers = tuple(args.screened_layers)
        if not screened_layers and args.n_layers > 1:
            screened_layers = (args.n_layers // 2,)
        screening = ScreeningConfig(
            d_model=args.d_model,
            d_slot=args.d_slot,
            d_k=args.d_k,
            d_v=args.d_v,
            n_slots=args.n_slots,
            screened_layers=screened_layers,
            bank_ids=default_bank_ids(args.n_slots),
            use_write_screening=args.phase == "read_write",
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
        param_dtype=args.param_dtype,
        lm_head_init=args.lm_head_init,
        vocab_parallel=args.vocab_parallel,
        remat_blocks=args.remat_blocks,
        sequence_chunk_size=args.sequence_chunk_size,
        use_screening=args.use_screening,
        screening=screening,
    )


def resolve_checkpoint_file(checkpoint):
    path = Path(checkpoint)
    if path.is_dir():
        return path / "model.safetensors"
    return path


def evaluate_binidx(args):
    if args.carry_state and args.sampling_mode not in {
        "sequential",
        "document_sequential",
    }:
        raise ValueError(
            "--carry-state requires --sampling-mode sequential or "
            "document_sequential"
        )
    if args.checkpoint:
        params, config, _ = load_model_safetensors(resolve_checkpoint_file(args.checkpoint))
        if config is None:
            raise ValueError("checkpoint safetensors is missing rwkv7m config metadata")
    else:
        config = build_config(args)
        params = None

    dataset = create_binidx_dataset(
        args.data_file,
        ctx_len=args.ctx_len,
        batch_size=args.batch_size,
        magic_prime=args.magic_prime,
        epoch_steps=args.steps,
        sampling_mode=args.sampling_mode,
        loss_mask_after_token=args.loss_mask_after_token,
    )
    runtime = create_runtime(
        jax.random.PRNGKey(args.seed),
        config,
        batch_size=args.batch_size,
    )
    if params is not None:
        load_runtime_params(runtime, params)

    if args.memory_off_counterfactual and not config.use_screening:
        raise ValueError(
            "--memory-off-counterfactual requires screening to be enabled"
        )

    @nnx.jit
    def eval_step(
        model,
        input_ids,
        target_ids,
        mask,
        rwkv_state,
        screen_state,
        residual_scale,
    ):
        logits, _, _, stats = model(
            input_ids,
            rwkv_state,
            screen_state,
            phase=args.phase,
            deterministic=True,
            screening_residual_scale=residual_scale,
        )
        return cross_entropy_loss(logits, target_ids, mask), stats, logits

    @nnx.jit
    def stateful_eval_step(
        model,
        input_ids,
        target_ids,
        mask,
        rwkv_state,
        screen_state,
        residual_scale,
    ):
        logits, new_rwkv_state, new_screen_state, stats = model(
            input_ids,
            rwkv_state,
            screen_state,
            phase=args.phase,
            deterministic=True,
            screening_residual_scale=residual_scale,
        )
        loss = cross_entropy_loss(logits, target_ids, mask)
        return loss, new_rwkv_state, new_screen_state, stats, logits

    losses = []
    loss_counts = []
    correct_counts = []
    memory_off_losses = []
    memory_off_correct_counts = []
    prediction_squared_error_sums = []
    prediction_value_counts = []
    rwkv_state = runtime.initial_rwkv_state
    screen_state = runtime.initial_screen_state
    memory_off_rwkv_state = runtime.initial_rwkv_state
    memory_off_screen_state = runtime.initial_screen_state
    try:
        for step in range(args.steps):
            if args.carry_state and dataset.should_reset_state_before_step(step):
                rwkv_state = runtime.initial_rwkv_state
                screen_state = runtime.initial_screen_state
                memory_off_rwkv_state = runtime.initial_rwkv_state
                memory_off_screen_state = runtime.initial_screen_state
            batch = dataset.get_batch(step)
            if args.carry_state and "state_reset_mask" in batch:
                reset_mask = batch["state_reset_mask"]
                rwkv_state = reset_state_rows(
                    rwkv_state,
                    runtime.initial_rwkv_state,
                    reset_mask,
                )
                screen_state = reset_state_rows(
                    screen_state,
                    runtime.initial_screen_state,
                    reset_mask,
                )
                memory_off_rwkv_state = reset_state_rows(
                    memory_off_rwkv_state,
                    runtime.initial_rwkv_state,
                    reset_mask,
                )
                memory_off_screen_state = reset_state_rows(
                    memory_off_screen_state,
                    runtime.initial_screen_state,
                    reset_mask,
                )
            if args.carry_state:
                loss, rwkv_state, screen_state, _, logits = stateful_eval_step(
                    runtime.model,
                    batch["input_ids"],
                    batch["target_ids"],
                    batch["mask"],
                    rwkv_state,
                    screen_state,
                    jnp.asarray(1.0, dtype=jnp.float32),
                )
            else:
                loss, _, logits = eval_step(
                    runtime.model,
                    batch["input_ids"],
                    batch["target_ids"],
                    batch["mask"],
                    runtime.initial_rwkv_state,
                    runtime.initial_screen_state,
                    jnp.asarray(1.0, dtype=jnp.float32),
                )
            losses.append(float(loss))
            loss_counts.append(float(jnp.sum(batch["mask"])))
            correct_counts.append(
                float(
                    jnp.sum(
                        (
                            jnp.argmax(logits, axis=-1)
                            == batch["target_ids"]
                        ).astype(jnp.float32)
                        * batch["mask"]
                    )
                )
            )
            if args.memory_off_counterfactual:
                if args.carry_state:
                    (
                        off_loss,
                        memory_off_rwkv_state,
                        memory_off_screen_state,
                        _,
                        off_logits,
                    ) = stateful_eval_step(
                        runtime.model,
                        batch["input_ids"],
                        batch["target_ids"],
                        batch["mask"],
                        memory_off_rwkv_state,
                        memory_off_screen_state,
                        jnp.asarray(0.0, dtype=jnp.float32),
                    )
                else:
                    off_loss, _, off_logits = eval_step(
                        runtime.model,
                        batch["input_ids"],
                        batch["target_ids"],
                        batch["mask"],
                        runtime.initial_rwkv_state,
                        runtime.initial_screen_state,
                        jnp.asarray(0.0, dtype=jnp.float32),
                    )
                memory_off_losses.append(float(off_loss))
                memory_off_correct_counts.append(
                    float(
                        jnp.sum(
                            (
                                jnp.argmax(off_logits, axis=-1)
                                == batch["target_ids"]
                            ).astype(jnp.float32)
                            * batch["mask"]
                        )
                    )
                )
                delta_mask = (
                    batch["mask"]
                    if args.loss_mask_after_token is not None
                    else jnp.ones_like(batch["mask"])
                )
                prediction_squared_error_sums.append(
                    float(
                        jnp.sum(
                            (
                                logits.astype(jnp.float32)
                                - off_logits.astype(jnp.float32)
                            )
                            ** 2
                            * delta_mask[..., None]
                        )
                    )
                )
                prediction_value_counts.append(
                    float(jnp.sum(delta_mask) * logits.shape[-1])
                )
            if args.print_every and (step % args.print_every == 0 or step == args.steps - 1):
                print(f"eval step={step} loss={losses[-1]:.6f}")
    finally:
        dataset.close()

    masked_count = sum(loss_counts)
    if args.loss_mask_after_token is None:
        mean_loss = sum(losses) / len(losses)
    else:
        if masked_count <= 0.0:
            raise ValueError(
                "loss mask selected no target positions in the evaluation run"
            )
        mean_loss = sum(
            loss * count for loss, count in zip(losses, loss_counts, strict=True)
        ) / masked_count
    result = {
        "steps": args.steps,
        "tokens": args.steps * args.batch_size * args.ctx_len,
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "carry_state": bool(args.carry_state),
        "sampling_mode": args.sampling_mode,
        "loss_mask_after_token": args.loss_mask_after_token,
        "loss_mask_positions": int(masked_count),
    }
    if args.loss_mask_after_token is not None:
        result["masked_accuracy"] = sum(correct_counts) / masked_count
    if memory_off_losses:
        memory_off_loss = (
            sum(memory_off_losses) / len(memory_off_losses)
            if args.loss_mask_after_token is None
            else sum(
                loss * count
                for loss, count in zip(
                    memory_off_losses,
                    loss_counts,
                    strict=True,
                )
            )
            / masked_count
        )
        prediction_rms_delta = math.sqrt(
            sum(prediction_squared_error_sums)
            / max(sum(prediction_value_counts), 1.0)
        )
        result.update(
            {
                "memory_off_loss": memory_off_loss,
                "memory_loss_delta": memory_off_loss - mean_loss,
                "prediction_rms_delta": prediction_rms_delta,
            }
        )
        if args.loss_mask_after_token is not None:
            memory_off_accuracy = (
                sum(memory_off_correct_counts) / masked_count
            )
            result.update(
                {
                    "memory_off_masked_accuracy": memory_off_accuracy,
                    "memory_accuracy_delta": (
                        result["masked_accuracy"] - memory_off_accuracy
                    ),
                }
            )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate rwkv7m next-token loss on binidx data.")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint dir or model.safetensors file")
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
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--magic-prime", type=int, default=None)
    parser.add_argument(
        "--sampling-mode",
        choices=["magic", "sequential", "document", "document_sequential"],
        default="magic",
    )
    parser.add_argument(
        "--loss-mask-after-token",
        type=int,
        default=None,
        help="evaluate only targets immediately following this input token",
    )
    parser.add_argument("--carry-state", action="store_true")
    parser.add_argument(
        "--memory-off-counterfactual",
        action="store_true",
        help=(
            "also evaluate with every Screening residual scaled to zero; "
            "positive memory_loss_delta means memory improved CE"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", choices=["read_screening_only", "read_write"], default="read_screening_only")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--param-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--lm-head-init", choices=["orthogonal", "variance_scaled"], default="orthogonal")
    parser.add_argument("--vocab-parallel", action="store_true")
    parser.add_argument("--remat-blocks", action="store_true")
    parser.add_argument("--sequence-chunk-size", type=int, default=None)
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
    parser.add_argument("--screened-layers", type=int, nargs="*", default=[])
    parser.add_argument("--print-every", type=int, default=1)
    return parse_args_with_config(parser, argv)


def main(argv=None):
    args = parse_args(argv)
    metrics = evaluate_binidx(args)
    print(
        "eval "
        f"steps={metrics['steps']} tokens={metrics['tokens']} "
        f"loss={metrics['loss']:.6f} perplexity={metrics['perplexity']:.6f}"
    )
    if "masked_accuracy" in metrics:
        print(
            "retrieval "
            f"masked_positions={metrics['loss_mask_positions']} "
            f"accuracy={metrics['masked_accuracy']:.6f}"
        )
    if "memory_off_loss" in metrics:
        print(
            "counterfactual "
            f"memory_off_loss={metrics['memory_off_loss']:.6f} "
            f"memory_loss_delta={metrics['memory_loss_delta']:.6f} "
            f"prediction_rms_delta={metrics['prediction_rms_delta']:.6f}"
        )
        if "memory_off_masked_accuracy" in metrics:
            print(
                "retrieval_counterfactual "
                f"memory_off_accuracy="
                f"{metrics['memory_off_masked_accuracy']:.6f} "
                f"memory_accuracy_delta="
                f"{metrics['memory_accuracy_delta']:.6f}"
            )


if __name__ == "__main__":
    main()
