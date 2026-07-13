"""Train the official RWKV-LM-V7 x070 model without Lightning or DeepSpeed launchers.

This is an experiment harness, not a PyTorch implementation of rwkv7m. It imports
the model, fused CUDA loss, dataset reader, and FusedAdam from an official
RWKV-LM-V7 checkout and only replaces the orchestration layer with a direct,
single-GPU training loop.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from types import SimpleNamespace


SAMPLES_PER_EPOCH = 40_320


def upstream_learning_rate(
    step: int,
    *,
    lr_init: float,
    lr_final: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Match RWKV-LM-V7's token-based cosine callback for positive exit tokens."""
    if total_steps <= warmup_steps:
        raise ValueError("total_steps must be greater than warmup_steps")
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = max(0.0, min(1.0, progress))
    final_factor = lr_final / lr_init
    multiplier = (0.5 + final_factor / 2) + (
        0.5 - final_factor / 2
    ) * math.cos(math.pi * progress)
    lr = lr_init * multiplier
    if step < warmup_steps:
        lr *= 0.01 + 0.99 * step / warmup_steps
    return lr


def optimizer_group(name: str, squeezed_ndim: int, weight_decay: float) -> str:
    """Return the exact group selected by upstream RWKV.configure_optimizers."""
    name = name.removeprefix("_forward_module.")
    if "att.w0" in name:
        return "2x"
    if squeezed_ndim >= 2 and weight_decay > 0 and ".weight" in name:
        return "decay"
    return "1x"


def epoch_and_index(global_sample_index: int) -> tuple[int, int]:
    """Map a global sample number to upstream's 40,320-sample epoch indexing."""
    return divmod(global_sample_index, SAMPLES_PER_EPOCH)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-repo", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--load-model")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--ctx-len", type=int, required=True)
    parser.add_argument("--magic-prime", type=int, required=True)
    parser.add_argument("--n-layer", type=int, required=True)
    parser.add_argument("--n-embd", type=int, required=True)
    parser.add_argument("--dim-att", type=int, required=True)
    parser.add_argument("--dim-ffn", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--kernel", default="@rwkv3")
    parser.add_argument("--precision", choices=("bf16",), default="bf16")
    parser.add_argument("--lr-init", type=float, default=1e-3)
    parser.add_argument("--lr-final", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=0)
    return parser.parse_args(argv)


def validate_args(args):
    upstream = Path(args.upstream_repo).resolve()
    data_file = Path(args.data_file).resolve()
    if not (upstream / "src" / "model.py").is_file():
        raise SystemExit(f"official model.py not found: {upstream / 'src' / 'model.py'}")
    for suffix in (".bin", ".idx"):
        if not Path(f"{data_file}{suffix}").is_file():
            raise SystemExit(f"dataset file not found: {data_file}{suffix}")
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.warmup_steps < 0 or args.steps <= args.warmup_steps:
        raise SystemExit("--steps must be greater than non-negative --warmup-steps")
    if args.global_batch_size <= 0 or args.micro_batch_size <= 0:
        raise SystemExit("batch sizes must be positive")
    if args.global_batch_size % args.micro_batch_size:
        raise SystemExit("--global-batch-size must be divisible by --micro-batch-size")
    if SAMPLES_PER_EPOCH % args.global_batch_size:
        raise SystemExit("--global-batch-size must divide 40320 for upstream sampling parity")
    if args.ctx_len <= 0 or args.ctx_len % 16:
        raise SystemExit("--ctx-len must be positive and divisible by x070 chunk length 16")
    if args.head_size != 64:
        raise SystemExit("official x070 fused CUDA currently requires --head-size 64")
    if args.n_embd % 32 or args.dim_att % 32 or args.dim_ffn % 32:
        raise SystemExit("n_embd, dim_att, and dim_ffn must be multiples of 32")
    if args.n_embd % args.head_size:
        raise SystemExit("--n-embd must be divisible by --head-size")
    if args.save_every < 0:
        raise SystemExit("--save-every must be non-negative")
    return upstream, data_file


def git_provenance(repo: Path) -> tuple[str, bool, str]:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--binary"],
        check=True,
        capture_output=True,
    ).stdout
    return commit, bool(diff), hashlib.sha256(diff).hexdigest()


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_official_state(torch, model, args, output_dir: Path):
    if args.load_model:
        checkpoint = Path(args.load_model).resolve()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        source = str(checkpoint)
    else:
        state = model.generate_init_weight()
        checkpoint = output_dir / "rwkv-init.pth"
        torch.save(state, checkpoint)
        source = str(checkpoint)
    normalized = {
        name.removeprefix("_forward_module."): value for name, value in state.items()
    }
    model.load_state_dict(normalized, strict=True)
    return source


def make_optimizer(model, args):
    from deepspeed.ops.adam import FusedAdam

    grouped = {"1x": [], "2x": [], "decay": []}
    names = {key: [] for key in grouped}
    for name, parameter in model.named_parameters():
        group = optimizer_group(name, len(parameter.squeeze().shape), args.weight_decay)
        grouped[group].append(parameter)
        names[group].append(name.removeprefix("_forward_module."))
    optim_groups = [
        {"params": grouped["1x"], "weight_decay": 0.0, "my_lr_scale": 1.0},
        {"params": grouped["2x"], "weight_decay": 0.0, "my_lr_scale": 2.0},
    ]
    if grouped["decay"]:
        optim_groups.append(
            {
                "params": grouped["decay"],
                "weight_decay": args.weight_decay,
                "my_lr_scale": 1.0,
            }
        )
    optimizer = FusedAdam(
        optim_groups,
        lr=args.lr_init,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        bias_correction=True,
        adam_w_mode=args.weight_decay > 0,
        amsgrad=False,
    )
    return optimizer, {key: sorted(value) for key, value in names.items()}


def main(argv=None):
    args = parse_args(argv)
    upstream, data_file = validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.update(
        RWKV_MY_TESTING="x070",
        RWKV_KERNEL=args.kernel,
        RWKV_CTXLEN=str(args.ctx_len),
        RWKV_HEAD_SIZE=str(args.head_size),
        RWKV_HEAD_L2WRAP_CE_CHUNK="0",
        RWKV_FLOAT_MODE=args.precision,
        RWKV_JIT_ON="1",
    )
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise SystemExit("the upstream venv with numpy and torch is required") from exc
    if not torch.cuda.is_available():
        raise SystemExit("the official fused x070 training path requires an NVIDIA CUDA GPU")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    old_cwd = Path.cwd()
    sys.path.insert(0, str(upstream))
    try:
        os.chdir(upstream)
        from src.dataset import MyDataset
        from src.model import RWKV, l2wrap_cross_entropy

        model_args = SimpleNamespace(
            accelerator="gpu",
            my_testing="x070",
            head_size=args.head_size,
            dim_att=args.dim_att,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            dim_ffn=args.dim_ffn,
            vocab_size=args.vocab_size,
            ctx_len=args.ctx_len,
            grad_cp=0,
            weight_decay=args.weight_decay,
            lr_init=args.lr_init,
            betas=(args.beta1, args.beta2),
            adam_eps=args.adam_eps,
        )
        dataset_args = SimpleNamespace(
            vocab_size=args.vocab_size,
            data_file=str(data_file),
            epoch_steps=SAMPLES_PER_EPOCH // args.global_batch_size,
            real_bsz=args.global_batch_size,
            micro_bsz=args.micro_batch_size,
            train_stage=3,
            ctx_len=args.ctx_len,
            magic_prime=args.magic_prime,
        )
        dataset = MyDataset(dataset_args)
        dataset.global_rank = 0
        dataset.world_size = 1
        dataset.real_epoch = 0

        model = RWKV(model_args)
        initial_checkpoint = load_official_state(torch, model, args, output_dir)
        model = model.to(device="cuda", dtype=torch.bfloat16).train()
        optimizer, optimizer_groups = make_optimizer(model, args)

        commit, upstream_dirty, upstream_diff_sha256 = git_provenance(upstream)
        config = {
            **vars(args),
            "upstream_repo": str(upstream),
            "data_file": str(data_file),
            "output_dir": str(output_dir),
            "upstream_commit": commit,
            "upstream_dirty": upstream_dirty,
            "upstream_diff_sha256": upstream_diff_sha256,
            "initial_checkpoint": initial_checkpoint,
            "launcher": "direct_single_gpu",
            "gradient_accumulation_steps": args.global_batch_size // args.micro_batch_size,
            "samples_per_epoch": SAMPLES_PER_EPOCH,
            "sampler": "official MyDataset cubic permutation, global sample order",
            "loss": "official fused l2wrap_cross_entropy",
            "optimizer": "deepspeed.ops.adam.FusedAdam",
            "optimizer_groups": optimizer_groups,
            "host": platform.node(),
            "platform": platform.platform(),
            "device": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        write_json(output_dir / "run_config.json", config)

        metric_fields = [
            "split", "step", "loss", "perplexity", "lr", "weight_decay",
            "grad_norm", "tokens", "tokens_per_sec", "elapsed_seconds",
        ]
        metrics_jsonl = (output_dir / "metrics.jsonl").open("w", encoding="utf-8")
        metrics_csv = (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(metrics_csv, fieldnames=metric_fields)
        csv_writer.writeheader()
        started = time.perf_counter()
        last_record = None

        for step in range(args.steps):
            lr = upstream_learning_rate(
                step,
                lr_init=args.lr_init,
                lr_final=args.lr_final,
                warmup_steps=args.warmup_steps,
                total_steps=args.steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr * group["my_lr_scale"]
                if group["weight_decay"] > 0:
                    group["weight_decay"] = args.weight_decay
            # DeepSpeed FusedAdam exposes zero_grad() without PyTorch's
            # set_to_none keyword on supported upstream versions.
            optimizer.zero_grad()
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            loss_sum = 0.0
            accumulation_steps = args.global_batch_size // args.micro_batch_size
            for accumulation in range(accumulation_steps):
                first_sample = step * args.global_batch_size + accumulation * args.micro_batch_size
                batch = []
                for offset in range(args.micro_batch_size):
                    epoch, index = epoch_and_index(first_sample + offset)
                    dataset.real_epoch = epoch
                    batch.append(dataset[index])
                inputs = torch.stack([item[0] for item in batch]).to("cuda", non_blocking=True)
                targets = torch.stack([item[1] for item in batch]).to("cuda", non_blocking=True)
                loss = l2wrap_cross_entropy(model(inputs), targets)
                loss_sum += float(loss.detach().float().cpu())
                (loss / accumulation_steps).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - step_started
            loss_value = loss_sum / accumulation_steps
            record = {
                "split": "train",
                "step": step + 1,
                "loss": loss_value,
                "perplexity": math.exp(min(loss_value, 80.0)),
                "lr": lr,
                "weight_decay": args.weight_decay,
                "grad_norm": float(grad_norm.detach().float().cpu()),
                "tokens": (step + 1) * args.global_batch_size * args.ctx_len,
                "tokens_per_sec": args.global_batch_size * args.ctx_len / elapsed,
                "elapsed_seconds": elapsed,
            }
            metrics_jsonl.write(json.dumps(record, sort_keys=True) + "\n")
            metrics_jsonl.flush()
            csv_writer.writerow(record)
            metrics_csv.flush()
            last_record = record
            print(
                f"step={step + 1}/{args.steps} loss={loss_value:.6f} "
                f"lr={lr:.8g} tok/s={record['tokens_per_sec']:.1f}"
            )
            if args.save_every and (step + 1) % args.save_every == 0:
                torch.save(model.state_dict(), output_dir / f"rwkv-step-{step + 1}.pth")

        metrics_jsonl.close()
        metrics_csv.close()
        final_checkpoint = output_dir / "rwkv-final.pth"
        torch.save(model.state_dict(), final_checkpoint)
        summary = {
            "status": "complete",
            "launcher": "direct_single_gpu",
            "upstream_commit": commit,
            "completed_steps": args.steps,
            "tokens": args.steps * args.global_batch_size * args.ctx_len,
            "elapsed_seconds": time.perf_counter() - started,
            "last_train": last_record,
            "latest_checkpoint": str(final_checkpoint),
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        write_json(output_dir / "run_summary.json", summary)
        print(f"wrote official direct-loop artifacts: {output_dir}")
    finally:
        os.chdir(old_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
