"""Measure official RWKV-LM-V7 train compute on a fixed GPU batch.

Run this with the official PyTorch/CUDA environment. Dataset sampling, host to
device transfer, compilation, scalar loss reads, logging, and checkpoints are
completed outside every timing window.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np

from benchmark_common import (
    COMPUTE_BENCHMARK_SCHEMA_VERSION,
    fixed_batch_content_sha256,
    gc_policy,
    measurement_contract,
    timing_summary,
)
from train_upstream_rwkv7_single_gpu import (
    git_provenance,
    make_optimizer,
    resolve_load_model,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-repo", required=True)
    parser.add_argument("--fixed-batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-model")
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
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-iterations", type=int, default=50)
    parser.add_argument("--disable-python-gc", action="store_true")
    args = parser.parse_args(argv)
    if args.head_size != 64:
        parser.error("official x070 fused CUDA currently requires head size 64")
    if args.benchmark_warmup < 0 or args.benchmark_iterations <= 0:
        parser.error("benchmark warmup must be non-negative and iterations positive")
    for name in ("n_layer", "n_embd", "dim_att", "dim_ffn", "vocab_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def _measure(function, *, warmup, iterations, torch):
    for _ in range(warmup):
        function()
        torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return timing_summary(samples)


def main(argv=None):
    args = parse_args(argv)
    upstream = Path(args.upstream_repo).resolve()
    if not (upstream / "src" / "model.py").is_file():
        raise SystemExit("official RWKV-LM-V7 src/model.py was not found")
    args.load_model = resolve_load_model(args.load_model)
    with np.load(args.fixed_batch, allow_pickle=False) as archive:
        input_ids_np = np.asarray(archive["input_ids"], dtype=np.int64)
        target_ids_np = np.asarray(archive["target_ids"], dtype=np.int64)
        mask_np = (
            np.asarray(archive["mask"], dtype=np.float32)
            if "mask" in archive
            else None
        )
        if mask_np is not None and not np.all(mask_np == 1):
            raise SystemExit(
                "official fused CE requires an all-ones fixed-batch mask"
            )
    if input_ids_np.shape != target_ids_np.shape or input_ids_np.ndim != 2:
        raise SystemExit("fixed batch must contain equally shaped rank-2 ids")
    if input_ids_np.min() < 0 or input_ids_np.max() >= args.vocab_size:
        raise SystemExit("fixed batch contains token ids outside the model vocabulary")
    batch_size, ctx_len = input_ids_np.shape
    if ctx_len % 16:
        raise SystemExit("official x070 requires fixed-batch token length divisible by 16")
    fixed_batch_file_sha256 = hashlib.sha256(
        args.fixed_batch.read_bytes()
    ).hexdigest()
    fixed_batch_content_fingerprint = fixed_batch_content_sha256(
        {
            "input_ids": input_ids_np,
            "target_ids": target_ids_np,
            **({"mask": mask_np} if mask_np is not None else {}),
        }
    )

    os.environ.update(
        RWKV_MY_TESTING="x070",
        RWKV_KERNEL=args.kernel,
        RWKV_CTXLEN=str(ctx_len),
        RWKV_HEAD_SIZE=str(args.head_size),
        RWKV_HEAD_L2WRAP_CE_CHUNK="0",
        RWKV_FLOAT_MODE=args.precision,
        RWKV_JIT_ON="1",
    )
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("the official PyTorch environment is required") from exc
    if not torch.cuda.is_available():
        raise SystemExit("an NVIDIA CUDA device is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    old_cwd = Path.cwd()
    sys.path.insert(0, str(upstream))
    try:
        os.chdir(upstream)
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
            ctx_len=ctx_len,
            grad_cp=0,
            weight_decay=args.weight_decay,
            lr_init=args.lr_init,
            betas=(args.beta1, args.beta2),
            adam_eps=args.adam_eps,
        )
        model = RWKV(model_args)
        if args.load_model:
            state = torch.load(
                args.load_model,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        else:
            state = model.generate_init_weight()
        model.load_state_dict(
            {name.removeprefix("_forward_module."): value for name, value in state.items()},
            strict=True,
        )
        model = model.to(device="cuda", dtype=torch.bfloat16).train()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        actual_dim_ffn = int(model.blocks[0].ffn.key.out_features)
        optimizer, optimizer_groups = make_optimizer(model, args)
        inputs = torch.as_tensor(input_ids_np, device="cuda", dtype=torch.long)
        targets = torch.as_tensor(target_ids_np, device="cuda", dtype=torch.long)
        torch.cuda.synchronize()

        def forward_phase():
            return l2wrap_cross_entropy(model(inputs), targets)

        def backward_phase(loss):
            loss.backward()

        def optimizer_phase():
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        def full_step():
            optimizer.zero_grad()
            loss = forward_phase()
            backward_phase(loss)
            optimizer_phase()
            return loss

        def measure_backward():
            samples = []
            total = args.benchmark_warmup + args.benchmark_iterations
            for index in range(total):
                optimizer.zero_grad()
                loss = forward_phase()
                torch.cuda.synchronize()
                started = time.perf_counter_ns()
                backward_phase(loss)
                torch.cuda.synchronize()
                if index >= args.benchmark_warmup:
                    samples.append(
                        (time.perf_counter_ns() - started) / 1_000_000.0
                    )
            return timing_summary(samples)

        with gc_policy(args.disable_python_gc):
            timings = {
                "forward": _measure(
                    forward_phase,
                    warmup=args.benchmark_warmup,
                    iterations=args.benchmark_iterations,
                    torch=torch,
                ),
                "backward": measure_backward(),
            }
            optimizer.zero_grad()
            backward_phase(forward_phase())
            torch.cuda.synchronize()
            timings["optimizer"] = _measure(
                optimizer_phase,
                warmup=args.benchmark_warmup,
                iterations=args.benchmark_iterations,
                torch=torch,
            )
            timings["full_step"] = _measure(
                full_step,
                warmup=args.benchmark_warmup,
                iterations=args.benchmark_iterations,
                torch=torch,
            )
        last_loss = full_step().detach()
        torch.cuda.synchronize()
        last_loss_value = float(last_loss.float().cpu())
        commit, dirty, diff_sha256 = git_provenance(upstream)
    finally:
        os.chdir(old_cwd)

    tokens = batch_size * ctx_len
    timings["full_step"]["tokens_per_second_median"] = (
        tokens * 1000.0 / timings["full_step"]["median_ms"]
    )
    payload = {
        "schema_version": COMPUTE_BENCHMARK_SCHEMA_VERSION,
        "benchmark_kind": "train_compute_only",
        "framework": "official_pytorch_cuda",
        "recorded_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "upstream_revision": commit,
        "upstream_dirty": dirty,
        "upstream_diff_sha256": diff_sha256,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "devices": [
            {
                "platform": "gpu",
                "device_kind": torch.cuda.get_device_name(0),
                "id": 0,
                "capability": list(torch.cuda.get_device_capability(0)),
            }
        ],
        "shape": {
            "batch": batch_size,
            "tokens": ctx_len,
            "variant": "official_x070",
            "dtype": "bfloat16",
            "parameter_count": parameter_count,
            "n_layers": args.n_layer,
            "d_model": args.n_embd,
            "d_ffn": actual_dim_ffn,
            "n_heads": args.dim_att // args.head_size,
            "head_size": args.head_size,
            "vocab_size": args.vocab_size,
            "actual_dim_ffn": actual_dim_ffn,
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
        "phase_details": {
            "forward": "autograd-enabled model and fused L2Wrap CE forward",
            "backward": "backward from a graph prepared before the timing window",
            "optimizer": "gradient clipping plus FusedAdam from precomputed gradients",
            "full_step": "forward, backward, clipping, and FusedAdam without phase barriers",
        },
        "optimizer": {
            "implementation": "deepspeed.ops.adam.FusedAdam",
            "lr_init": args.lr_init,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "betas": [args.beta1, args.beta2],
            "eps": args.adam_eps,
            "groups": optimizer_groups,
        },
        "last_loss": last_loss_value,
        "timings": timings,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
