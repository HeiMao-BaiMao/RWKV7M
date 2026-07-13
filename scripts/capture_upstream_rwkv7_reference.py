"""Capture official RWKV-LM-V7 x070 fused-CUDA outputs for local parity checks.

Run this script with the upstream Python environment, not the rwkv7m JAX
environment. It intentionally imports the official checkout and executes its
actual CUDA kernels; it does not provide a PyTorch runtime for rwkv7m.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_checkpoint_spec(state, head_size):
    embedding = state["emb.weight"]
    block_ids = sorted({int(name.split(".")[1]) for name in state if name.startswith("blocks.")})
    if block_ids != list(range(len(block_ids))):
        raise ValueError(f"non-contiguous block ids in checkpoint: {block_ids}")
    vocab_size, d_model = embedding.shape
    d_ffn = state["blocks.0.ffn.key.weight"].shape[0]
    if d_model % head_size:
        raise ValueError(f"d_model={d_model} is not divisible by head_size={head_size}")
    return len(block_ids), int(d_model), int(d_ffn), int(vocab_size)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-ffn", type=int, default=224)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--kernel", default="@rwkv3")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.head_size != 64:
        raise SystemExit("official x070 fused path currently requires --head-size 64")
    if args.tokens <= 0 or args.tokens % 16:
        raise SystemExit("--tokens must be positive and divisible by the official chunk length 16")
    upstream = Path(args.upstream_repo).resolve()
    model_file = upstream / "src" / "model.py"
    if not model_file.is_file():
        raise SystemExit(f"official model.py not found: {model_file}")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is required in the upstream environment") from exc
    if not torch.cuda.is_available():
        raise SystemExit("official fused parity capture requires an NVIDIA CUDA device")

    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    state = None
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        args.n_layer, args.d_model, args.d_ffn, args.vocab_size = infer_checkpoint_spec(
            state, args.head_size
        )

    os.environ.update(
        RWKV_MY_TESTING="x070",
        RWKV_KERNEL=args.kernel,
        RWKV_CTXLEN=str(args.tokens),
        RWKV_HEAD_SIZE=str(args.head_size),
        RWKV_HEAD_L2WRAP_CE_CHUNK="0",
        RWKV_FLOAT_MODE="bf16",
        RWKV_JIT_ON="0",
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    old_cwd = Path.cwd()
    sys.path.insert(0, str(upstream))
    try:
        os.chdir(upstream)
        from src.model import RWKV

        model_args = SimpleNamespace(
            my_testing="x070",
            head_size=args.head_size,
            dim_att=args.d_model,
            n_embd=args.d_model,
            n_layer=args.n_layer,
            dim_ffn=args.d_ffn,
            vocab_size=args.vocab_size,
            grad_cp=0,
            weight_decay=0.0,
        )
        model = RWKV(model_args)
        if state is not None:
            model.load_state_dict(state, strict=True)
        model = model.to(device="cuda", dtype=torch.bfloat16).eval()
        rng = np.random.default_rng(args.seed)
        input_ids_np = rng.integers(
            0, args.vocab_size, size=(args.batch_size, args.tokens), dtype=np.int32
        )
        target_ids_np = np.roll(input_ids_np, -1, axis=1)
        input_ids = torch.as_tensor(input_ids_np, device="cuda", dtype=torch.long)
        layer_outputs = []
        with torch.inference_mode():
            x = model.emb(input_ids)
            v_first = torch.empty_like(x)
            for block in model.blocks:
                x, v_first = block(x, v_first)
                layer_outputs.append(x.float().cpu().numpy())
            logits = model.head(model.ln_out(x)).float().cpu().numpy()
        official_state = {
            name: tensor.detach().float().cpu().numpy()
            for name, tensor in model.state_dict().items()
        }
    finally:
        os.chdir(old_cwd)

    commit = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metadata = {
        "format": "rwkv7m-upstream-x070-parity",
        "format_version": 1,
        "upstream_commit": commit,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "n_layers": args.n_layer,
        "d_model": args.d_model,
        "d_ffn": args.d_ffn,
        "vocab_size": args.vocab_size,
        "head_size": args.head_size,
        "batch_size": args.batch_size,
        "tokens": args.tokens,
        "seed": args.seed,
        "official_dtype": "bfloat16",
        "local_dtype": "bfloat16",
        "scope": "official fused CUDA zero-initial-state sequence forward",
    }
    payload = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "input_ids": input_ids_np,
        "target_ids": target_ids_np,
        "reference_logits": logits,
    }
    payload.update({f"weight::{name}": value for name, value in official_state.items()})
    payload.update({f"layer::{i}": value for i, value in enumerate(layer_outputs)})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    print(f"wrote official RWKV-7 parity archive: {output}")
    print(f"upstream_commit={commit} tensors={len(official_state)} logits_shape={logits.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
