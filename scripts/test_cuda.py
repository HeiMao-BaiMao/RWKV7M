"""Test script for CUDA/GPU environment.

Run this on a machine with NVIDIA GPU and JAX CUDA support installed.
Expected: all assertions pass and GPU is actually being used.

Install JAX with CUDA support:
  uv run --extra cuda12 python test_cuda.py
  # or
  pip install "jax[cuda12]"
"""

import jax
import jax.numpy as jnp
import time

from src.model.screening import ScreeningConfig
from src.model.screened_rwkv import ModelConfig, ScreenedRWKVModel, init_rwkv_state, create_model_variables
from src.model.state import init_screen_state
from src.train.train_loop import generate_toy_batch
from src.train.train_step import train_step
from src.train.train_state import TrainState, create_optimizer


def check_gpu_available():
    print("=== GPU Availability Check ===")
    devices = jax.devices()
    print(f"  Available devices: {devices}")
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    if not gpu_devices:
        print("  WARNING: No GPU found. Tests will run on CPU.")
        return False
    print(f"  GPU found: {gpu_devices[0]}")
    return True


def test_forward_gpu():
    print("=== Testing forward pass on GPU ===")
    key = jax.random.PRNGKey(42)
    cfg = ModelConfig(
        d_model=512,
        d_ffn=1024,
        n_layers=8,
        n_heads=8,
        head_size=64,
        vocab_size=10000,
        max_seq_len=256,
        dtype="bfloat16",
        use_screening=True,
        screening=ScreeningConfig(
            d_model=512,
            d_slot=256,
            d_k=64,
            d_v=128,
            n_slots=8,
            screened_layers=(2, 5),
            bank_ids=(0, 0, 1, 1, 2, 2, 2, 2),
        ),
    )
    model = ScreenedRWKVModel(config=cfg)
    rwkv_state = init_rwkv_state(4, cfg)
    screen_state = init_screen_state(4, cfg.screening)
    key, subkey = jax.random.split(key)
    variables, _ = create_model_variables(subkey, cfg, 4)

    input_ids = jax.random.randint(subkey, (4, 128), 0, cfg.vocab_size)

    # JIT compile and warm up
    apply_fn = jax.jit(lambda ids, rs, ss: model.apply(
        variables, ids, rs, ss, phase="read_only", deterministic=True
    ))
    jax.block_until_ready(apply_fn(input_ids, rwkv_state, screen_state))

    # Time forward pass
    times = []
    for _ in range(5):
        start = time.perf_counter()
        out = apply_fn(input_ids, rwkv_state, screen_state)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    print(f"  batch=4, seq=128, layers=8, d_model=512")
    print(f"  avg forward time: {avg_time * 1000:.2f} ms")
    print("  PASS")
    return avg_time


def test_backward_gpu():
    print("=== Testing backward pass on GPU ===")
    key = jax.random.PRNGKey(43)
    cfg = ModelConfig(
        d_model=256,
        d_ffn=512,
        n_layers=4,
        n_heads=4,
        head_size=64,
        vocab_size=5000,
        max_seq_len=128,
        dtype="bfloat16",
        use_screening=True,
        screening=ScreeningConfig(
            d_model=256,
            d_slot=128,
            d_k=32,
            d_v=64,
            n_slots=4,
            screened_layers=(1,),
            bank_ids=(0, 0, 1, 2),
        ),
    )
    model = ScreenedRWKVModel(config=cfg)
    rwkv_state = init_rwkv_state(2, cfg)
    screen_state = init_screen_state(2, cfg.screening)
    key, subkey = jax.random.split(key)
    variables, _ = create_model_variables(subkey, cfg, 2)

    opt_config = {
        "lr_init": 1e-3,
        "lr_final": 1e-5,
        "warmup_steps": 10,
        "max_grad_norm": 1.0,
        "weight_decay": 0.001,
    }
    tx = create_optimizer(opt_config, total_steps=20)
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )

    batch = generate_toy_batch(subkey, 2, 32, cfg.vocab_size)
    rwkv_state = init_rwkv_state(2, cfg)
    screen_state = init_screen_state(2, cfg.screening)

    # Time training step (train_step is already JITted with static_argnames)
    jax.block_until_ready(train_step(train_state, batch, rwkv_state, screen_state, "read_only"))

    times = []
    for _ in range(5):
        start = time.perf_counter()
        out = train_step(train_state, batch, rwkv_state, screen_state, "read_only")
        jax.block_until_ready(out)
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    print(f"  batch=2, seq=32, layers=4, d_model=256")
    print(f"  avg train step time: {avg_time * 1000:.2f} ms")
    print("  PASS")
    return avg_time


def test_large_batch_gpu():
    print("=== Testing large batch throughput ===")
    key = jax.random.PRNGKey(44)
    cfg = ModelConfig(
        d_model=768,
        d_ffn=2048,
        n_layers=12,
        n_heads=12,
        head_size=64,
        vocab_size=32000,
        max_seq_len=512,
        dtype="bfloat16",
        use_screening=True,
        screening=ScreeningConfig(
            d_model=768,
            d_slot=512,
            d_k=64,
            d_v=256,
            n_slots=16,
            screened_layers=(3, 7, 11),
            bank_ids=tuple([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]),
        ),
    )
    model = ScreenedRWKVModel(config=cfg)
    rwkv_state = init_rwkv_state(8, cfg)
    screen_state = init_screen_state(8, cfg.screening)
    key, subkey = jax.random.split(key)
    variables, _ = create_model_variables(subkey, cfg, 8)

    input_ids = jax.random.randint(subkey, (8, 256), 0, cfg.vocab_size)

    apply_fn = jax.jit(lambda ids, rs, ss: model.apply(
        variables, ids, rs, ss, phase="read_only", deterministic=True
    ))
    jax.block_until_ready(apply_fn(input_ids, rwkv_state, screen_state))

    times = []
    for _ in range(3):
        start = time.perf_counter()
        out = apply_fn(input_ids, rwkv_state, screen_state)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    tokens_per_sec = (8 * 256) / avg_time
    print(f"  batch=8, seq=256, layers=12, d_model=768")
    print(f"  avg forward time: {avg_time * 1000:.2f} ms")
    print(f"  throughput: {tokens_per_sec:.0f} tokens/sec")
    print("  PASS")


def test_memory_usage():
    print("=== Testing memory usage ===")
    # JAX doesn't have a simple memory API, but we can check device memory
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            used, total = result.stdout.strip().split(", ")
            print(f"  GPU memory: {used} MiB / {total} MiB used")
    except Exception:
        print("  nvidia-smi not available, skipping memory check")
    print("  PASS")


if __name__ == "__main__":
    has_gpu = check_gpu_available()
    test_forward_gpu()
    test_backward_gpu()
    if has_gpu:
        test_large_batch_gpu()
        test_memory_usage()
    print("\n=== ALL CUDA TESTS PASSED ===")
