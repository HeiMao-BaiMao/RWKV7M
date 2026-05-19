"""Toy training script for quick smoke testing."""

import jax
from src.model.screening import ScreeningConfig
from src.model.screened_rwkv import ModelConfig
from src.train.train_loop import run_toy_training


def main():
    key = jax.random.PRNGKey(42)

    screening = ScreeningConfig(
        d_model=64,
        d_slot=32,
        d_k=16,
        d_v=16,
        n_slots=4,
        screened_layers=(1,),
        bank_ids=(0, 0, 1, 2),
    )
    cfg = ModelConfig(
        d_model=64,
        d_ffn=128,
        n_layers=3,
        n_heads=4,
        head_size=16,
        vocab_size=256,
        max_seq_len=16,
        dtype="float32",
        use_screening=True,
        screening=screening,
    )

    print("Running toy training...")
    losses, train_state = run_toy_training(
        key,
        cfg,
        batch_size=2,
        seq_len=8,
        num_steps=100,
        print_every=20,
    )
    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"Loss finite: {all(loss == loss for loss in losses)}")


if __name__ == "__main__":
    main()
