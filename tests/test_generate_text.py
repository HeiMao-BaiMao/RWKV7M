import jax

from rwkv7m import create_runtime, generate_text, tiny_config


def test_generate_text_runs_with_tokenizer_prompt():
    config = tiny_config(vocab_size=512, d_model=32, n_layers=2, n_heads=2, head_size=16)
    runtime = create_runtime(jax.random.PRNGKey(0), config, batch_size=1)
    text = generate_text(runtime, "a", max_new_tokens=1, temperature=0.0)
    assert text.startswith("a")
    assert len(text) >= 1
