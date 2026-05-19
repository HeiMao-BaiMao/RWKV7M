import jax
import jax.numpy as jnp
from ..model.state import init_screen_state, ModelScreenState


def prefill(
    model,
    variables,
    prompt_ids,
    rwkv_state,
    screen_state,
    *,
    phase="read_screening_only",
):
    logits, new_rwkv_state, new_screen_state, stats = model.apply(
        variables,
        prompt_ids,
        rwkv_state,
        screen_state,
        phase=phase,
        deterministic=True,
    )
    return logits, new_rwkv_state, new_screen_state, stats


def decode_one(
    model,
    variables,
    token_id,
    rwkv_state,
    screen_state,
    *,
    phase="read_screening_only",
):
    if token_id.ndim == 2 and token_id.shape[1] == 1:
        input_ids = token_id
    else:
        input_ids = token_id[:, None]
    logits, new_rwkv_state, new_screen_state, stats = model.apply(
        variables,
        input_ids,
        rwkv_state,
        screen_state,
        phase=phase,
        deterministic=True,
    )
    next_logits = logits[:, -1, :]
    return next_logits, new_rwkv_state, new_screen_state, stats


def generate(
    model,
    variables,
    prompt_ids,
    rwkv_state,
    screen_state,
    *,
    max_new_tokens=50,
    temperature=1.0,
    top_p=0.9,
    rng_key=None,
    phase="read_screening_only",
):
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    logits, rwkv_state, screen_state, stats = prefill(
        model, variables, prompt_ids, rwkv_state, screen_state, phase=phase
    )

    last_logits = logits[:, -1, :]
    generated_ids = []

    for _ in range(max_new_tokens):
        rng_key, subkey = jax.random.split(rng_key)

        if temperature > 0.0:
            last_logits = last_logits / temperature
            if top_p < 1.0:
                sorted_logits = jnp.sort(last_logits, axis=-1)[:, ::-1]
                cumulative_probs = jnp.cumsum(
                    jax.nn.softmax(sorted_logits, axis=-1), axis=-1
                )
                cutoff = jnp.sum(cumulative_probs < top_p, axis=-1, keepdims=True)
                mask = jnp.zeros_like(last_logits)
                for b in range(last_logits.shape[0]):
                    threshold = sorted_logits[b, jnp.minimum(cutoff[b], sorted_logits.shape[-1] - 1)]
                    mask = mask.at[b].set(last_logits[b] >= threshold)
                last_logits = jnp.where(mask, last_logits, -1e10)

            next_token = jax.random.categorical(subkey, last_logits)[:, None]
        else:
            next_token = jnp.argmax(last_logits, axis=-1)[:, None]

        generated_ids.append(next_token)

        last_logits, rwkv_state, screen_state, _ = decode_one(
            model, variables, next_token, rwkv_state, screen_state, phase=phase
        )

    generated = jnp.concatenate(generated_ids, axis=1)
    return generated, rwkv_state, screen_state
