from dataclasses import replace

from ..train.train_step import train_step
from .sharding import host_batch_to_global_arrays


def train_batch_data_parallel(
    dist,
    host_batch,
    layout,
    *,
    phase="read_screening_only",
):
    global_batch = host_batch_to_global_arrays(host_batch, dist.batch_sharding, layout)
    train_state, rwkv_state, screen_state, metrics = train_step(
        dist.train_state,
        global_batch,
        dist.rwkv_state,
        dist.screen_state,
        phase=phase,
    )
    return (
        replace(
            dist,
            train_state=train_state,
            rwkv_state=rwkv_state,
            screen_state=screen_state,
        ),
        metrics,
    )
