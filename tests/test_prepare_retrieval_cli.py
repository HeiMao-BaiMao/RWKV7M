import json

from rwkv7m.cli.prepare_retrieval import (
    ANSWER_MARKER_TOKEN,
    build_retrieval_dataset,
)
from rwkv7m.data import create_binidx_dataset


def test_retrieval_dataset_has_document_resets_and_answer_only_mask(tmp_path):
    prefix = str(tmp_path / "retrieval")
    metadata = build_retrieval_dataset(
        prefix,
        documents=4,
        ctx_len=16,
        chunks_per_document=3,
        vocab_size=128,
        key_count=32,
        distractors=2,
        seed=7,
    )
    assert metadata["training_contract"] == {
        "sampling_mode": "document",
        "carry_state": False,
        "ctx_len": 48,
        "loss_mask_after_token": ANSWER_MARKER_TOKEN,
    }
    stored = json.loads(
        (tmp_path / "retrieval.retrieval.json").read_text(encoding="utf-8")
    )
    assert stored["format"] == "rwkv7m-delayed-kv-v1"

    dataset = create_binidx_dataset(
        prefix,
        ctx_len=16,
        batch_size=2,
        sampling_mode="document_sequential",
        loss_mask_after_token=ANSWER_MARKER_TOKEN,
    )
    try:
        batches = [dataset.get_batch(step) for step in range(3)]
        assert batches[0]["state_reset_mask"].tolist() == [True, True]
        assert batches[1]["state_reset_mask"].tolist() == [False, False]
        assert batches[2]["state_reset_mask"].tolist() == [False, False]
        assert [float(batch["mask"].sum()) for batch in batches] == [0.0, 0.0, 2.0]
    finally:
        dataset.close()

    training_dataset = create_binidx_dataset(
        prefix,
        ctx_len=48,
        batch_size=2,
        sampling_mode="document",
        loss_mask_after_token=ANSWER_MARKER_TOKEN,
    )
    try:
        training_batch = training_dataset.get_batch(0)
        assert training_batch["state_reset_mask"].tolist() == [True, True]
        assert float(training_batch["mask"].sum()) == 2.0
    finally:
        training_dataset.close()
