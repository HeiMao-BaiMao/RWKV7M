import json

from rwkv7m.cli.audit_distributed_run import main as audit_main, parse_args
from rwkv7m.distributed import audit_distributed_run, audit_report_to_dict


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def write_good_run(output_dir, checkpoint_dir=None):
    checkpoint_root = output_dir if checkpoint_dir is None else checkpoint_dir
    run_config = {
        "args": {
            "checkpoint_backend": "flax",
            "checkpoint_dir": (
                None if checkpoint_dir is None else str(checkpoint_dir)
            ),
            "ctx_len": 4,
            "global_batch_size": 2,
            "log_jsonl": None,
            "summary_json": None,
        },
        "model_config": {"vocab_size": 32},
        "process_info": {"process_index": 0, "process_count": 1},
    }
    run_summary = {
        "status": "completed",
        "start_step": 0,
        "current_step": 2,
        "completed_steps": 2,
        "requested_steps": 2,
        "tokens_per_step": 8,
        "tokens_seen": 16,
        "checkpoint_backend": "flax",
        "latest_checkpoint": str(checkpoint_root / "ckpt-00000002"),
        "best_eval": {
            "step": 2,
            "metric": "loss",
            "mode": "min",
            "value": 1.5,
            "checkpoint": str(checkpoint_root / "ckpt-00000002"),
            "metrics": {"loss": 1.5, "perplexity": 4.4816890703380645},
        },
    }
    checkpoint_payload = {
        "format": "rwkv7m_train_checkpoint",
        "format_version": 1,
        "backend": "flax",
        "step": 2,
        "config": {"vocab_size": 32},
        "rng_key": [0, 0],
        "dataset_position": {"step": 2},
        "metadata": {
            "distributed": True,
            "process_count": 1,
            "local_device_count": 1,
            "device_count": 1,
        },
    }
    metrics = [
        {"split": "train", "step": 1, "tokens": 8, "tokens_per_sec": 100.0, "loss": 2.0},
        {"split": "train", "step": 2, "tokens": 8, "tokens_per_sec": 90.0, "loss": 1.8},
        {"split": "eval", "step": 2, "tokens": 8, "tokens_per_sec": None, "loss": 1.5},
    ]
    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "run_summary.json", run_summary)
    write_json(output_dir / "best_eval.json", run_summary["best_eval"])
    (output_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in metrics),
        encoding="utf-8",
    )
    checkpoint_path = checkpoint_root / "ckpt-00000002"
    write_json(checkpoint_path / "checkpoint.json", checkpoint_payload)
    (checkpoint_path / "train_state.msgpack").write_bytes(b"state")
    (checkpoint_path / "model.safetensors").write_bytes(b"weights")


def test_audit_distributed_run_accepts_complete_run(tmp_path):
    output_dir = tmp_path / "out"
    write_good_run(output_dir)

    report = audit_distributed_run(
        output_dir,
        require_complete=True,
        require_best_checkpoint=True,
        min_train_records=2,
    )

    assert report.ok
    summary = audit_report_to_dict(report)
    assert summary["error_count"] == 0
    assert summary["train_records"] == 2
    assert summary["eval_records"] == 1
    assert summary["current_step"] == 2


def test_audit_distributed_run_uses_separate_checkpoint_root(tmp_path):
    output_dir = tmp_path / "out"
    checkpoint_dir = tmp_path / "shared-checkpoints"
    write_good_run(output_dir, checkpoint_dir)

    report = audit_distributed_run(
        output_dir,
        require_complete=True,
        require_best_checkpoint=True,
    )

    assert report.ok
    assert report.checkpoints == (
        checkpoint_dir / "ckpt-00000002",
    )


def test_audit_distributed_run_reports_missing_best_checkpoint(tmp_path):
    output_dir = tmp_path / "out"
    write_good_run(output_dir)
    for path in (output_dir / "ckpt-00000002").iterdir():
        path.unlink()
    (output_dir / "ckpt-00000002").rmdir()

    report = audit_distributed_run(
        output_dir,
        require_complete=True,
        require_best_checkpoint=True,
    )

    assert not report.ok
    assert any(issue.code == "latest_checkpoint_missing" for issue in report.issues)
    assert any(issue.code == "best_checkpoint_missing" for issue in report.issues)


def test_audit_distributed_run_reports_metric_step_mismatch(tmp_path):
    output_dir = tmp_path / "out"
    write_good_run(output_dir)
    (output_dir / "metrics.jsonl").write_text(
        json.dumps({"split": "train", "step": 1, "tokens": 8, "tokens_per_sec": 1.0, "loss": 2.0}) + "\n",
        encoding="utf-8",
    )

    report = audit_distributed_run(output_dir, require_complete=True)

    assert not report.ok
    assert any(issue.code == "summary_current_step_mismatch" for issue in report.issues)


def test_audit_cli_parse_and_exit_code(tmp_path, capsys):
    output_dir = tmp_path / "out"
    write_good_run(output_dir)
    args = parse_args([str(output_dir), "--require-complete", "--min-train-records", "2"])
    assert args.output_dir == str(output_dir)
    assert args.require_complete is True
    assert args.min_train_records == 2

    try:
        audit_main([str(output_dir), "--require-complete", "--json"])
    except SystemExit as exc:
        assert exc.code == 0

    output = capsys.readouterr().out
    assert '"ok": true' in output
