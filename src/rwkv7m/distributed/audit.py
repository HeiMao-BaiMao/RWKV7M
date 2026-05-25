from dataclasses import dataclass
import json
import math
from pathlib import Path

from ..io.flax_checkpoint import MODEL_SAFETENSORS, TRAIN_STATE_MSGPACK
from .checkpoint import CHECKPOINT_JSON, ORBAX_TRAIN_STATE_DIR, list_checkpoint_dirs


@dataclass(frozen=True)
class DistributedRunAuditIssue:
    severity: str
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class DistributedRunAuditReport:
    output_dir: Path
    issues: tuple[DistributedRunAuditIssue, ...]
    run_config: dict | None
    run_summary: dict | None
    best_eval: dict | None
    metrics: tuple[dict, ...]
    checkpoints: tuple[Path, ...]

    @property
    def ok(self):
        return not any(issue.severity == "error" for issue in self.issues)


def _issue(issues, severity, code, message, path=None):
    issues.append(
        DistributedRunAuditIssue(
            severity=severity,
            code=code,
            message=message,
            path=None if path is None else str(path),
        )
    )


def _load_json(path, issues, *, code, required=True):
    path = Path(path)
    if not path.exists():
        if required:
            _issue(issues, "error", code, f"missing JSON file: {path}", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        _issue(issues, "error", code, f"invalid JSON: {exc}", path)
    return None


def _load_jsonl(path, issues, *, required=True):
    path = Path(path)
    if not path.exists():
        if required:
            _issue(issues, "error", "metrics_missing", f"missing metrics JSONL: {path}", path)
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                _issue(
                    issues,
                    "error",
                    "metrics_invalid_json",
                    f"invalid JSONL record at line {lineno}: {exc}",
                    path,
                )
    return records


def _checkpoint_step(path):
    try:
        return int(Path(path).name.removeprefix("ckpt-"))
    except ValueError:
        return None


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_recorded_path(output_dir, recorded):
    if recorded is None:
        return None
    path = Path(recorded)
    if path.is_absolute() or path.exists():
        return path
    output_dir = Path(output_dir)
    candidate = output_dir / path
    if candidate.exists():
        return candidate
    if path.name.startswith("ckpt-"):
        return output_dir / path.name
    return candidate


def _configured_path(output_dir, run_config, arg_name, default_name):
    args = {}
    if run_config is not None:
        args = run_config.get("args", {})
    configured = args.get(arg_name)
    if configured is None:
        return Path(output_dir) / default_name
    return _resolve_recorded_path(output_dir, configured)


def _is_finite_number(value):
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _check_metrics(issues, metrics, run_config, run_summary, *, min_train_records=None):
    train_records = [record for record in metrics if record.get("split") == "train"]
    if min_train_records is not None and len(train_records) < int(min_train_records):
        _issue(
            issues,
            "error",
            "metrics_too_few_train_records",
            f"expected at least {int(min_train_records)} train records, found {len(train_records)}",
        )

    train_steps = []
    for record in train_records:
        if "step" not in record:
            _issue(issues, "error", "metrics_step_missing", "train record is missing step")
            continue
        try:
            train_steps.append(int(record["step"]))
        except (TypeError, ValueError):
            _issue(issues, "error", "metrics_step_invalid", f"invalid train step: {record.get('step')}")
        if not _is_finite_number(record.get("loss")):
            _issue(issues, "error", "metrics_loss_invalid", "train record has non-finite loss")
        if record.get("tokens_per_sec") is not None and not _is_finite_number(record.get("tokens_per_sec")):
            _issue(
                issues,
                "error",
                "metrics_tokens_per_sec_invalid",
                "train record has non-finite tokens_per_sec",
            )

    if train_steps and train_steps != sorted(train_steps):
        _issue(issues, "error", "metrics_train_steps_not_monotonic", "train steps are not monotonic")

    args = {} if run_config is None else run_config.get("args", {})
    if "global_batch_size" in args and "ctx_len" in args:
        expected_tokens = int(args["global_batch_size"]) * int(args["ctx_len"])
        for record in train_records:
            if record.get("tokens") != expected_tokens:
                _issue(
                    issues,
                    "warning",
                    "metrics_tokens_mismatch",
                    f"train record tokens={record.get('tokens')} but expected {expected_tokens}",
                )

    if run_summary is not None and train_steps:
        current_step = run_summary.get("current_step")
        if current_step is not None and int(current_step) != max(train_steps):
            _issue(
                issues,
                "error",
                "summary_current_step_mismatch",
                f"summary current_step={current_step} but max train step is {max(train_steps)}",
            )


def _check_summary(issues, run_summary, *, require_complete=False):
    if run_summary is None:
        return
    status = run_summary.get("status")
    if status not in {"running", "completed", "failed"}:
        _issue(issues, "error", "summary_status_invalid", f"invalid summary status: {status}")
    if require_complete and status != "completed":
        _issue(issues, "error", "summary_not_completed", f"run status is {status!r}, not 'completed'")

    start_step = run_summary.get("start_step")
    current_step = run_summary.get("current_step")
    completed_steps = run_summary.get("completed_steps")
    if None not in (start_step, current_step, completed_steps):
        expected_completed = int(current_step) - int(start_step)
        if int(completed_steps) != expected_completed:
            _issue(
                issues,
                "error",
                "summary_completed_steps_mismatch",
                f"completed_steps={completed_steps} but current-start is {expected_completed}",
            )

    tokens_seen = run_summary.get("tokens_seen")
    tokens_per_step = run_summary.get("tokens_per_step")
    if None not in (tokens_seen, tokens_per_step, completed_steps):
        expected_tokens = int(tokens_per_step) * int(completed_steps)
        if int(tokens_seen) != expected_tokens:
            _issue(
                issues,
                "error",
                "summary_tokens_seen_mismatch",
                f"tokens_seen={tokens_seen} but expected {expected_tokens}",
            )


def _check_checkpoint_artifacts(issues, checkpoint_dir, payload):
    backend = payload.get("backend", "flax")
    if backend == "orbax":
        state_dir = checkpoint_dir / ORBAX_TRAIN_STATE_DIR
        if not state_dir.exists():
            _issue(
                issues,
                "error",
                "checkpoint_artifact_missing",
                f"missing Orbax train-state directory: {state_dir}",
                state_dir,
            )
        return
    if backend != "flax":
        _issue(issues, "error", "checkpoint_backend_unknown", f"unknown checkpoint backend: {backend}")
        return

    for filename in (TRAIN_STATE_MSGPACK, MODEL_SAFETENSORS):
        path = checkpoint_dir / filename
        if not path.exists():
            _issue(
                issues,
                "error",
                "checkpoint_artifact_missing",
                f"missing checkpoint artifact: {path}",
                path,
            )


def _check_checkpoints(issues, output_dir, run_summary):
    checkpoints = tuple(list_checkpoint_dirs(output_dir))
    payloads = {}
    for checkpoint_dir in checkpoints:
        step_from_name = _checkpoint_step(checkpoint_dir)
        payload = _load_json(
            checkpoint_dir / CHECKPOINT_JSON,
            issues,
            code="checkpoint_metadata_missing",
            required=True,
        )
        if payload is None:
            continue
        payloads[checkpoint_dir.resolve()] = payload
        payload_step = _int_or_none(payload.get("step"))
        if payload_step is None:
            _issue(
                issues,
                "error",
                "checkpoint_step_invalid",
                f"invalid checkpoint step: {payload.get('step')!r}",
                checkpoint_dir,
            )
        elif step_from_name is not None and payload_step != step_from_name:
            _issue(
                issues,
                "error",
                "checkpoint_step_mismatch",
                f"{checkpoint_dir.name} metadata step={payload.get('step')}",
                checkpoint_dir,
            )
        dataset_position = payload.get("dataset_position")
        if isinstance(dataset_position, dict) and dataset_position.get("step") is not None:
            dataset_step = _int_or_none(dataset_position["step"])
            if dataset_step is None:
                _issue(
                    issues,
                    "error",
                    "checkpoint_dataset_position_invalid",
                    f"invalid dataset_position step={dataset_position['step']!r}",
                    checkpoint_dir,
                )
            elif payload_step is not None and dataset_step != payload_step:
                _issue(
                    issues,
                    "error",
                    "checkpoint_dataset_position_mismatch",
                    f"dataset_position step={dataset_position['step']} but checkpoint step={payload.get('step')}",
                    checkpoint_dir,
                )
        metadata = payload.get("metadata", {})
        if metadata.get("distributed") is not True:
            _issue(
                issues,
                "warning",
                "checkpoint_not_marked_distributed",
                f"checkpoint metadata is not marked distributed: {checkpoint_dir}",
                checkpoint_dir,
            )
        _check_checkpoint_artifacts(issues, checkpoint_dir, payload)

    if run_summary is not None:
        latest_checkpoint = _resolve_recorded_path(output_dir, run_summary.get("latest_checkpoint"))
        if latest_checkpoint is not None:
            if not latest_checkpoint.exists():
                _issue(
                    issues,
                    "error",
                    "latest_checkpoint_missing",
                    f"latest checkpoint does not exist: {latest_checkpoint}",
                    latest_checkpoint,
                )
            else:
                payload = payloads.get(latest_checkpoint.resolve())
                current_step = run_summary.get("current_step")
                payload_step = None if payload is None else _int_or_none(payload.get("step"))
                current_step_int = _int_or_none(current_step)
                if payload_step is not None and current_step_int is not None and payload_step != current_step_int:
                    _issue(
                        issues,
                        "error",
                        "latest_checkpoint_step_mismatch",
                        f"latest checkpoint step={payload.get('step')} but summary current_step={current_step}",
                        latest_checkpoint,
                    )
    return checkpoints


def _check_best_eval(issues, output_dir, run_summary, best_eval, *, require_best_checkpoint=False):
    summary_best = None if run_summary is None else run_summary.get("best_eval")
    active_best = best_eval if best_eval is not None else summary_best

    if require_best_checkpoint and active_best is None:
        _issue(issues, "error", "best_eval_missing", "best eval record is required but missing")
        return
    if active_best is None:
        return

    if not _is_finite_number(active_best.get("value")):
        _issue(issues, "error", "best_eval_value_invalid", "best eval value is non-finite")

    if best_eval is not None and summary_best is not None:
        for key in ("step", "metric", "mode", "value", "checkpoint"):
            if best_eval.get(key) != summary_best.get(key):
                _issue(
                    issues,
                    "error",
                    "best_eval_summary_mismatch",
                    f"best_eval.json {key}={best_eval.get(key)!r} differs from summary {key}={summary_best.get(key)!r}",
                )

    checkpoint = active_best.get("checkpoint")
    if checkpoint is None:
        if require_best_checkpoint:
            _issue(issues, "error", "best_checkpoint_missing", "best eval checkpoint is required but unset")
        return
    checkpoint_path = _resolve_recorded_path(output_dir, checkpoint)
    if not checkpoint_path.exists():
        _issue(
            issues,
            "error",
            "best_checkpoint_missing",
            f"best eval checkpoint does not exist: {checkpoint_path}",
            checkpoint_path,
        )


def audit_distributed_run(
    output_dir,
    *,
    require_complete=False,
    require_best_checkpoint=False,
    min_train_records=None,
    require_metrics=True,
    require_run_config=True,
    require_run_summary=True,
):
    output_dir = Path(output_dir)
    issues = []
    if not output_dir.exists():
        _issue(issues, "error", "output_dir_missing", f"output directory does not exist: {output_dir}", output_dir)
        return DistributedRunAuditReport(output_dir, tuple(issues), None, None, None, tuple(), tuple())
    if not output_dir.is_dir():
        _issue(issues, "error", "output_dir_not_directory", f"output path is not a directory: {output_dir}", output_dir)
        return DistributedRunAuditReport(output_dir, tuple(issues), None, None, None, tuple(), tuple())

    run_config = _load_json(
        output_dir / "run_config.json",
        issues,
        code="run_config_missing",
        required=require_run_config,
    )
    summary_path = _configured_path(output_dir, run_config, "summary_json", "run_summary.json")
    run_summary = _load_json(
        summary_path,
        issues,
        code="run_summary_missing",
        required=require_run_summary,
    )
    metrics_path = _configured_path(output_dir, run_config, "log_jsonl", "metrics.jsonl")
    metrics = tuple(_load_jsonl(metrics_path, issues, required=require_metrics))
    best_eval = _load_json(
        output_dir / "best_eval.json",
        issues,
        code="best_eval_missing",
        required=False,
    )

    _check_summary(issues, run_summary, require_complete=require_complete)
    _check_metrics(
        issues,
        metrics,
        run_config,
        run_summary,
        min_train_records=min_train_records,
    )
    checkpoints = _check_checkpoints(issues, output_dir, run_summary)
    _check_best_eval(
        issues,
        output_dir,
        run_summary,
        best_eval,
        require_best_checkpoint=require_best_checkpoint,
    )

    if run_config is not None and run_summary is not None:
        args = run_config.get("args", {})
        if args.get("checkpoint_backend") != run_summary.get("checkpoint_backend"):
            _issue(
                issues,
                "warning",
                "checkpoint_backend_summary_mismatch",
                "run_config checkpoint_backend differs from run_summary",
            )

    return DistributedRunAuditReport(
        output_dir=output_dir,
        issues=tuple(issues),
        run_config=run_config,
        run_summary=run_summary,
        best_eval=best_eval,
        metrics=metrics,
        checkpoints=checkpoints,
    )


def audit_report_to_dict(report):
    return {
        "ok": report.ok,
        "output_dir": str(report.output_dir),
        "issue_count": len(report.issues),
        "error_count": sum(1 for issue in report.issues if issue.severity == "error"),
        "warning_count": sum(1 for issue in report.issues if issue.severity == "warning"),
        "issues": [
            {
                "severity": issue.severity,
                "code": issue.code,
                "message": issue.message,
                "path": issue.path,
            }
            for issue in report.issues
        ],
        "metric_records": len(report.metrics),
        "train_records": sum(1 for record in report.metrics if record.get("split") == "train"),
        "eval_records": sum(1 for record in report.metrics if record.get("split") == "eval"),
        "checkpoints": [str(path) for path in report.checkpoints],
        "run_status": None if report.run_summary is None else report.run_summary.get("status"),
        "current_step": None if report.run_summary is None else report.run_summary.get("current_step"),
        "latest_checkpoint": None if report.run_summary is None else report.run_summary.get("latest_checkpoint"),
        "best_eval": None if report.run_summary is None else report.run_summary.get("best_eval"),
    }
