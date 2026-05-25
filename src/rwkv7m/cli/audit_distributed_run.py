import argparse
import json

from ..distributed.audit import audit_distributed_run, audit_report_to_dict


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit rwkv7m distributed training run artifacts."
    )
    parser.add_argument("output_dir")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-best-checkpoint", action="store_true")
    parser.add_argument("--min-train-records", type=int, default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args(argv)


def _print_text_report(report):
    payload = audit_report_to_dict(report)
    status = "ok" if payload["ok"] else "failed"
    print(
        f"audit {status}: {payload['output_dir']} "
        f"errors={payload['error_count']} warnings={payload['warning_count']}"
    )
    print(
        f"records train={payload['train_records']} eval={payload['eval_records']} "
        f"checkpoints={len(payload['checkpoints'])} status={payload['run_status']} "
        f"current_step={payload['current_step']}"
    )
    for issue in payload["issues"]:
        path = "" if issue["path"] is None else f" path={issue['path']}"
        print(f"{issue['severity']}: {issue['code']}: {issue['message']}{path}")


def main(argv=None):
    args = parse_args(argv)
    report = audit_distributed_run(
        args.output_dir,
        require_complete=args.require_complete,
        require_best_checkpoint=args.require_best_checkpoint,
        min_train_records=args.min_train_records,
    )
    if args.as_json:
        print(json.dumps(audit_report_to_dict(report), indent=2, sort_keys=True))
    else:
        _print_text_report(report)
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
