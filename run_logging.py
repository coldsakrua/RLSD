from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from transformers import TrainerCallback


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_run_id(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    value = "".join(out).strip("_.")
    return value or "run"


def _normalize_report_to(report_to: Any) -> list[str]:
    if report_to is None:
        return []
    if isinstance(report_to, str):
        raw = report_to.strip()
        if not raw or raw.lower() == "none":
            return []
        if "," in raw:
            items = [x.strip() for x in raw.split(",") if x.strip()]
            return items
        return [raw]
    if isinstance(report_to, (list, tuple, set)):
        out: list[str] = []
        for item in report_to:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    return [str(report_to)]


def _coerce_log_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return str(value)
    return str(value)


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# Longest prefixes first so e.g. strict_split_flip/ is stripped before strict/.
_METRIC_PREFIXES_TO_STRIP = (
    "strict_split_flip/",
    "strict_split/",
    "strict/",
    "rlsd/",
    "opsd/",
)


def normalize_metric_key(key: str) -> str:
    """Drop trainer-variant prefixes so dashboards share one schema (e.g. wrong_weight/mean)."""
    k = str(key).strip()
    if not k:
        return k
    while True:
        stripped = False
        for prefix in _METRIC_PREFIXES_TO_STRIP:
            if k.startswith(prefix):
                k = k[len(prefix) :]
                stripped = True
                break
        if not stripped:
            break
    return k


def _json_dump(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def configure_wandb_offline(
    training_args,
    *,
    disable_wandb: bool,
    run_name: Optional[str] = None,
    default_project: str = "RLSD",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    output_dir = os.path.abspath(getattr(training_args, "output_dir", ".") or ".")
    os.makedirs(output_dir, exist_ok=True)

    if run_name:
        training_args.run_name = run_name

    metrics_jsonl_path = os.path.join(output_dir, "train_metrics.jsonl")
    meta_path = os.path.join(output_dir, "wandb_run_meta.json")

    if disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        training_args.report_to = []
        payload = {
            "timestamp_utc": _utc_now_iso(),
            "output_dir": output_dir,
            "run_name": getattr(training_args, "run_name", None),
            "wandb_enabled": False,
            "wandb_mode": "disabled",
            "report_to": [],
            "metrics_jsonl_path": metrics_jsonl_path,
        }
        if extra_meta:
            payload["extra"] = extra_meta
        _json_dump(meta_path, payload)
        return {"output_dir": output_dir, "metrics_jsonl_path": metrics_jsonl_path, "meta_path": meta_path}

    # Ensure W&B is enabled and fully local by default.
    os.environ.pop("WANDB_DISABLED", None)
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_PROJECT", default_project)
    os.environ.setdefault("WANDB_DIR", output_dir)

    wandb_data_dir = os.path.join(output_dir, ".wandb_data")
    wandb_cache_dir = os.path.join(output_dir, ".wandb_cache")
    wandb_artifact_dir = os.path.join(output_dir, "wandb_artifacts")
    for p in (wandb_data_dir, wandb_cache_dir, wandb_artifact_dir):
        os.makedirs(p, exist_ok=True)
    os.environ.setdefault("WANDB_DATA_DIR", wandb_data_dir)
    os.environ.setdefault("WANDB_CACHE_DIR", wandb_cache_dir)
    os.environ.setdefault("WANDB_ARTIFACT_DIR", wandb_artifact_dir)

    if run_name:
        os.environ.setdefault("WANDB_NAME", str(run_name))

    if not os.environ.get("WANDB_RUN_ID"):
        run_id_seed = f"{getattr(training_args, 'run_name', '')}_{os.path.basename(output_dir)}"
        os.environ["WANDB_RUN_ID"] = _sanitize_run_id(run_id_seed)[:128]
    os.environ.setdefault("WANDB_RESUME", "allow")

    report_to = _normalize_report_to(getattr(training_args, "report_to", None))
    report_to_lower = {x.lower() for x in report_to}
    if "all" in report_to_lower:
        report_to = ["wandb"]
    elif "wandb" not in report_to_lower:
        report_to.append("wandb")
    training_args.report_to = report_to

    wandb_root = os.path.join(os.environ.get("WANDB_DIR", output_dir), "wandb")
    sync_glob = os.path.join(wandb_root, "offline-run-*")
    payload = {
        "timestamp_utc": _utc_now_iso(),
        "output_dir": output_dir,
        "run_name": getattr(training_args, "run_name", None),
        "wandb_enabled": True,
        "report_to": training_args.report_to,
        "wandb_mode": os.environ.get("WANDB_MODE"),
        "wandb_project": os.environ.get("WANDB_PROJECT"),
        "wandb_name": os.environ.get("WANDB_NAME"),
        "wandb_run_id": os.environ.get("WANDB_RUN_ID"),
        "wandb_resume": os.environ.get("WANDB_RESUME"),
        "wandb_dir": os.environ.get("WANDB_DIR"),
        "wandb_data_dir": os.environ.get("WANDB_DATA_DIR"),
        "wandb_cache_dir": os.environ.get("WANDB_CACHE_DIR"),
        "wandb_artifact_dir": os.environ.get("WANDB_ARTIFACT_DIR"),
        "wandb_sync_glob": sync_glob,
        "wandb_sync_hint": f"unset WANDB_MODE && wandb sync \"{sync_glob}\"",
        "metrics_jsonl_path": metrics_jsonl_path,
    }
    if extra_meta:
        payload["extra"] = extra_meta
    _json_dump(meta_path, payload)

    return {"output_dir": output_dir, "metrics_jsonl_path": metrics_jsonl_path, "meta_path": meta_path}


class StructuredJsonMetricsCallback(TrainerCallback):
    def __init__(self, jsonl_path: str):
        self.jsonl_path = os.path.abspath(jsonl_path)
        self.dir_path = os.path.dirname(self.jsonl_path)
        os.makedirs(self.dir_path, exist_ok=True)
        self.latest_path = os.path.join(self.dir_path, "train_metrics.latest.json")
        self.summary_path = os.path.join(self.dir_path, "train_metrics.summary.json")
        self._start_time = time.time()
        self._scalar_stats: Dict[str, Dict[str, float]] = {}
        self._event_count = 0

    def _should_write(self, state) -> bool:
        return bool(getattr(state, "is_world_process_zero", True))

    def _update_scalar_stats(self, logs: Dict[str, Any]) -> None:
        for key, raw_value in logs.items():
            if not _is_numeric_scalar(raw_value):
                continue
            value = float(raw_value)
            stat = self._scalar_stats.get(key)
            if stat is None:
                self._scalar_stats[key] = {
                    "count": 1.0,
                    "sum": value,
                    "min": value,
                    "max": value,
                    "last": value,
                }
                continue
            stat["count"] += 1.0
            stat["sum"] += value
            stat["min"] = min(stat["min"], value)
            stat["max"] = max(stat["max"], value)
            stat["last"] = value

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not self._should_write(state):
            return
        clean_logs: Dict[str, Any] = {}
        for k, v in logs.items():
            cv = _coerce_log_value(v)
            if cv is not None:
                clean_logs[normalize_metric_key(k)] = cv

        record = {
            "timestamp_utc": _utc_now_iso(),
            "wall_time_sec": float(max(0.0, time.time() - self._start_time)),
            "step": int(getattr(state, "global_step", 0) or 0),
            "epoch": float(state.epoch) if state.epoch is not None else None,
        }
        record.update(clean_logs)

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

        self._event_count += 1
        self._update_scalar_stats(clean_logs)
        _json_dump(self.latest_path, record)

    def on_train_end(self, args, state, control, **kwargs):
        if not self._should_write(state):
            return
        scalar_summary: Dict[str, Dict[str, float]] = {}
        for key, stat in self._scalar_stats.items():
            count = max(1.0, stat["count"])
            scalar_summary[key] = {
                "last": float(stat["last"]),
                "mean": float(stat["sum"] / count),
                "min": float(stat["min"]),
                "max": float(stat["max"]),
                "count": int(stat["count"]),
            }
        payload = {
            "timestamp_utc": _utc_now_iso(),
            "events": int(self._event_count),
            "total_wall_time_sec": float(max(0.0, time.time() - self._start_time)),
            "final_step": int(getattr(state, "global_step", 0) or 0),
            "scalar_summary": scalar_summary,
            "source_jsonl": self.jsonl_path,
        }
        _json_dump(self.summary_path, payload)
