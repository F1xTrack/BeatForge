"""Small crash-tolerant JSONL job event store for detached generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class JobEvent:
    kind: str
    payload: dict[str, Any]
    recorded_at: str


class JobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, job_id: str) -> tuple[Path, Path]:
        if not job_id or any(char in job_id for char in "\\/:\n\r"):
            raise ValueError("job_id must be a simple path-safe identifier")
        directory = self.root / job_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "events.jsonl", directory / "latest.json"

    def append(self, job_id: str, kind: str, payload: dict[str, Any] | None = None) -> JobEvent:
        events_path, latest_path = self._paths(job_id)
        event = JobEvent(kind, dict(payload or {}), datetime.now(timezone.utc).isoformat())
        line = json.dumps(asdict(event), ensure_ascii=False, sort_keys=True) + "\n"
        with events_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
        latest_path.write_text(json.dumps(asdict(event), ensure_ascii=False, indent=2), encoding="utf-8")
        return event

    def _live_path(self, job_id: str) -> Path:
        _, latest_path = self._paths(job_id)
        return latest_path.with_name("live.json")

    def write_live(self, job_id: str, status: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Publish the current provider state without adding an event-log entry.

        This file is intentionally a replaceable snapshot, unlike
        ``events.jsonl``.  Callers must keep secrets out of ``payload``.
        """
        if not status or not isinstance(status, str):
            raise ValueError("live status must be a non-empty string")
        value = {"status": status, "recorded_at": datetime.now(timezone.utc).isoformat(),
                 **dict(payload or {})}
        self._live_path(job_id).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        return value

    def live(self, job_id: str) -> dict[str, Any] | None:
        """Read the latest live provider snapshot, if one exists."""
        path = self._live_path(job_id)
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None

    def latest(self, job_id: str) -> JobEvent | None:
        _, latest_path = self._paths(job_id)
        if not latest_path.exists():
            return None
        value = json.loads(latest_path.read_text(encoding="utf-8"))
        return JobEvent(value["kind"], value["payload"], value["recorded_at"])
