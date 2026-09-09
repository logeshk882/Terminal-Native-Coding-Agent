"""
Observability logger persisting structured JSONL traces of agent execution runs.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, Optional


class RunLogger:
    """Records structured execution events into .tmca/runs/<run_id>.jsonl."""

    def __init__(self, run_id: str, base_dir: Path = Path(".")) -> None:
        self.run_id = run_id
        self.runs_dir = base_dir / ".tmca" / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.runs_dir / f"{run_id}.jsonl"

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Write a single JSON event to the trace log file."""
        record = {
            "run_id": self.run_id,
            "timestamp": time.time(),
            "event": event_type,
            "payload": payload,
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
