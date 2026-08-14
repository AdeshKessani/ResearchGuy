"""
Minimal reasoning-trace logger.

Every stage calls `log_stage(...)` with its input/output. This produces a
JSONL file you can replay later to show *why* the agent made each decision --
this is the artifact that makes the project demo well in interviews.
"""

import json
import time
from pathlib import Path
from typing import Any

TRACE_DIR = Path("traces")


class Tracer:
    def __init__(self, run_id: str):
        TRACE_DIR.mkdir(exist_ok=True)
        self.path = TRACE_DIR / f"{run_id}.jsonl"

    def log_stage(self, stage: str, input_data: Any, output_data: Any) -> None:
        record = {
            "timestamp": time.time(),
            "stage": stage,
            "input": _to_jsonable(input_data),
            "output": _to_jsonable(output_data),
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(o) for o in obj]
    return obj
