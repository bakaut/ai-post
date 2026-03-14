from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from common.types import BenchmarkResult, TranscriptEvent


class ResultWriter:
    def __init__(self, base_dir: str | Path = "results") -> None:
        self.base_dir = Path(base_dir)

    def prepare_run_dir(self, run_id: str) -> Path:
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def write_event_jsonl(
        self,
        run_id: str,
        event: TranscriptEvent,
        filename: str = "events.jsonl",
    ) -> Path:
        run_dir = self.prepare_run_dir(run_id)
        path = run_dir / filename
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return path

    def write_events_jsonl(
        self,
        run_id: str,
        events: Iterable[TranscriptEvent],
        filename: str = "events.jsonl",
    ) -> Path:
        run_dir = self.prepare_run_dir(run_id)
        path = run_dir / filename
        with path.open("a", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return path

    def write_summary_json(
        self,
        result: BenchmarkResult,
        filename: str = "summary.json",
    ) -> Path:
        run_dir = self.prepare_run_dir(result.run_id)
        path = run_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    def write_text_artifact(
        self,
        run_id: str,
        name: str,
        content: str,
    ) -> Path:
        run_dir = self.prepare_run_dir(run_id)
        path = run_dir / name
        with path.open("w", encoding="utf-8") as f:
            f.write(content)
        return path


def format_console_line(event: TranscriptEvent) -> str:
    cost = f"${event.cost_estimate_usd:.6f}" if event.cost_estimate_usd is not None else "-"
    conf = f"{event.provider_confidence:.2f}" if event.provider_confidence is not None else "-"
    return (
        f"[{event.backend}][{event.status.value}] {event.text}\n"
        f"latency={event.latency_ms}ms "
        f"audio={event.audio_sec:.2f}s "
        f"rtf={event.rtf:.2f} "
        f"conf={conf} "
        f"cost={cost} "
        f"restarts={event.restart_count} "
        f"errors={event.error_count} "
        f"dropped={event.dropped_chunks}"
    )
