from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventStatus(str, Enum):
    PARTIAL = "partial"
    FINAL = "final"
    ERROR = "error"
    INFO = "info"


class CommitStrategy(str, Enum):
    NONE = "none"
    VAD = "vad"
    MANUAL = "manual"
    PROVIDER_NATIVE = "provider_native"


@dataclass(slots=True)
class RunConfig:
    run_id: str
    backend: str
    language: str = "ru"
    audio_device: str = "default"
    sample_rate_hz: int = 16000
    channels: int = 1
    chunk_ms: int = 100
    mode: str = "realtime"
    commit_strategy: CommitStrategy = CommitStrategy.NONE
    reference_text_path: str | None = None
    pricing_profile: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BackendCapabilities:
    backend: str
    supports_partial: bool
    supports_final: bool
    supports_word_timestamps: bool = False
    supports_speaker_diarization: bool = False
    supports_confidence: bool = False
    supports_language_detection: bool = False
    supports_vad_server_side: bool = False
    supports_manual_commit: bool = False
    requires_stream_reconnect: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TranscriptEvent:
    run_id: str
    backend: str
    session_id: str
    segment_id: str

    status: EventStatus
    text: str

    started_at_iso: str
    emitted_at_iso: str
    started_at_monotonic_ms: int
    emitted_at_monotonic_ms: int
    latency_ms: int
    audio_sec: float
    rtf: float

    provider_confidence: float | None = None
    provider_avg_logprob: float | None = None
    language: str | None = None
    partial_index: int | None = None

    restart_count: int = 0
    error_count: int = 0
    dropped_chunks: int = 0
    is_timeout: bool = False
    cost_estimate_usd: float | None = None

    provider_event_type: str | None = None
    raw_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass(slots=True)
class LatencyMetrics:
    partial_count: int = 0
    final_count: int = 0
    avg_partial_latency_ms: float | None = None
    avg_final_latency_ms: float | None = None
    p50_final_latency_ms: float | None = None
    p95_final_latency_ms: float | None = None
    p99_final_latency_ms: float | None = None
    min_final_latency_ms: int | None = None
    max_final_latency_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StabilityMetrics:
    reconnects: int = 0
    restarts: int = 0
    errors: int = 0
    timeouts: int = 0
    dropped_chunks: int = 0
    total_segments: int = 0
    finalized_segments: int = 0
    partial_segments: int = 0
    failed_segments: int = 0
    success_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CostMetrics:
    audio_seconds_total: float = 0.0
    audio_minutes_total: float = 0.0
    cost_estimate_usd: float = 0.0
    pricing_profile: str | None = None
    pricing_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QualityMetrics:
    reference_text_path: str | None = None
    reference_text: str | None = None
    wer: float | None = None
    cer: float | None = None
    punctuation_score_5: int | None = None
    capitalization_score_5: int | None = None
    readability_score_5: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ApiErgonomics:
    auth_score_5: int | None = None
    integration_score_5: int | None = None
    docs_score_5: int | None = None
    reconnect_complexity_score_5: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RealtimeFit:
    supports_partial: bool
    supports_final: bool
    supports_server_vad: bool = False
    supports_manual_commit: bool = False
    estimated_fit_score_10: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkResult:
    run_id: str
    backend: str
    started_at_iso: str
    finished_at_iso: str
    config: RunConfig
    capabilities: BackendCapabilities
    total_events: int
    total_final_events: int
    total_partial_events: int
    latency: LatencyMetrics
    stability: StabilityMetrics
    cost: CostMetrics
    quality: QualityMetrics
    api: ApiErgonomics
    realtime_fit: RealtimeFit
    final_text_joined: str = ""
    final_text_segments: list[str] = field(default_factory=list)
    raw_files: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "backend": self.backend,
            "started_at_iso": self.started_at_iso,
            "finished_at_iso": self.finished_at_iso,
            "config": self.config.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "total_events": self.total_events,
            "total_final_events": self.total_final_events,
            "total_partial_events": self.total_partial_events,
            "latency": self.latency.to_dict(),
            "stability": self.stability.to_dict(),
            "cost": self.cost.to_dict(),
            "quality": self.quality.to_dict(),
            "api": self.api.to_dict(),
            "realtime_fit": self.realtime_fit.to_dict(),
            "final_text_joined": self.final_text_joined,
            "final_text_segments": self.final_text_segments,
            "raw_files": self.raw_files,
            "notes": self.notes,
        }
