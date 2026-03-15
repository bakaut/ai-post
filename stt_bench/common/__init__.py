from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS = {
    "AudioChunk": ".audio_capture",
    "MicrophoneAudioSource": ".audio_capture",
    "list_input_devices": ".audio_capture",
    "resolve_input_device": ".audio_capture",
    "ResultWriter": ".result_writer",
    "format_console_line": ".result_writer",
    "ApiErgonomics": ".types",
    "BackendCapabilities": ".types",
    "BenchmarkResult": ".types",
    "CommitStrategy": ".types",
    "CostMetrics": ".types",
    "EventStatus": ".types",
    "LatencyMetrics": ".types",
    "QualityMetrics": ".types",
    "RealtimeFit": ".types",
    "RunConfig": ".types",
    "StabilityMetrics": ".types",
    "TranscriptEvent": ".types",
}

__all__ = [
    "AudioChunk",
    "MicrophoneAudioSource",
    "list_input_devices",
    "resolve_input_device",
    "ResultWriter",
    "format_console_line",
    "ApiErgonomics",
    "BackendCapabilities",
    "BenchmarkResult",
    "CommitStrategy",
    "CostMetrics",
    "EventStatus",
    "LatencyMetrics",
    "QualityMetrics",
    "RealtimeFit",
    "RunConfig",
    "StabilityMetrics",
    "TranscriptEvent",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
