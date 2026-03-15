from __future__ import annotations

import importlib
from typing import Any

from .base import BaseSTTBackend

_LAZY_EXPORTS = {
    "LocalSTTBackend": ".local_backend",
    "OpenAIRealtimeSTTBackend": ".openai_backend",
    "GoogleGeminiLiveSTTBackend": ".google_backend",
    "YandexStreamingSTTBackend": ".yandex_backend",
    "ElevenLabsRealtimeSTTBackend": ".elevenlabs_backend",
    "SpeechmaticsRealtimeSTTBackend": ".speechmatics_backend",
    "DeepgramRealtimeSTTBackend": ".deepgram_backend",
}

__all__ = [
    "BaseSTTBackend",
    "LocalSTTBackend",
    "OpenAIRealtimeSTTBackend",
    "GoogleGeminiLiveSTTBackend",
    "YandexStreamingSTTBackend",
    "ElevenLabsRealtimeSTTBackend",
    "SpeechmaticsRealtimeSTTBackend",
    "DeepgramRealtimeSTTBackend",
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
