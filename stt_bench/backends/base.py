from __future__ import annotations

import abc
import asyncio
import time
import uuid
from collections.abc import AsyncIterator

from common.types import BackendCapabilities, EventStatus, RunConfig, TranscriptEvent, utc_now_iso


class BaseSTTBackend(abc.ABC):
    name: str = "base"

    def __init__(self) -> None:
        self.config: RunConfig | None = None
        self.session_id: str = uuid.uuid4().hex
        self._events_queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self._started = False
        self._stopped = False
        self._restart_count = 0
        self._error_count = 0
        self._dropped_chunks = 0

    @abc.abstractmethod
    def capabilities(self) -> BackendCapabilities:
        raise NotImplementedError

    async def start(self, config: RunConfig) -> None:
        if self._started:
            raise RuntimeError(f"{self.name}: backend already started")
        self.config = config
        self._started = True
        self._stopped = False
        await self._on_start()

    @abc.abstractmethod
    async def _on_start(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def feed_audio(self, pcm_chunk: bytes) -> None:
        raise NotImplementedError

    async def finish_audio(self) -> None:
        await self._on_finish_audio()

    async def _on_finish_audio(self) -> None:
        return None

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            item = await self._events_queue.get()
            if item is None:
                break
            yield item

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            await self._on_stop()
        finally:
            await self._events_queue.put(None)

    async def _on_stop(self) -> None:
        return None

    async def emit_event(self, event: TranscriptEvent) -> None:
        await self._events_queue.put(event)

    async def emit_partial(
        self,
        *,
        segment_id: str,
        text: str,
        started_at_monotonic_ms: int,
        audio_sec: float,
        partial_index: int | None = None,
        provider_confidence: float | None = None,
        provider_avg_logprob: float | None = None,
        language: str | None = None,
        provider_event_type: str | None = None,
        raw_meta: dict | None = None,
    ) -> TranscriptEvent:
        event = self._build_event(
            status=EventStatus.PARTIAL,
            segment_id=segment_id,
            text=text,
            started_at_monotonic_ms=started_at_monotonic_ms,
            audio_sec=audio_sec,
            partial_index=partial_index,
            provider_confidence=provider_confidence,
            provider_avg_logprob=provider_avg_logprob,
            language=language,
            provider_event_type=provider_event_type,
            raw_meta=raw_meta or {},
        )
        await self.emit_event(event)
        return event

    async def emit_final(
        self,
        *,
        segment_id: str,
        text: str,
        started_at_monotonic_ms: int,
        audio_sec: float,
        provider_confidence: float | None = None,
        provider_avg_logprob: float | None = None,
        language: str | None = None,
        cost_estimate_usd: float | None = None,
        provider_event_type: str | None = None,
        raw_meta: dict | None = None,
    ) -> TranscriptEvent:
        event = self._build_event(
            status=EventStatus.FINAL,
            segment_id=segment_id,
            text=text,
            started_at_monotonic_ms=started_at_monotonic_ms,
            audio_sec=audio_sec,
            provider_confidence=provider_confidence,
            provider_avg_logprob=provider_avg_logprob,
            language=language,
            cost_estimate_usd=cost_estimate_usd,
            provider_event_type=provider_event_type,
            raw_meta=raw_meta or {},
        )
        await self.emit_event(event)
        return event

    async def emit_error(
        self,
        *,
        segment_id: str,
        text: str,
        started_at_monotonic_ms: int,
        raw_meta: dict | None = None,
    ) -> TranscriptEvent:
        self._error_count += 1
        event = self._build_event(
            status=EventStatus.ERROR,
            segment_id=segment_id,
            text=text,
            started_at_monotonic_ms=started_at_monotonic_ms,
            audio_sec=0.0,
            raw_meta=raw_meta or {},
        )
        await self.emit_event(event)
        return event

    def increment_restart(self) -> None:
        self._restart_count += 1

    def increment_dropped_chunks(self, count: int = 1) -> None:
        self._dropped_chunks += count

    def _build_event(
        self,
        *,
        status: EventStatus,
        segment_id: str,
        text: str,
        started_at_monotonic_ms: int,
        audio_sec: float,
        partial_index: int | None = None,
        provider_confidence: float | None = None,
        provider_avg_logprob: float | None = None,
        language: str | None = None,
        cost_estimate_usd: float | None = None,
        provider_event_type: str | None = None,
        raw_meta: dict | None = None,
    ) -> TranscriptEvent:
        if self.config is None:
            raise RuntimeError(f"{self.name}: backend config is not set")

        emitted_at_monotonic_ms = self.now_monotonic_ms()
        processing_sec = max(0.0, (emitted_at_monotonic_ms - started_at_monotonic_ms) / 1000.0)
        rtf = processing_sec / audio_sec if audio_sec > 0 else 0.0

        return TranscriptEvent(
            run_id=self.config.run_id,
            backend=self.config.backend,
            session_id=self.session_id,
            segment_id=segment_id,
            status=status,
            text=text,
            started_at_iso=utc_now_iso(),
            emitted_at_iso=utc_now_iso(),
            started_at_monotonic_ms=started_at_monotonic_ms,
            emitted_at_monotonic_ms=emitted_at_monotonic_ms,
            latency_ms=max(0, emitted_at_monotonic_ms - started_at_monotonic_ms),
            audio_sec=audio_sec,
            rtf=rtf,
            provider_confidence=provider_confidence,
            provider_avg_logprob=provider_avg_logprob,
            language=language,
            partial_index=partial_index,
            restart_count=self._restart_count,
            error_count=self._error_count,
            dropped_chunks=self._dropped_chunks,
            cost_estimate_usd=cost_estimate_usd,
            provider_event_type=provider_event_type,
            raw_meta=raw_meta or {},
        )

    @staticmethod
    def now_monotonic_ms() -> int:
        return int(time.monotonic() * 1000)

    @staticmethod
    def make_segment_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def pcm_duration_sec(
        pcm_chunk: bytes,
        *,
        sample_rate_hz: int = 16000,
        sample_width_bytes: int = 2,
        channels: int = 1,
    ) -> float:
        bytes_per_second = sample_rate_hz * sample_width_bytes * channels
        if bytes_per_second <= 0:
            return 0.0
        return len(pcm_chunk) / bytes_per_second
