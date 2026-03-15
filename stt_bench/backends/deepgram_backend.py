from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import Any
from urllib.parse import urlencode

import websockets

from backends.base import BaseSTTBackend
from common.types import BackendCapabilities, RunConfig


class DeepgramRealtimeSTTBackend(BaseSTTBackend):
    name = "deepgram"

    def __init__(self) -> None:
        super().__init__()
        self._ws = None
        self._recv_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._stream_started_ms: int | None = None
        self._current_segment_id: str | None = None
        self._current_segment_started_ms: int | None = None
        self._current_partial_index: int = 0
        self._current_partial_text: str = ""
        self._current_partial_audio_sec: float = 0.0

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.name,
            supports_partial=True,
            supports_final=True,
            supports_word_timestamps=True,
            supports_speaker_diarization=True,
            supports_confidence=True,
            supports_language_detection=True,
            supports_vad_server_side=True,
            supports_manual_commit=True,
            requires_stream_reconnect=False,
            notes="Deepgram Live Audio WebSocket backend (nova-3 by default)",
        )

    async def _on_start(self) -> None:
        if self.config is None:
            raise RuntimeError("DeepgramRealtimeSTTBackend: config is not set")
        if self.config.channels != 1:
            raise ValueError(f"Expected mono input, got channels={self.config.channels}")

        api_key = self.config.extra.get("api_key") or os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")

        self._stream_started_ms = self.now_monotonic_ms()
        self._ws = await websockets.connect(
            self._build_ws_url(self.config),
            additional_headers={"Authorization": f"Token {api_key}"},
            max_size=8 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        )
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def feed_audio(self, pcm_chunk: bytes) -> None:
        if not self._started or self.config is None:
            raise RuntimeError("DeepgramRealtimeSTTBackend: backend is not started")
        if self._ws is None:
            raise RuntimeError("DeepgramRealtimeSTTBackend: websocket is not connected")
        if not pcm_chunk:
            return
        await self._ws.send(bytes(pcm_chunk))

    async def _on_finish_audio(self) -> None:
        if self._ws is not None:
            await self._send_json({"type": "Finalize"})

    async def _on_stop(self) -> None:
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
            self._keepalive_task = None

        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._send_json({"type": "CloseStream"})

        if self._recv_task is not None:
            try:
                await asyncio.wait_for(self._recv_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._recv_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._recv_task
            finally:
                self._recv_task = None

        await self._flush_current_segment(reason="deepgram_stop_flush")

        if self._ws is not None:
            try:
                await self._ws.close()
            finally:
                self._ws = None

    def _build_ws_url(self, config: RunConfig) -> str:
        extra = config.extra
        endpointing_ms = extra.get("endpointing_ms", 300)
        utterance_end_ms = extra.get("utterance_end_ms")
        version = extra.get("version")

        params: dict[str, Any] = {
            "model": extra.get("model", "nova-3"),
            "language": extra.get("language", config.language),
            "encoding": extra.get("encoding", "linear16"),
            "sample_rate": int(config.sample_rate_hz),
            "channels": int(config.channels),
            "interim_results": str(bool(extra.get("interim_results", True))).lower(),
            "punctuate": str(bool(extra.get("punctuate", True))).lower(),
            "smart_format": str(bool(extra.get("smart_format", True))).lower(),
            "vad_events": str(bool(extra.get("vad_events", True))).lower(),
            "diarize": str(bool(extra.get("diarize", False))).lower(),
            "profanity_filter": str(bool(extra.get("profanity_filter", False))).lower(),
            "numerals": str(bool(extra.get("numerals", False))).lower(),
        }
        if endpointing_ms is not None:
            params["endpointing"] = int(endpointing_ms)
        if utterance_end_ms is not None:
            params["utterance_end_ms"] = str(utterance_end_ms)
        if version:
            params["version"] = str(version)
        keywords = extra.get("keywords") or []
        if isinstance(keywords, (list, tuple)):
            params["keywords"] = [str(item) for item in keywords if str(item).strip()]
        return f"wss://api.deepgram.com/v1/listen?{urlencode(params, doseq=True)}"

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("DeepgramRealtimeSTTBackend: websocket is not connected")
        await self._ws.send(json.dumps(payload, ensure_ascii=False))

    async def _keepalive_loop(self) -> None:
        interval_sec = float(self.config.extra.get("keepalive_interval_sec", 3.0)) if self.config else 3.0
        try:
            while self._ws is not None:
                await asyncio.sleep(interval_sec)
                if self._ws is None:
                    break
                await self._send_json({"type": "KeepAlive"})
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _recv_loop(self) -> None:
        if self._ws is None:
            return
        try:
            async for raw_message in self._ws:
                if isinstance(raw_message, bytes):
                    continue
                try:
                    event = json.loads(raw_message)
                except json.JSONDecodeError:
                    await self.emit_error(
                        segment_id=f"{self.session_id}_invalid_json",
                        text="deepgram realtime: invalid JSON event",
                        started_at_monotonic_ms=self._stream_started_ms or self.now_monotonic_ms(),
                        raw_meta={"raw_message": str(raw_message)[:500]},
                    )
                    continue
                await self._handle_server_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.emit_error(
                segment_id=f"{self.session_id}_recv_loop_error",
                text=f"deepgram realtime receive loop failed: {exc}",
                started_at_monotonic_ms=self._stream_started_ms or self.now_monotonic_ms(),
                raw_meta={"exception_type": type(exc).__name__},
            )

    async def _handle_server_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", "")).strip()
        if event_type == "Results":
            await self._handle_results_event(event)
            return
        if event_type == "Metadata":
            request_id = event.get("request_id")
            if isinstance(request_id, str) and request_id:
                self.session_id = request_id
            return
        if event_type == "SpeechStarted":
            self._begin_segment(timestamp_sec=self._safe_float(event.get("timestamp")), force_reset=True)
            return
        if event_type == "UtteranceEnd":
            await self._flush_current_segment(reason="deepgram_utterance_end")
            return
        if event_type == "Error":
            await self.emit_error(
                segment_id=self._current_segment_id or f"{self.session_id}_error",
                text=f"deepgram realtime error: {event.get('description') or event.get('message') or 'unknown error'}",
                started_at_monotonic_ms=self._current_segment_started_ms or self._stream_started_ms or self.now_monotonic_ms(),
                raw_meta=event,
            )

    async def _handle_results_event(self, event: dict[str, Any]) -> None:
        channel = event.get("channel") or {}
        alternatives = channel.get("alternatives") or []
        if not alternatives:
            return

        alternative = alternatives[0] or {}
        transcript = str(alternative.get("transcript") or "").strip()
        if not transcript:
            return

        metadata = event.get("metadata") or {}
        request_id = metadata.get("request_id")
        if isinstance(request_id, str) and request_id:
            self.session_id = request_id

        start_sec = self._safe_float(event.get("start"))
        duration_sec = max(0.0, self._safe_float(event.get("duration")))
        self._begin_segment(timestamp_sec=start_sec)

        confidence = self._safe_optional_float(alternative.get("confidence"))
        language = self._extract_language(alternative)
        started_at_monotonic_ms = self._current_segment_started_ms or self._timestamp_to_monotonic_ms(start_sec)
        provider_event_type = self._provider_event_type(event)

        if bool(event.get("is_final")):
            await self.emit_final(
                segment_id=self._current_segment_id or self.make_segment_id(),
                text=transcript,
                started_at_monotonic_ms=started_at_monotonic_ms,
                audio_sec=duration_sec,
                provider_confidence=confidence,
                language=language,
                cost_estimate_usd=self._estimate_cost_usd(duration_sec),
                provider_event_type=provider_event_type,
                raw_meta=event,
            )
            self._reset_segment()
            return

        if transcript == self._current_partial_text:
            return

        self._current_partial_index += 1
        self._current_partial_text = transcript
        self._current_partial_audio_sec = duration_sec
        await self.emit_partial(
            segment_id=self._current_segment_id or self.make_segment_id(),
            text=transcript,
            started_at_monotonic_ms=started_at_monotonic_ms,
            audio_sec=duration_sec,
            partial_index=self._current_partial_index,
            provider_confidence=confidence,
            language=language,
            provider_event_type=provider_event_type,
            raw_meta=event,
        )

    async def _flush_current_segment(self, *, reason: str) -> None:
        if not self._current_partial_text:
            self._reset_segment()
            return
        await self.emit_final(
            segment_id=self._current_segment_id or self.make_segment_id(),
            text=self._current_partial_text,
            started_at_monotonic_ms=self._current_segment_started_ms or self.now_monotonic_ms(),
            audio_sec=self._current_partial_audio_sec,
            language=self.config.language if self.config else None,
            cost_estimate_usd=self._estimate_cost_usd(self._current_partial_audio_sec),
            provider_event_type=reason,
            raw_meta={"reason": reason},
        )
        self._reset_segment()

    def _begin_segment(self, *, timestamp_sec: float | None, force_reset: bool = False) -> None:
        if self._current_segment_id is not None and not force_reset:
            return
        self._current_segment_id = self.make_segment_id()
        self._current_segment_started_ms = self._timestamp_to_monotonic_ms(timestamp_sec)
        self._current_partial_index = 0
        self._current_partial_text = ""
        self._current_partial_audio_sec = 0.0

    def _reset_segment(self) -> None:
        self._current_segment_id = None
        self._current_segment_started_ms = None
        self._current_partial_index = 0
        self._current_partial_text = ""
        self._current_partial_audio_sec = 0.0

    def _timestamp_to_monotonic_ms(self, timestamp_sec: float | None) -> int:
        if self._stream_started_ms is None or timestamp_sec is None:
            return self.now_monotonic_ms()
        return self._stream_started_ms + int(timestamp_sec * 1000)

    def _provider_event_type(self, event: dict[str, Any]) -> str:
        flags: list[str] = ["Results"]
        if bool(event.get("is_final")):
            flags.append("final")
        if bool(event.get("speech_final")):
            flags.append("speech_final")
        if bool(event.get("from_finalize")):
            flags.append("from_finalize")
        return ".".join(flags)

    def _extract_language(self, alternative: dict[str, Any]) -> str | None:
        languages = alternative.get("languages")
        if isinstance(languages, list):
            for item in languages:
                if isinstance(item, str) and item:
                    return item
        return self.config.language if self.config else None

    def _estimate_cost_usd(self, audio_sec: float) -> float | None:
        if self.config is None:
            return None
        pricing = self.config.extra.get("pricing_per_min_usd")
        if pricing is None:
            return None
        try:
            return float(pricing) * (audio_sec / 60.0)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _safe_optional_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
