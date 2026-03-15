from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import websockets

from backends.base import BaseSTTBackend
from common.types import BackendCapabilities, RunConfig


class SpeechmaticsRealtimeSTTBackend(BaseSTTBackend):
    name = "speechmatics"

    def __init__(self) -> None:
        super().__init__()
        self._ws = None
        self._recv_task: asyncio.Task | None = None
        self._recognition_started = asyncio.Event()
        self._transcript_finished = asyncio.Event()
        self._stream_started_ms: int | None = None
        self._current_segment_id: str | None = None
        self._current_segment_started_ms: int | None = None
        self._current_segment_audio_sec: float = 0.0
        self._current_partial_index: int = 0
        self._current_partial_text: str = ""

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.name,
            supports_partial=True,
            supports_final=True,
            supports_word_timestamps=True,
            supports_speaker_diarization=False,
            supports_confidence=True,
            supports_language_detection=False,
            supports_vad_server_side=False,
            supports_manual_commit=True,
            requires_stream_reconnect=False,
            notes="Speechmatics realtime transcription via WebSocket",
        )

    async def _on_start(self) -> None:
        if self.config is None:
            raise RuntimeError("SpeechmaticsRealtimeSTTBackend: config is not set")
        if self.config.channels != 1:
            raise ValueError(f"Expected mono input, got channels={self.config.channels}")

        self._stream_started_ms = self.now_monotonic_ms()
        auth_token = (
            self.config.extra.get("auth_token")
            or self.config.extra.get("api_key")
            or os.environ.get("SPEECHMATICS_AUTH_TOKEN")
            or os.environ.get("SPEECHMATICS_API_KEY")
        )
        if not auth_token:
            raise RuntimeError("Set SPEECHMATICS_AUTH_TOKEN or SPEECHMATICS_API_KEY")

        url = str(
            self.config.extra.get("url")
            or os.environ.get("SPEECHMATICS_RT_URL")
            or "wss://eu2.rt.speechmatics.com/v2"
        )

        self._recognition_started.clear()
        self._transcript_finished.clear()
        self._ws = await websockets.connect(
            url,
            additional_headers={"Authorization": f"Bearer {auth_token}"},
            max_size=8 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        )
        self._recv_task = asyncio.create_task(self._recv_loop())
        await self._send_json(self._build_start_recognition_message(self.config))

        try:
            await asyncio.wait_for(self._recognition_started.wait(), timeout=10)
        except asyncio.TimeoutError:
            await self.emit_error(
                segment_id=f"{self.session_id}_recognition_start_timeout",
                text="speechmatics realtime: timed out waiting for RecognitionStarted",
                started_at_monotonic_ms=self._stream_started_ms or self.now_monotonic_ms(),
                raw_meta={"stage": "start"},
            )

    async def feed_audio(self, pcm_chunk: bytes) -> None:
        if not self._started or self.config is None:
            raise RuntimeError("SpeechmaticsRealtimeSTTBackend: backend is not started")
        if self._ws is None:
            raise RuntimeError("SpeechmaticsRealtimeSTTBackend: websocket is not connected")
        if not pcm_chunk:
            return

        self._ensure_segment_started()
        self._current_segment_audio_sec += self.pcm_duration_sec(
            pcm_chunk,
            sample_rate_hz=self.config.sample_rate_hz,
            sample_width_bytes=2,
            channels=self.config.channels,
        )
        await self._ws.send(bytes(pcm_chunk))

    async def _on_finish_audio(self) -> None:
        if self._ws is None:
            return
        await self._send_json({"message": "EndOfStream"})
        try:
            await asyncio.wait_for(
                self._transcript_finished.wait(),
                timeout=float(self.config.extra.get("finish_timeout_sec", 15.0)) if self.config else 15.0,
            )
        except asyncio.TimeoutError:
            await self.emit_error(
                segment_id=f"{self.session_id}_end_of_transcript_timeout",
                text="speechmatics realtime: timed out waiting for EndOfTranscript",
                started_at_monotonic_ms=self._stream_started_ms or self.now_monotonic_ms(),
                raw_meta={"stage": "finish_audio"},
            )

    async def _on_stop(self) -> None:
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            finally:
                self._recv_task = None

        await self._flush_current_segment(reason="speechmatics_stop_flush")

        if self._ws is not None:
            try:
                await self._ws.close()
            finally:
                self._ws = None

    def _build_start_recognition_message(self, config: RunConfig) -> dict[str, Any]:
        extra = config.extra
        return {
            "message": "StartRecognition",
            "audio_format": {
                "type": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": int(config.sample_rate_hz),
            },
            "transcription_config": {
                "language": extra.get("language", config.language),
                "operating_point": extra.get("operating_point", "enhanced"),
                "enable_partials": bool(extra.get("enable_partials", True)),
                "max_delay": float(extra.get("max_delay", 1.0)),
                "enable_entities": bool(extra.get("enable_entities", False)),
                "diarization": str(extra.get("diarization", "none")),
            },
        }

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("SpeechmaticsRealtimeSTTBackend: websocket is not connected")
        await self._ws.send(json.dumps(payload, ensure_ascii=False))

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
                        text="speechmatics realtime: invalid JSON event",
                        started_at_monotonic_ms=self._stream_started_ms or self.now_monotonic_ms(),
                        raw_meta={"raw_message": str(raw_message)[:500]},
                    )
                    continue
                await self._handle_server_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self.emit_error(
                segment_id=f"{self.session_id}_recv_loop_error",
                text=f"speechmatics realtime receive loop failed: {e}",
                started_at_monotonic_ms=self._stream_started_ms or self.now_monotonic_ms(),
                raw_meta={"exception_type": type(e).__name__},
            )

    async def _handle_server_event(self, event: dict[str, Any]) -> None:
        message_type = str(event.get("message", "")).strip()
        if message_type == "RecognitionStarted":
            session_id = event.get("id")
            if isinstance(session_id, str) and session_id:
                self.session_id = session_id
            self._recognition_started.set()
            return
        if message_type == "AddPartialTranscript":
            text = self._extract_transcript_text(event)
            if not text:
                return
            self._ensure_segment_started()
            self._current_partial_index += 1
            self._current_partial_text = text
            await self.emit_partial(
                segment_id=self._current_segment_id or self.make_segment_id(),
                text=text,
                started_at_monotonic_ms=self._current_segment_started_ms or self.now_monotonic_ms(),
                audio_sec=self._current_segment_audio_sec,
                partial_index=self._current_partial_index,
                provider_confidence=self._extract_avg_confidence(event),
                language=self._configured_language(),
                provider_event_type=message_type,
                raw_meta=event,
            )
            return
        if message_type == "AddTranscript":
            text = self._extract_transcript_text(event)
            if not text:
                return
            self._ensure_segment_started()
            await self.emit_final(
                segment_id=self._current_segment_id or self.make_segment_id(),
                text=text,
                started_at_monotonic_ms=self._current_segment_started_ms or self.now_monotonic_ms(),
                audio_sec=self._current_segment_audio_sec,
                provider_confidence=self._extract_avg_confidence(event),
                language=self._configured_language(),
                cost_estimate_usd=self._estimate_cost_usd(self._current_segment_audio_sec),
                provider_event_type=message_type,
                raw_meta=event,
            )
            self._reset_segment()
            return
        if message_type == "EndOfTranscript":
            self._transcript_finished.set()
            await self._flush_current_segment(reason="speechmatics_end_of_transcript_flush")
            return
        if message_type == "Error":
            detail = event.get("reason") or event.get("error") or event.get("detail") or "unknown speechmatics realtime error"
            await self.emit_error(
                segment_id=self._current_segment_id or f"{self.session_id}_error",
                text=f"speechmatics transcription failed: {detail}",
                started_at_monotonic_ms=self._current_segment_started_ms or self._stream_started_ms or self.now_monotonic_ms(),
                raw_meta=event,
            )
            return

    async def _flush_current_segment(self, *, reason: str) -> None:
        if not self._current_partial_text:
            self._reset_segment()
            return
        await self.emit_final(
            segment_id=self._current_segment_id or self.make_segment_id(),
            text=self._current_partial_text,
            started_at_monotonic_ms=self._current_segment_started_ms or self.now_monotonic_ms(),
            audio_sec=self._current_segment_audio_sec,
            provider_confidence=None,
            language=self._configured_language(),
            cost_estimate_usd=self._estimate_cost_usd(self._current_segment_audio_sec),
            provider_event_type=reason,
            raw_meta={"flush_reason": reason},
        )
        self._reset_segment()

    def _ensure_segment_started(self) -> None:
        if self._current_segment_id is None:
            self._current_segment_id = self.make_segment_id()
            self._current_segment_started_ms = self.now_monotonic_ms()
            self._current_segment_audio_sec = 0.0
            self._current_partial_index = 0
            self._current_partial_text = ""

    def _reset_segment(self) -> None:
        self._current_segment_id = None
        self._current_segment_started_ms = None
        self._current_segment_audio_sec = 0.0
        self._current_partial_index = 0
        self._current_partial_text = ""

    def _configured_language(self) -> str | None:
        if self.config is None:
            return None
        return str(self.config.extra.get("language", self.config.language))

    def _estimate_cost_usd(self, audio_sec: float) -> float | None:
        if self.config is None:
            return None
        price_per_min = self.config.extra.get("pricing_per_min_usd")
        if price_per_min is None:
            return None
        try:
            price_per_min = float(price_per_min)
        except (TypeError, ValueError):
            return None
        if audio_sec <= 0:
            return 0.0
        return (audio_sec / 60.0) * price_per_min

    def _extract_transcript_text(self, event: dict[str, Any]) -> str:
        metadata = event.get("metadata")
        if isinstance(metadata, dict):
            transcript = metadata.get("transcript")
            if isinstance(transcript, str) and transcript.strip():
                return transcript.strip()

        results = event.get("results")
        if not isinstance(results, list):
            return ""

        parts: list[str] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            alternatives = result.get("alternatives")
            if not isinstance(alternatives, list) or not alternatives:
                continue
            first_alt = alternatives[0]
            if not isinstance(first_alt, dict):
                continue
            content = first_alt.get("content") or first_alt.get("text")
            if not isinstance(content, str) or not content:
                continue
            self._append_result_part(parts, content)
        return " ".join(parts).strip()

    def _append_result_part(self, parts: list[str], content: str) -> None:
        if not parts:
            parts.append(content)
            return
        if re.fullmatch(r"[\]\)\}\.,!?;:%]+", content):
            parts[-1] += content
            return
        if parts[-1] and parts[-1][-1] in "([{":
            parts[-1] += content
            return
        parts.append(content)

    def _extract_avg_confidence(self, event: dict[str, Any]) -> float | None:
        results = event.get("results")
        if not isinstance(results, list):
            return None
        values: list[float] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            alternatives = result.get("alternatives")
            if not isinstance(alternatives, list) or not alternatives:
                continue
            first_alt = alternatives[0]
            if not isinstance(first_alt, dict):
                continue
            confidence = first_alt.get("confidence")
            if isinstance(confidence, (int, float)):
                values.append(float(confidence))
        if not values:
            return None
        return sum(values) / len(values)
