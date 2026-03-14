from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any
from urllib.parse import urlencode

import websockets

from backends.base import BaseSTTBackend
from common.types import BackendCapabilities, RunConfig


class ElevenLabsRealtimeSTTBackend(BaseSTTBackend):
    name = "elevenlabs"

    def __init__(self) -> None:
        super().__init__()
        self._ws = None
        self._recv_task: asyncio.Task | None = None
        self._session_started = asyncio.Event()
        self._api_key: str | None = None
        self._sample_rate: int = 16000
        self._commit_strategy: str = "manual"
        self._stream_started_ms: int | None = None
        self._current_segment_id: str | None = None
        self._current_segment_started_ms: int | None = None
        self._current_segment_audio_sec: float = 0.0
        self._current_partial_index: int = 0
        self._current_partial_text: str = ""
        self._sent_first_chunk = False
        self._previous_text_sent = False

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.name,
            supports_partial=True,
            supports_final=True,
            supports_word_timestamps=True,
            supports_speaker_diarization=False,
            supports_confidence=False,
            supports_language_detection=True,
            supports_vad_server_side=True,
            supports_manual_commit=True,
            requires_stream_reconnect=False,
            notes="ElevenLabs Realtime STT via WebSocket",
        )

    async def _on_start(self) -> None:
        if self.config is None:
            raise RuntimeError("ElevenLabsRealtimeSTTBackend: config is not set")
        self._stream_started_ms = self.now_monotonic_ms()
        self._api_key = self.config.extra.get("api_key") or os.environ.get("ELEVENLABS_API_KEY")
        if not self._api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")
        if self.config.channels != 1:
            raise ValueError(f"Expected mono input, got channels={self.config.channels}")

        self._sample_rate = int(self.config.sample_rate_hz)
        self._commit_strategy = str(self.config.extra.get("commit_strategy", "manual")).strip().lower()
        if self._commit_strategy not in {"manual", "vad"}:
            raise ValueError("commit_strategy must be 'manual' or 'vad'")

        url = self._build_ws_url(self.config)
        headers = {"xi-api-key": self._api_key}
        self._ws = await websockets.connect(
            url,
            additional_headers=headers,
            max_size=8 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        )
        self._recv_task = asyncio.create_task(self._recv_loop())

        try:
            await asyncio.wait_for(self._session_started.wait(), timeout=10)
        except asyncio.TimeoutError:
            await self.emit_error(
                segment_id=f"{self.session_id}_session_start_timeout",
                text="elevenlabs realtime: timed out waiting for session_started",
                started_at_monotonic_ms=self._stream_started_ms or self.now_monotonic_ms(),
                raw_meta={"stage": "start"},
            )

    async def feed_audio(self, pcm_chunk: bytes) -> None:
        if not self._started or self.config is None:
            raise RuntimeError("ElevenLabsRealtimeSTTBackend: backend is not started")
        if self._ws is None:
            raise RuntimeError("ElevenLabsRealtimeSTTBackend: websocket is not connected")
        if not pcm_chunk:
            return

        if self._current_segment_id is None:
            self._start_new_segment()
        self._current_segment_audio_sec += self.pcm_duration_sec(
            pcm_chunk,
            sample_rate_hz=self.config.sample_rate_hz,
            sample_width_bytes=2,
            channels=self.config.channels,
        )

        payload: dict[str, Any] = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(pcm_chunk).decode("ascii"),
            "sample_rate": self._sample_rate,
        }
        previous_text = self.config.extra.get("previous_text")
        if previous_text and not self._previous_text_sent and not self._sent_first_chunk:
            payload["previous_text"] = str(previous_text)
            self._previous_text_sent = True
        await self._send_json(payload)
        self._sent_first_chunk = True

    async def _on_finish_audio(self) -> None:
        if self._ws is None:
            return
        if self._commit_strategy == "manual":
            await self._send_json(
                {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": "",
                    "sample_rate": self._sample_rate,
                    "commit": True,
                }
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
        if self._ws is not None:
            try:
                await self._ws.close()
            finally:
                self._ws = None

    def _build_ws_url(self, config: RunConfig) -> str:
        extra = config.extra
        params: dict[str, Any] = {
            "model_id": extra.get("model_id", "scribe_v2_realtime"),
            "audio_format": extra.get("audio_format", f"pcm_{config.sample_rate_hz}"),
            "language_code": extra.get("language_code", config.language),
            "commit_strategy": extra.get("commit_strategy", "manual"),
            "include_timestamps": str(bool(extra.get("include_timestamps", True))).lower(),
            "include_language_detection": str(bool(extra.get("include_language_detection", True))).lower(),
            "enable_logging": str(bool(extra.get("enable_logging", True))).lower(),
        }
        if str(params["commit_strategy"]).lower() == "vad":
            params["vad_silence_threshold_secs"] = float(extra.get("vad_silence_threshold_secs", 1.5))
            params["vad_threshold"] = float(extra.get("vad_threshold", 0.4))
            params["min_speech_duration_ms"] = int(extra.get("min_speech_duration_ms", 100))
            params["min_silence_duration_ms"] = int(extra.get("min_silence_duration_ms", 100))
        return "wss://api.elevenlabs.io/v1/speech-to-text/realtime?" + urlencode(params)

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("ElevenLabsRealtimeSTTBackend: websocket is not connected")
        await self._ws.send(json.dumps(payload, ensure_ascii=False))

    async def _recv_loop(self) -> None:
        if self._ws is None:
            return
        try:
            async for raw_message in self._ws:
                try:
                    event = json.loads(raw_message)
                except json.JSONDecodeError:
                    await self.emit_error(
                        segment_id=f"{self.session_id}_invalid_json",
                        text="elevenlabs realtime: invalid JSON event",
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
                text=f"elevenlabs realtime receive loop failed: {e}",
                started_at_monotonic_ms=self._stream_started_ms or self.now_monotonic_ms(),
                raw_meta={"exception_type": type(e).__name__},
            )

    async def _handle_server_event(self, event: dict[str, Any]) -> None:
        message_type = str(event.get("message_type", "")).strip()
        if message_type == "session_started":
            self._session_started.set()
            return
        if message_type == "partial_transcript":
            text = str(event.get("text", "") or "").strip()
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
                language=event.get("language_code"),
                provider_event_type=message_type,
                raw_meta=event,
            )
            return
        if message_type == "committed_transcript":
            text = str(event.get("text", "") or "").strip()
            if not text:
                return
            self._ensure_segment_started()
            await self.emit_final(
                segment_id=self._current_segment_id or self.make_segment_id(),
                text=text,
                started_at_monotonic_ms=self._current_segment_started_ms or self.now_monotonic_ms(),
                audio_sec=self._current_segment_audio_sec,
                language=event.get("language_code"),
                cost_estimate_usd=self._estimate_cost_usd(self._current_segment_audio_sec),
                provider_event_type=message_type,
                raw_meta=event,
            )
            self._reset_segment()
            return
        if message_type == "committed_transcript_with_timestamps":
            text = str(event.get("text", "") or "").strip()
            if not text:
                return
            self._ensure_segment_started()
            await self.emit_final(
                segment_id=self._current_segment_id or self.make_segment_id(),
                text=text,
                started_at_monotonic_ms=self._current_segment_started_ms or self.now_monotonic_ms(),
                audio_sec=self._current_segment_audio_sec,
                language=event.get("language_code"),
                cost_estimate_usd=self._estimate_cost_usd(self._current_segment_audio_sec),
                provider_event_type=message_type,
                raw_meta=event,
            )
            self._reset_segment()
            return
        if "error" in event or message_type.endswith("_error") or message_type == "error":
            msg = event.get("message") or event.get("error") or event.get("detail") or "unknown elevenlabs realtime error"
            await self.emit_error(
                segment_id=self._current_segment_id or f"{self.session_id}_error",
                text=f"elevenlabs transcription failed: {msg}",
                started_at_monotonic_ms=self._current_segment_started_ms or self._stream_started_ms or self.now_monotonic_ms(),
                raw_meta=event,
            )
            return

    def _start_new_segment(self) -> None:
        self._current_segment_id = self.make_segment_id()
        self._current_segment_started_ms = self.now_monotonic_ms()
        self._current_segment_audio_sec = 0.0
        self._current_partial_index = 0
        self._current_partial_text = ""

    def _ensure_segment_started(self) -> None:
        if self._current_segment_id is None:
            self._start_new_segment()

    def _reset_segment(self) -> None:
        self._current_segment_id = None
        self._current_segment_started_ms = None
        self._current_segment_audio_sec = 0.0
        self._current_partial_index = 0
        self._current_partial_text = ""

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
