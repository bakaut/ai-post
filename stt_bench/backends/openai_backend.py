from __future__ import annotations

import asyncio
import base64
import json
import os
from array import array
from typing import Any
from urllib.parse import quote_plus

import websockets

from backends.base import BaseSTTBackend
from common.types import BackendCapabilities, RunConfig


class OpenAIRealtimeSTTBackend(BaseSTTBackend):
    name = "openai"

    def __init__(self) -> None:
        super().__init__()
        self._ws = None
        self._recv_task: asyncio.Task | None = None
        self._session_updated = asyncio.Event()
        self._api_key: str | None = None
        self._ratecv_state: tuple[int, float] | None = None
        self._item_started_ms: dict[str, int] = {}
        self._item_audio_sec: dict[str, float] = {}
        self._item_partial_text: dict[str, str] = {}
        self._item_partial_index: dict[str, int] = {}
        self._item_done_events: dict[str, asyncio.Event] = {}
        self._session_started_ms: int | None = None
        self._include_logprobs = False

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.name,
            supports_partial=True,
            supports_final=True,
            supports_word_timestamps=False,
            supports_speaker_diarization=False,
            supports_confidence=False,
            supports_language_detection=False,
            supports_vad_server_side=True,
            supports_manual_commit=True,
            requires_stream_reconnect=False,
            notes="OpenAI Realtime WebSocket STT backend",
        )

    async def _on_start(self) -> None:
        if self.config is None:
            raise RuntimeError("OpenAIRealtimeSTTBackend: config is not set")
        self._session_started_ms = self.now_monotonic_ms()
        self._api_key = self.config.extra.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        realtime_model = self.config.extra.get("realtime_model", "gpt-realtime")
        self._include_logprobs = bool(self.config.extra.get("include_logprobs", False))
        url = f"wss://api.openai.com/v1/realtime?model={quote_plus(str(realtime_model))}"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        self._ws = await websockets.connect(
            url,
            additional_headers=headers,
            max_size=8 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        )

        self._recv_task = asyncio.create_task(self._recv_loop())
        await self._send_json(self._build_session_update_event(self.config))

        try:
            await asyncio.wait_for(self._session_updated.wait(), timeout=10)
        except asyncio.TimeoutError:
            await self.emit_error(
                segment_id=f"{self.session_id}_session_update_timeout",
                text="openai realtime: timed out waiting for session.updated",
                started_at_monotonic_ms=self._session_started_ms or self.now_monotonic_ms(),
                raw_meta={"stage": "start"},
            )

    async def feed_audio(self, pcm_chunk: bytes) -> None:
        if not self._started or self.config is None:
            raise RuntimeError("OpenAIRealtimeSTTBackend: backend is not started")
        if self._ws is None:
            raise RuntimeError("OpenAIRealtimeSTTBackend: websocket is not connected")
        if not pcm_chunk:
            return
        if self.config.channels != 1:
            raise ValueError(f"Expected mono input, got channels={self.config.channels}")

        pcm24 = self._normalize_input_pcm_to_24k_mono_le(pcm_chunk, self.config)
        payload = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm24).decode("ascii"),
        }
        await self._send_json(payload)

    async def _on_finish_audio(self) -> None:
        if self._ws is not None:
            await self._send_json({"type": "input_audio_buffer.commit"})
            await self._wait_for_pending_items()

    async def _on_stop(self) -> None:
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            finally:
                self._recv_task = None
        await self._flush_pending_items_as_finals(reason="stop")
        if self._ws is not None:
            try:
                await self._ws.close()
            finally:
                self._ws = None

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("OpenAIRealtimeSTTBackend: websocket is not connected")
        await self._ws.send(json.dumps(payload, ensure_ascii=False))

    def _build_session_update_event(self, config: RunConfig) -> dict[str, Any]:
        extra = config.extra
        transcription_model = extra.get("transcription_model", "gpt-4o-mini-transcribe")
        transcription_prompt = extra.get("transcription_prompt")
        noise_reduction_type = extra.get("noise_reduction_type", "near_field")
        turn_detection_enabled = extra.get("turn_detection_enabled", True)
        silence_duration_ms = int(extra.get("silence_duration_ms", 500))
        prefix_padding_ms = int(extra.get("prefix_padding_ms", 300))
        threshold = float(extra.get("threshold", 0.5))

        session: dict[str, Any] = {
            "type": "realtime",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "noise_reduction": {"type": noise_reduction_type},
                    "transcription": {"model": transcription_model, "language": config.language},
                }
            },
        }
        if transcription_prompt:
            session["audio"]["input"]["transcription"]["prompt"] = transcription_prompt
        if self._include_logprobs:
            session["include"] = ["item.input_audio_transcription.logprobs"]
        if turn_detection_enabled:
            session["audio"]["input"]["turn_detection"] = {
                "type": "server_vad",
                "create_response": False,
                "prefix_padding_ms": prefix_padding_ms,
                "silence_duration_ms": silence_duration_ms,
                "threshold": threshold,
            }
        else:
            session["audio"]["input"]["turn_detection"] = None
        return {"type": "session.update", "session": session}

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
                        text="openai realtime: invalid JSON event",
                        started_at_monotonic_ms=self._session_started_ms or self.now_monotonic_ms(),
                        raw_meta={"raw_message": str(raw_message)[:500]},
                    )
                    continue
                await self._handle_server_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self.emit_error(
                segment_id=f"{self.session_id}_recv_loop_error",
                text=f"openai realtime receive loop failed: {e}",
                started_at_monotonic_ms=self._session_started_ms or self.now_monotonic_ms(),
                raw_meta={"exception_type": type(e).__name__},
            )

    async def _handle_server_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type", "")
        if event_type == "session.updated":
            self._session_updated.set()
            return
        if event_type == "error":
            error = event.get("error", {}) or {}
            await self.emit_error(
                segment_id=event.get("event_id", f"{self.session_id}_server_error"),
                text=f"openai realtime error: {error.get('message', 'unknown error')}",
                started_at_monotonic_ms=self._session_started_ms or self.now_monotonic_ms(),
                raw_meta=event,
            )
            return
        if event_type == "input_audio_buffer.speech_started":
            item_id = event.get("item_id")
            if item_id:
                self._item_started_ms[item_id] = self.now_monotonic_ms()
                self._item_partial_text.setdefault(item_id, "")
                self._item_partial_index.setdefault(item_id, 0)
                self._item_done_events.setdefault(item_id, asyncio.Event())
            return
        if event_type == "input_audio_buffer.speech_stopped":
            item_id = event.get("item_id")
            if item_id:
                start_ms = float(event.get("audio_start_ms", 0.0) or 0.0)
                end_ms = float(event.get("audio_end_ms", 0.0) or 0.0)
                if end_ms >= start_ms:
                    self._item_audio_sec[item_id] = (end_ms - start_ms) / 1000.0
            return
        if event_type == "conversation.item.input_audio_transcription.delta":
            item_id = event.get("item_id")
            delta = event.get("delta", "") or ""
            if not item_id or not delta:
                return
            prev = self._item_partial_text.get(item_id, "")
            full_text = prev + delta
            self._item_partial_text[item_id] = full_text
            self._item_partial_index[item_id] = self._item_partial_index.get(item_id, 0) + 1
            start_ms = self._item_started_ms.get(item_id, self._session_started_ms or self.now_monotonic_ms())
            avg_logprob = self._extract_avg_logprob(event.get("logprobs"))
            await self.emit_partial(
                segment_id=item_id,
                text=full_text.strip(),
                started_at_monotonic_ms=start_ms,
                audio_sec=self._item_audio_sec.get(item_id, 0.0),
                partial_index=self._item_partial_index[item_id],
                provider_avg_logprob=avg_logprob,
                language=self.config.language if self.config else None,
                provider_event_type=event_type,
                raw_meta=event,
            )
            return
        if event_type == "conversation.item.input_audio_transcription.completed":
            item_id = event.get("item_id") or self.make_segment_id()
            transcript = ((event.get("transcript") or "").strip() or self._item_partial_text.get(item_id, "").strip())
            start_ms = self._item_started_ms.get(item_id, self._session_started_ms or self.now_monotonic_ms())
            usage = event.get("usage", {}) or {}
            audio_sec = self._extract_audio_seconds(usage)
            if audio_sec is None:
                audio_sec = self._item_audio_sec.get(item_id, 0.0)
            avg_logprob = self._extract_avg_logprob(event.get("logprobs"))
            cost_estimate_usd = self._estimate_cost_usd(audio_sec)
            await self.emit_final(
                segment_id=item_id,
                text=transcript,
                started_at_monotonic_ms=start_ms,
                audio_sec=audio_sec,
                provider_avg_logprob=avg_logprob,
                language=self.config.language if self.config else None,
                cost_estimate_usd=cost_estimate_usd,
                provider_event_type=event_type,
                raw_meta=event,
            )
            self._mark_item_done(item_id)
            return
        if event_type == "conversation.item.input_audio_transcription.failed":
            item_id = event.get("item_id") or self.make_segment_id()
            err = event.get("error", {}) or {}
            await self.emit_error(
                segment_id=item_id,
                text=f"openai transcription failed: {err.get('message', 'unknown transcription error')}",
                started_at_monotonic_ms=self._item_started_ms.get(item_id, self._session_started_ms or self.now_monotonic_ms()),
                raw_meta=event,
            )
            self._mark_item_done(item_id)
            return

    def _mark_item_done(self, item_id: str) -> None:
        self._item_done_events.setdefault(item_id, asyncio.Event()).set()

    def _pending_item_ids(self) -> list[str]:
        known_item_ids = set(self._item_started_ms) | set(self._item_partial_text) | set(self._item_done_events)
        return [
            item_id
            for item_id in known_item_ids
            if not (self._item_done_events.get(item_id) and self._item_done_events[item_id].is_set())
        ]

    async def _wait_for_pending_items(self) -> None:
        pending = self._pending_item_ids()
        if not pending:
            return

        timeout_sec = 10.0
        if self.config is not None:
            try:
                timeout_sec = float(self.config.extra.get("finalize_timeout_sec", timeout_sec))
            except (TypeError, ValueError):
                timeout_sec = 10.0
        if timeout_sec <= 0:
            return

        wait_tasks = [
            asyncio.create_task(self._item_done_events.setdefault(item_id, asyncio.Event()).wait())
            for item_id in pending
        ]
        try:
            await asyncio.wait_for(asyncio.gather(*wait_tasks), timeout=timeout_sec)
        except asyncio.TimeoutError:
            await self._flush_pending_items_as_finals(reason="finish_audio_timeout")
        finally:
            for task in wait_tasks:
                task.cancel()
            for task in wait_tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _flush_pending_items_as_finals(self, *, reason: str) -> None:
        for item_id in self._pending_item_ids():
            text = self._item_partial_text.get(item_id, "").strip()
            if not text:
                self._mark_item_done(item_id)
                continue

            start_ms = self._item_started_ms.get(item_id, self._session_started_ms or self.now_monotonic_ms())
            audio_sec = self._item_audio_sec.get(item_id, 0.0)
            await self.emit_final(
                segment_id=item_id,
                text=text,
                started_at_monotonic_ms=start_ms,
                audio_sec=audio_sec,
                language=self.config.language if self.config else None,
                cost_estimate_usd=self._estimate_cost_usd(audio_sec),
                provider_event_type="conversation.item.input_audio_transcription.completed_fallback",
                raw_meta={"fallback_reason": reason},
            )
            self._mark_item_done(item_id)

    def _normalize_input_pcm_to_24k_mono_le(self, pcm_chunk: bytes, config: RunConfig) -> bytes:
        if config.channels != 1:
            raise ValueError(f"Expected mono PCM input, got channels={config.channels}")
        if config.sample_rate_hz == 24000:
            return pcm_chunk
        if config.sample_rate_hz <= 0:
            raise ValueError(f"Invalid input sample rate: {config.sample_rate_hz}")
        return self._resample_pcm16_mono(
            pcm_chunk,
            input_rate_hz=config.sample_rate_hz,
            output_rate_hz=24000,
        )

    def _resample_pcm16_mono(self, pcm_chunk: bytes, *, input_rate_hz: int, output_rate_hz: int) -> bytes:
        if len(pcm_chunk) % 2 != 0:
            raise ValueError("Expected PCM16 audio with an even byte length")

        samples = array("h")
        samples.frombytes(pcm_chunk)

        prev_sample, position = self._ratecv_state or (None, 0.0)
        if prev_sample is None:
            extended = [int(sample) for sample in samples]
        else:
            extended = [prev_sample, *[int(sample) for sample in samples]]

        if len(extended) < 2:
            if extended:
                self._ratecv_state = (extended[-1], position)
            return b""

        step = input_rate_hz / output_rate_hz
        max_index = len(extended) - 1
        output = array("h")

        while position < max_index:
            left_index = int(position)
            fraction = position - left_index
            left = extended[left_index]
            right = extended[left_index + 1]
            sample = int(round(left + (right - left) * fraction))
            output.append(max(-32768, min(32767, sample)))
            position += step

        self._ratecv_state = (extended[-1], position - max_index)
        return output.tobytes()

    def _extract_avg_logprob(self, logprobs: Any) -> float | None:
        if not logprobs or not isinstance(logprobs, list):
            return None
        values: list[float] = []
        for item in logprobs:
            if isinstance(item, dict):
                value = item.get("logprob")
                if isinstance(value, (int, float)):
                    values.append(float(value))
        return (sum(values) / len(values)) if values else None

    def _extract_audio_seconds(self, usage: dict[str, Any]) -> float | None:
        if not usage or not isinstance(usage, dict):
            return None
        if usage.get("type") == "duration":
            seconds = usage.get("seconds")
            if isinstance(seconds, (int, float)):
                return float(seconds)
        return None

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
