from __future__ import annotations

import asyncio
import os
from typing import Any

from backends.base import BaseSTTBackend
from common.types import BackendCapabilities, RunConfig


class GoogleGeminiLiveSTTBackend(BaseSTTBackend):
    name = "google"

    def __init__(self) -> None:
        super().__init__()
        self._client = None
        self._types = None
        self._session_cm = None
        self._session = None
        self._recv_task: asyncio.Task | None = None
        self._setup_complete = asyncio.Event()
        self._stream_started_ms: int | None = None
        self._auto_activity_detection = True
        self._activity_open = False
        self._total_audio_sent_sec = 0.0
        self._current_segment_id: str | None = None
        self._current_segment_started_ms: int | None = None
        self._current_segment_audio_start_sec = 0.0
        self._current_partial_index = 0
        self._current_partial_text = ""

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
            notes="Google Gemini Live API input-audio transcription backend",
        )

    async def _on_start(self) -> None:
        if self.config is None:
            raise RuntimeError("GoogleGeminiLiveSTTBackend: config is not set")
        if self.config.channels != 1:
            raise ValueError(f"Expected mono input, got channels={self.config.channels}")

        self._stream_started_ms = self.now_monotonic_ms()
        self._setup_complete.clear()

        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise RuntimeError(
                "google-genai is not installed. Install it with: pip install google-genai"
            ) from e

        api_key = (
            self.config.extra.get("api_key")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            raise RuntimeError("Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set")

        self._types = types
        self._auto_activity_detection = bool(self.config.extra.get("turn_detection_enabled", True))
        if not self._auto_activity_detection:
            raise RuntimeError(
                "Google Gemini Live backend currently requires turn detection enabled; "
                "manual VAD is not supported by the installed google-genai Gemini API client."
            )
        api_version = str(self.config.extra.get("api_version", "v1beta"))
        model = str(
            self.config.extra.get(
                "model",
                "models/gemini-2.5-flash-native-audio-preview-12-2025",
            )
        )

        self._client = genai.Client(
            api_key=api_key,
            http_options={"api_version": api_version},
        )
        self._session_cm = self._client.aio.live.connect(
            model=model,
            config=self._build_connect_config(self.config),
        )
        self._session = await self._session_cm.__aenter__()
        self._recv_task = asyncio.create_task(self._recv_loop())

        try:
            await asyncio.wait_for(self._setup_complete.wait(), timeout=10)
        except asyncio.TimeoutError:
            await self.emit_error(
                segment_id=f"{self.session_id}_setup_timeout",
                text="google live: timed out waiting for setup_complete",
                started_at_monotonic_ms=self._stream_started_ms or self.now_monotonic_ms(),
                raw_meta={"stage": "start"},
            )

    async def feed_audio(self, pcm_chunk: bytes) -> None:
        if not self._started or self.config is None:
            raise RuntimeError("GoogleGeminiLiveSTTBackend: backend is not started")
        if self._session is None or self._types is None:
            raise RuntimeError("GoogleGeminiLiveSTTBackend: session is not connected")
        if not pcm_chunk:
            return

        if self._current_segment_id is None:
            self._start_new_segment()

        if not self._auto_activity_detection and not self._activity_open:
            await self._session.send_realtime_input(activity_start=self._types.ActivityStart())
            self._activity_open = True

        self._total_audio_sent_sec += self.pcm_duration_sec(
            pcm_chunk,
            sample_rate_hz=self.config.sample_rate_hz,
            sample_width_bytes=2,
            channels=self.config.channels,
        )

        await self._session.send_realtime_input(
            audio=self._types.Blob(
                data=bytes(pcm_chunk),
                mime_type=f"audio/pcm;rate={self.config.sample_rate_hz}",
            )
        )

    async def _on_finish_audio(self) -> None:
        if self._session is None or self._types is None:
            return
        if not self._auto_activity_detection and self._activity_open:
            await self._session.send_realtime_input(activity_end=self._types.ActivityEnd())
            self._activity_open = False
        await self._session.send_realtime_input(audio_stream_end=True)

    async def _on_stop(self) -> None:
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            finally:
                self._recv_task = None

        await self._flush_current_segment(reason="google_stop_flush")

        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            finally:
                self._session_cm = None
                self._session = None

    def _build_connect_config(self, config: RunConfig):
        if self._types is None:
            raise RuntimeError("GoogleGeminiLiveSTTBackend: types module is not initialized")

        types = self._types
        turn_detection_enabled = bool(config.extra.get("turn_detection_enabled", True))

        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=not turn_detection_enabled,
                    prefix_padding_ms=int(config.extra.get("prefix_padding_ms", 200)),
                    silence_duration_ms=int(config.extra.get("silence_duration_ms", 500)),
                )
            ),
        )

    async def _recv_loop(self) -> None:
        if self._session is None:
            return
        try:
            while True:
                async for message in self._session.receive():
                    await self._handle_server_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self.emit_error(
                segment_id=f"{self.session_id}_recv_loop_error",
                text=f"google live receive loop failed: {e}",
                started_at_monotonic_ms=self._stream_started_ms or self.now_monotonic_ms(),
                raw_meta={"exception_type": type(e).__name__},
            )

    async def _handle_server_message(self, message: Any) -> None:
        setup_complete = getattr(message, "setup_complete", None)
        if setup_complete is not None:
            session_id = getattr(setup_complete, "session_id", None)
            if session_id:
                self.session_id = session_id
            self._setup_complete.set()

        server_content = getattr(message, "server_content", None)
        if server_content is None:
            return

        transcription = getattr(server_content, "input_transcription", None)
        if transcription is not None:
            await self._handle_transcription(
                text=(getattr(transcription, "text", None) or "").strip(),
                finished=bool(getattr(transcription, "finished", False)),
                provider_event_type="input_transcription",
                raw_meta={
                    "turn_complete": bool(getattr(server_content, "turn_complete", False)),
                    "waiting_for_input": bool(getattr(server_content, "waiting_for_input", False)),
                },
            )

        if transcription is None:
            model_turn_text = self._extract_model_turn_text(server_content)
            if model_turn_text:
                await self._handle_transcription(
                    text=model_turn_text,
                    finished=bool(getattr(server_content, "turn_complete", False)),
                    provider_event_type="model_turn_text",
                    raw_meta={
                        "turn_complete": bool(getattr(server_content, "turn_complete", False)),
                    },
                )

        if bool(getattr(server_content, "turn_complete", False)):
            await self._flush_current_segment(reason="google_turn_complete_flush")

    async def _handle_transcription(
        self,
        *,
        text: str,
        finished: bool,
        provider_event_type: str,
        raw_meta: dict[str, Any],
    ) -> None:
        if not text:
            if finished:
                await self._flush_current_segment(reason=f"{provider_event_type}_empty_finish")
            return

        self._ensure_segment_started()
        assert self._current_segment_id is not None
        assert self._current_segment_started_ms is not None

        prev_text = self._current_partial_text
        self._current_partial_text = text
        audio_sec = self._current_segment_audio_sec()

        if finished:
            await self.emit_final(
                segment_id=self._current_segment_id,
                text=text,
                started_at_monotonic_ms=self._current_segment_started_ms,
                audio_sec=audio_sec,
                language=self.config.language if self.config else None,
                cost_estimate_usd=self._estimate_cost_usd(audio_sec),
                provider_event_type=provider_event_type,
                raw_meta=raw_meta,
            )
            self._reset_segment()
            return

        if text == prev_text:
            return

        self._current_partial_index += 1
        await self.emit_partial(
            segment_id=self._current_segment_id,
            text=text,
            started_at_monotonic_ms=self._current_segment_started_ms,
            audio_sec=audio_sec,
            partial_index=self._current_partial_index,
            language=self.config.language if self.config else None,
            provider_event_type=provider_event_type,
            raw_meta=raw_meta,
        )

    async def _flush_current_segment(self, *, reason: str) -> None:
        if not self._current_segment_id or not self._current_partial_text:
            self._reset_segment()
            return
        assert self._current_segment_started_ms is not None
        audio_sec = self._current_segment_audio_sec()
        await self.emit_final(
            segment_id=self._current_segment_id,
            text=self._current_partial_text,
            started_at_monotonic_ms=self._current_segment_started_ms,
            audio_sec=audio_sec,
            language=self.config.language if self.config else None,
            cost_estimate_usd=self._estimate_cost_usd(audio_sec),
            provider_event_type=reason,
            raw_meta={"flushed": True, "reason": reason},
        )
        self._reset_segment()

    def _start_new_segment(self) -> None:
        self._current_segment_id = self.make_segment_id()
        self._current_segment_started_ms = self.now_monotonic_ms()
        self._current_segment_audio_start_sec = self._total_audio_sent_sec
        self._current_partial_index = 0
        self._current_partial_text = ""

    def _ensure_segment_started(self) -> None:
        if self._current_segment_id is None or self._current_segment_started_ms is None:
            self._start_new_segment()

    def _reset_segment(self) -> None:
        self._current_segment_id = None
        self._current_segment_started_ms = None
        self._current_segment_audio_start_sec = self._total_audio_sent_sec
        self._current_partial_index = 0
        self._current_partial_text = ""

    def _current_segment_audio_sec(self) -> float:
        return max(0.0, self._total_audio_sent_sec - self._current_segment_audio_start_sec)

    def _estimate_cost_usd(self, audio_sec: float) -> float | None:
        if self.config is None:
            return None
        pricing_per_min_usd = self.config.extra.get("pricing_per_min_usd")
        if pricing_per_min_usd is None:
            return None
        return max(0.0, float(pricing_per_min_usd) * (audio_sec / 60.0))

    @staticmethod
    def _extract_model_turn_text(server_content: Any) -> str:
        model_turn = getattr(server_content, "model_turn", None)
        parts = getattr(model_turn, "parts", None) or []
        texts = [str(getattr(part, "text", "") or "").strip() for part in parts]
        return "\n".join(text for text in texts if text).strip()
