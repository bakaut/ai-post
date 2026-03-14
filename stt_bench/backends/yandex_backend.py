from __future__ import annotations

import asyncio
import os
import queue
import threading
from typing import Any

import grpc

from backends.base import BaseSTTBackend
from common.types import BackendCapabilities, RunConfig


class YandexStreamingSTTBackend(BaseSTTBackend):
    name = "yandex"

    def __init__(self) -> None:
        super().__init__()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=200)
        self._grpc_thread: threading.Thread | None = None
        self._thread_exc: Exception | None = None
        self._channel = None
        self._stub = None
        self._stt_pb2 = None
        self._stt_service_pb2_grpc = None
        self._stream_started_ms: int | None = None
        self._total_audio_sent_sec: float = 0.0
        self._segment_counter = 0
        self._current_segment_id: str | None = None
        self._current_segment_started_ms: int | None = None
        self._current_segment_audio_start_sec: float = 0.0
        self._last_partial_index: int = 0
        self._pending_final_text: str | None = None

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.name,
            supports_partial=True,
            supports_final=True,
            supports_word_timestamps=False,
            supports_speaker_diarization=False,
            supports_confidence=False,
            supports_language_detection=False,
            supports_vad_server_side=False,
            supports_manual_commit=False,
            requires_stream_reconnect=False,
            notes="Yandex SpeechKit STT API v3 streaming via gRPC",
        )

    async def _on_start(self) -> None:
        if self.config is None:
            raise RuntimeError("YandexStreamingSTTBackend: config is not set")
        self._loop = asyncio.get_running_loop()
        self._stream_started_ms = self.now_monotonic_ms()
        self._import_generated_stubs()
        self._start_grpc_thread()

    async def feed_audio(self, pcm_chunk: bytes) -> None:
        if not self._started or self.config is None:
            raise RuntimeError("YandexStreamingSTTBackend: backend is not started")
        if not isinstance(pcm_chunk, (bytes, bytearray)):
            raise TypeError("YandexStreamingSTTBackend.feed_audio expects bytes")
        if not pcm_chunk:
            return
        if self.config.channels != 1:
            raise ValueError(f"Expected mono input, got channels={self.config.channels}")
        try:
            self._audio_queue.put_nowait(bytes(pcm_chunk))
        except queue.Full:
            self.increment_dropped_chunks(1)
            return
        self._total_audio_sent_sec += self.pcm_duration_sec(
            pcm_chunk,
            sample_rate_hz=self.config.sample_rate_hz,
            sample_width_bytes=2,
            channels=self.config.channels,
        )

    async def _on_finish_audio(self) -> None:
        try:
            self._audio_queue.put_nowait(None)
        except queue.Full:
            self._audio_queue.put(None)

    async def _on_stop(self) -> None:
        try:
            self._audio_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._grpc_thread is not None:
            self._grpc_thread.join(timeout=5)
            self._grpc_thread = None
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:
                pass
            self._channel = None
        if self._thread_exc is not None:
            await self.emit_error(
                segment_id=f"{self.session_id}_grpc_thread_error",
                text=f"yandex grpc thread failed: {self._thread_exc}",
                started_at_monotonic_ms=self._stream_started_ms or self.now_monotonic_ms(),
                raw_meta={"exception_type": type(self._thread_exc).__name__},
            )

    def _import_generated_stubs(self) -> None:
        try:
            import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
            import yandex.cloud.ai.stt.v3.stt_service_pb2_grpc as stt_service_pb2_grpc
        except ImportError as e:
            raise RuntimeError(
                "Yandex SpeechKit gRPC stubs are not available. Expected imports: "
                "yandex.cloud.ai.stt.v3.stt_pb2 and stt_service_pb2_grpc"
            ) from e
        self._stt_pb2 = stt_pb2
        self._stt_service_pb2_grpc = stt_service_pb2_grpc

    def _start_grpc_thread(self) -> None:
        self._grpc_thread = threading.Thread(target=self._grpc_worker, name="yandex-stt-grpc-thread", daemon=True)
        self._grpc_thread.start()

    def _grpc_worker(self) -> None:
        try:
            if self.config is None or self._stt_pb2 is None or self._stt_service_pb2_grpc is None:
                raise RuntimeError("Yandex backend is not initialized")
            endpoint = self.config.extra.get("endpoint") or os.environ.get("YANDEX_STT_ENDPOINT")
            if not endpoint:
                raise RuntimeError("Yandex STT endpoint is not set")
            auth_header = self._build_auth_header(self.config)
            cred = grpc.ssl_channel_credentials()
            self._channel = grpc.secure_channel(endpoint, cred)
            self._stub = self._stt_service_pb2_grpc.RecognizerStub(self._channel)
            response_iter = self._stub.RecognizeStreaming(
                self._request_generator(), metadata=(("authorization", auth_header),)
            )
            for response in response_iter:
                self._handle_response_sync(response)
            self._flush_pending_final_sync(reason="stream_end")
        except Exception as e:
            self._thread_exc = e
            if self._loop is not None:
                fut = asyncio.run_coroutine_threadsafe(
                    self.emit_error(
                        segment_id=f"{self.session_id}_grpc_error",
                        text=f"yandex grpc failed: {e}",
                        started_at_monotonic_ms=self._stream_started_ms or self.now_monotonic_ms(),
                        raw_meta={"exception_type": type(e).__name__},
                    ),
                    self._loop,
                )
                try:
                    fut.result(timeout=5)
                except Exception:
                    pass

    def _request_generator(self):
        if self.config is None or self._stt_pb2 is None:
            raise RuntimeError("Yandex backend is not initialized")
        yield self._build_initial_request(self.config)
        while True:
            chunk = self._audio_queue.get()
            if chunk is None:
                break
            yield self._stt_pb2.StreamingRequest(chunk=self._stt_pb2.AudioChunk(data=chunk))

    def _build_initial_request(self, config: RunConfig):
        stt_pb2 = self._stt_pb2
        assert stt_pb2 is not None
        profanity_filter = bool(config.extra.get("profanity_filter", False))
        literature_text = bool(config.extra.get("literature_text", False))
        normalize_text = bool(config.extra.get("text_normalization", True))
        language_code = config.extra.get("language_code", "ru-RU" if config.language == "ru" else f"{config.language}-RU")

        return stt_pb2.StreamingRequest(
            session_options=stt_pb2.StreamingOptions(
                recognition_model=stt_pb2.RecognitionModelOptions(
                    audio_format=stt_pb2.AudioFormatOptions(
                        raw_audio=stt_pb2.RawAudio(
                            audio_encoding=stt_pb2.RawAudio.LINEAR16_PCM,
                            sample_rate_hertz=config.sample_rate_hz,
                            audio_channel_count=config.channels,
                        )
                    ),
                    text_normalization=stt_pb2.TextNormalizationOptions(
                        text_normalization=(
                            stt_pb2.TextNormalizationOptions.TEXT_NORMALIZATION_ENABLED
                            if normalize_text
                            else stt_pb2.TextNormalizationOptions.TEXT_NORMALIZATION_DISABLED
                        ),
                        profanity_filter=profanity_filter,
                        literature_text=literature_text,
                    ),
                    language_restriction=stt_pb2.LanguageRestrictionOptions(
                        restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                        language_code=[language_code],
                    ),
                    audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME,
                )
            )
        )

    def _build_auth_header(self, config: RunConfig) -> str:
        api_key = config.extra.get("api_key") or os.environ.get("YANDEX_API_KEY")
        iam_token = config.extra.get("iam_token") or os.environ.get("YANDEX_IAM_TOKEN")
        if api_key:
            return f"Api-Key {api_key}"
        if iam_token:
            return f"Bearer {iam_token}"
        raise RuntimeError("Neither YANDEX_API_KEY nor YANDEX_IAM_TOKEN is set")

    def _handle_response_sync(self, response) -> None:
        event_type = response.WhichOneof("Event")
        if not event_type or event_type == "status_code":
            return
        if event_type == "partial":
            text = self._extract_alternatives_text(getattr(response, "partial", None))
            if not text:
                return
            self._ensure_segment_started_sync()
            self._last_partial_index += 1
            self._emit_partial_from_thread(
                segment_id=self._current_segment_id or self.make_segment_id(),
                text=text,
                started_at_monotonic_ms=self._current_segment_started_ms or self.now_monotonic_ms(),
                audio_sec=self._current_segment_audio_sec(),
                partial_index=self._last_partial_index,
                provider_event_type="partial",
                raw_meta={"event_type": "partial"},
            )
            return
        if event_type == "final":
            text = self._extract_alternatives_text(getattr(response, "final", None))
            if text:
                self._ensure_segment_started_sync()
                self._pending_final_text = text
            return
        if event_type == "final_refinement":
            refined = getattr(response, "final_refinement", None)
            text = self._extract_refined_text(refined)
            if not text:
                return
            self._ensure_segment_started_sync()
            self._emit_final_from_thread(
                segment_id=self._current_segment_id or self.make_segment_id(),
                text=text,
                started_at_monotonic_ms=self._current_segment_started_ms or self.now_monotonic_ms(),
                audio_sec=self._current_segment_audio_sec(),
                cost_estimate_usd=self._estimate_cost_usd(self._current_segment_audio_sec()),
                provider_event_type="final_refinement",
                raw_meta={"event_type": "final_refinement"},
            )
            self._close_current_segment_sync()
            return
        if event_type == "eou_update":
            self._flush_pending_final_sync(reason="eou_update")
            self._close_current_segment_sync()
            return

    def _ensure_segment_started_sync(self) -> None:
        if self._current_segment_id is not None:
            return
        self._segment_counter += 1
        self._current_segment_id = f"{self.session_id}_{self._segment_counter:04d}"
        self._current_segment_started_ms = self.now_monotonic_ms()
        self._current_segment_audio_start_sec = self._total_audio_sent_sec
        self._last_partial_index = 0
        self._pending_final_text = None

    def _close_current_segment_sync(self) -> None:
        self._current_segment_id = None
        self._current_segment_started_ms = None
        self._current_segment_audio_start_sec = self._total_audio_sent_sec
        self._last_partial_index = 0
        self._pending_final_text = None

    def _flush_pending_final_sync(self, *, reason: str) -> None:
        if not self._pending_final_text:
            return
        self._ensure_segment_started_sync()
        self._emit_final_from_thread(
            segment_id=self._current_segment_id or self.make_segment_id(),
            text=self._pending_final_text,
            started_at_monotonic_ms=self._current_segment_started_ms or self.now_monotonic_ms(),
            audio_sec=self._current_segment_audio_sec(),
            cost_estimate_usd=self._estimate_cost_usd(self._current_segment_audio_sec()),
            provider_event_type="final",
            raw_meta={"event_type": "final", "flush_reason": reason},
        )
        self._pending_final_text = None

    def _current_segment_audio_sec(self) -> float:
        return max(0.0, self._total_audio_sent_sec - self._current_segment_audio_start_sec)

    def _emit_partial_from_thread(self, *, segment_id: str, text: str, started_at_monotonic_ms: int,
                                  audio_sec: float, partial_index: int, provider_event_type: str,
                                  raw_meta: dict[str, Any]) -> None:
        if self._loop is None:
            return
        fut = asyncio.run_coroutine_threadsafe(
            self.emit_partial(
                segment_id=segment_id,
                text=text,
                started_at_monotonic_ms=started_at_monotonic_ms,
                audio_sec=audio_sec,
                partial_index=partial_index,
                language=self.config.language if self.config else None,
                provider_event_type=provider_event_type,
                raw_meta=raw_meta,
            ),
            self._loop,
        )
        try:
            fut.result(timeout=5)
        except Exception:
            pass

    def _emit_final_from_thread(self, *, segment_id: str, text: str, started_at_monotonic_ms: int,
                                audio_sec: float, cost_estimate_usd: float | None, provider_event_type: str,
                                raw_meta: dict[str, Any]) -> None:
        if self._loop is None:
            return
        fut = asyncio.run_coroutine_threadsafe(
            self.emit_final(
                segment_id=segment_id,
                text=text,
                started_at_monotonic_ms=started_at_monotonic_ms,
                audio_sec=audio_sec,
                language=self.config.language if self.config else None,
                cost_estimate_usd=cost_estimate_usd,
                provider_event_type=provider_event_type,
                raw_meta=raw_meta,
            ),
            self._loop,
        )
        try:
            fut.result(timeout=5)
        except Exception:
            pass

    def _extract_alternatives_text(self, event_obj) -> str:
        if event_obj is None:
            return ""
        alternatives = getattr(event_obj, "alternatives", None)
        if not alternatives:
            return ""
        texts = []
        for alt in alternatives:
            text = getattr(alt, "text", None)
            if text:
                texts.append(str(text).strip())
        return texts[0] if texts else ""

    def _extract_refined_text(self, event_obj) -> str:
        if event_obj is None:
            return ""
        normalized_text = getattr(event_obj, "normalized_text", None)
        if normalized_text is not None:
            text = self._extract_alternatives_text(normalized_text)
            if text:
                return text
        alternatives = getattr(event_obj, "alternatives", None)
        if alternatives:
            texts = []
            for alt in alternatives:
                text = getattr(alt, "text", None)
                if text:
                    texts.append(str(text).strip())
            if texts:
                return texts[0]
        return ""

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
