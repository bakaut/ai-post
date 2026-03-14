from __future__ import annotations

import os
import tempfile
import wave
from typing import Any

from backends.base import BaseSTTBackend
from common.types import BackendCapabilities, RunConfig


class LocalSTTBackend(BaseSTTBackend):
    name = "local"

    def __init__(self) -> None:
        super().__init__()
        self._audio_buffer = bytearray()
        self._model = None
        self._model_loaded = False
        self._transcription_started_ms: int | None = None
        self._tmp_files: list[str] = []

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.name,
            supports_partial=False,
            supports_final=True,
            supports_word_timestamps=True,
            supports_speaker_diarization=False,
            supports_confidence=False,
            supports_language_detection=True,
            supports_vad_server_side=False,
            supports_manual_commit=False,
            requires_stream_reconnect=False,
            notes="Local faster-whisper MVP: final-only after finish_audio()",
        )

    async def _on_start(self) -> None:
        if self.config is None:
            raise RuntimeError("LocalSTTBackend: config is not set")

        model_name = self.config.extra.get("model_name", "small")
        device = self.config.extra.get("device", "cpu")
        compute_type = self.config.extra.get("compute_type", "int8")
        cpu_threads = self.config.extra.get("cpu_threads")
        download_root = self.config.extra.get("download_root")
        local_files_only = self.config.extra.get("local_files_only", False)

        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper is not installed. Install it with: pip install faster-whisper"
            ) from e

        kwargs: dict[str, Any] = {
            "model_size_or_path": model_name,
            "device": device,
            "compute_type": compute_type,
        }
        if cpu_threads is not None:
            kwargs["cpu_threads"] = int(cpu_threads)
        if download_root is not None:
            kwargs["download_root"] = str(download_root)
        if local_files_only is not None:
            kwargs["local_files_only"] = bool(local_files_only)

        self._model = WhisperModel(**kwargs)
        self._model_loaded = True

    async def feed_audio(self, pcm_chunk: bytes) -> None:
        if not self._started or self.config is None:
            raise RuntimeError("LocalSTTBackend: backend is not started")
        if not isinstance(pcm_chunk, (bytes, bytearray)):
            raise TypeError("LocalSTTBackend.feed_audio expects bytes")
        if not pcm_chunk:
            return
        self._audio_buffer.extend(pcm_chunk)

    async def _on_finish_audio(self) -> None:
        if self.config is None:
            raise RuntimeError("LocalSTTBackend: config is not set")
        if not self._model_loaded or self._model is None:
            raise RuntimeError("LocalSTTBackend: model is not loaded")
        if not self._audio_buffer:
            return

        wav_path = self._write_temp_wav(
            pcm_bytes=bytes(self._audio_buffer),
            sample_rate_hz=self.config.sample_rate_hz,
            channels=self.config.channels,
            sample_width_bytes=2,
        )
        self._tmp_files.append(wav_path)
        self._transcription_started_ms = self.now_monotonic_ms()

        try:
            transcribe_kwargs = self._build_transcribe_kwargs(self.config)
            segments, info = self._model.transcribe(wav_path, **transcribe_kwargs)
            segment_index = 0
            detected_language = getattr(info, "language", None)

            for seg in segments:
                segment_index += 1
                text = (seg.text or "").strip()
                if not text:
                    continue

                start_sec = float(getattr(seg, "start", 0.0) or 0.0)
                end_sec = float(getattr(seg, "end", 0.0) or 0.0)
                audio_sec = max(0.0, end_sec - start_sec)
                raw_meta: dict[str, Any] = {
                    "segment_start_sec": start_sec,
                    "segment_end_sec": end_sec,
                    "language_probability": getattr(info, "language_probability", None),
                }

                words = getattr(seg, "words", None)
                if words:
                    raw_meta["words"] = [
                        {
                            "start": getattr(w, "start", None),
                            "end": getattr(w, "end", None),
                            "word": getattr(w, "word", None),
                            "probability": getattr(w, "probability", None),
                        }
                        for w in words
                    ]

                await self.emit_final(
                    segment_id=f"{self.session_id}_{segment_index:04d}",
                    text=text,
                    started_at_monotonic_ms=self._transcription_started_ms,
                    audio_sec=audio_sec,
                    language=detected_language,
                    provider_event_type="local_final_segment",
                    raw_meta=raw_meta,
                )

        except Exception as e:
            await self.emit_error(
                segment_id=f"{self.session_id}_transcribe_error",
                text=f"local transcription failed: {e}",
                started_at_monotonic_ms=self._transcription_started_ms or self.now_monotonic_ms(),
                raw_meta={"exception_type": type(e).__name__},
            )

    async def _on_stop(self) -> None:
        for path in self._tmp_files:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass
        self._tmp_files.clear()

    def _write_temp_wav(
        self,
        *,
        pcm_bytes: bytes,
        sample_rate_hz: int,
        channels: int,
        sample_width_bytes: int,
    ) -> str:
        fd, path = tempfile.mkstemp(prefix="stt_local_", suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width_bytes)
            wf.setframerate(sample_rate_hz)
            wf.writeframes(pcm_bytes)
        return path

    def _build_transcribe_kwargs(self, config: RunConfig) -> dict[str, Any]:
        extra = config.extra
        kwargs: dict[str, Any] = {
            "language": extra.get("language", config.language),
            "beam_size": int(extra.get("beam_size", 5)),
            "vad_filter": bool(extra.get("vad_filter", True)),
            "word_timestamps": bool(extra.get("word_timestamps", False)),
        }

        initial_prompt = extra.get("initial_prompt")
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt

        condition_on_previous_text = extra.get("condition_on_previous_text")
        if condition_on_previous_text is not None:
            kwargs["condition_on_previous_text"] = bool(condition_on_previous_text)

        temperature = extra.get("temperature")
        if temperature is not None:
            kwargs["temperature"] = float(temperature)

        vad_parameters = extra.get("vad_parameters")
        if vad_parameters:
            kwargs["vad_parameters"] = vad_parameters

        return kwargs
