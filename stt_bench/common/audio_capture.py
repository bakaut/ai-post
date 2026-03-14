from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import sounddevice as sd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class AudioChunk:
    pcm: bytes
    frames: int
    sample_rate_hz: int
    channels: int
    device: str
    created_at_iso: str
    created_at_monotonic_ms: int
    input_overflow: bool = False

    @property
    def duration_sec(self) -> float:
        if self.sample_rate_hz <= 0:
            return 0.0
        return self.frames / self.sample_rate_hz


def list_input_devices() -> list[dict[str, Any]]:
    devices = sd.query_devices()
    result: list[dict[str, Any]] = []

    for idx, dev in enumerate(devices):
        max_in = int(dev.get("max_input_channels", 0) or 0)
        if max_in <= 0:
            continue

        result.append(
            {
                "index": idx,
                "name": dev.get("name", ""),
                "max_input_channels": max_in,
                "default_samplerate": dev.get("default_samplerate"),
                "hostapi": dev.get("hostapi"),
            }
        )
    return result


def resolve_input_device(device: str | int | None) -> str | int | None:
    if device is None or device == "default":
        return None

    if isinstance(device, int):
        return device

    device_str = str(device).strip()
    if device_str.isdigit():
        return int(device_str)

    candidates = list_input_devices()

    for dev in candidates:
        if dev["name"] == device_str:
            return int(dev["index"])

    lower = device_str.lower()
    for dev in candidates:
        if lower in dev["name"].lower():
            return int(dev["index"])

    available = ", ".join(f'{d["index"]}:{d["name"]}' for d in candidates)
    raise ValueError(
        f"Input device '{device}' not found. Available input devices: {available}"
    )


class MicrophoneAudioSource:
    def __init__(
        self,
        *,
        device: str | int | None = "default",
        sample_rate_hz: int = 16000,
        channels: int = 1,
        chunk_ms: int = 100,
        dtype: str = "int16",
        queue_maxsize: int = 100,
    ) -> None:
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if chunk_ms <= 0:
            raise ValueError("chunk_ms must be > 0")

        self.device = device
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self.chunk_ms = chunk_ms
        self.dtype = dtype

        self.blocksize = int(sample_rate_hz * chunk_ms / 1000)
        if self.blocksize <= 0:
            raise ValueError("blocksize resolved to 0")

        self._stream: sd.RawInputStream | None = None
        self._queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue(maxsize=queue_maxsize)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False

        self.total_chunks = 0
        self.total_frames = 0
        self.dropped_chunks = 0
        self.overflow_events = 0

    async def start(self) -> None:
        if self._running:
            raise RuntimeError("MicrophoneAudioSource is already started")

        self._loop = asyncio.get_running_loop()
        resolved_device = resolve_input_device(self.device)

        sd.check_input_settings(
            device=resolved_device,
            samplerate=self.sample_rate_hz,
            channels=self.channels,
            dtype=self.dtype,
        )

        self._stream = sd.RawInputStream(
            device=resolved_device,
            samplerate=self.sample_rate_hz,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self._stream.start()
        self._running = True

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None

        await self._queue.put(None)

    def stats(self) -> dict[str, Any]:
        total_audio_sec = self.total_frames / self.sample_rate_hz if self.sample_rate_hz > 0 else 0.0
        return {
            "device": self.device,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "chunk_ms": self.chunk_ms,
            "total_chunks": self.total_chunks,
            "total_frames": self.total_frames,
            "total_audio_sec": total_audio_sec,
            "dropped_chunks": self.dropped_chunks,
            "overflow_events": self.overflow_events,
        }

    async def __aenter__(self) -> "MicrophoneAudioSource":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    def __aiter__(self):
        return self.iter_chunks()

    async def iter_chunks(self):
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item

    def _callback(self, indata, frames: int, time_info, status) -> None:
        if self._loop is None or not self._running:
            return

        pcm = bytes(indata)
        input_overflow = bool(status.input_overflow) if status else False

        chunk = AudioChunk(
            pcm=pcm,
            frames=frames,
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            device=str(self.device),
            created_at_iso=utc_now_iso(),
            created_at_monotonic_ms=int(time.monotonic() * 1000),
            input_overflow=input_overflow,
        )

        self._loop.call_soon_threadsafe(self._enqueue_chunk, chunk)

    def _enqueue_chunk(self, chunk: AudioChunk) -> None:
        if not self._running:
            return

        self.total_chunks += 1
        self.total_frames += chunk.frames
        if chunk.input_overflow:
            self.overflow_events += 1

        try:
            self._queue.put_nowait(chunk)
        except asyncio.QueueFull:
            self.dropped_chunks += 1
