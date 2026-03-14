#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys

from backends.local_backend import LocalSTTBackend
from common.audio_capture import MicrophoneAudioSource
from common.result_writer import ResultWriter, format_console_line
from common.types import RunConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record microphone audio and transcribe it locally with faster-whisper.")
    parser.add_argument("--device", default="default", help='Input device name/index. Default: "default"')
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds. Default: 5.0")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate. Default: 16000")
    parser.add_argument("--channels", type=int, default=1, help="Audio channels. Default: 1")
    parser.add_argument("--chunk-ms", type=int, default=100, help="Audio chunk size in milliseconds. Default: 100")
    parser.add_argument("--model", default="small", help='faster-whisper model name/path. Default: "small"')
    parser.add_argument("--fw-device", default="cpu", help='faster-whisper device. Default: "cpu"')
    parser.add_argument("--compute-type", default="int8", help='faster-whisper compute type. Default: "int8"')
    parser.add_argument("--language", default="ru", help='Language hint for whisper. Default: "ru"')
    parser.add_argument("--word-timestamps", action="store_true", help="Enable word timestamps in raw_meta.")
    parser.add_argument("--no-vad-filter", action="store_true", help="Disable whisper vad_filter.")
    parser.add_argument("--results-dir", default="results", help='Directory for events/txt files. Default: "results"')
    parser.add_argument("--run-id", default="local_demo_001", help='Run ID. Default: "local_demo_001"')
    return parser.parse_args()


async def drain_events(backend: LocalSTTBackend, writer: ResultWriter, run_id: str) -> list[str]:
    final_segments: list[str] = []
    async for event in backend.events():
        print(format_console_line(event))
        print()
        writer.write_event_jsonl(run_id, event)
        if event.status.value == "final" and event.text.strip():
            final_segments.append(event.text.strip())
    return final_segments


async def main() -> int:
    args = parse_args()
    writer = ResultWriter(args.results_dir)
    config = RunConfig(
        run_id=args.run_id,
        backend="local",
        language=args.language,
        audio_device=str(args.device),
        sample_rate_hz=args.sample_rate,
        channels=args.channels,
        chunk_ms=args.chunk_ms,
        extra={
            "model_name": args.model,
            "device": args.fw_device,
            "compute_type": args.compute_type,
            "vad_filter": not args.no_vad_filter,
            "word_timestamps": args.word_timestamps,
        },
    )

    backend = LocalSTTBackend()
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        stop_requested.set()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, _request_stop)

    await backend.start(config)

    print("Recording started.")
    print(f"device={args.device} duration={args.duration}s sample_rate={args.sample_rate}")
    print("Speak now...\n")

    try:
        async with MicrophoneAudioSource(
            device=args.device,
            sample_rate_hz=args.sample_rate,
            channels=args.channels,
            chunk_ms=args.chunk_ms,
        ) as mic:
            started = loop.time()
            async for chunk in mic:
                await backend.feed_audio(chunk.pcm)
                if stop_requested.is_set():
                    break
                if loop.time() - started >= args.duration:
                    break

            print("\nRecording finished.")
            print("Microphone stats:", mic.stats())

        await backend.finish_audio()
        await backend.stop()
        final_segments = await drain_events(backend, writer, config.run_id)
        final_text = "\n".join(final_segments).strip()
        txt_path = writer.write_text_artifact(config.run_id, "final_text.txt", final_text + ("\n" if final_text else ""))

        print("\nFinal transcript:\n")
        print(final_text if final_text else "(empty)")
        print(f"\nSaved text to: {txt_path}")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        with contextlib.suppress(Exception):
            await backend.stop()
        return 130
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        with contextlib.suppress(Exception):
            await backend.stop()
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
