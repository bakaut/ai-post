#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import sys

from backends.elevenlabs_backend import ElevenLabsRealtimeSTTBackend
from common.audio_capture import MicrophoneAudioSource
from common.result_writer import ResultWriter, format_console_line
from common.types import RunConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record microphone audio and transcribe it with ElevenLabs Realtime STT.")
    parser.add_argument("--device", default="default")
    parser.add_argument("--duration", type=float, default=7.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--language", default="ru")
    parser.add_argument("--language-code", default="ru")
    parser.add_argument("--model-id", default="scribe_v2_realtime")
    parser.add_argument("--commit-strategy", default="manual", choices=["manual", "vad"])
    parser.add_argument("--vad-silence-threshold-secs", type=float, default=1.5)
    parser.add_argument("--vad-threshold", type=float, default=0.4)
    parser.add_argument("--min-speech-duration-ms", type=int, default=100)
    parser.add_argument("--min-silence-duration-ms", type=int, default=100)
    parser.add_argument("--disable-timestamps", action="store_true")
    parser.add_argument("--disable-language-detection", action="store_true")
    parser.add_argument("--disable-logging", action="store_true")
    parser.add_argument("--previous-text", default=None)
    parser.add_argument("--pricing-per-min-usd", type=float, default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-id", default="elevenlabs_demo_001")
    return parser.parse_args()


async def drain_events(backend: ElevenLabsRealtimeSTTBackend, writer: ResultWriter, run_id: str) -> tuple[list[str], int, int, int]:
    final_segments: list[str] = []
    partial_count = 0
    final_count = 0
    error_count = 0
    async for event in backend.events():
        print(format_console_line(event))
        print()
        writer.write_event_jsonl(run_id, event)
        if event.status.value == "partial":
            partial_count += 1
        elif event.status.value == "final":
            final_count += 1
            if event.text.strip():
                final_segments.append(event.text.strip())
        elif event.status.value == "error":
            error_count += 1
    return final_segments, partial_count, final_count, error_count


async def main() -> int:
    args = parse_args()
    writer = ResultWriter(args.results_dir)
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY is not set", file=sys.stderr)
        return 2

    config = RunConfig(
        run_id=args.run_id,
        backend="elevenlabs",
        language=args.language,
        audio_device=str(args.device),
        sample_rate_hz=args.sample_rate,
        channels=args.channels,
        chunk_ms=args.chunk_ms,
        extra={
            "api_key": api_key,
            "model_id": args.model_id,
            "audio_format": f"pcm_{args.sample_rate}",
            "language_code": args.language_code,
            "commit_strategy": args.commit_strategy,
            "include_timestamps": not args.disable_timestamps,
            "include_language_detection": not args.disable_language_detection,
            "enable_logging": not args.disable_logging,
            "vad_silence_threshold_secs": args.vad_silence_threshold_secs,
            "vad_threshold": args.vad_threshold,
            "min_speech_duration_ms": args.min_speech_duration_ms,
            "min_silence_duration_ms": args.min_silence_duration_ms,
            "previous_text": args.previous_text,
            "pricing_per_min_usd": args.pricing_per_min_usd,
        },
    )

    backend = ElevenLabsRealtimeSTTBackend()
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
    print(f"device={args.device} duration={args.duration}s sample_rate={args.sample_rate} backend=elevenlabs")
    print("Speak now...\n")

    try:
        async with MicrophoneAudioSource(device=args.device, sample_rate_hz=args.sample_rate, channels=args.channels, chunk_ms=args.chunk_ms) as mic:
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
        await asyncio.sleep(1.5)
        await backend.stop()

        final_segments, partial_count, final_count, error_count = await drain_events(backend, writer, config.run_id)
        final_text = "\n".join(final_segments).strip()
        txt_path = writer.write_text_artifact(config.run_id, "final_text.txt", final_text + ("\n" if final_text else ""))

        print("\nSummary:")
        print(f"partial_events={partial_count} final_events={final_count} errors={error_count}")
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
