#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uv_bootstrap import ensure_uv_venv

ensure_uv_venv(__file__)

import argparse
import asyncio
import contextlib
import os
import signal

from backends.speechmatics_backend import SpeechmaticsRealtimeSTTBackend
from common.audio_capture import MicrophoneAudioSource
from common.result_writer import ResultWriter, format_console_line
from common.types import RunConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record microphone audio and transcribe it with Speechmatics realtime STT.")
    parser.add_argument("--device", default="default")
    parser.add_argument("--duration", type=float, default=7.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--language", default="ru")
    parser.add_argument("--rt-url", default=None)
    parser.add_argument("--operating-point", default="enhanced")
    parser.add_argument("--max-delay", type=float, default=1.0)
    parser.add_argument("--disable-partials", action="store_true")
    parser.add_argument("--enable-entities", action="store_true")
    parser.add_argument("--diarization", default="none")
    parser.add_argument("--pricing-per-min-usd", type=float, default=None)
    parser.add_argument("--finish-timeout-sec", type=float, default=15.0)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-id", default="speechmatics_demo_001")
    return parser.parse_args()


async def drain_events(backend: SpeechmaticsRealtimeSTTBackend, writer: ResultWriter, run_id: str) -> tuple[list[str], int, int, int]:
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
    auth_token = os.environ.get("SPEECHMATICS_AUTH_TOKEN") or os.environ.get("SPEECHMATICS_API_KEY")
    if not auth_token:
        print("ERROR: SPEECHMATICS_AUTH_TOKEN or SPEECHMATICS_API_KEY is not set", file=sys.stderr)
        return 2

    config = RunConfig(
        run_id=args.run_id,
        backend="speechmatics",
        language=args.language,
        audio_device=str(args.device),
        sample_rate_hz=args.sample_rate,
        channels=args.channels,
        chunk_ms=args.chunk_ms,
        extra={
            "auth_token": auth_token,
            "url": args.rt_url or os.environ.get("SPEECHMATICS_RT_URL"),
            "language": args.language,
            "operating_point": args.operating_point,
            "enable_partials": not args.disable_partials,
            "max_delay": args.max_delay,
            "enable_entities": args.enable_entities,
            "diarization": args.diarization,
            "pricing_per_min_usd": args.pricing_per_min_usd,
            "finish_timeout_sec": args.finish_timeout_sec,
        },
    )

    backend = SpeechmaticsRealtimeSTTBackend()
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
    print(f"device={args.device} duration={args.duration}s sample_rate={args.sample_rate} backend=speechmatics")
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
