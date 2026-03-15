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

from backends.deepgram_backend import DeepgramRealtimeSTTBackend
from common.audio_capture import MicrophoneAudioSource
from common.result_writer import ResultWriter, format_console_line
from common.types import RunConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record microphone audio and transcribe it with Deepgram Live Audio.")
    parser.add_argument("--device", default="default")
    parser.add_argument("--duration", type=float, default=7.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--language", default="ru")
    parser.add_argument("--model", default="nova-3")
    parser.add_argument("--endpointing-ms", type=int, default=300)
    parser.add_argument("--utterance-end-ms", default=None)
    parser.add_argument("--disable-interim-results", action="store_true")
    parser.add_argument("--disable-punctuate", action="store_true")
    parser.add_argument("--disable-smart-format", action="store_true")
    parser.add_argument("--disable-vad-events", action="store_true")
    parser.add_argument("--diarize", action="store_true")
    parser.add_argument("--pricing-per-min-usd", type=float, default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-id", default="deepgram_demo_001")
    return parser.parse_args()


async def drain_events(
    backend: DeepgramRealtimeSTTBackend,
    writer: ResultWriter,
    run_id: str,
) -> tuple[list[str], int, int]:
    final_segments: list[str] = []
    partial_count = 0
    final_count = 0
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
    return final_segments, partial_count, final_count


async def main() -> int:
    args = parse_args()
    writer = ResultWriter(args.results_dir)
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        print("ERROR: DEEPGRAM_API_KEY is not set", file=sys.stderr)
        return 2

    config = RunConfig(
        run_id=args.run_id,
        backend="deepgram",
        language=args.language,
        audio_device=str(args.device),
        sample_rate_hz=args.sample_rate,
        channels=args.channels,
        chunk_ms=args.chunk_ms,
        extra={
            "api_key": api_key,
            "model": args.model,
            "language": args.language,
            "endpointing_ms": args.endpointing_ms,
            "utterance_end_ms": args.utterance_end_ms,
            "interim_results": not args.disable_interim_results,
            "punctuate": not args.disable_punctuate,
            "smart_format": not args.disable_smart_format,
            "vad_events": not args.disable_vad_events,
            "diarize": args.diarize,
            "pricing_per_min_usd": args.pricing_per_min_usd,
        },
    )

    backend = DeepgramRealtimeSTTBackend()
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
    print(f"device={args.device} duration={args.duration}s sample_rate={args.sample_rate} backend=deepgram")
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
        await asyncio.sleep(1.5)
        await backend.stop()
        final_segments, partial_count, final_count = await drain_events(backend, writer, config.run_id)
        final_text = "\n".join(final_segments).strip()
        txt_path = writer.write_text_artifact(config.run_id, "final_text.txt", final_text + ("\n" if final_text else ""))
        print("\nSummary:")
        print(f"partial_events={partial_count} final_events={final_count}")
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
