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
import json
import re
import signal
import statistics
import time
import wave
from dataclasses import asdict, dataclass
from typing import Any

from common.audio_capture import MicrophoneAudioSource
from common.result_writer import ResultWriter, format_console_line
from common.types import RunConfig, TranscriptEvent


@dataclass(slots=True)
class CompareSummary:
    backend: str
    run_id: str
    ok: bool
    error: str | None
    final_text: str
    final_segments: list[str]
    partial_events: int
    final_events: int
    error_events: int
    avg_partial_latency_ms: float | None
    avg_final_latency_ms: float | None
    p95_final_latency_ms: float | None
    total_audio_sec_from_events: float
    estimated_cost_usd: float | None
    wer: float | None
    cer: float | None
    restarts_max: int
    dropped_chunks_max: int
    provider_errors_sum: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the same STT comparison across local/openai/yandex/elevenlabs.")
    parser.add_argument("--backends", default="local,openai,yandex,elevenlabs")
    parser.add_argument("--device", default="default")
    parser.add_argument("--duration", type=float, default=7.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--reference-file", default=None)
    parser.add_argument("--wav-file", default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-prefix", default="compare")
    parser.add_argument("--post-finish-wait-sec", type=float, default=1.5)

    parser.add_argument("--local-model", default="small")
    parser.add_argument("--local-fw-device", default="cpu")
    parser.add_argument("--local-compute-type", default="int8")
    parser.add_argument("--local-word-timestamps", action="store_true")
    parser.add_argument("--local-no-vad-filter", action="store_true")

    parser.add_argument("--openai-realtime-model", default="gpt-realtime")
    parser.add_argument("--openai-transcription-model", default="gpt-4o-mini-transcribe")
    parser.add_argument("--openai-transcription-prompt", default=None)
    parser.add_argument("--openai-noise-reduction-type", default="near_field", choices=["near_field", "far_field"])
    parser.add_argument("--openai-disable-turn-detection", action="store_true")
    parser.add_argument("--openai-silence-duration-ms", type=int, default=500)
    parser.add_argument("--openai-prefix-padding-ms", type=int, default=300)
    parser.add_argument("--openai-threshold", type=float, default=0.5)
    parser.add_argument("--openai-include-logprobs", action="store_true")
    parser.add_argument("--openai-pricing-per-min-usd", type=float, default=None)

    parser.add_argument("--yandex-endpoint", default=None)
    parser.add_argument("--yandex-language-code", default="ru-RU")
    parser.add_argument("--yandex-partial-results", action="store_true")
    parser.add_argument("--yandex-disable-text-normalization", action="store_true")
    parser.add_argument("--yandex-profanity-filter", action="store_true")
    parser.add_argument("--yandex-literature-text", action="store_true")
    parser.add_argument("--yandex-pricing-per-min-usd", type=float, default=None)

    parser.add_argument("--elevenlabs-language-code", default="ru")
    parser.add_argument("--elevenlabs-model-id", default="scribe_v2_realtime")
    parser.add_argument("--elevenlabs-commit-strategy", default="manual", choices=["manual", "vad"])
    parser.add_argument("--elevenlabs-vad-silence-threshold-secs", type=float, default=1.5)
    parser.add_argument("--elevenlabs-vad-threshold", type=float, default=0.4)
    parser.add_argument("--elevenlabs-min-speech-duration-ms", type=int, default=100)
    parser.add_argument("--elevenlabs-min-silence-duration-ms", type=int, default=100)
    parser.add_argument("--elevenlabs-disable-timestamps", action="store_true")
    parser.add_argument("--elevenlabs-disable-language-detection", action="store_true")
    parser.add_argument("--elevenlabs-disable-logging", action="store_true")
    parser.add_argument("--elevenlabs-previous-text", default=None)
    parser.add_argument("--elevenlabs-pricing-per-min-usd", type=float, default=None)
    return parser.parse_args()


def load_reference_text(args: argparse.Namespace) -> str:
    if args.reference_text:
        return args.reference_text.strip()
    if args.reference_file:
        return Path(args.reference_file).read_text(encoding="utf-8").strip()
    return ""


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()
    return text


def levenshtein(seq1: list[str] | str, seq2: list[str] | str) -> int:
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        a = seq1[i - 1]
        for j in range(1, m + 1):
            b = seq2[j - 1]
            cost = 0 if a == b else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def compute_wer(reference: str, hypothesis: str) -> float | None:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return None
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return None
    return levenshtein(ref_words, hyp_words) / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float | None:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return None
    return levenshtein(ref, hyp) / len(ref)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = int((len(values) - 1) * p)
    return values[idx]


def make_run_id(prefix: str, backend: str) -> str:
    ts = time.strftime("%Y%m%dT%H%M%S")
    return f"{prefix}_{backend}_{ts}"


def parse_backends(raw: str) -> list[str]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    valid = {"local", "openai", "yandex", "elevenlabs"}
    bad = [x for x in items if x not in valid]
    if bad:
        raise ValueError(f"Unsupported backends: {', '.join(bad)}")
    if not items:
        raise ValueError("No backends selected")
    return items


def load_wav_file(path: str) -> tuple[bytes, int, int]:
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        pcm = wf.readframes(frames)
    if sample_width != 2:
        raise ValueError("Only PCM16 WAV is supported for --wav-file")
    if channels != 1:
        raise ValueError("Only mono WAV is supported for --wav-file")
    return pcm, sample_rate, channels


def chunk_pcm(pcm: bytes, *, sample_rate_hz: int, channels: int, chunk_ms: int, sample_width_bytes: int = 2) -> list[bytes]:
    bytes_per_second = sample_rate_hz * channels * sample_width_bytes
    chunk_size = int(bytes_per_second * chunk_ms / 1000)
    if chunk_size <= 0:
        raise ValueError("chunk_size resolved to 0")
    return [pcm[i:i + chunk_size] for i in range(0, len(pcm), chunk_size)]


async def replay_wav_realtime(backend, *, pcm: bytes, sample_rate_hz: int, channels: int, chunk_ms: int) -> None:
    chunks = chunk_pcm(pcm, sample_rate_hz=sample_rate_hz, channels=channels, chunk_ms=chunk_ms)
    chunk_sleep = chunk_ms / 1000.0
    for chunk in chunks:
        await backend.feed_audio(chunk)
        await asyncio.sleep(chunk_sleep)


async def record_from_mic(backend, *, device: str, sample_rate_hz: int, channels: int, chunk_ms: int,
                          duration_sec: float, stop_requested: asyncio.Event) -> dict[str, Any]:
    async with MicrophoneAudioSource(device=device, sample_rate_hz=sample_rate_hz, channels=channels, chunk_ms=chunk_ms) as mic:
        started = asyncio.get_running_loop().time()
        async for chunk in mic:
            await backend.feed_audio(chunk.pcm)
            if stop_requested.is_set():
                break
            if asyncio.get_running_loop().time() - started >= duration_sec:
                break
        return mic.stats()


async def drain_events(backend, writer: ResultWriter, run_id: str) -> list[TranscriptEvent]:
    events: list[TranscriptEvent] = []
    async for event in backend.events():
        print(format_console_line(event))
        print()
        writer.write_event_jsonl(run_id, event)
        events.append(event)
    return events


def summarize_events(*, backend_name: str, run_id: str, events: list[TranscriptEvent], reference_text: str,
                     error: str | None = None) -> CompareSummary:
    partials = [e for e in events if e.status.value == "partial"]
    finals = [e for e in events if e.status.value == "final"]
    errors = [e for e in events if e.status.value == "error"]
    final_segments = [e.text.strip() for e in finals if e.text.strip()]
    final_text = "\n".join(final_segments).strip()
    partial_lat = [float(e.latency_ms) for e in partials]
    final_lat = [float(e.latency_ms) for e in finals]
    cost_values = [e.cost_estimate_usd for e in finals if e.cost_estimate_usd is not None]
    estimated_cost_usd = sum(cost_values) if cost_values else None
    wer = compute_wer(reference_text, final_text) if reference_text else None
    cer = compute_cer(reference_text, final_text) if reference_text else None
    return CompareSummary(
        backend=backend_name,
        run_id=run_id,
        ok=error is None,
        error=error,
        final_text=final_text,
        final_segments=final_segments,
        partial_events=len(partials),
        final_events=len(finals),
        error_events=len(errors),
        avg_partial_latency_ms=statistics.mean(partial_lat) if partial_lat else None,
        avg_final_latency_ms=statistics.mean(final_lat) if final_lat else None,
        p95_final_latency_ms=percentile(final_lat, 0.95),
        total_audio_sec_from_events=sum(e.audio_sec for e in finals),
        estimated_cost_usd=estimated_cost_usd,
        wer=wer,
        cer=cer,
        restarts_max=max((e.restart_count for e in events), default=0),
        dropped_chunks_max=max((e.dropped_chunks for e in events), default=0),
        provider_errors_sum=sum(e.error_count for e in events),
    )


def print_summary(summary: CompareSummary) -> None:
    print("=" * 80)
    print(f"backend={summary.backend} ok={summary.ok}")
    if summary.error:
        print(f"error={summary.error}")
    print(f"partial_events={summary.partial_events}")
    print(f"final_events={summary.final_events}")
    print(f"error_events={summary.error_events}")
    print(f"avg_partial_latency_ms={summary.avg_partial_latency_ms}")
    print(f"avg_final_latency_ms={summary.avg_final_latency_ms}")
    print(f"p95_final_latency_ms={summary.p95_final_latency_ms}")
    print(f"total_audio_sec_from_events={summary.total_audio_sec_from_events:.3f}")
    print(f"estimated_cost_usd={summary.estimated_cost_usd}")
    print(f"wer={summary.wer}")
    print(f"cer={summary.cer}")
    print(f"restarts_max={summary.restarts_max}")
    print(f"dropped_chunks_max={summary.dropped_chunks_max}")
    print(f"provider_errors_sum={summary.provider_errors_sum}")
    print("\nfinal_text:\n")
    print(summary.final_text if summary.final_text else "(empty)")
    print("=" * 80)
    print()


def print_compare_table(summaries: list[CompareSummary]) -> None:
    print("\nCOMPARE\n")
    header = f"{'backend':12} {'ok':3} {'wer':>8} {'cer':>8} {'avg_final_ms':>12} {'p95_final_ms':>12} {'cost_usd':>10} {'finals':>6} {'partials':>8} {'errors':>6}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        wer = f"{s.wer:.3f}" if s.wer is not None else "-"
        cer = f"{s.cer:.3f}" if s.cer is not None else "-"
        avgf = f"{s.avg_final_latency_ms:.1f}" if s.avg_final_latency_ms is not None else "-"
        p95f = f"{s.p95_final_latency_ms:.1f}" if s.p95_final_latency_ms is not None else "-"
        cost = f"{s.estimated_cost_usd:.6f}" if s.estimated_cost_usd is not None else "-"
        print(f"{s.backend:12} {str(s.ok):3} {wer:>8} {cer:>8} {avgf:>12} {p95f:>12} {cost:>10} {s.final_events:>6} {s.partial_events:>8} {s.error_events:>6}")
    print()


def build_backend_and_config(backend_name: str, args: argparse.Namespace, run_id: str, sample_rate_hz: int,
                             channels: int) -> tuple[Any, RunConfig]:
    if backend_name == "local":
        from backends import LocalSTTBackend

        return LocalSTTBackend(), RunConfig(
            run_id=run_id,
            backend="local",
            language="ru",
            audio_device=str(args.device),
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            chunk_ms=args.chunk_ms,
            extra={
                "model_name": args.local_model,
                "device": args.local_fw_device,
                "compute_type": args.local_compute_type,
                "vad_filter": not args.local_no_vad_filter,
                "word_timestamps": args.local_word_timestamps,
            },
        )
    if backend_name == "openai":
        from backends import OpenAIRealtimeSTTBackend

        return OpenAIRealtimeSTTBackend(), RunConfig(
            run_id=run_id,
            backend="openai",
            language="ru",
            audio_device=str(args.device),
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            chunk_ms=args.chunk_ms,
            extra={
                "api_key": __import__("os").environ.get("OPENAI_API_KEY"),
                "realtime_model": args.openai_realtime_model,
                "transcription_model": args.openai_transcription_model,
                "transcription_prompt": args.openai_transcription_prompt,
                "noise_reduction_type": args.openai_noise_reduction_type,
                "turn_detection_enabled": not args.openai_disable_turn_detection,
                "silence_duration_ms": args.openai_silence_duration_ms,
                "prefix_padding_ms": args.openai_prefix_padding_ms,
                "threshold": args.openai_threshold,
                "include_logprobs": args.openai_include_logprobs,
                "pricing_per_min_usd": args.openai_pricing_per_min_usd,
            },
        )
    if backend_name == "yandex":
        import os
        from backends import YandexStreamingSTTBackend

        return YandexStreamingSTTBackend(), RunConfig(
            run_id=run_id,
            backend="yandex",
            language="ru",
            audio_device=str(args.device),
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            chunk_ms=args.chunk_ms,
            extra={
                "endpoint": args.yandex_endpoint or os.environ.get("YANDEX_STT_ENDPOINT"),
                "api_key": os.environ.get("YANDEX_API_KEY"),
                "iam_token": os.environ.get("YANDEX_IAM_TOKEN"),
                "language_code": args.yandex_language_code,
                "partial_results": args.yandex_partial_results,
                "text_normalization": not args.yandex_disable_text_normalization,
                "profanity_filter": args.yandex_profanity_filter,
                "literature_text": args.yandex_literature_text,
                "pricing_per_min_usd": args.yandex_pricing_per_min_usd,
            },
        )
    if backend_name == "elevenlabs":
        import os
        from backends import ElevenLabsRealtimeSTTBackend

        return ElevenLabsRealtimeSTTBackend(), RunConfig(
            run_id=run_id,
            backend="elevenlabs",
            language="ru",
            audio_device=str(args.device),
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            chunk_ms=args.chunk_ms,
            extra={
                "api_key": os.environ.get("ELEVENLABS_API_KEY"),
                "model_id": args.elevenlabs_model_id,
                "audio_format": f"pcm_{sample_rate_hz}",
                "language_code": args.elevenlabs_language_code,
                "commit_strategy": args.elevenlabs_commit_strategy,
                "include_timestamps": not args.elevenlabs_disable_timestamps,
                "include_language_detection": not args.elevenlabs_disable_language_detection,
                "enable_logging": not args.elevenlabs_disable_logging,
                "vad_silence_threshold_secs": args.elevenlabs_vad_silence_threshold_secs,
                "vad_threshold": args.elevenlabs_vad_threshold,
                "min_speech_duration_ms": args.elevenlabs_min_speech_duration_ms,
                "min_silence_duration_ms": args.elevenlabs_min_silence_duration_ms,
                "previous_text": args.elevenlabs_previous_text,
                "pricing_per_min_usd": args.elevenlabs_pricing_per_min_usd,
            },
        )
    raise ValueError(f"Unsupported backend: {backend_name}")


async def run_single_backend(backend_name: str, args: argparse.Namespace, writer: ResultWriter, reference_text: str,
                             *, wav_pcm: bytes | None, wav_sample_rate_hz: int | None, wav_channels: int | None,
                             stop_requested: asyncio.Event) -> CompareSummary:
    sample_rate_hz = wav_sample_rate_hz if wav_pcm is not None else args.sample_rate
    channels = wav_channels if wav_pcm is not None else args.channels
    run_id = make_run_id(args.run_prefix, backend_name)
    events: list[TranscriptEvent] = []
    backend = None
    error: str | None = None
    try:
        backend, config = build_backend_and_config(backend_name, args, run_id, sample_rate_hz, channels)
        await backend.start(config)
        if wav_pcm is not None:
            print(f"\n[{backend_name}] replaying WAV in realtime...\n")
            await replay_wav_realtime(backend, pcm=wav_pcm, sample_rate_hz=sample_rate_hz, channels=channels, chunk_ms=args.chunk_ms)
        else:
            print("\n" + "=" * 80)
            print(f"[{backend_name}] Прочитай этот текст:\n")
            print(reference_text if reference_text else "(reference text is empty)")
            print()
            input("Нажми Enter и читай вслух...")
            print("3...")
            await asyncio.sleep(1)
            print("2...")
            await asyncio.sleep(1)
            print("1...")
            await asyncio.sleep(1)
            print("Старт.\n")
            mic_stats = await record_from_mic(
                backend,
                device=args.device,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
                chunk_ms=args.chunk_ms,
                duration_sec=args.duration,
                stop_requested=stop_requested,
            )
            print(f"[{backend_name}] microphone stats: {mic_stats}")
        await backend.finish_audio()
        await asyncio.sleep(args.post_finish_wait_sec)
        await backend.stop()
        events = await drain_events(backend, writer, run_id)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if backend is not None:
            with contextlib.suppress(Exception):
                await backend.stop()
            with contextlib.suppress(Exception):
                events = await drain_events(backend, writer, run_id)

    summary = summarize_events(backend_name=backend_name, run_id=run_id, events=events, reference_text=reference_text, error=error)
    writer.write_text_artifact(run_id, "final_text.txt", summary.final_text + ("\n" if summary.final_text else ""))
    Path(writer.prepare_run_dir(run_id) / "compare_summary.json").write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


async def main() -> int:
    args = parse_args()
    writer = ResultWriter(args.results_dir)
    selected_backends = parse_backends(args.backends)
    reference_text = load_reference_text(args)

    wav_pcm: bytes | None = None
    wav_sample_rate_hz: int | None = None
    wav_channels: int | None = None
    if args.wav_file:
        wav_pcm, wav_sample_rate_hz, wav_channels = load_wav_file(args.wav_file)
        print("Using WAV mode.")
        print(f"wav_file={args.wav_file}")
        print(f"sample_rate={wav_sample_rate_hz} channels={wav_channels}")
    else:
        print("Using microphone mode.")

    if not reference_text:
        print("WARNING: reference text is empty, WER/CER will be skipped.", file=sys.stderr)

    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        stop_requested.set()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, _request_stop)

    summaries: list[CompareSummary] = []
    for backend_name in selected_backends:
        if stop_requested.is_set():
            break
        summary = await run_single_backend(
            backend_name,
            args,
            writer,
            reference_text,
            wav_pcm=wav_pcm,
            wav_sample_rate_hz=wav_sample_rate_hz,
            wav_channels=wav_channels,
            stop_requested=stop_requested,
        )
        summaries.append(summary)
        print_summary(summary)

    print_compare_table(summaries)
    compare_path = Path(args.results_dir) / f"{args.run_prefix}_compare.json"
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    compare_path.write_text(json.dumps([s.to_dict() for s in summaries], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved compare summary to: {compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
