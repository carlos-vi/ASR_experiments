"""Live transcription from the microphone.

Usage:  python -m whisper_transcriber.transcribe_mic [--model base.en] [--threshold 0.005] [--device N]

How it works: the microphone produces 30 ms chunks, the segmenter detects when
the speaker starts and stops talking, and each complete utterance is sent to
Whisper. Press Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import queue
import sys
import time

import numpy as np

from .model import AVAILABLE_MODELS, duration_seconds, load_model, transcribe_audio
from .segmenter import SAMPLE_RATE, SpeechSegmenter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live English microphone transcriber.")
    parser.add_argument("--model", default="base.en", choices=AVAILABLE_MODELS,
                        help="base.en keeps up with live speech on a typical laptop CPU")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="Minimum energy to detect speech (lower it if speech is missed, "
                             "raise it if noise is picked up)")
    parser.add_argument("--device", type=int, default=None,
                        help="Microphone index (see python -m sounddevice)")
    parser.add_argument("--silence-ms", type=int, default=700, help="Pause that ends an utterance")
    parser.add_argument("--output", help="File to append the transcript to")
    args = parser.parse_args(argv)

    import sounddevice as sd  # lazy import: requires PortAudio

    print(f"Loading model '{args.model}'...", file=sys.stderr)
    model = load_model(args.model)
    chunk_ms = 30
    chunk_samples = SAMPLE_RATE * chunk_ms // 1000
    segmenter = SpeechSegmenter(threshold=args.threshold, chunk_ms=chunk_ms,
                                end_silence_ms=args.silence_ms)
    chunks: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):  # runs on an audio thread
        if status:
            print(status, file=sys.stderr)
        chunks.put(indata[:, 0].copy())

    output = open(args.output, "a", encoding="utf-8") if args.output else None
    print("Start speaking (Ctrl+C to exit)...", file=sys.stderr)
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=chunk_samples, callback=callback,
                            device=args.device):
            while True:
                for utterance in segmenter.process(chunks.get()):
                    start = time.time()
                    text = transcribe_audio(model, utterance)
                    if text:
                        print(f"{text}   [{duration_seconds(utterance):.1f}s of audio, "
                              f"{time.time() - start:.1f}s of compute]")
                        if output:
                            output.write(text + "\n")
                            output.flush()
    except KeyboardInterrupt:
        for utterance in segmenter.flush():
            text = transcribe_audio(model, utterance)
            if text:
                print(text)
    finally:
        if output:
            output.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
