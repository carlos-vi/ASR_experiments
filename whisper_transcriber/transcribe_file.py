"""Transcribe an English audio file.

Usage:  python -m whisper_transcriber.transcribe_file audio.wav [--model small.en]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .audio import load_audio
from .model import AVAILABLE_MODELS, load_model, transcribe_audio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe an English audio file.")
    parser.add_argument("audio", type=Path, help="Audio file (wav, flac, ogg...)")
    parser.add_argument("--model", default="small.en", choices=AVAILABLE_MODELS)
    parser.add_argument("--output", type=Path, help="Save the transcript to this file")
    args = parser.parse_args(argv)

    audio = load_audio(args.audio)
    model = load_model(args.model)
    text = transcribe_audio(model, audio, use_vad=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
