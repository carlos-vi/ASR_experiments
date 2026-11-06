"""Synthetic data to verify the whole pipeline without downloading anything.

This is NOT human speech. Each letter is "pronounced" as a tone with its own
frequency. It is only a smoke test: if the model, the CTC loss and the decoder
are correct, the model learns to read these tones within minutes, even on a
CPU. Real speech is a far harder problem.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .data import write_manifest
from .features import SAMPLE_RATE
from .vocabulary import CHARACTERS

ALPHABET = "abcdeilmnorstu "  # small subset so that it converges quickly


def _tone(char: str, rng: np.random.Generator) -> np.ndarray:
    duration = rng.uniform(0.10, 0.16)
    t = np.arange(int(duration * SAMPLE_RATE)) / SAMPLE_RATE
    if char == " ":
        return np.zeros_like(t)
    frequency = 250 + 90 * CHARACTERS.index(char)
    return 0.4 * np.hanning(len(t)) * np.sin(2 * np.pi * frequency * t)


def generate(directory: str | Path, count: int, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(directory)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(count):
        length = int(rng.integers(3, 9))
        text = "".join(rng.choice(list(ALPHABET), size=length)).strip() or "a"
        text = " ".join(text.split())
        pieces = []
        for char in text:
            pieces += [_tone(char, rng), np.zeros(int(0.03 * SAMPLE_RATE))]
        wave = np.concatenate(pieces) + rng.normal(0, 0.01, sum(map(len, pieces)))
        name = f"audio/{i:05d}.wav"
        sf.write(root / name, wave.astype(np.float32), SAMPLE_RATE)
        rows.append({"audio": name, "text": text, "duration": len(wave) / SAMPLE_RATE})
    manifest = root / "manifest.jsonl"
    write_manifest(manifest, rows)
    return manifest
