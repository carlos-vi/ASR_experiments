"""Whisper model loading (faster-whisper), tuned for CPUs with limited RAM."""
from __future__ import annotations

import numpy as np

from .segmenter import SAMPLE_RATE

# The ".en" checkpoints are English-only and more accurate than their multilingual
# counterparts of the same size.
AVAILABLE_MODELS = ("tiny.en", "base.en", "small.en", "medium.en")


def load_model(name: str = "small.en", device: str = "cpu", compute_type: str = "int8"):
    """Return a faster-whisper ``WhisperModel``.

    ``int8`` quantizes the weights to 8-bit integers: the model needs about four
    times less memory than in float32 and runs faster on CPU, with a small loss
    of quality. This is what makes it practical on a laptop without a GPU.
    """
    from faster_whisper import WhisperModel  # lazy import: it is a heavy dependency

    return WhisperModel(name, device=device, compute_type=compute_type)


def transcribe_audio(model, audio: np.ndarray, language: str = "en", use_vad: bool = False) -> str:
    """Transcribe a 16 kHz mono float32 array and return the text."""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    segments, _info = model.transcribe(
        audio,
        language=language,
        beam_size=5,
        vad_filter=use_vad,
        condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def duration_seconds(audio: np.ndarray) -> float:
    return len(audio) / SAMPLE_RATE
