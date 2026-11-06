"""Energy-based speech segmenter (a minimal voice activity detector).

VAD stands for Voice Activity Detection: deciding, for every small chunk of
audio, whether someone is speaking or there is only silence or noise. This
implementation uses the energy (RMS) of each chunk, the simplest approach that
works. Whisper performs better on complete sentences than on arbitrary slices,
so the VAD cuts the microphone stream into utterances: from the moment the
speaker starts talking until they pause.
"""
from __future__ import annotations

from collections import deque

import numpy as np

SAMPLE_RATE = 16_000  # Whisper expects 16 kHz mono audio


def rms(frame: np.ndarray) -> float:
    """Root mean square of the samples: a measure of how loud the chunk is."""
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


class SpeechSegmenter:
    """Consumes audio chunks and returns complete utterances.

    Parameters
    ----------
    threshold: RMS energy above which a chunk counts as speech.
    chunk_ms: duration of each chunk passed to ``process``.
    end_silence_ms: continuous silence that closes an utterance.
    min_speech_ms: shorter utterances are discarded (clicks, coughs).
    padding_ms: audio kept from before the first speech chunk, so the start of
        the first word is not cut off.
    max_ms: maximum utterance duration; longer ones are cut.
    """

    def __init__(
        self,
        threshold: float = 0.015,
        chunk_ms: int = 30,
        end_silence_ms: int = 700,
        min_speech_ms: int = 250,
        padding_ms: int = 300,
        max_ms: int = 25_000,
    ) -> None:
        self.threshold = threshold
        self.chunk_ms = chunk_ms
        self.end_silence_chunks = max(1, end_silence_ms // chunk_ms)
        self.min_speech_chunks = max(1, min_speech_ms // chunk_ms)
        self.max_chunks = max(1, max_ms // chunk_ms)
        self._padding: deque[np.ndarray] = deque(maxlen=max(1, padding_ms // chunk_ms))
        self._current: list[np.ndarray] = []
        self._speech_chunks = 0
        self._silent_chunks = 0
        self._speaking = False

    @property
    def speaking(self) -> bool:
        return self._speaking

    def process(self, chunk: np.ndarray) -> list[np.ndarray]:
        """Process one chunk and return any utterances it completed."""
        finished: list[np.ndarray] = []
        is_speech = rms(chunk) >= self.threshold

        if not self._speaking:
            if is_speech:
                self._speaking = True
                self._current = list(self._padding)
                self._current.append(chunk)
                self._speech_chunks = 1
                self._silent_chunks = 0
            else:
                self._padding.append(chunk)
            return finished

        self._current.append(chunk)
        if is_speech:
            self._speech_chunks += 1
            self._silent_chunks = 0
        else:
            self._silent_chunks += 1

        if self._silent_chunks >= self.end_silence_chunks or len(self._current) >= self.max_chunks:
            utterance = self._close()
            if utterance is not None:
                finished.append(utterance)
        return finished

    def flush(self) -> list[np.ndarray]:
        """Close any utterance in progress (when the recording ends)."""
        if self._speaking:
            utterance = self._close()
            if utterance is not None:
                return [utterance]
        return []

    def _close(self) -> np.ndarray | None:
        # Drop trailing silence, keeping a small margin.
        excess = max(0, self._silent_chunks - 2)
        chunks = self._current[: len(self._current) - excess] if excess else self._current
        valid = self._speech_chunks >= self.min_speech_chunks
        self._current = []
        self._speech_chunks = 0
        self._silent_chunks = 0
        self._speaking = False
        self._padding.clear()
        if not valid:
            return None
        return np.concatenate(chunks).astype(np.float32)
