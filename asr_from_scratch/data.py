"""Dataset, duration-bucketed batching and collation from a JSONL manifest.

Each manifest line:  {"audio": "path/to/file.flac", "text": "hello world", "duration": 2.1}
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from .features import SAMPLE_RATE
from .vocabulary import encode, normalize


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_manifest(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_waveform(path: str | Path) -> np.ndarray:
    wave, rate = sf.read(str(path), dtype="float32", always_2d=True)
    wave = wave.mean(1)
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        divisor = gcd(rate, SAMPLE_RATE)
        wave = resample_poly(wave, SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
    return wave


class SpeechDataset(Dataset):
    """Audio clips from one or more manifests; relative paths resolve per manifest."""

    def __init__(self, manifests: str | Path | Sequence[str | Path],
                 max_seconds: float = 20.0, max_clips: int = 0) -> None:
        if isinstance(manifests, (str, Path)):
            manifests = [manifests]
        self.rows: list[dict] = []
        for manifest in manifests:
            base = Path(manifest).parent
            for row in read_manifest(manifest):
                # Very long clips waste memory; empty transcripts cannot be trained on.
                if row["duration"] <= max_seconds and normalize(row["text"]):
                    audio = Path(row["audio"])
                    self.rows.append({**row, "audio": str(audio if audio.is_absolute() else base / audio)})
        if max_clips and max_clips < len(self.rows):
            # Evenly spaced, so a subset still spans all speakers in a sorted manifest.
            step = len(self.rows) / max_clips
            self.rows = [self.rows[int(i * step)] for i in range(max_clips)]
        self.durations = [row["duration"] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        wave = torch.from_numpy(np.ascontiguousarray(load_waveform(row["audio"])))
        return wave, torch.tensor(encode(row["text"]), dtype=torch.long), normalize(row["text"])


def collate(batch):
    waves, targets, texts = zip(*batch)
    wave_lengths = torch.tensor([wave.size(0) for wave in waves])
    target_lengths = torch.tensor([target.size(0) for target in targets])
    return (pad_sequence(waves, batch_first=True), wave_lengths,
            pad_sequence(targets, batch_first=True), target_lengths, list(texts))


class DurationBatchSampler(Sampler[list[int]]):
    """Forms batches with a fixed amount of audio out of clips of similar duration.

    Grouping similar durations minimizes padding, so GPU time goes to real audio,
    and a limit in seconds keeps memory use stable whatever the clip lengths are.
    Shuffling happens at two levels (clips into pools, then batches) to preserve
    randomness. With several processes, each one receives a disjoint slice of the
    same batch list, and all of them get the same number of batches.
    """

    def __init__(self, durations: Sequence[float], max_seconds: float, rank: int = 0,
                 world_size: int = 1, pool_batches: int = 50, seed: int = 0) -> None:
        self.durations = durations
        self.max_seconds = max_seconds
        self.rank, self.world_size = rank, world_size
        self.seed = seed
        self.epoch = 0
        average = max(sum(durations) / max(1, len(durations)), 1e-3)
        self.pool_size = max(1, int(pool_batches * max_seconds / average))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.durations), generator=generator).tolist()
        batches: list[list[int]] = []
        for start in range(0, len(order), self.pool_size):
            pool = sorted(order[start:start + self.pool_size], key=self.durations.__getitem__)
            batch: list[int] = []
            seconds = 0.0
            for index in pool:
                if batch and seconds + self.durations[index] > self.max_seconds:
                    batches.append(batch)
                    batch, seconds = [], 0.0
                batch.append(index)
                seconds += self.durations[index]
            if batch:
                batches.append(batch)
        shuffled = torch.randperm(len(batches), generator=generator).tolist()
        usable = len(batches) // self.world_size * self.world_size
        return [batches[i] for i in shuffled[:usable]][self.rank :: self.world_size]

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())
