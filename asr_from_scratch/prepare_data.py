"""Download LibriSpeech from the Hugging Face Hub and build the manifests.

The corpus is stored as Parquet shards that embed the original 16 kHz FLAC files.
Shards are fetched in parallel, unpacked without re-encoding and deleted right
away, so peak disk usage stays close to the size of the final data.

    python -m asr_from_scratch.prepare_data --output data \
        --splits train.100 train.360 validation test

Available splits: train.100 (100 h), train.360 (360 h), validation and test
(about 5 h each). ``--max-hours`` caps every split, which is useful for quick tests.
"""
from __future__ import annotations

import argparse
import io
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

from .data import write_manifest
from .features import SAMPLE_RATE

REPOSITORY = "openslr/librispeech_asr"


class _AudioBudget:
    """Thread-safe count of the audio extracted so far from a split."""

    def __init__(self, seconds: float) -> None:
        self.limit, self.total = seconds, 0.0
        self._lock = threading.Lock()

    def take(self, seconds: float) -> bool:
        with self._lock:
            if self.limit and self.total >= self.limit:
                return False
            self.total += seconds
            return True


def _extract_shard(shard: str, output: Path, budget: _AudioBudget) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    if budget.limit and budget.total >= budget.limit:
        return []
    path = Path(hf_hub_download(REPOSITORY, shard, repo_type="dataset",
                                local_dir=output / ".shards"))
    rows = []
    try:
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=256, columns=["audio", "text", "id"]):
                audio_column = batch.column("audio").to_pylist()
                for audio, text, clip_id in zip(audio_column, batch.column("text").to_pylist(),
                                                batch.column("id").to_pylist()):
                    info = sf.info(io.BytesIO(audio["bytes"]))
                    if info.samplerate != SAMPLE_RATE:
                        raise ValueError(f"{clip_id}: expected {SAMPLE_RATE} Hz, got {info.samplerate} Hz")
                    if not budget.take(info.duration):
                        return rows
                    name = f"audio/{clip_id}.flac"
                    (output / name).write_bytes(audio["bytes"])
                    rows.append({"audio": name, "text": text, "duration": info.duration})
        return rows
    finally:
        path.unlink(missing_ok=True)


def prepare_split(split: str, output: Path, max_hours: float, workers: int) -> None:
    from huggingface_hub import HfApi

    prefix = f"clean/{split}/"
    shards = sorted(name for name in HfApi().list_repo_files(REPOSITORY, repo_type="dataset")
                    if name.startswith(prefix) and name.endswith(".parquet"))
    if not shards:
        raise ValueError(f"No shards found for split '{split}' in {REPOSITORY}")

    directory = output / split
    (directory / "audio").mkdir(parents=True, exist_ok=True)
    budget = _AudioBudget(max_hours * 3600)
    rows: list[dict] = []
    with ThreadPoolExecutor(workers) as pool:
        for finished, shard_rows in enumerate(
            pool.map(lambda shard: _extract_shard(shard, directory, budget), shards), 1
        ):
            rows += shard_rows
            print(f"{split}: shard {finished}/{len(shards)}, {len(rows)} clips, "
                  f"{budget.total / 3600:.1f} h", flush=True)
    shutil.rmtree(directory / ".shards", ignore_errors=True)
    write_manifest(directory / "manifest.jsonl", rows)
    print(f"{split}: done, {len(rows)} clips in {directory}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download LibriSpeech and build manifests.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+",
                        default=["train.100", "train.360", "validation", "test"])
    parser.add_argument("--max-hours", type=float, default=0.0, help="Cap per split (0 = no cap)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    for split in args.splits:
        prepare_split(split, args.output, args.max_hours, args.workers)


if __name__ == "__main__":
    main()
