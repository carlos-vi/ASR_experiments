"""Evaluate a checkpoint on a manifest and report WER and CER.

    python -m asr_from_scratch.evaluate checkpoints/best.pt data/test/manifest.jsonl

    # Beam search with a language model, tuning its weights on the validation set
    python -m asr_from_scratch.evaluate checkpoints/best.pt data/test/manifest.jsonl \
        --lm lm.arpa.gz --tune-manifest data/validation/manifest.jsonl

    # Compare with faster-whisper on the same clips
    python -m asr_from_scratch.evaluate checkpoints/best.pt data/test/manifest.jsonl \
        --max-clips 200 --whisper small.en
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DurationBatchSampler, SpeechDataset, collate
from .decoder import beam_search_decode, greedy_decode
from .inference import load_checkpoint
from .language_model import ArpaLanguageModel
from .metrics import cer, wer
from .vocabulary import normalize

ALPHA_GRID = (0.4, 0.8, 1.2)
BETA_GRID = (0.0, 1.5, 3.0)
_worker_lm: ArpaLanguageModel | None = None


@torch.no_grad()
def collect_log_probs(model, dataset: SpeechDataset, device: str | torch.device,
                      batch_seconds: float = 300.0, workers: int = 2) -> tuple[list[np.ndarray], list[str]]:
    """Run the model over a dataset, returning per-clip log-probabilities and references."""
    was_training = model.training
    model.eval()
    device_type = torch.device(device).type
    sampler = DurationBatchSampler(dataset.durations, batch_seconds)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=workers)
    outputs, references = [], []
    for waves, lengths, _targets, _target_lengths, texts in loader:
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=device_type == "cuda"):
            log_probs, output_lengths = model(waves.to(device), lengths.to(device))
        log_probs = log_probs.float().cpu().numpy()
        outputs += [row[:n] for row, n in zip(log_probs, output_lengths.tolist())]
        references += texts
    model.train(was_training)
    return outputs, references


def greedy_transcripts(outputs: list[np.ndarray]) -> list[str]:
    return [greedy_decode(torch.from_numpy(row)[None], torch.tensor([len(row)]))[0] for row in outputs]


def _init_worker(lm_path: str) -> None:
    global _worker_lm
    if _worker_lm is None:  # already inherited when the process start method is fork
        _worker_lm = ArpaLanguageModel(lm_path)


def _decode_clip(task: tuple[np.ndarray, float, float]) -> str:
    log_probs, alpha, beta = task
    return beam_search_decode(log_probs, _worker_lm, alpha, beta)


def beam_transcripts(pool, outputs: list[np.ndarray], alpha: float, beta: float) -> list[str]:
    return pool.map(_decode_clip, [(row, alpha, beta) for row in outputs], chunksize=8)


def tune_weights(pool, outputs: list[np.ndarray], references: list[str]) -> tuple[float, float]:
    """Grid search for the language model weights that minimize WER."""
    best, best_wer = (ALPHA_GRID[0], BETA_GRID[0]), float("inf")
    for alpha, beta in itertools.product(ALPHA_GRID, BETA_GRID):
        score = wer(references, beam_transcripts(pool, outputs, alpha, beta))
        print(f"  alpha {alpha:.1f} | beta {beta:.1f} | WER {score:.4f}", flush=True)
        if score < best_wer:
            best, best_wer = (alpha, beta), score
    return best


def compare_with_whisper(dataset: SpeechDataset, name: str) -> None:
    from whisper_transcriber.audio import load_audio
    from whisper_transcriber.model import load_model, transcribe_audio

    model = load_model(name)
    hypotheses = [normalize(transcribe_audio(model, load_audio(row["audio"]))) for row in dataset.rows]
    references = [normalize(row["text"]) for row in dataset.rows]
    print(f"Whisper {name:<9} | WER {wer(references, hypotheses):.3f} | "
          f"CER {cer(references, hypotheses):.3f} | {len(dataset)} clips")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("manifest")
    parser.add_argument("--max-clips", type=int, default=0, help="Limit the clips evaluated (0 = all)")
    parser.add_argument("--lm", help="ARPA language model for beam search")
    parser.add_argument("--alpha", type=float, default=0.8, help="Language model weight")
    parser.add_argument("--beta", type=float, default=3.0, help="Per-word bonus")
    parser.add_argument("--tune-manifest", help="Manifest used to tune alpha and beta")
    parser.add_argument("--tune-clips", type=int, default=100)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--whisper", help="Whisper model to compare with (e.g. small.en)")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, device)
    dataset = SpeechDataset(args.manifest, max_seconds=40.0, max_clips=args.max_clips)
    outputs, references = collect_log_probs(model, dataset, device)
    hypotheses = greedy_transcripts(outputs)
    print(f"Greedy            | WER {wer(references, hypotheses):.3f} | "
          f"CER {cer(references, hypotheses):.3f} | {len(dataset)} clips", flush=True)

    if args.lm:
        global _worker_lm
        _worker_lm = ArpaLanguageModel(args.lm)
        with multiprocessing.Pool(args.workers, _init_worker, (args.lm,)) as pool:
            alpha, beta = args.alpha, args.beta
            if args.tune_manifest:
                tune_set = SpeechDataset(args.tune_manifest, max_seconds=40.0, max_clips=args.tune_clips)
                alpha, beta = tune_weights(pool, *collect_log_probs(model, tune_set, device))
                print(f"Best weights: alpha {alpha:.1f}, beta {beta:.1f}", flush=True)
            hypotheses = beam_transcripts(pool, outputs, alpha, beta)
        print(f"Beam search + LM  | WER {wer(references, hypotheses):.3f} | "
              f"CER {cer(references, hypotheses):.3f} | {len(dataset)} clips", flush=True)

    for reference, hypothesis in list(zip(references, hypotheses))[:5]:
        print(f"  ref: {reference}\n  hyp: {hypothesis}")

    if args.whisper:
        compare_with_whisper(dataset, args.whisper)


if __name__ == "__main__":
    main()
