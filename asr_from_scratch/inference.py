"""Transcribe an audio file with a trained checkpoint.

    python -m asr_from_scratch.inference checkpoints/best.pt audio.wav
    python -m asr_from_scratch.inference checkpoints/best.pt audio.wav --lm lm.arpa.gz
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .data import load_waveform
from .decoder import beam_search_decode, greedy_decode
from .language_model import ArpaLanguageModel
from .model import ConformerCTC


def load_checkpoint(path: str, device: str = "cpu") -> ConformerCTC:
    checkpoint = torch.load(path, map_location=device)
    model = ConformerCTC(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


@torch.no_grad()
def log_probabilities(model: ConformerCTC, wave: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    samples = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32))[None]
    return model(samples, torch.tensor([samples.size(1)]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Transcribe an audio file.")
    parser.add_argument("checkpoint")
    parser.add_argument("audio")
    parser.add_argument("--lm", help="ARPA language model (.arpa or .arpa.gz) for beam search")
    parser.add_argument("--alpha", type=float, default=0.7, help="Language model weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Per-word bonus")
    args = parser.parse_args(argv)

    log_probs, lengths = log_probabilities(load_checkpoint(args.checkpoint), load_waveform(args.audio))
    if args.lm:
        lm = ArpaLanguageModel(args.lm)
        print(beam_search_decode(log_probs[0].numpy(), lm, args.alpha, args.beta))
    else:
        print(greedy_decode(log_probs, lengths)[0])


if __name__ == "__main__":
    main()
