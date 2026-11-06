"""From waveform to log-Mel spectrogram, the representation the model sees.

1. Audio is a 1D signal: air pressure sampled 16,000 times per second.
2. It is cut into 25 ms windows that advance every 10 ms.
3. The Fourier transform of each window gives the energy at each frequency.
4. Frequencies are grouped into 80 bands on the Mel scale, which mimics human
   hearing (better resolution at low frequencies).
5. The logarithm compresses the dynamic range, as loudness perception does.

The result is a (time x 80) matrix, effectively an image of the sound. Both
classes operate on padded batches so they can run on the GPU, which removes the
main CPU bottleneck during training.
"""
from __future__ import annotations

import math

import torch
from torch import nn

SAMPLE_RATE = 16_000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160


def length_mask(lengths: torch.Tensor, size: int) -> torch.Tensor:
    """Boolean mask of shape (batch, size), True on valid positions."""
    return torch.arange(size, device=lengths.device)[None, :] < lengths[:, None]


def mel_filterbank(n_mels: int, n_fft: int = N_FFT, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Triangular filters evenly spaced on the Mel scale: (n_mels, n_fft // 2 + 1)."""
    top_mel = 2595.0 * math.log10(1.0 + (sample_rate / 2) / 700.0)
    edges = 700.0 * (10.0 ** (torch.linspace(0.0, top_mel, n_mels + 2) / 2595.0) - 1.0)
    lower, center, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    frequencies = torch.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)[None, :]
    rising = (frequencies - lower) / (center - lower)
    falling = (upper - frequencies) / (upper - center)
    return torch.minimum(rising, falling).clamp(min=0.0)


class LogMel(nn.Module):
    def __init__(self, n_mels: int = N_MELS) -> None:
        super().__init__()
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        self.register_buffer("filters", mel_filterbank(n_mels), persistent=False)

    def forward(self, waves: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """waves: (batch, samples) -> (batch, frames, n_mels), normalized per utterance."""
        spectrum = torch.stft(waves, N_FFT, HOP_LENGTH, window=self.window, return_complex=True)
        logmel = torch.log(self.filters @ spectrum.abs().square() + 1e-6).transpose(1, 2)
        frames = lengths // HOP_LENGTH + 1
        mask = length_mask(frames, logmel.size(1)).unsqueeze(-1)
        count = frames[:, None, None].to(logmel.dtype)
        mean = (logmel * mask).sum(1, keepdim=True) / count
        variance = (((logmel - mean) ** 2) * mask).sum(1, keepdim=True) / count
        return (logmel - mean) / (variance + 1e-5).sqrt() * mask, frames


class SpecAugment(nn.Module):
    """Data augmentation that hides random frequency bands and time spans.

    It prevents the model from relying on any specific region of the spectrogram.
    The time mask width adapts to each utterance so short clips stay intelligible.
    """

    def __init__(self, freq_masks: int = 2, freq_width: int = 20,
                 time_masks: int = 2, time_width: int = 50, time_ratio: float = 0.1) -> None:
        super().__init__()
        self.freq_masks, self.freq_width = freq_masks, freq_width
        self.time_masks, self.time_width, self.time_ratio = time_masks, time_width, time_ratio

    @staticmethod
    def _random_spans(count: int, size: int, widths: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
        """Union of ``count`` random spans per item, as a (batch, size) boolean mask."""
        positions = torch.arange(size, device=widths.device)[None, :]
        mask = torch.zeros(widths.size(0), size, dtype=torch.bool, device=widths.device)
        for _ in range(count):
            width = (torch.rand_like(widths) * (widths + 1)).floor()
            start = (torch.rand_like(widths) * (limits - width).clamp(min=0)).floor()
            mask |= (positions >= start[:, None]) & (positions < (start + width)[:, None])
        return mask

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: (batch, frames, n_mels)."""
        batch, frames, mels = x.shape
        limits = lengths.to(x.dtype)
        time_widths = (limits * self.time_ratio).clamp(max=self.time_width)
        time_mask = self._random_spans(self.time_masks, frames, time_widths, limits)
        freq_widths = torch.full((batch,), float(self.freq_width), device=x.device, dtype=x.dtype)
        freq_limits = torch.full((batch,), float(mels), device=x.device, dtype=x.dtype)
        freq_mask = self._random_spans(self.freq_masks, mels, freq_widths, freq_limits)
        return x.masked_fill(time_mask[:, :, None] | freq_mask[:, None, :], 0.0)
