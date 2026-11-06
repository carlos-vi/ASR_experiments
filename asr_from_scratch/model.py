"""Conformer acoustic model trained with CTC.

The Conformer (Gulati et al., 2020) combines self-attention, which captures
long-range context, with convolution, which captures local patterns such as the
transitions between phonemes. A block is arranged as:

    half feed-forward -> self-attention -> convolution -> half feed-forward

The model owns its feature extraction, so it takes raw waveforms as input.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .features import N_MELS, LogMel, SpecAugment, length_mask
from .vocabulary import VOCAB_SIZE


def subsampled_lengths(lengths: torch.Tensor) -> torch.Tensor:
    """Two convolutions (kernel 3, stride 2, no padding) divide the length by about 4."""
    for _ in range(2):
        lengths = (lengths - 3) // 2 + 1
    return lengths


def sinusoids(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encoding of a 1D tensor of (possibly negative) positions."""
    scales = torch.exp(torch.arange(0, dim, 2, device=positions.device)
                       * (-math.log(10_000.0) / dim))
    angles = positions[:, None].float() * scales
    encoding = torch.zeros(positions.size(0), dim, device=positions.device)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles)
    return encoding


class RelativeSelfAttention(nn.Module):
    """Multi-head self-attention with relative positional encoding.

    Attention scores depend on the distance between two frames rather than on
    their absolute positions, which generalizes better across utterance lengths.
    """

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, d_model // heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(d_model)
        self.query_key_value = nn.Linear(d_model, 3 * d_model)
        self.position = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        batch, frames, d_model = x.shape
        q, k, v = self.query_key_value(self.norm(x)).chunk(3, dim=-1)
        q, k, v = (t.view(batch, frames, self.heads, self.head_dim).transpose(1, 2)
                   for t in (q, k, v))

        # One learned-projection vector per distance in [-(T-1), T-1].
        offsets = torch.arange(-(frames - 1), frames, device=x.device)
        relative = self.position(sinusoids(offsets, d_model))
        relative = relative.view(2 * frames - 1, self.heads, self.head_dim).permute(1, 2, 0)
        scores = q @ relative  # (batch, heads, frames, 2 * frames - 1)
        positions = torch.arange(frames, device=x.device)
        index = positions[:, None] - positions[None, :] + frames - 1
        bias = scores.gather(3, index.expand(batch, self.heads, frames, frames))
        bias = bias / math.sqrt(self.head_dim)
        bias = bias.masked_fill(padding_mask[:, None, None, :], float("-inf"))

        # The float mask is added to the content scores before the softmax.
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0
        )
        attended = attended.transpose(1, 2).reshape(batch, frames, d_model)
        return self.output_dropout(self.output(attended))


class ConvolutionModule(nn.Module):
    """Gated depthwise convolution that models local temporal patterns."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.expand = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size,
                                   padding=kernel_size // 2, groups=d_model)
        # LayerNorm rather than BatchNorm: it is unaffected by padding and needs
        # no synchronization across GPUs.
        self.conv_norm = nn.LayerNorm(d_model)
        self.project = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).masked_fill(padding_mask[:, :, None], 0.0).transpose(1, 2)
        x = self.depthwise(F.glu(self.expand(x), dim=1))
        x = F.silu(self.conv_norm(x.transpose(1, 2))).transpose(1, 2)
        return self.dropout(self.project(x)).transpose(1, 2)


class FeedForward(nn.Sequential):
    def __init__(self, d_model: int, dropout: float, expansion: int = 4) -> None:
        super().__init__(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, expansion * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * d_model, d_model),
            nn.Dropout(dropout),
        )


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.feed_forward_1 = FeedForward(d_model, dropout)
        self.attention = RelativeSelfAttention(d_model, heads, dropout)
        self.convolution = ConvolutionModule(d_model, kernel_size, dropout)
        self.feed_forward_2 = FeedForward(d_model, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.feed_forward_1(x)
        x = x + self.attention(x, padding_mask)
        x = x + self.convolution(x, padding_mask)
        x = x + 0.5 * self.feed_forward_2(x)
        return self.norm(x)


class ConformerCTC(nn.Module):
    def __init__(self, d_model: int = 144, layers: int = 12, heads: int = 4,
                 conv_kernel: int = 31, dropout: float = 0.1, n_mels: int = N_MELS) -> None:
        super().__init__()
        self.features = LogMel(n_mels)
        self.augment = SpecAugment()
        # 10 ms -> 40 ms per frame. The sequence is four times shorter, which makes
        # attention far cheaper since its cost grows with the square of the length.
        self.subsampling = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
        )
        mels_after = ((n_mels - 3) // 2 + 1 - 3) // 2 + 1
        self.projection = nn.Linear(32 * mels_after, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            ConformerBlock(d_model, heads, conv_kernel, dropout) for _ in range(layers)
        )
        self.output = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, waves: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """waves: (batch, samples). Returns log-probabilities (batch, T, vocab) and lengths."""
        with torch.no_grad(), torch.autocast(device_type=waves.device.type, enabled=False):
            x, frames = self.features(waves, lengths)
            if self.training:
                x = self.augment(x, frames)

        x = self.subsampling(x.unsqueeze(1))  # (batch, channels, T, mels)
        batch, channels, steps, mels = x.shape
        x = self.projection(x.permute(0, 2, 1, 3).reshape(batch, steps, channels * mels))
        x = self.dropout(x)
        output_lengths = subsampled_lengths(frames).clamp(min=1, max=steps)
        padding_mask = ~length_mask(output_lengths, steps)
        for block in self.blocks:
            x = block(x, padding_mask)
        return self.output(x).log_softmax(-1), output_lengths


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
