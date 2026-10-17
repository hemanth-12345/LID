"""
Simplified RawNet2-style LID network.

Unlike the MFCC/TDNN pipeline, RawNet's core idea is to skip hand-crafted
features entirely and learn a front end directly from the raw waveform, via
a bank of learnable band-pass ("sinc") filters. This module implements that
front end plus residual conv blocks, a GRU for longer-range temporal
context, and attentive statistics pooling into a classifier -- a compact
version of RawNet2/RawNet3 adapted for LID rather than speaker verification.

Expects raw waveform input (B, T) or (B, 1, T), e.g. (B, 48000) for a 3s
clip at 16kHz -- use feature_type="raw" in the dataset for this model.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv1d(nn.Module):
    """Learnable band-pass sinc filterbank front end (RawNet-style)."""

    def __init__(self, out_channels: int, kernel_size: int, sample_rate: int = 16000,
                 min_low_hz: float = 50.0, min_band_hz: float = 50.0):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30.0
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        mel = np.linspace(self._to_mel(low_hz), self._to_mel(high_hz), out_channels + 1)
        hz = self._to_hz(mel)

        self.low_hz_ = nn.Parameter(torch.tensor(hz[:-1], dtype=torch.float32).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.tensor(np.diff(hz), dtype=torch.float32).view(-1, 1))

        n_lin = torch.linspace(0, kernel_size / 2 - 1, steps=int(kernel_size / 2))
        self.register_buffer("window_", 0.54 - 0.46 * torch.cos(2 * np.pi * n_lin / kernel_size))
        n = (kernel_size - 1) / 2.0
        self.register_buffer("n_", 2 * np.pi * torch.arange(-n, 0).view(1, -1) / sample_rate)

    @staticmethod
    def _to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def _to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, x):
        # x: (B, 1, T)
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                            self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) /
                           (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None] + 1e-8)

        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)


class ResBlock(nn.Module):
    """Pre-activation residual block with max-pool downsampling."""

    def __init__(self, in_channels: int, out_channels: int, pool: int = 3):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.downsample = (nn.Conv1d(in_channels, out_channels, kernel_size=1)
                            if in_channels != out_channels else None)
        self.pool = nn.MaxPool1d(pool)
        self.act = nn.LeakyReLU(0.3)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(self.act(self.bn1(x)))
        out = self.conv2(self.act(self.bn2(out)))
        return self.pool(out + residual)


class AttentivePool1d(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, bottleneck, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        w = self.attn(x)
        mu = torch.sum(x * w, dim=2)
        sigma = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-5))
        return torch.cat([mu, sigma], dim=1)


class RawNetLID(nn.Module):
    def __init__(self, num_languages: int, sample_rate: int = 16000, sinc_channels: int = 64):
        super().__init__()
        self.sinc = SincConv1d(sinc_channels, kernel_size=251, sample_rate=sample_rate)
        self.bn0 = nn.BatchNorm1d(sinc_channels)
        self.act0 = nn.LeakyReLU(0.3)
        self.maxpool0 = nn.MaxPool1d(3)

        self.block1 = ResBlock(sinc_channels, 128, pool=3)
        self.block2 = ResBlock(128, 128, pool=3)
        self.block3 = ResBlock(128, 256, pool=3)
        self.block4 = ResBlock(256, 256, pool=3)

        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
        self.pool = AttentivePool1d(256)
        self.fc = nn.Linear(256 * 2, num_languages)

    def forward(self, x, return_embedding: bool = False):
        # x: (B, T) or (B, 1, T) raw waveform
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.maxpool0(self.act0(self.bn0(self.sinc(x))))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)  # (B, C, T')

        gru_out, _ = self.gru(x.transpose(1, 2))
        gru_out = gru_out.transpose(1, 2)

        stats = self.pool(gru_out)
        out = self.fc(stats)
        if return_embedding:
            return out, stats
        return out
