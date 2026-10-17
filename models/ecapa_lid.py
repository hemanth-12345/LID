"""
Baseline ECAPA-TDNN-style architecture for on-device spoken language
identification (LID). This serves as the baseline required by MS1 of the
work-let before benchmarking newer architectures (RawNet, AASIST, ECAPA2)
against it.

Components:
  - SEModule: squeeze-and-excitation channel attention
  - Res2Dilated blocks: dilated TDNN layers with SE re-calibration
  - Attentive statistics pooling: attention-weighted mean + std over time
"""
import torch
import torch.nn as nn


class SEModule(nn.Module):
    """Squeeze-and-Excitation block for 1D channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        return x * y


class DilatedTDNNBlock(nn.Module):
    """Dilated 1D conv block with SE re-calibration (simplified Res2Net-TDNN)."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int,
                 se_reduction: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3,
                      dilation=dilation, padding=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            SEModule(out_channels, reduction=se_reduction),
        )

    def forward(self, x):
        return self.block(x)


class AttentiveStatsPool(nn.Module):
    """Attention-weighted mean + std pooling over the time dimension."""

    def __init__(self, in_channels: int, bottleneck: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, bottleneck, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, in_channels, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        w = self.attention(x)                                       # (B, C, T)
        mu = torch.sum(x * w, dim=2)                                 # (B, C)
        sigma = torch.sqrt(
            (torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-5)
        )
        return torch.cat([mu, sigma], dim=1)                         # (B, 2C)


class ECAPA_TDNN_LID(nn.Module):
    """
    Baseline ECAPA-TDNN-style LID network.

    Args:
        num_languages: number of target languages (classifier output size)
        input_channels: feature dimension (40 for MFCC, 80 for mel-spectrogram)
        channels: hidden channel width for the TDNN blocks
    """

    def __init__(self, num_languages: int, input_channels: int = 40,
                 channels: int = 512, se_reduction: int = 8):
        super().__init__()
        self.num_languages = num_languages

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
        )

        self.layer1 = DilatedTDNNBlock(channels, channels, dilation=2, se_reduction=se_reduction)
        self.layer2 = DilatedTDNNBlock(channels, channels, dilation=3, se_reduction=se_reduction)
        self.layer3 = DilatedTDNNBlock(channels, channels, dilation=4, se_reduction=se_reduction)

        concat_channels = channels * 3
        self.se_multiscale = SEModule(concat_channels, reduction=se_reduction)
        self.pool = AttentiveStatsPool(concat_channels)

        self.fc = nn.Linear(concat_channels * 2, num_languages)

    def forward(self, x, return_embedding: bool = False):
        # x: (B, C_in, T), e.g. (B, 40, 301) for 3s @ 16kHz MFCCs
        x = self.stem(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = torch.cat([x1, x2, x3], dim=1)          # (B, 3*channels, T)
        x = self.se_multiscale(x)

        stats = self.pool(x)                         # (B, 6*channels)
        out = self.fc(stats)

        if return_embedding:
            return out, stats
        return out


# Backwards-compatible alias for the original prototype name
TinyLID = ECAPA_TDNN_LID
