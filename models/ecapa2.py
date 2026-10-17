"""
Simplified ECAPA2-style LID network.

ECAPA2's contribution over ECAPA-TDNN is largely about efficiency: replacing
plain dilated Conv1d layers with depthwise-separable dilated convs (a
"macro" long-range context step decoupled from a "micro" pointwise/SE
channel-mixing step). Same receptive field and multiscale-concat +
attentive-pooling design as the ECAPA baseline, but noticeably fewer
parameters -- which is exactly the axis the work-let's comparison report
cares about.
"""
import torch
import torch.nn as nn

from models.ecapa_lid import SEModule, AttentiveStatsPool


class MacroMicroBlock(nn.Module):
    """Macro: depthwise dilated conv (long-range context, cheap).
    Micro: pointwise conv + SE (channel mixing/recalibration, local)."""

    def __init__(self, channels: int, dilation: int, se_reduction: int = 8):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation,
                                    padding=dilation, groups=channels, bias=False)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.se = SEModule(channels, reduction=se_reduction)

    def forward(self, x):
        out = self.pointwise(self.depthwise(x))
        out = self.act(self.bn(out))
        out = self.se(out)
        return out + x  # residual


class ECAPA2LID(nn.Module):
    def __init__(self, num_languages: int, input_channels: int = 40,
                 channels: int = 512, se_reduction: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
        )

        self.block1 = MacroMicroBlock(channels, dilation=2, se_reduction=se_reduction)
        self.block2 = MacroMicroBlock(channels, dilation=3, se_reduction=se_reduction)
        self.block3 = MacroMicroBlock(channels, dilation=4, se_reduction=se_reduction)

        concat_channels = channels * 3
        self.se_multiscale = SEModule(concat_channels, reduction=se_reduction)
        self.pool = AttentiveStatsPool(concat_channels)
        self.fc = nn.Linear(concat_channels * 2, num_languages)

    def forward(self, x, return_embedding: bool = False):
        x = self.stem(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        cat = torch.cat([x1, x2, x3], dim=1)
        cat = self.se_multiscale(cat)

        stats = self.pool(cat)
        out = self.fc(stats)
        if return_embedding:
            return out, stats
        return out
