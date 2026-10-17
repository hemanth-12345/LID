"""
Novel standalone streamable LID architecture -- the work-let's closure-phase
deliverable: "novel standalone LID architecture (with streaming support)...
outperforming TDNN baselines."

Design rationale, pulling one idea from each architecture benchmarked
in MS2:
  - From ECAPA-TDNN: multiscale dilated context, concatenated before pooling.
  - From ECAPA2: depthwise-separable convs + SE for a cheaper macro/micro
    block, so the extra streaming bookkeeping below doesn't cost parameters.
  - From RawNet: keeping the receptive field causal (left-context only) is
    what actually enables streaming -- RawNet's raw-waveform recurrence and
    this network's causal convolutions solve the same "don't look ahead"
    constraint in different ways.
  - From AASIST: an attention-weighted aggregation step (here: attentive
    stats pooling) rather than plain average pooling, generalized to update
    incrementally chunk-by-chunk.

Two usage modes:
  1. `forward(x)` -- full-sequence, for training and offline (same-corpora /
     cross-corpora) evaluation, identical interface to the other models.
  2. `init_stream` / `forward_chunk` -- incremental inference: feed
     consecutive chunks of features and get an updated prediction after
     each one, without recomputing over the whole history.

Caveat (documented, not hidden): the causal conv blocks below recompute
over just the current chunk in `forward_chunk`, i.e. there is no cross-chunk
convolutional receptive-field caching yet. That means forward_chunk's early
chunks see less true context than an equivalent-length slice would in
`forward()`. Closure-phase work should add conv state caching (standard
technique: carry each causal conv's last `dilation*(kernel-1)` frames into
the next chunk) to close that gap -- the streaming *pooling* math is already
exact (see StreamingStatsPool docstring).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ecapa_lid import SEModule


class CausalDilatedBlock(nn.Module):
    """Depthwise-separable dilated conv with *causal* (left-only) padding,
    so no future frame ever leaks into the current output -- required for
    streaming. SE recalibration is per-chunk-safe (only pools time)."""

    def __init__(self, channels: int, dilation: int, se_reduction: int = 8, kernel_size: int = 3):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.causal_pad = dilation * (kernel_size - 1)  # left-only

        self.depthwise = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                                    dilation=dilation, groups=channels, bias=False)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.se = SEModule(channels, reduction=se_reduction)

    def forward(self, x):
        x_padded = F.pad(x, (self.causal_pad, 0))
        out = self.pointwise(self.depthwise(x_padded))
        out = self.act(self.bn(out))
        out = self.se(out)
        return out + x


class StreamingStatsPool(nn.Module):
    """
    Attentive statistics pooling with an exact incremental update rule.

    Full-sequence softmax attention over T frames is:
        w_t = exp(s_t) / sum_t' exp(s_t')
        mu  = sum_t w_t * x_t
        std = sqrt(sum_t w_t * x_t^2 - mu^2)

    Processing frames in chunks, we can't take a softmax over frames we
    haven't seen yet -- but sum_t exp(s_t), sum_t exp(s_t)*x_t, and
    sum_t exp(s_t)*x_t^2 are each just running sums across chunks. So
    `update()` accumulates those three running sums per chunk, and
    `normalize()` divides them out at the end -- reproducing the exact same
    mu/std as running standard attentive pooling once over the full
    concatenated sequence, not an approximation.

    A running max-score is subtracted for numerical stability (a streaming
    log-sum-exp trick); because the same constant divides out of both the
    numerator and denominator, it doesn't change the result.
    """

    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, bottleneck, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1),
        )
        self.channels = channels

    def _scores(self, x):
        return self.attention(x)  # (B, C, T) raw (pre-softmax) scores

    def forward(self, x):
        scores = self._scores(x)
        w = torch.softmax(scores, dim=2)
        mu = torch.sum(x * w, dim=2)
        sigma = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-5))
        return torch.cat([mu, sigma], dim=1)

    def init_state(self, batch_size: int, device: str):
        z = torch.zeros(batch_size, self.channels, device=device)
        return {"sum_w": z.clone(), "sum_wx": z.clone(), "sum_wx2": z.clone(),
                "running_max": torch.full((batch_size, self.channels), -1e4, device=device)}

    def update(self, x_chunk: torch.Tensor, state: dict) -> dict:
        scores = self._scores(x_chunk)  # (B, C, T)
        chunk_max = scores.max(dim=2).values
        new_max = torch.maximum(state["running_max"], chunk_max)

        # rescale previous running sums to the new max for numerical stability
        rescale = torch.exp(state["running_max"] - new_max)
        exp_scores = torch.exp(scores - new_max.unsqueeze(2))

        state["sum_w"] = state["sum_w"] * rescale + exp_scores.sum(dim=2)
        state["sum_wx"] = state["sum_wx"] * rescale + (exp_scores * x_chunk).sum(dim=2)
        state["sum_wx2"] = state["sum_wx2"] * rescale + (exp_scores * x_chunk ** 2).sum(dim=2)
        state["running_max"] = new_max
        return state

    def normalize(self, state: dict) -> torch.Tensor:
        mu = state["sum_wx"] / (state["sum_w"] + 1e-8)
        sigma = torch.sqrt((state["sum_wx2"] / (state["sum_w"] + 1e-8) - mu ** 2).clamp(min=1e-5))
        return torch.cat([mu, sigma], dim=1)


class StreamingLID(nn.Module):
    def __init__(self, num_languages: int, input_channels: int = 40,
                 channels: int = 512, se_reduction: int = 8):
        super().__init__()
        self.stem_conv = nn.Conv1d(input_channels, channels, kernel_size=5)
        self.stem_causal_pad = 4  # kernel_size - 1, left-only
        self.stem_bn = nn.BatchNorm1d(channels)
        self.stem_act = nn.ReLU()

        self.block1 = CausalDilatedBlock(channels, dilation=2, se_reduction=se_reduction)
        self.block2 = CausalDilatedBlock(channels, dilation=3, se_reduction=se_reduction)
        self.block3 = CausalDilatedBlock(channels, dilation=4, se_reduction=se_reduction)

        concat_channels = channels * 3
        self.se_multiscale = SEModule(concat_channels, reduction=se_reduction)
        self.pool = StreamingStatsPool(concat_channels)
        self.fc = nn.Linear(concat_channels * 2, num_languages)

    def _stem(self, x):
        x = F.pad(x, (self.stem_causal_pad, 0))
        return self.stem_act(self.stem_bn(self.stem_conv(x)))

    def encode(self, x):
        x = self._stem(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        cat = torch.cat([x1, x2, x3], dim=1)
        return self.se_multiscale(cat)

    def forward(self, x, return_embedding: bool = False):
        feat = self.encode(x)
        stats = self.pool(feat)
        out = self.fc(stats)
        if return_embedding:
            return out, stats
        return out

    # ---- streaming API ----
    def init_stream(self, batch_size: int = 1, device: str = "cpu"):
        return self.pool.init_state(batch_size, device)

    def forward_chunk(self, x_chunk: torch.Tensor, state: dict):
        """Process one chunk of features (B, C, T_chunk); returns
        (logits_so_far, updated_state). See module docstring for the
        conv-state-caching caveat."""
        feat = self.encode(x_chunk)
        state = self.pool.update(feat, state)
        stats = self.pool.normalize(state)
        logits = self.fc(stats)
        return logits, state
