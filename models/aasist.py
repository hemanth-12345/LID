"""
Simplified AASIST-style LID network.

AASIST's core idea is to model spectral and temporal structure as two
separate graphs of nodes (frequency bins, time frames), run attention-based
message passing on each, then fuse both graphs through a shared "master"
node that aggregates global context -- rather than pooling a single 1D
sequence like TDNN/ECAPA. This is a compact reimplementation of that
heterogeneous-graph-attention idea using standard multi-head self-attention
as the message-passing operator, adapted for LID rather than spoof
detection.

Expects MFCC/mel-spectrogram input (B, C, T), reused here as a
(frequency, time) map for the 2D conv frontend.
"""
import torch
import torch.nn as nn


class GraphAttentionLayer(nn.Module):
    """Self-attention message passing between graph nodes + a feed-forward
    update, i.e. a standard Transformer encoder layer used as the graph
    attention operator (as in AASIST's attention-based graph module)."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, nodes):
        # nodes: (B, N, D)
        attn_out, _ = self.attn(nodes, nodes, nodes)
        nodes = self.norm1(nodes + attn_out)
        nodes = self.norm2(nodes + self.ff(nodes))
        return nodes


class AASISTLid(nn.Module):
    def __init__(self, num_languages: int, input_channels: int = 40, embed_dim: int = 64):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

        self.spec_gat = GraphAttentionLayer(embed_dim)   # spectral-branch graph
        self.temp_gat = GraphAttentionLayer(embed_dim)   # temporal-branch graph
        self.fusion_gat = GraphAttentionLayer(embed_dim)  # heterogeneous fusion graph

        self.master_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.fc = nn.Linear(embed_dim, num_languages)

    def forward(self, x, return_embedding: bool = False):
        # x: (B, C, T), e.g. (B, 40, 301) MFCC, treated as a (freq, time) map
        feat = self.frontend(x.unsqueeze(1))  # (B, D, C, T)

        spec_nodes = feat.mean(dim=3).transpose(1, 2)  # pool over time  -> (B, C, D) nodes = freq bins
        temp_nodes = feat.mean(dim=2).transpose(1, 2)  # pool over freq  -> (B, T, D) nodes = time frames

        spec_nodes = self.spec_gat(spec_nodes)
        temp_nodes = self.temp_gat(temp_nodes)

        batch_size = feat.size(0)
        master = self.master_token.expand(batch_size, -1, -1)
        fused = torch.cat([master, spec_nodes, temp_nodes], dim=1)
        fused = self.fusion_gat(fused)

        master_out = fused[:, 0, :]  # master node's representation after fusion
        out = self.fc(master_out)
        if return_embedding:
            return out, master_out
        return out
