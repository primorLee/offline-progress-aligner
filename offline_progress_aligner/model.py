"""The trained 512-D relation-token temporal adapter (unchanged tensor layout)."""
import torch
from torch import nn
from torch.nn import functional as F


class RelationTemporalAdapter(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.pair = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, dim), nn.GELU())
        self.score = nn.Linear(dim, 1)
        self.temporal = nn.Conv1d(dim, dim, 3)
        self.readout = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim),
                                     nn.GELU(), nn.Linear(dim, dim))

    def forward(self, pairs, valid):
        # No object IDs, source frame index or normalized timestamp enters here.
        x = self.pair(pairs)
        scores = self.score(x).squeeze(-1).masked_fill(~valid, -1e4)
        pooled = (scores.softmax(-1).unsqueeze(-1) * x).sum(-2)
        previous = torch.cat((pooled[:, :1], pooled[:, :-1]), dim=1)
        motion = pooled - previous
        history = self.temporal(F.pad(pooled.transpose(1, 2), (2, 0), mode="replicate")).transpose(1, 2)
        return F.normalize(self.readout(torch.cat((pooled, motion, history), dim=-1)), dim=-1)
