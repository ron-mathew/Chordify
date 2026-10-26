"""
ChordNet — Chord recognition transformer.

Architecture:
  Single BaseTransformer conditioned on BeatNet logits.
  Linear head → 122 chord classes (argmax).
  Also exposes logit weights for downstream SegmentNet conditioning.

Chord label space (122 classes):
  N, X, then 12 roots × 10 qualities:
  maj, min, 7, maj7, min7, 6, m6, sus2, sus4, 5

Input:  (batch, channel, time, freq) — magnitude spectrogram of accompaniment
Output: chord indices (batch, time), logit weight for SegmentNet

Based on the Lamucal open-source project.
"""
import torch
import torch.nn as nn
from models.transformers import BaseTransformer


class ChordNet(nn.Module):
    def __init__(
        self,
        n_freq: int = 2048,
        n_classes: int = 122,
        n_group: int = 32,
        f_layers: int = 5,
        f_nhead: int = 8,
        t_layers: int = 5,
        t_nhead: int = 8,
        d_layers: int = 5,
        d_nhead: int = 8,
        dropout: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.transformers = BaseTransformer(
            n_freq=n_freq,
            n_group=n_group,
            f_layers=f_layers,
            f_nhead=f_nhead,
            f_dropout=dropout,
            t_layers=t_layers,
            t_nhead=t_nhead,
            t_dropout=dropout,
            d_layers=d_layers,
            d_nhead=d_nhead,
            d_dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_freq, n_classes)

    def forward(self, x: torch.Tensor, weight: torch.Tensor | None = None):
        """
        x      : (batch, channel, time, freq)
        weight : optional beat logits (batch, time, freq)
        returns:
          chord_indices (batch, time)
          logit_weight  (batch, time, freq)
        """
        output, weight_logits = self.transformers(x, weight)
        output = self.dropout(output)
        output = self.fc(output)
        output = output.argmax(dim=-1)
        return output, weight_logits


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = ChordNet()
    x = torch.randn(2, 2, 64, 2048)
    chords, weight = model(x)
    print("ChordNet output:", chords.shape, weight.shape)
