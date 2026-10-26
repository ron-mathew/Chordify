"""
BeatNet — Beat and downbeat detection transformer.

Architecture:
  3 parallel BaseTransformer branches (drum, bass, other stems).
  Weighted sum: 0.4 drums + 0.3 bass + 0.3 other.
  Linear head → 3 classes: 0=no-beat, 1=beat, 2=downbeat.

Input:  (batch, source=3, channel, time, freq) magnitude spectrograms
Output: beat predictions (batch, time), logit weight for downstream conditioning

Based on the Lamucal open-source project.
"""
import torch
import torch.nn as nn
from models.transformers import BaseTransformer


class BeatNet(nn.Module):
    def __init__(
        self,
        source: int = 3,
        n_classes: int = 3,
        weights=(0.4, 0.3, 0.3),
        n_freq: int = 2048,
        n_group: int = 32,
        f_layers: int = 2,
        f_nhead: int = 4,
        t_layers: int = 2,
        t_nhead: int = 4,
        d_layers: int = 2,
        d_nhead: int = 8,
        dropout: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.weights = weights
        self.transformer_layers = nn.ModuleList(
            BaseTransformer(
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
            for _ in range(source)
        )
        self.dropout = nn.Dropout(dropout)
        self.beat_fc = nn.Linear(n_freq, n_classes)
        self._reset_parameters(confidence=0.05)

    def _reset_parameters(self, confidence: float = 0.05) -> None:
        self.beat_fc.bias.data.fill_(
            -torch.log(torch.tensor(1 / confidence - 1))
        )

    def forward(self, inp: torch.Tensor):
        """
        inp: (batch, source, channel, time, freq)
        returns:
          beats        (batch, time) — argmax class per frame
          logit_weight (batch, time, freq) — conditioning for ChordNet
        """
        y_list, logits_weight_list = [], []
        for i, layer in enumerate(self.transformer_layers):
            x = inp[:, i, :, :, :]          # (batch, channel, time, freq)
            x, _f = layer(x)
            w = self.weights[i]
            y_list.append(x * w)
            logits_weight_list.append(_f * w)

        y = torch.sum(torch.stack(y_list, dim=0), dim=0)
        logits_weight = torch.sum(torch.stack(logits_weight_list, dim=0), dim=0)

        y = self.dropout(y)
        beats = self.beat_fc(y)
        beats = torch.argmax(beats, dim=-1)
        return beats, logits_weight


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = BeatNet()
    x = torch.randn(2, 3, 2, 64, 2048)
    beats, weight = model(x)
    print("BeatNet output:", beats.shape, weight.shape)
