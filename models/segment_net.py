"""
SegmentNet — Music structure segmentation.

Architecture:
  SegmentEmbeddings: 4-layer conv/pool stack that reduces the freq axis by /4.
  BaseTransformer conditioned on ChordNet logits.
  Linear head → 10 structural segment classes (argmax).

Segment label space (10 classes):
  start, end, intro, outro, verse, chorus, solo, break, bridge, inst

Input:  (batch, channel, time, freq) — magnitude spectrogram of full mix
Output: segment class indices (batch, time)

Based on the Lamucal open-source project.
"""
import torch
import torch.nn as nn
from models.transformers import BaseTransformer


class SegmentEmbeddings(nn.Module):
    """4-layer conv/pool stack for spectral dimensionality reduction (freq ÷ 4)."""

    def __init__(
        self,
        n_channel: int = 2,
        n_hidden: int = 128,
        d_model: int = 2048,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.act_fn = nn.ReLU()

        self.conv1 = nn.Conv2d(n_channel, n_hidden // 2, kernel_size=(1, 1))
        self.pool0 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(n_hidden // 2, n_hidden, kernel_size=(1, 1))
        self.drop2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(n_hidden, n_hidden // 2, kernel_size=(1, 1))
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop3 = nn.Dropout(dropout)

        self.conv4 = nn.Conv2d(n_hidden // 2, n_channel, kernel_size=(1, 1))
        self.drop4 = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(d_model // 4)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channel, time, freq)
        x = self.conv1(x)
        x = self.pool0(x)
        x = self.act_fn(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.act_fn(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.act_fn(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.act_fn(x)
        x = self.drop4(x)

        x = self.norm(x)
        x = self.drop(x)
        return x


class SegmentNet(nn.Module):
    def __init__(
        self,
        n_freq: int = 2048,
        n_channel: int = 2,
        n_classes: int = 10,
        emb_hidden: int = 128,
        n_group: int = 32,
        f_layers: int = 2,
        f_nhead: int = 8,
        t_layers: int = 2,
        t_nhead: int = 8,
        d_layers: int = 2,
        d_nhead: int = 8,
        dropout: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__()
        divisor = 4
        d_model = n_freq // divisor

        self.embeddings = SegmentEmbeddings(
            n_channel=n_channel, n_hidden=emb_hidden, d_model=n_freq, dropout=dropout
        )
        self.wfc = nn.Linear(n_freq, d_model)
        self.transformer = BaseTransformer(
            n_channel=n_channel,
            n_freq=d_model,
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
        self.norm = nn.LayerNorm(d_model)
        self.segment_classifier = nn.Linear(d_model, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (batch, channel, time, freq)
        if weight is None:
            B, C, T, F = x.shape
            weight = torch.ones(B, T, F, device=x.device)
        x = self.embeddings(x)
        weight = self.wfc(weight)
        x, _ = self.transformer(x, weight=weight)
        x = self.norm(x)
        x = self.segment_classifier(x)
        x = torch.argmax(x, dim=-1)
        return x


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    net = SegmentNet(n_freq=2048)
    x = torch.randn(1, 2, 64, 2048)
    y = net(x)
    print("SegmentNet output:", y.shape)
