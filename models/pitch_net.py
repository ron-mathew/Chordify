"""
PitchNet — Vocal pitch / melody tracking.

Architecture:
  PitchEmbedding : 4-layer conv/pool stack, reduces freq by /16.
  PitchEncoder   : EncoderFre (0.6) + EncoderTime (0.4) weighted combination.
  Custom decoder : self-attention + lyric-conditioned cross-attention.
  Linear head    → 850 classes (85 semitones × 10 sub-semitone bins, argmax).

The lyrics_matrix conditioning input comes from Wav2Vec2 frame-level features.
When lyrics are not available (None), pitch is estimated unconditionally.

Input:  (batch, channel, time, freq) — magnitude spectrogram of vocal stem
Output: pitch indices (batch, time), logit weight

Based on the Lamucal open-source project.
"""
import torch
import torch.nn as nn
from models.transformers import EncoderFre, EncoderTime, FeedForward, positional_encoding


class PitchEmbedding(nn.Module):
    """4-layer conv/pool, freq dimension ÷ 16."""

    def __init__(
        self,
        n_channel: int,
        d_model: int,
        n_hidden: int = 32,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.act_fn = nn.ReLU()

        self.conv1 = nn.Conv2d(n_channel, n_hidden // 4, kernel_size=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(n_hidden // 4, n_hidden // 2, kernel_size=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(n_hidden // 2, n_hidden, kernel_size=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop3 = nn.Dropout(dropout)

        self.conv4 = nn.Conv2d(n_hidden, n_hidden, kernel_size=(1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop4 = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(d_model // 16)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x); x = self.pool1(x); x = self.act_fn(x); x = self.drop1(x)
        x = self.conv2(x); x = self.pool2(x); x = self.act_fn(x); x = self.drop2(x)
        x = self.conv3(x); x = self.pool3(x); x = self.act_fn(x); x = self.drop3(x)
        x = self.conv4(x); x = self.pool4(x); x = self.act_fn(x); x = self.drop4(x)
        x = self.norm(x)
        x = self.drop(x)
        return x


class PitchEncoder(nn.Module):
    """Weighted combination of frequency and time encoders."""

    def __init__(
        self,
        n_freq: int = 2048,
        n_group: int = 32,
        weights=(0.6, 0.4),
        f_layers: int = 2,
        f_nhead: int = 8,
        f_pr: float = 0.01,
        t_layers: int = 2,
        t_nhead: int = 8,
        t_pr: float = 0.01,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.weights = weights
        self.encoder_fre = EncoderFre(
            n_freq=n_freq, n_group=n_group, nhead=f_nhead,
            n_layers=f_layers, dropout=dropout, pr=f_pr,
        )
        self.encoder_time = EncoderTime(
            n_freq=n_freq, nhead=t_nhead, n_layers=t_layers, dropout=dropout, pr=t_pr,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.encoder_fre(x)
        y2 = self.encoder_time(x)
        y = y1 * self.weights[0] + y2 * self.weights[1]
        y = self.dropout(y)
        y = self.norm(y)
        return y


class PitchNet(nn.Module):
    def __init__(
        self,
        n_freq: int = 2048,
        n_channel: int = 2,
        n_classes: int = 850,
        emb_hidden: int = 16,
        wr: float = 0.3,
        pr: float = 0.02,
        n_group: int = 32,
        f_layers: int = 2,
        f_nhead: int = 8,
        t_layers: int = 2,
        t_nhead: int = 8,
        enc_weights=(0.6, 0.4),
        d_layers: int = 2,
        d_nhead: int = 8,
        dropout: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.wr = wr
        self.pr = pr

        self.embedding = PitchEmbedding(n_channel, n_freq, emb_hidden, dropout)
        # After PitchEmbedding: channels=emb_hidden, freq=n_freq//16
        d_model = (n_freq // 16) * emb_hidden

        self.encoder = PitchEncoder(
            n_freq=d_model,
            n_group=n_group,
            weights=enc_weights,
            f_layers=f_layers,
            f_nhead=f_nhead,
            t_layers=t_layers,
            t_nhead=t_nhead,
            dropout=dropout,
        )

        self.attn1_layer = nn.ModuleList()
        self.attn2_layer = nn.ModuleList()
        self.ff_layer = nn.ModuleList()
        for _ in range(d_layers):
            self.attn1_layer.append(
                nn.MultiheadAttention(d_model, d_nhead, batch_first=True)
            )
            self.attn2_layer.append(
                nn.MultiheadAttention(d_model, d_nhead, batch_first=True)
            )
            self.ff_layer.append(FeedForward(d_model, dropout=dropout))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor | None = None,
    ):
        """
        x      : (batch, channel, time, freq)
        weight : optional lyric matrix (batch, time, d_model) from Wav2Vec2
        returns:
          pitch_indices (batch, time)
          logit_weight  (batch, time, d_model)
        """
        y = self.embedding(x)

        B, C, T, F = y.shape
        y = y.permute(0, 2, 1, 3)     # (B, T, C, F)
        y = y.reshape(B, T, C * F)    # (B, T, C*F)
        y = self.encoder(y)

        if weight is None:
            weight = torch.zeros_like(y)
        y_w = y + weight * self.wr
        B, T, Fout = y_w.shape
        y_w = y_w + positional_encoding(B, T, Fout).to(y_w.device) * self.pr

        for attn1, attn2, ff in zip(self.attn1_layer, self.attn2_layer, self.ff_layer):
            y, _ = attn1(y, y, y, need_weights=False)
            y, _ = attn1(y, y_w, y_w, need_weights=False)
            y = ff(y)

        output = self.dropout(y)
        output = self.classifier(output)
        output = torch.argmax(output, dim=-1)
        return output, y


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = PitchNet()
    x = torch.randn(1, 2, 64, 2048)
    p, w = model(x, None)
    print("PitchNet output:", p.shape, w.shape)
