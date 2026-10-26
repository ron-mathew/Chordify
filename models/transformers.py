"""
Shared factorised axial-attention transformer building blocks.

Architecture:
  EncoderFre  – frequency-axis multi-head self-attention (groups bins into
                sub-bands to reduce O((T·F)²) to O(T·G²))
  EncoderTime – time-axis multi-head self-attention over full frequency vectors
  Decoder     – fuses frequency and time encodings; optionally conditioned on
                an external logit weight tensor
  BaseTransformer – per-channel EncoderFre + EncoderTime → Decoder

Based on the Lamucal open-source project.
Original authors retain copyright of the original architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
def positional_encoding(
    batch_size: int,
    n_time: int,
    n_feature: int,
    zero_pad: bool = False,
    scale: bool = False,
    dtype=torch.float32,
) -> torch.Tensor:
    """Standard sinusoidal positional encoding."""
    pos_indices = torch.tile(
        torch.unsqueeze(torch.arange(n_time), 0), [batch_size, 1]
    )
    pos = torch.arange(n_time, dtype=dtype).reshape(-1, 1)
    pos_enc = pos / torch.pow(
        10000, 2 * torch.arange(0, n_feature, dtype=dtype) / n_feature
    )
    pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2])
    pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2])

    if zero_pad:
        pos_enc = torch.cat(
            [torch.zeros(size=[1, n_feature]), pos_enc[1:, :]], dim=0
        )

    outputs = F.embedding(pos_indices, pos_enc)
    if scale:
        outputs = outputs * (n_feature**0.5)
    return outputs


# ---------------------------------------------------------------------------
# Feed-forward sublayer
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, n_feature: int = 2048, n_hidden: int = 512, dropout: float = 0.5):
        super().__init__()
        self.linear1 = nn.Linear(n_feature, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_feature)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_layer = nn.LayerNorm(n_feature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear1(x)
        y = F.relu(y)
        y = self.dropout1(y)
        y = self.linear2(y)
        y = self.dropout2(y)
        y = self.norm_layer(y)
        return y


# ---------------------------------------------------------------------------
# Frequency encoder
# ---------------------------------------------------------------------------
class EncoderFre(nn.Module):
    """
    Factorised frequency-axis attention.
    Groups n_freq bins into n_group sub-bands of size d_model = n_freq // n_group
    and applies multi-head attention within each sub-band independently.
    """

    def __init__(
        self,
        n_freq: int,
        n_group: int,
        nhead: int = 8,
        n_layers: int = 5,
        dropout: float = 0.5,
        pr: float = 0.01,
    ):
        super().__init__()
        assert n_freq % n_group == 0, "n_freq must be divisible by n_group"
        self.d_model = d_model = n_freq // n_group
        self.n_freq = n_freq
        self.n_group = n_group
        self.pr = pr
        self.n_layers = n_layers

        self.attn_layer = nn.ModuleList()
        self.ff_layer = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layer.append(
                nn.MultiheadAttention(d_model, nhead, batch_first=True)
            )
            self.ff_layer.append(FeedForward(d_model, dropout=dropout))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_freq, n_freq)
        self.norm_layer = nn.LayerNorm(n_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_time, n_freq)
        B, T, Fr = x.shape
        x_reshape = x.reshape(B * T, self.n_group, self.d_model)
        x_reshape = x_reshape + positional_encoding(
            batch_size=x_reshape.shape[0],
            n_time=x_reshape.shape[1],
            n_feature=x_reshape.shape[2],
        ).to(x.device) * self.pr

        for attn, ff in zip(self.attn_layer, self.ff_layer):
            x_reshape, _ = attn(x_reshape, x_reshape, x_reshape, need_weights=False)
            x_reshape = ff(x_reshape)

        y = x_reshape.reshape(B, T, self.n_freq)
        y = self.dropout(y)
        y = self.fc(y)
        y = self.norm_layer(y)
        return y


# ---------------------------------------------------------------------------
# Time encoder
# ---------------------------------------------------------------------------
class EncoderTime(nn.Module):
    """Time-axis multi-head self-attention over full frequency vectors."""

    def __init__(
        self,
        n_freq: int,
        nhead: int = 8,
        n_layers: int = 5,
        dropout: float = 0.5,
        pr: float = 0.02,
    ):
        super().__init__()
        self.n_freq = n_freq
        self.n_layers = n_layers
        self.pr = pr

        self.attn_layer = nn.ModuleList()
        self.ff_layer = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layer.append(
                nn.MultiheadAttention(n_freq, nhead, batch_first=True)
            )
            self.ff_layer.append(FeedForward(n_freq, dropout=dropout))

        self.norm_layer = nn.LayerNorm(n_freq)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_freq, n_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_time, n_freq)
        B, T, Fr = x.shape
        x = x + positional_encoding(B, T, Fr).to(x.device) * self.pr

        for attn, ff in zip(self.attn_layer, self.ff_layer):
            x, _ = attn(x, x, x, need_weights=False)
            x = ff(x)

        x = self.dropout(x)
        x = self.fc(x)
        x = self.norm_layer(x)
        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class Decoder(nn.Module):
    """
    Fuses frequency and time encodings via weighted sum + optional external
    conditioning logit.  Two stacked self-attention + cross-attention blocks.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        n_layers: int = 5,
        dropout: float = 0.5,
        r1: float = 1.0,
        r2: float = 1.0,
        wr: float = 1.0,
        pr: float = 0.01,
    ):
        super().__init__()
        self.r1 = r1
        self.r2 = r2
        self.wr = wr
        self.n_layers = n_layers
        self.pr = pr

        self.attn1_layer = nn.ModuleList()
        self.attn2_layer = nn.ModuleList()
        self.ff_layer = nn.ModuleList()
        for _ in range(n_layers):
            self.attn1_layer.append(
                nn.MultiheadAttention(d_model, nhead, batch_first=True)
            )
            self.attn2_layer.append(
                nn.MultiheadAttention(d_model, nhead, batch_first=True)
            )
            self.ff_layer.append(FeedForward(d_model, dropout=dropout))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weight: torch.Tensor | None = None,
    ):
        y = x1 * self.r1 + x2 * self.r2
        if weight is not None:
            y = y + weight * self.wr

        y = y + positional_encoding(y.shape[0], y.shape[1], y.shape[2]).to(y.device) * self.pr + self.pr

        for attn1, attn2, ff in zip(self.attn1_layer, self.attn2_layer, self.ff_layer):
            y, _ = attn1(y, y, y, need_weights=False)
            y, _ = attn2(y, x2, x2, need_weights=False)
            y = ff(y)

        output = self.dropout(y)
        output = self.fc(output)
        return output, y


# ---------------------------------------------------------------------------
# BaseTransformer
# ---------------------------------------------------------------------------
class BaseTransformer(nn.Module):
    """
    Per-channel EncoderFre + EncoderTime, followed by a shared Decoder.
    Accepts optional conditioning weight for cross-modal guidance.

    Input:  (batch, channel, n_time, n_freq)
    Output: (batch, n_time, n_freq), logit_weight (batch, n_time, n_freq)
    """

    def __init__(
        self,
        n_channel: int = 2,
        n_freq: int = 2048,
        # EncoderFre params
        n_group: int = 32,
        f_layers: int = 2,
        f_nhead: int = 4,
        f_dropout: float = 0.5,
        f_pr: float = 0.01,
        # EncoderTime params
        t_layers: int = 2,
        t_nhead: int = 4,
        t_dropout: float = 0.5,
        t_pr: float = 0.01,
        # Decoder params
        d_layers: int = 2,
        d_nhead: int = 4,
        d_dropout: float = 0.5,
        d_pr: float = 0.02,
        r1: float = 1.0,
        r2: float = 1.0,
        wr: float = 0.2,
    ):
        super().__init__()
        self.n_channel = n_channel

        self.encoder_fre_layers = nn.ModuleList()
        self.encoder_time_layers = nn.ModuleList()
        for _ in range(n_channel):
            self.encoder_fre_layers.append(
                EncoderFre(
                    n_freq=n_freq, n_group=n_group, nhead=f_nhead,
                    n_layers=f_layers, dropout=f_dropout, pr=f_pr,
                )
            )
            self.encoder_time_layers.append(
                EncoderTime(
                    n_freq=n_freq, nhead=t_nhead, n_layers=t_layers,
                    dropout=t_dropout, pr=t_pr,
                )
            )

        self.decoder = Decoder(
            d_model=n_freq, nhead=d_nhead, n_layers=d_layers,
            dropout=d_dropout, r1=r1, r2=r2, wr=wr, pr=d_pr,
        )

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor | None = None,
    ):
        ff_list, tf_list = [], []
        for i in range(self.n_channel):
            x1 = self.encoder_fre_layers[i](x[:, i, :, :])
            x2 = self.encoder_time_layers[i](x[:, i, :, :])
            ff_list.append(x1)
            tf_list.append(x2)

        y1 = torch.sum(torch.stack(ff_list, dim=0), dim=0)
        y2 = torch.sum(torch.stack(tf_list, dim=0), dim=0)
        y, w = self.decoder(y1, y2, weight=weight)
        return y, w


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    net = BaseTransformer(n_freq=2048)
    x = torch.randn(1, 2, 64, 2048)
    y, logits = net(x)
    print("BaseTransformer output:", y.shape, logits.shape)
