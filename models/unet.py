"""
U-Net source separation model.

UNet:  Single-source encoder-decoder with 6 conv/deconv levels, ELU
       activations, BatchNorm and symmetric skip connections.
UNets: N parallel UNet branches (one per source).  Applies channel-wise
       softmax so the masks across sources sum to 1.0 — energy-conserving.

Architecture:
  Input  → (batch, channel, time, freq)  magnitude spectrogram
  Output → (batch, sources, channel, time, freq) soft masks

Based on the Lamucal open-source project.
Original authors retain copyright of the original architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class UPad(nn.Module):
    """Asymmetric zero-padding before a strided convolution."""

    def __init__(self, padding_setting=(1, 2, 1, 2)):
        super().__init__()
        self.padding_setting = padding_setting

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, self.padding_setting, "constant", 0)


class UTransposedPad(nn.Module):
    """Removes the border pixels added by the transposed convolution."""

    def __init__(self, padding_setting=(1, 2, 1, 2)):
        super().__init__()
        self.padding_setting = padding_setting

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l, r, t, b = self.padding_setting
        return x[:, :, l:-r, t:-b]


def get_activation(name: str) -> nn.Module:
    mapping = {
        "ReLU": nn.ReLU(),
        "ELU": nn.ELU(),
    }
    return mapping.get(name, nn.LeakyReLU(0.2))


# ---------------------------------------------------------------------------
# Single-source U-Net
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(
        self,
        n_channel: int = 2,
        conv_n_filters=(16, 32, 64, 128, 256, 512),
        down_activation: str = "ELU",
        up_activation: str = "ELU",
        down_dropouts=None,
        up_dropouts=None,
    ):
        super().__init__()
        conv_num = len(conv_n_filters)
        down_act = get_activation(down_activation)
        up_act = get_activation(up_activation)
        down_dropouts = [0] * conv_num if down_dropouts is None else down_dropouts
        up_dropouts = [0] * conv_num if up_dropouts is None else up_dropouts

        # Encoder path
        self.down_layers = nn.ModuleList()
        for i in range(conv_num):
            in_ch = n_channel if i == 0 else conv_n_filters[i - 1]
            out_ch = conv_n_filters[i]
            layers = [
                UPad(),
                nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=2, padding=0),
                nn.BatchNorm2d(out_ch, track_running_stats=True, eps=1e-3, momentum=0.01),
                down_act,
            ]
            if down_dropouts[i] > 0:
                layers.append(nn.Dropout(down_dropouts[i]))
            self.down_layers.append(nn.Sequential(*layers))

        # Decoder path
        self.up_layers = nn.ModuleList()
        for i in range(conv_num - 1, -1, -1):
            in_ch = conv_n_filters[conv_num - 1] if i == conv_num - 1 else conv_n_filters[i + 1]
            out_ch = 1 if i == 0 else conv_n_filters[i - 1]
            layers = [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=5, stride=2, padding=0),
                UTransposedPad(),
                up_act,
                nn.BatchNorm2d(out_ch, track_running_stats=True, eps=1e-3, momentum=0.01),
            ]
            if up_dropouts[i] > 0:
                layers.append(nn.Dropout(up_dropouts[i]))
            self.up_layers.append(nn.Sequential(*layers))

        self.last_layer = nn.Conv2d(1, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_convs = []
        for layer in self.down_layers:
            x = layer(x)
            d_convs.append(x)

        n = len(self.up_layers)
        for i, layer in enumerate(self.up_layers):
            if i == 0:
                x = layer(x)
            else:
                x1 = d_convs[n - i - 1]
                x = torch.cat([x, x1], dim=1)
                x = layer(x)

        x = self.last_layer(x)
        return x


# ---------------------------------------------------------------------------
# Multi-source U-Net ensemble
# ---------------------------------------------------------------------------
class UNets(nn.Module):
    """
    N parallel UNet branches, one per source.
    Applies softmax across the source dimension so masks sum to 1.
    """

    def __init__(
        self,
        sources: int,
        n_channel: int = 2,
        conv_n_filters=(16, 32, 64, 128, 256, 512),
        down_activation: str = "ELU",
        up_activation: str = "ELU",
        down_dropouts=None,
        up_dropouts=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.unet_layers = nn.ModuleList(
            UNet(
                n_channel=n_channel,
                conv_n_filters=conv_n_filters,
                down_activation=down_activation,
                up_activation=up_activation,
                down_dropouts=down_dropouts,
                up_dropouts=up_dropouts,
            )
            for _ in range(sources)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channel, time, freq)
        masks = torch.stack([layer(x) for layer in self.unet_layers], dim=1)
        # masks: (batch, sources, channel, time, freq)
        masks = F.softmax(masks, dim=1)
        return masks


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    net = UNets(sources=4, n_channel=2)
    x = torch.randn(1, 2, 64, 2049)
    y = net(x)
    print("UNets output shape:", y.shape)  # (1, 4, 2, 64, 2049)
    print("Mask sum (should ≈ 1):", y.sum(dim=1).mean().item())
