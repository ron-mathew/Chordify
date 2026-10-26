"""
Chordify spectrogram utilities.

Wrappers around torch.stft / torch.istft with Hann window.
STFT output convention: (channel, time, freq)
"""
import torch


def stft(
    x: torch.Tensor,
    n_fft: int = 4096,
    hop_length: int = 1024,
    pad: bool = True,
) -> torch.Tensor:
    """
    Compute STFT of a waveform.

    Parameters
    ----------
    x          : (channel, samples) or (batch*channel, samples)
    n_fft      : FFT size
    hop_length : hop in samples
    pad        : whether to centre-pad (reflect) before STFT

    Returns
    -------
    complex tensor (channel, time, freq) with freq = n_fft // 2 + 1
    """
    z = torch.stft(
        x,
        n_fft,
        hop_length or n_fft // 4,
        window=torch.hann_window(n_fft).to(device=x.device),
        win_length=n_fft,
        normalized=True,
        center=pad,
        return_complex=True,
        pad_mode="reflect",
    )
    # torch.stft returns (channel, freq, time) → transpose to (channel, time, freq)
    z = torch.transpose(z, 1, 2)
    return z


def istft(
    stft_feature: torch.Tensor,
    n_fft: int = 4096,
    hop_length: int = 1024,
    pad: bool = True,
) -> torch.Tensor:
    """
    Inverse STFT.

    Parameters
    ----------
    stft_feature : (channel, time, freq) complex tensor

    Returns
    -------
    waveform (channel, samples)
    """
    stft_feature = torch.transpose(stft_feature, 1, 2)  # → (channel, freq, time)
    waveform = torch.istft(
        stft_feature,
        n_fft,
        hop_length,
        window=torch.hann_window(n_fft).to(device=stft_feature.device),
        win_length=n_fft,
        normalized=True,
        center=pad,
    )
    return waveform


def get_spec(waveform: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Single waveform → STFT. cfg must have n_fft, hop_length, pad."""
    return stft(waveform, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"], pad=cfg["pad"])


def get_specs(waveforms: torch.Tensor, cfg: dict) -> torch.Tensor:
    """
    Batch of waveforms (sources, channel, samples) → (sources, channel, time, freq).
    """
    S, C, T = waveforms.shape
    _wav = waveforms.view(S * C, T)
    specs = stft(_wav, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"], pad=cfg["pad"])
    return specs.view(S, C, specs.shape[-2], specs.shape[-1])


def get_mixed_spec(waveforms: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Sum sources then compute STFT. Returns (channel, time, freq)."""
    mixed = torch.sum(waveforms, dim=0)
    return stft(mixed, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"], pad=cfg["pad"])


def build_masked_stft(
    masks: torch.Tensor,
    stft_feature: torch.Tensor,
    n_fft: int = 4096,
) -> list:
    """
    Apply soft masks to complex STFT to produce per-source STFTs.

    Parameters
    ----------
    masks       : (sources, channel, time, freq_half) — output of UNets
    stft_feature: (channel, time, freq_full) complex STFT of the mix

    Returns
    -------
    List of (channel, time, freq_full) complex tensors, one per source.
    """
    import torch.nn.functional as F
    out = []
    for i in range(masks.shape[0]):
        mask = masks[i]                              # (channel, time, freq_half)
        pad_num = n_fft // 2 + 1 - mask.size(-1)
        mask = F.pad(mask, (0, pad_num, 0, 0, 0, 0))
        inst_stft = mask.type(stft_feature.dtype) * stft_feature
        out.append(inst_stft)
    return out
