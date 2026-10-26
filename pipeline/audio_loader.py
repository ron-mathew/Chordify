"""
Chordify audio loader.

Entry point for loading and validating audio files before they enter
the main analysis pipeline.
"""
from pathlib import Path

import torch

from utils.audio import read_audio, SUPPORTED_EXTENSIONS


def load_audio(
    filepath: str,
    sample_rate: int = 44100,
    n_channel: int = 2,
    device: str = "cpu",
) -> tuple[torch.Tensor, int, float]:
    """
    Load an audio file and return the waveform with metadata.

    Parameters
    ----------
    filepath    : path to audio file
    sample_rate : target sample rate
    n_channel   : target channel count (2 = stereo)
    device      : 'cpu' or 'cuda'

    Returns
    -------
    waveform   : (n_channel, samples) tensor
    sample_rate: int
    duration   : float (seconds)
    """
    ext = Path(filepath).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"[Chordify] Unsupported audio format '{ext}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    waveform, sr = read_audio(filepath, sample_rate=sample_rate,
                               n_channel=n_channel, device=device)
    duration = waveform.shape[-1] / sr
    print(f"[Chordify] Loaded  : {Path(filepath).name}")
    print(f"[Chordify] Duration: {duration:.2f}s  |  SR: {sr} Hz  |  Channels: {waveform.shape[0]}")
    return waveform, sr, duration
