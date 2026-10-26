"""
Chordify audio I/O utilities.

read_audio  – load any audio file, resample to target rate, validate channels
write_audio – save a waveform tensor
gen_audio   – generate random noise waveform (for testing)

NOTE: Uses soundfile for WAV/FLAC I/O to avoid dependency on torchcodec
(required by torchaudio >= 2.5 for file I/O).  Falls back to torchaudio
for formats soundfile does not support.
"""
from pathlib import Path

import numpy as np
import torch


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def _resample_numpy(wav_np: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    """Resample a (samples, channels) numpy array using soxr or scipy."""
    if src_sr == tgt_sr:
        return wav_np
    try:
        import soxr
        return soxr.resample(wav_np, src_sr, tgt_sr)
    except ImportError:
        pass
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(src_sr, tgt_sr)
        return resample_poly(wav_np, tgt_sr // g, src_sr // g).astype(np.float32)
    except ImportError:
        raise ImportError(
            "[Chordify] Neither soxr nor scipy found for resampling. "
            "Run: pip install soxr"
        )


def read_audio(
    filepath: str | Path,
    sample_rate: int = 44100,
    n_channel: int = 2,
    device: str = "cpu",
) -> tuple[torch.Tensor, int]:
    """
    Load an audio file, validate channels, resample if necessary.

    Parameters
    ----------
    filepath    : path to audio file (.wav, .flac, etc.)
    sample_rate : target sample rate (Hz)
    n_channel   : expected number of channels (2 = stereo)
    device      : 'cpu' or 'cuda'

    Returns
    -------
    waveform (n_channel, samples) float32 tensor, sample_rate int
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"[Chordify] Audio file not found: {filepath}")

    # ---- Try soundfile first (fast, no extra deps for WAV/FLAC)
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        # data: (samples, channels)
    except Exception:
        # ---- Fallback: try torchaudio (needs torchcodec or legacy backend)
        try:
            import torchaudio
            wav_t, sr = torchaudio.load(str(path))          # (channels, samples)
            data = wav_t.numpy().T                           # → (samples, channels)
        except Exception as e:
            raise RuntimeError(
                f"[Chordify] Could not load audio '{path.name}': {e}\n"
                "  For .stem.mp4 files, install ffmpeg system-wide and stempeg:\n"
                "    pip install stempeg"
            )

    # Resample if needed
    if sr != sample_rate:
        data = _resample_numpy(data, sr, sample_rate)

    # Shape: (samples, channels) → (channels, samples)
    waveform = torch.from_numpy(data.T.copy()).float()

    if waveform.ndim != 2:
        raise ValueError(
            f"[Chordify] Expected 2-D waveform (channel, samples), got {waveform.shape}"
        )

    # Mono → stereo
    if waveform.shape[0] == 1 and n_channel == 2:
        waveform = waveform.repeat(2, 1)

    if waveform.shape[0] != n_channel:
        raise ValueError(
            f"[Chordify] Expected {n_channel}-channel audio, got {waveform.shape[0]}."
        )

    return waveform.to(device), sample_rate


def write_audio(
    path: str | Path,
    waveform: torch.Tensor,
    sample_rate: int = 44100,
) -> None:
    """Save a (channels, samples) waveform tensor to a WAV file."""
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = waveform.cpu().numpy().T   # (samples, channels)
    sf.write(str(path), data, sample_rate)


def gen_audio(
    sample_rate: int = 44100,
    n_channel: int = 2,
    duration: int = 5,
    device: str = "cpu",
) -> tuple[torch.Tensor, int]:
    """Generate a random noise waveform — used only for unit tests."""
    waveform = torch.randn(n_channel, sample_rate * duration).to(device)
    return waveform, sample_rate
