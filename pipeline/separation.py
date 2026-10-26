"""
Chordify source separation pipeline stage.

Uses a U-Net to separate a stereo mixture into 4 stems:
  index 0 → vocal
  index 1 → bass
  index 2 → drums
  index 3 → other

Processing uses a sliding window to handle arbitrary audio lengths without
running out of memory on large spectrograms.
"""
import torch
import torch.nn.functional as F

from utils.spectrogram import get_spec, istft, build_masked_stft
from models import get_model


# Source ordering (matches MUSDB18 / Lamucal convention)
SOURCE_NAMES = ["vocal", "bass", "drums", "other"]


def separate_sources(
    waveform: torch.Tensor,
    sample_rate: int,
    config: dict,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Run U-Net source separation on a stereo waveform.

    Parameters
    ----------
    waveform    : (2, samples) stereo tensor
    sample_rate : must match config['sample_rate']
    config      : full Chordify config dict
    device      : compute device

    Returns
    -------
    inst_waveforms : (4, 2, samples) — vocal, bass, drums, other
    """
    assert sample_rate == config["audio"]["sample_rate"], (
        f"[Chordify] Sample rate mismatch: got {sample_rate}, "
        f"expected {config['audio']['sample_rate']}"
    )

    sep_cfg   = config["separate"]
    spec_cfg  = sep_cfg["spec"]
    n_fft     = spec_cfg["n_fft"]
    hop_length = spec_cfg["hop_length"]
    n_time    = spec_cfg["n_time"]
    n_sources = config["audio"]["sources"]

    model_cfg = {
        "sources": n_sources,
        "n_channel": config["audio"]["n_channel"],
    }
    model_cfg.update(sep_cfg["model"])

    print("[Chordify] Loading U-Net source separator …")
    unet = get_model(
        sep_cfg["model_name"],
        model_cfg,
        model_path=sep_cfg.get("model_path") or None,
        is_train=False,
        device=device,
    )

    wav_len = waveform.shape[-1]
    # Chunk size: (n_time - 5) frames * hop + n_fft
    split_len = (n_time - 5) * hop_length + n_fft

    output_waveforms = [[] for _ in range(n_sources)]

    for i in range(0, wav_len, split_len):
        with torch.no_grad():
            x = waveform[:, i : i + split_len]
            pad_num = 0
            if x.shape[-1] < split_len:
                pad_num = split_len - (wav_len - i)
                x = F.pad(x, (0, pad_num))

            z = get_spec(x, spec_cfg)           # (2, time, freq)
            mag_z = torch.abs(z).unsqueeze(0)   # (1, 2, time, freq)
            masks = unet(mag_z)                 # (1, 4, 2, time, freq)
            masks = masks.squeeze(0)            # (4, 2, time, freq)

            masked_stfts = build_masked_stft(masks, z, n_fft=n_fft)
            for j, ms in enumerate(masked_stfts):
                wav_j = istft(ms, n_fft=n_fft, hop_length=hop_length, pad=True)
                if pad_num > 0:
                    wav_j = wav_j[:, :-pad_num]
                output_waveforms[j].append(wav_j)

    inst_waveforms = torch.stack(
        [torch.cat(wlist, dim=-1) for wlist in output_waveforms], dim=0
    )
    print(f"[Chordify] Separation complete. Stems shape: {inst_waveforms.shape}")
    return inst_waveforms
