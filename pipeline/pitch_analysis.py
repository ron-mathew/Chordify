"""
Chordify vocal pitch / melody tracking pipeline stage.

PitchNet accepts the vocal spectrogram, optionally conditioned on a
Wav2Vec2 lyric matrix, and predicts per-frame pitch class (0–849).

Class → MIDI mapping:
  midi_note   = 21 + class_index // 10   (A0 = MIDI 21)
  cents_offset = (class_index % 10) * 10
"""
import torch

from utils.spectrogram import get_spec
from models import get_model


def run_pitch_analysis(
    vocal_waveform: torch.Tensor,
    lyrics_matrix,
    config: dict,
    device: str = "cpu",
) -> tuple[list[int], torch.Tensor]:
    """
    Run PitchNet on the separated vocal stem.

    Parameters
    ----------
    vocal_waveform : (2, samples) vocal stem waveform
    lyrics_matrix  : (batch, time, d_model) Wav2Vec2 features or None
    config         : full Chordify config dict
    device         : compute device

    Returns
    -------
    pitch_preds  : list[int] — per-frame pitch class index (0–849)
    pitch_logits : (batch, time, d_model) — for downstream use
    """
    spec_cfg  = config["spec"]
    pitch_cfg = config["pitch"]

    print("[Chordify] Running Pitch-Net …")
    pitch_net = get_model(
        pitch_cfg["model_name"],
        pitch_cfg["model"],
        model_path=pitch_cfg.get("model_path") or None,
        is_train=False,
        device=device,
    )

    vocal_spec = get_spec(vocal_waveform, spec_cfg)    # (2, time, freq)

    with torch.no_grad():
        n_fft = spec_cfg["n_fft"]
        pitch_features = vocal_spec[:, :, : n_fft // 2].unsqueeze(0)  # (1,2,T,F)
        pitch_features = pitch_features.to(device)
        pitch_mag = torch.abs(pitch_features)

        pitch_pred, pitch_logits = pitch_net(pitch_mag, lyrics_matrix)

    pitch_idx = pitch_pred.squeeze(0).cpu().tolist()
    print(f"[Chordify] Pitch analysis complete ({len(pitch_idx)} frames).")
    return pitch_idx, pitch_logits
