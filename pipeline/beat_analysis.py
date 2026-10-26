"""
Chordify beat / tempo analysis pipeline stage.

Supports:
1. Neural BeatNet (primary):
   Accepts spectrograms from 3 stems (drums=index 2, bass=index 1, other=index 3)
   and predicts per-frame beat class (0=none, 1=beat, 2=downbeat).
2. DSP/Librosa Fallback (automatic):
   Triggered when no valid trained checkpoint is found. Uses onset strength envelope
   and dynamic programming beat tracking via librosa.beat.beat_track.
"""
from __future__ import annotations

import torch
import numpy as np

from utils.spectrogram import get_specs
from pipeline.dsp_fallback import estimate_beats_dsp
from models import get_model
from models.load import has_valid_checkpoint


def run_beat_analysis(
    inst_waveforms: torch.Tensor,
    config: dict,
    device: str = "cpu",
    sr: int = 44100,
) -> tuple[list[int], float, torch.Tensor, str]:
    """
    Run BeatNet on separated stems or DSP fallback.

    Parameters
    ----------
    inst_waveforms : (4, 2, samples) — vocal, bass, drums, other
    config         : full Chordify config dict
    device         : compute device
    sr             : sample rate (default 44100)

    Returns
    -------
    beat_preds   : list[int]  — per-frame beat class (0/1/2)
    bpm          : float
    beat_logits  : (batch, time, freq) — conditioning tensor for ChordNet
    method       : str        — "neural" or "dsp_librosa"
    """
    spec_cfg   = config["spec"]
    beat_cfg   = config["beat"]
    model_path = beat_cfg.get("model_path") or None

    n_fft = spec_cfg["n_fft"]
    hop_length = spec_cfg["hop_length"]

    # Stem ordering: vocal=0, bass=1, drums=2, other=3
    # BeatNet expects (drums, bass, other) -> indices 2, 1, 3
    stems_for_beat = inst_waveforms[[2, 1, 3]]   # (3, 2, samples)
    inst_specs = get_specs(stems_for_beat, spec_cfg)   # (3, 2, time, freq)
    n_frames = inst_specs.shape[2]

    # Check if a valid trained neural checkpoint exists
    if has_valid_checkpoint(model_path):
        print("[Chordify] BeatNet checkpoint found — using neural model")
        beat_net = get_model(
            beat_cfg["model_name"],
            beat_cfg["model"],
            model_path=model_path,
            is_train=False,
            device=device,
        )

        with torch.no_grad():
            beat_features = inst_specs[:, :, :, : n_fft // 2].unsqueeze(0)   # (1,3,2,T,F)
            beat_features = beat_features.to(device)
            beat_mag = torch.abs(beat_features)

            beat_pred, beat_logits = beat_net(beat_mag)

        beats_np = beat_pred.squeeze(0).cpu().numpy()   # (time,)
        beat_preds = beats_np.tolist()

        # Derive tempo from predicted beats
        from librosa.feature import tempo as librosa_tempo
        try:
            bpm = float(librosa_tempo(onset_envelope=beats_np.astype(float), sr=sr, hop_length=hop_length)[0])
        except Exception:
            bpm = 120.0
        method = "neural"

    else:
        print("[Chordify] BeatNet checkpoint unavailable — using DSP/Librosa fallback")
        # Sum drums + bass + other for rhythm tracking
        rhythm_wave = torch.sum(inst_waveforms[1:], dim=0)  # (2, samples)

        bpm, beat_preds, _, beat_frames = estimate_beats_dsp(
            audio=rhythm_wave,
            sr=sr,
            hop_length=hop_length,
            expected_frames=n_frames,
        )

        # Synthetic neutral beat logits tensor (1, time, 2048) for downstream conditioning
        beat_logits = torch.zeros(1, n_frames, n_fft // 2, device=device)
        method = "dsp_librosa"

    print(f"[Chordify] BPM: {bpm:.1f}  [{method}]")
    return beat_preds, bpm, beat_logits, method
