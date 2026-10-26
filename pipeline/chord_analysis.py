"""
Chordify chord recognition and key estimation pipeline stage.

Supports:
1. Neural ChordNet (primary):
   Accepts accompaniment (bass+drums+other) spectrogram and BeatNet beat logits
   as cross-modal conditioning, producing per-frame chord class indices (0–121).
2. DSP/Chroma Fallback (automatic):
   Triggered when no valid trained checkpoint is found. Uses beat-synchronous
   dual-band chroma CQT, 24-triad temperature-scaled Viterbi HMM decoding,
   Krumhansl-Schmuckler key profiling, and key-aware enharmonic spelling.
"""
from __future__ import annotations

import numpy as np
import torch

from utils.spectrogram import get_mixed_spec
from utils.labels import CHORD_LABELS, canonicalize_chord_label
from pipeline.modulation import search_key
from pipeline.dsp_fallback import estimate_chords_dsp
from models import get_model
from models.load import has_valid_checkpoint


def idx_to_chord_label(idx: int) -> str:
    if 0 <= idx < len(CHORD_LABELS):
        return CHORD_LABELS[idx]
    return "N"


def run_chord_analysis(
    inst_waveforms: torch.Tensor,
    beat_logits: torch.Tensor | None,
    config: dict,
    device: str = "cpu",
    sr: int = 44100,
    beat_preds: list[int] | np.ndarray | None = None,
) -> tuple[list[str], str, int, torch.Tensor, str, list[dict] | None]:
    """
    Run ChordNet or DSP fallback.

    Parameters
    ----------
    inst_waveforms : (4, 2, samples) — vocal, bass, drums, other
    beat_logits    : (batch, time, freq) conditioning from BeatNet (or None)
    config         : full Chordify config dict
    device         : compute device
    sr             : sample rate (default 44100)
    beat_preds     : optional per-frame beat classes for beat-synchronous decoding

    Returns
    -------
    chord_labels  : list[str]  — per-frame chord label
    song_key      : str
    key_score     : int        — diatonic fit score
    chord_logits  : (batch, time, freq) — conditioning for SegmentNet
    method        : str        — "neural" or "dsp_chroma"
    chord_events  : list[dict] — optional structured chord segments
    """
    spec_cfg  = config["spec"]
    chord_cfg = config["chord"]
    model_path = chord_cfg.get("model_path") or None

    n_fft = spec_cfg["n_fft"]
    hop_length = spec_cfg["hop_length"]

    # Accompaniment: bass(1) + drums(2) + other(3)
    accomp_spec = get_mixed_spec(inst_waveforms[1:], spec_cfg)   # (2, time, freq)
    n_frames = accomp_spec.shape[1]
    chord_events = None

    # Check if a valid trained neural checkpoint exists
    if has_valid_checkpoint(model_path):
        print("[Chordify] ChordNet checkpoint found — using neural model")
        chord_net = get_model(
            chord_cfg["model_name"],
            chord_cfg["model"],
            model_path=model_path,
            is_train=False,
            device=device,
        )

        with torch.no_grad():
            chord_features = accomp_spec[:, :, : n_fft // 2].unsqueeze(0)  # (1,2,T,F)
            chord_features = chord_features.to(device)
            chord_mag = torch.abs(chord_features)

            if beat_logits is None:
                beat_logits = torch.zeros(1, n_frames, n_fft // 2, device=device)

            chord_pred, chord_logits = chord_net(chord_mag, beat_logits)

        chord_idx = chord_pred.squeeze(0).cpu().tolist()
        chord_labels = [idx_to_chord_label(int(i)) for i in chord_idx]

        # Strip "X" and "N" when computing key
        filtered_for_key = [c.split(":")[0] for c in chord_labels if c not in ("N", "X")]
        song_key, key_score = search_key(filtered_for_key)
        method = "neural"

    else:
        print("[Chordify] ChordNet checkpoint unavailable — using DSP/Chroma fallback")
        # Sum accompaniment stems: bass(1) + drums(2) + other(3)
        accomp_wave = torch.sum(inst_waveforms[1:], dim=0)  # (2, samples)

        # Extract beat frame positions if provided
        beat_indices = None
        if beat_preds is not None:
            if isinstance(beat_preds, torch.Tensor):
                beat_preds = beat_preds.cpu().numpy()
            b_arr = np.asarray(beat_preds)
            nz = np.where(b_arr > 0)[0]
            if len(nz) > 2:
                beat_indices = nz

        frame_labels, song_key, key_score_float, chord_events = estimate_chords_dsp(
            audio=accomp_wave,
            sr=sr,
            hop_length=hop_length,
            beats=beat_indices,
            expected_frames=n_frames,
        )
        chord_labels = [canonicalize_chord_label(l) for l in frame_labels]
        key_score = int(round(key_score_float * 10))

        # Synthetic neutral chord logits tensor (1, time, 2048) for downstream SegmentNet
        chord_logits = torch.zeros(1, n_frames, n_fft // 2, device=device)
        method = "dsp_chroma"

    print(f"[Chordify] Key: {song_key}  (score {key_score})  [{method}]")
    return chord_labels, song_key, key_score, chord_logits, method, chord_events
