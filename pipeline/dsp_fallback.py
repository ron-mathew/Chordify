"""
Chordify DSP-based fallback module for Chord, Key, and Beat estimation.

Used automatically when trained neural-network checkpoints (ChordNet / BeatNet)
are unavailable, missing, or untrained.

Advanced MIR Methods:
1. Beat-Synchronous Chromagram:
   - Harmonic extraction (librosa.effects.harmonic)
   - Dual-band chroma: full-spectrum CQT + bass-spectrum CQT (C1-C3)
   - Beat-synchronous temporal aggregation (librosa.util.sync)
2. Musical Key Estimation:
   - Krumhansl-Schmuckler key profile correlation over global chromagram
3. Key-Aware Enharmonic Spelling:
   - Formats chords according to the detected tonal center (e.g. Bb, Eb, Gm instead of A#, D#, G:min)
4. Temporal Optimization:
   - Temperature-scaled Viterbi HMM decoding across beats
   - Self-transition regularization to prevent rapid jitter while allowing genuine musical changes
5. Tempo Ambiguity Resolution:
   - Dynamic beat tracking with tempo octave selection (BPM vs BPM/2 for slow ballads / 6/8 meter)
"""
from __future__ import annotations

import numpy as np
import torch
import librosa
from scipy.ndimage import median_filter

from utils.labels import CANONICAL_CHORD_MAP, canonicalize_chord_label


PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler Key Profiles (12 pitch classes)
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)

# Enharmonic spelling maps
_FLAT_KEYS = {"A#", "Bb", "D#", "Eb", "G#", "Ab", "C#", "Db", "F", "Gm", "Cm", "Fm", "Bbm", "Dm", "Ebm"}
_ENHARMONIC_FLAT_MAP = {
    "A#:maj": "Bb", "D#:maj": "Eb", "G#:maj": "Ab", "C#:maj": "Db", "F#:maj": "Gb",
    "A#:min": "Bbm", "D#:min": "Ebm", "G#:min": "Abm", "C#:min": "Dbm", "F#:min": "Gbm",
    "A#": "Bb", "D#": "Eb", "G#": "Ab", "C#": "Db", "F#": "Gb",
    "A#m": "Bbm", "D#m": "Ebm", "G#m": "Abm", "C#m": "Dbm", "F#m": "Gbm",
}


def _build_triad_templates() -> tuple[np.ndarray, list[str]]:
    """Build L2-normalized 12-bin chroma templates for 12 Major and 12 Minor chords."""
    templates = []
    labels = []

    # 12 Major chords (root, root+4, root+7)
    for i, root in enumerate(PITCH_CLASSES):
        t = np.zeros(12, dtype=np.float32)
        t[i] = 1.0
        t[(i + 4) % 12] = 0.8
        t[(i + 7) % 12] = 0.8
        t /= np.linalg.norm(t) + 1e-8
        templates.append(t)
        labels.append(f"{root}:maj")

    # 12 Minor chords (root, root+3, root+7)
    for i, root in enumerate(PITCH_CLASSES):
        t = np.zeros(12, dtype=np.float32)
        t[i] = 1.0
        t[(i + 3) % 12] = 0.8
        t[(i + 7) % 12] = 0.8
        t /= np.linalg.norm(t) + 1e-8
        templates.append(t)
        labels.append(f"{root}:min")

    return np.array(templates, dtype=np.float32), labels


_TEMPLATES, _TEMPLATE_LABELS = _build_triad_templates()


def _to_mono_numpy(audio: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert (channels, samples) or (samples,) audio to 1D float32 numpy array."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)
    return np.asarray(audio, dtype=np.float32)


def spell_chord(raw_label: str, key_str: str) -> str:
    """Format chord spelling according to the detected key signature."""
    if raw_label in ("N", "X"):
        return "N"
    
    # Check if key prefers flat notation
    is_flat_key = any(k in key_str for k in ["b", "Bb", "Eb", "Ab", "Db", "F", "Gm", "Cm", "Fm", "Bbm", "Dm"])
    if is_flat_key and raw_label in _ENHARMONIC_FLAT_MAP:
        return _ENHARMONIC_FLAT_MAP[raw_label]

    # Standard clean name: "C:maj" -> "C", "A:min" -> "Am"
    if ":" in raw_label:
        root, quality = raw_label.split(":", 1)
        if is_flat_key and root in _ENHARMONIC_FLAT_MAP:
            root = _ENHARMONIC_FLAT_MAP[root]
        return root if quality == "maj" else f"{root}m"

    return raw_label


# ---------------------------------------------------------------------------
# Key Estimation (Krumhansl-Schmuckler)
# ---------------------------------------------------------------------------
def estimate_key_dsp(chroma_mean: np.ndarray) -> tuple[str, float]:
    """
    Estimate musical key via Krumhansl-Schmuckler correlation.
    Returns clean key string (e.g. "Bb" or "Gm") and correlation score.
    """
    if np.sum(chroma_mean) < 1e-6:
        return "C", 0.0

    chroma_norm = (chroma_mean - np.mean(chroma_mean)) / (np.std(chroma_mean) + 1e-8)
    best_corr = -1.0
    best_key = "C"

    for shift in range(12):
        shifted_maj = np.roll(_KS_MAJOR, shift)
        s_maj_norm = (shifted_maj - np.mean(shifted_maj)) / (np.std(shifted_maj) + 1e-8)
        corr_maj = float(np.corrcoef(chroma_norm, s_maj_norm)[0, 1])
        if corr_maj > best_corr:
            best_corr = corr_maj
            root = PITCH_CLASSES[shift]
            best_key = _ENHARMONIC_FLAT_MAP.get(root, root)

        shifted_min = np.roll(_KS_MINOR, shift)
        s_min_norm = (shifted_min - np.mean(shifted_min)) / (np.std(shifted_min) + 1e-8)
        corr_min = float(np.corrcoef(chroma_norm, s_min_norm)[0, 1])
        if corr_min > best_corr:
            best_corr = corr_min
            root = PITCH_CLASSES[shift]
            clean_root = _ENHARMONIC_FLAT_MAP.get(root, root)
            best_key = f"{clean_root}m"

    return best_key, round(best_corr, 2)


# ---------------------------------------------------------------------------
# Beat & Tempo Fallback with Ambiguity Resolution
# ---------------------------------------------------------------------------
def estimate_beats_dsp(
    audio: torch.Tensor | np.ndarray,
    sr: int = 44100,
    hop_length: int = 1024,
    expected_frames: int | None = None,
) -> tuple[float, list[int], list[dict], np.ndarray]:
    """
    DSP-based Beat and Tempo tracking using Librosa onset envelope and dynamic programming.
    Resolves double-time / half-time tempo ambiguity.
    """
    y = _to_mono_numpy(audio)
    duration = len(y) / sr

    if len(y) < hop_length * 4 or np.max(np.abs(y)) < 1e-4:
        n_f = expected_frames or max(1, int(duration * sr / hop_length))
        return 120.0, [0] * n_f, [], np.array([0])

    # 1. Onset envelope
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    except Exception:
        onset_env = np.ones(max(1, int(duration * sr / hop_length)), dtype=np.float32)

    # 2. Beat tracking
    try:
        tempo_val, raw_beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
        )
        if hasattr(tempo_val, "__len__"):
            raw_bpm = float(tempo_val[0]) if len(tempo_val) > 0 else 120.0
        elif hasattr(tempo_val, "item"):
            raw_bpm = float(tempo_val.item())
        else:
            raw_bpm = float(tempo_val)
    except Exception:
        raw_bpm = 120.0
        raw_beats = np.arange(0, len(onset_env), max(1, int(sr * 60 / (120 * hop_length))))

    # 3. Tempo Octave / Ballad Ambiguity Resolution
    # Ballads and 6/8 songs frequently trigger double-time detection (e.g. 150-180 BPM instead of 75-90 BPM)
    if raw_bpm > 135.0:
        bpm = raw_bpm / 2.0
        beats = raw_beats[::2]
    else:
        bpm = raw_bpm
        beats = raw_beats

    bpm = round(bpm, 1)
    if bpm <= 0:
        bpm = 120.0

    n_frames = expected_frames or len(onset_env)
    beat_classes = [0] * n_frames
    beat_events = []

    for i, f in enumerate(beats):
        if f < n_frames:
            btype = "downbeat" if (i % 4 == 0) else "beat"
            beat_classes[f] = 2 if btype == "downbeat" else 1
            t = (f * hop_length) / sr
            beat_events.append({
                "time": round(float(t), 3),
                "type": btype,
            })

    return bpm, beat_classes, beat_events, beats


# ---------------------------------------------------------------------------
# Beat-Synchronous Viterbi Chord Recognition
# ---------------------------------------------------------------------------
def estimate_chords_dsp(
    audio: torch.Tensor | np.ndarray,
    sr: int = 44100,
    hop_length: int = 1024,
    beats: np.ndarray | None = None,
    expected_frames: int | None = None,
    min_segment_dur: float = 1.0,
) -> tuple[list[str], str, float, list[dict]]:
    """
    Beat-Synchronous Viterbi Chord Recognition using Dual-Band Chroma CQT.

    Returns
    -------
    frame_labels : list[str]  — per-frame canonical chord label
    song_key     : str        — estimated global key
    key_score    : float      — key correlation score
    chord_events : list[dict] — timestamped chord segments with confidence
    """
    y = _to_mono_numpy(audio)
    duration = len(y) / sr

    if len(y) < hop_length or np.max(np.abs(y)) < 1e-4:
        n_f = expected_frames or max(1, int(duration * sr / hop_length))
        frame_labels = ["N"] * n_f
        chord_events = [{"start": 0.0, "end": round(duration, 2), "chord": "N", "label": "N", "confidence": 1.0, "source": "dsp_chroma"}]
        return frame_labels, "C", 0.0, chord_events

    # 1. Harmonic source extraction
    try:
        y_harm = librosa.effects.harmonic(y, margin=3.0)
    except Exception:
        y_harm = y

    # 2. Dual-band Chroma feature extraction
    # Full spectrum chroma
    try:
        chroma_full = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop_length, n_chroma=12)
    except Exception:
        chroma_full = librosa.feature.chroma_stft(y=y_harm, sr=sr, hop_length=hop_length, n_chroma=12)

    # Bass chroma (C1 to C3, approx 32 to 130 Hz) for accurate root pitch identification
    try:
        chroma_bass = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=hop_length, n_chroma=12,
            fmin=float(librosa.note_to_hz("C1")), n_octaves=2,
        )
    except Exception:
        chroma_bass = chroma_full

    n_frames = chroma_full.shape[1]

    # Global Key Detection
    chroma_mean = np.mean(chroma_full, axis=1)
    song_key, key_score = estimate_key_dsp(chroma_mean)

    # 3. Synchronize Chroma to Beat Grid
    if beats is None or len(beats) < 2:
        _, _, _, beats = estimate_beats_dsp(y, sr=sr, hop_length=hop_length)

    beat_frames = np.unique(np.clip(beats, 0, n_frames - 1))
    if len(beat_frames) == 0 or beat_frames[0] != 0:
        beat_frames = np.insert(beat_frames, 0, 0)
    if beat_frames[-1] != n_frames - 1:
        beat_frames = np.append(beat_frames, n_frames - 1)

    try:
        chroma_f_b = librosa.util.sync(chroma_full, beat_frames, aggregate=np.median)
        chroma_b_b = librosa.util.sync(chroma_bass, beat_frames, aggregate=np.median)
    except Exception:
        chroma_f_b = chroma_full
        chroma_b_b = chroma_bass

    # Weighted fusion: 65% harmonic mid/treble + 35% bass root
    norm_f = chroma_f_b / (np.linalg.norm(chroma_f_b, axis=0, keepdims=True) + 1e-8)
    norm_b = chroma_b_b / (np.linalg.norm(chroma_b_b, axis=0, keepdims=True) + 1e-8)
    chroma_comb = 0.65 * norm_f + 0.35 * norm_b
    chroma_comb /= (np.linalg.norm(chroma_comb, axis=0, keepdims=True) + 1e-8)

    # 4. Template Matching (24 states x n_beats)
    sim = np.maximum(np.dot(_TEMPLATES, chroma_comb), 0.0)

    # 5. Temperature-scaled Viterbi HMM Decoding
    gamma = 14.0
    log_B = gamma * sim
    n_states = 24
    n_obs = chroma_comb.shape[1]

    # Transition matrix with self-transition regularization
    log_A = np.full((n_states, n_states), -3.0, dtype=np.float32)
    np.fill_diagonal(log_A, 0.0)

    viterbi = np.zeros((n_states, n_obs), dtype=np.float32)
    backpointer = np.zeros((n_states, n_obs), dtype=int)
    viterbi[:, 0] = log_B[:, 0]

    for t in range(1, n_obs):
        for s in range(n_states):
            prob = viterbi[:, t - 1] + log_A[:, s] + log_B[s, t]
            best_prev = int(np.argmax(prob))
            viterbi[s, t] = prob[best_prev]
            backpointer[s, t] = best_prev

    best_path = [int(np.argmax(viterbi[:, -1]))]
    for t in range(n_obs - 1, 0, -1):
        best_path.append(backpointer[best_path[-1], t])
    best_path.reverse()

    # 6. Extract Segments with Key-Aware Spelling
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    if len(beat_times) < n_obs:
        beat_times = np.linspace(0, duration, n_obs)

    raw_segments = []
    curr_s = 0.0
    curr_raw = _TEMPLATE_LABELS[best_path[0]]
    curr_label = spell_chord(curr_raw, song_key)
    confs = [float(sim[best_path[0], 0])]

    for i in range(1, n_obs):
        r_lbl = _TEMPLATE_LABELS[best_path[i]]
        lbl = spell_chord(r_lbl, song_key)
        t_cur = float(beat_times[i])
        if lbl != curr_label:
            raw_segments.append({
                "start": round(curr_s, 2),
                "end": round(t_cur, 2),
                "chord": curr_label,
                "label": curr_label,
                "raw_label": curr_raw,
                "confidence": round(float(np.mean(confs)), 2),
                "source": "dsp_chroma",
            })
            curr_label = lbl
            curr_raw = r_lbl
            curr_s = t_cur
            confs = [float(sim[best_path[i], i])]
        else:
            confs.append(float(sim[best_path[i], i]))

    raw_segments.append({
        "start": round(curr_s, 2),
        "end": round(duration, 2),
        "chord": curr_label,
        "label": curr_label,
        "raw_label": curr_raw,
        "confidence": round(float(np.mean(confs)), 2),
        "source": "dsp_chroma",
    })

    # 7. Merge sub-second noise segments (< min_segment_dur)
    pruned_segments = []
    for s in raw_segments:
        if not pruned_segments:
            pruned_segments.append(s)
        elif (s["end"] - s["start"]) < min_segment_dur and len(pruned_segments) > 0:
            pruned_segments[-1]["end"] = s["end"]
        else:
            if s["label"] == pruned_segments[-1]["label"]:
                pruned_segments[-1]["end"] = s["end"]
            else:
                pruned_segments.append(s)

    # 8. Reconstruct frame-level labels for downstream pipeline consistency
    target_frames = expected_frames or n_frames
    frame_labels = []
    for f in range(target_frames):
        t = (f * hop_length) / sr
        matched = "N"
        for seg in pruned_segments:
            if seg["start"] <= t <= seg["end"]:
                # Map to standard canonical name for 122-class model compatibility
                matched = canonicalize_chord_label(seg.get("raw_label", seg["label"]))
                break
        frame_labels.append(matched)

    return frame_labels, song_key, float(key_score), pruned_segments
