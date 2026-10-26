"""
Chordify multimodal fusion stage.

This is the key academic contribution of the project: all modality
outputs (beats, chords, pitch, structure, lyrics, tempo, key) are merged
onto a single common timeline.

The unified representation makes cross-modal relationships explicit and
is the foundation for AI tab generation.

Output schema
-------------
{
  "metadata": {
    "duration":    float,
    "sample_rate": int,
    "bpm":         float,
    "key":         str,
    "key_score":   int
  },
  "beats":      [{"time": float, "type": "beat"|"downbeat"}, ...],
  "downbeats":  [{"time": float}, ...],
  "chords":     [{"start": float, "end": float, "label": str}, ...],
  "sections":   [{"start": float, "end": float, "label": str}, ...],
  "pitch":      [{"time": float, "midi_note": int, "cents_offset": int}, ...],
  "lyrics":     [{"start": float, "end": float, "text": str}, ...],
  "timeline":   [ ... unified sorted event list ... ]
}
"""
from __future__ import annotations

from utils.timestamps import (
    beat_frames_to_events,
    chord_frames_to_segments,
    segment_frames_to_sections,
    pitch_frames_to_events,
    build_unified_timeline,
)


def fuse(
    beat_preds: list[int],
    bpm: float,
    chord_labels: list[str],
    song_key: str,
    key_score: int,
    segment_labels: list[str],
    pitch_preds: list[int],
    lyrics: list[dict],
    duration: float,
    sample_rate: int,
    hop_length: int,
    metadata_extra: dict | None = None,
    override_chord_segments: list[dict] | None = None,
    words: list[dict] | None = None,
) -> dict:
    """
    Build the complete Chordify unified output dictionary.

    Parameters
    ----------
    beat_preds     : per-frame beat class list  (0/1/2)
    bpm            : estimated tempo
    chord_labels   : per-frame chord label list
    song_key       : estimated key string
    key_score      : diatonic fit score
    segment_labels : per-frame section label list
    pitch_preds    : per-frame pitch class index list (0–849)
    lyrics         : list of {"start", "end", "text"} dicts
    duration       : total audio duration in seconds
    sample_rate    : audio sample rate
    hop_length     : STFT hop length (for frame→time conversion)

    Returns
    -------
    Structured dict matching the Chordify output schema.
    """
    print("[Chordify] Fusing multimodal outputs …")

    # ------------------------------------------------------------------ beats
    beat_events = beat_frames_to_events(beat_preds, hop_length, sample_rate)
    beats     = [e for e in beat_events if e["type"] == "beat"]
    downbeats = [{"time": e["time"]} for e in beat_events if e["type"] == "downbeat"]

    # ----------------------------------------------------------------- chords
    if override_chord_segments:
        chord_segments = override_chord_segments
    else:
        chord_segments = chord_frames_to_segments(chord_labels, hop_length, sample_rate)

    # --------------------------------------------------------------- sections
    sections = segment_frames_to_sections(segment_labels, hop_length, sample_rate)

    # ------------------------------------------------------------------ pitch
    pitch_events = pitch_frames_to_events(pitch_preds, hop_length, sample_rate)

    # --------------------------------------------------------- unified timeline
    timeline = build_unified_timeline(
        beats=beat_events,
        chords=chord_segments,
        sections=sections,
        pitch=pitch_events,
        lyrics=lyrics,
        duration=duration,
    )

    meta = {
        "duration":    round(duration, 3),
        "sample_rate": sample_rate,
        "bpm":         round(bpm, 2),
        "key":         song_key,
        "key_score":   key_score,
    }
    if metadata_extra:
        meta.update(metadata_extra)

    result = {
        "metadata":  meta,
        "beats":     beats,
        "downbeats": downbeats,
        "chords":    chord_segments,
        "sections":  sections,
        "pitch":     pitch_events,
        "lyrics":    lyrics,
        "words":     words or [],
        "timeline":  timeline,
    }

    print(
        f"[Chordify] Fusion complete — "
        f"{len(beats)} beats | "
        f"{len(chord_segments)} chord segments | "
        f"{len(sections)} sections | "
        f"{len(lyrics)} lyric segments"
    )
    return result
