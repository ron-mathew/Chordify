"""
Chordify timestamp utilities.

Converts frame-index arrays from model outputs into human-readable
timestamp-based representations, and builds the unified multimodal
timeline that is central to the Chordify output format.
"""
from __future__ import annotations

from typing import Any


def frames_to_time(frame_idx: int, hop_length: int, sample_rate: int) -> float:
    """Convert a STFT frame index to seconds."""
    return round(frame_idx * hop_length / sample_rate, 4)


def beat_frames_to_events(
    beat_predictions: list[int],
    hop_length: int,
    sample_rate: int,
) -> list[dict]:
    """
    Convert BeatNet per-frame predictions to timestamped beat events.

    beat class mapping:
      0 → no beat
      1 → beat
      2 → downbeat

    Returns list of dicts:
      {"time": float, "type": "beat" | "downbeat"}
    """
    events = []
    for i, cls in enumerate(beat_predictions):
        if cls == 1:
            events.append({
                "time": frames_to_time(i, hop_length, sample_rate),
                "type": "beat",
            })
        elif cls == 2:
            events.append({
                "time": frames_to_time(i, hop_length, sample_rate),
                "type": "downbeat",
            })
    return events


def chord_frames_to_segments(
    chord_predictions: list[str],
    hop_length: int,
    sample_rate: int,
) -> list[dict]:
    """
    RLE-compress per-frame chord labels into timed segments.

    Returns list of dicts:
      {"start": float, "end": float, "label": str}
    """
    if not chord_predictions:
        return []

    segments = []
    current_label = chord_predictions[0]
    start_frame = 0

    for i, label in enumerate(chord_predictions[1:], start=1):
        if label != current_label:
            segments.append({
                "start": frames_to_time(start_frame, hop_length, sample_rate),
                "end": frames_to_time(i, hop_length, sample_rate),
                "label": current_label,
            })
            current_label = label
            start_frame = i

    # Flush last segment
    segments.append({
        "start": frames_to_time(start_frame, hop_length, sample_rate),
        "end": frames_to_time(len(chord_predictions), hop_length, sample_rate),
        "label": current_label,
    })
    return segments


def segment_frames_to_sections(
    segment_predictions: list[str],
    hop_length: int,
    sample_rate: int,
) -> list[dict]:
    """
    RLE-compress per-frame segment labels into timed sections.

    Returns list of dicts:
      {"start": float, "end": float, "label": str}
    """
    return chord_frames_to_segments(segment_predictions, hop_length, sample_rate)


def pitch_frames_to_events(
    pitch_predictions: list[int],
    hop_length: int,
    sample_rate: int,
    semitone_base: int = 21,      # MIDI note 21 = A0
    bins_per_semitone: int = 10,
) -> list[dict]:
    """
    Convert PitchNet 850-class per-frame predictions to timestamped pitch events.

    Class index k maps to:
      semitone = semitone_base + k // bins_per_semitone
      cent_offset = (k % bins_per_semitone) * 10   (0-90 cents)

    Returns list of dicts:
      {"time": float, "midi_note": int, "cents_offset": int}
    """
    events = []
    for i, cls in enumerate(pitch_predictions):
        midi_note = semitone_base + cls // bins_per_semitone
        cents = (cls % bins_per_semitone) * 10
        events.append({
            "time": frames_to_time(i, hop_length, sample_rate),
            "midi_note": int(midi_note),
            "cents_offset": int(cents),
        })
    return events


def build_unified_timeline(
    beats: list[dict],
    chords: list[dict],
    sections: list[dict],
    pitch: list[dict],
    lyrics: list[dict],
    duration: float,
) -> list[dict[str, Any]]:
    """
    Merge all modalities onto a common sorted event timeline.

    Each event is tagged with its modality type so they remain
    distinguishable in the output.
    """
    timeline: list[dict] = []

    for e in beats:
        timeline.append({"time": e["time"], "type": e["type"]})

    for e in chords:
        timeline.append({
            "time": e["start"],
            "type": "chord_start",
            "chord": e["label"],
            "end": e["end"],
        })

    for e in sections:
        timeline.append({
            "time": e["start"],
            "type": "section_start",
            "section": e["label"],
            "end": e["end"],
        })

    for e in pitch:
        timeline.append({
            "time": e["time"],
            "type": "pitch",
            "midi_note": e["midi_note"],
            "cents_offset": e["cents_offset"],
        })

    for e in lyrics:
        timeline.append({
            "time": e.get("start", 0.0),
            "type": "lyric",
            "text": e.get("text", ""),
            "end": e.get("end", e.get("start", 0.0)),
        })

    timeline.sort(key=lambda x: x["time"])
    return timeline
