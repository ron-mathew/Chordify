"""
Chordify AI guitar tab generation.

v2 improvements:
  - Measure-based layout: chords are grouped into bars (4 beats each)
  - Chord names shown once per bar, not per frame
  - Enharmonic-friendly display (Bb not A#)
  - Synchronized view: lyrics aligned beneath chord names
  - Readable even at 20–30 chords for a 3-minute song
"""
from __future__ import annotations

import textwrap
from utils.labels import canonicalize_chord_label

# ---------------------------------------------------------------------------
# Guitar chord voicing dictionary
# String order: e  B  G  D  A  E  (high → low, fret numbers or 'x')
# ---------------------------------------------------------------------------
_CHORD_VOICINGS: dict[str, tuple] = {
    "N":  ("x","x","x","x","x","x"),
    "X":  ("x","x","x","x","x","x"),
    # ── Major ──────────────────────────────────────────────────────────────
    "C:maj":  (0, 1, 0, 2, 3,"x"),  "Db:maj": ("x",4,6,6,6,4),
    "D:maj":  (2, 3, 2, 0,"x","x"),  "Eb:maj": (3, 4, 3, 1,"x","x"),
    "E:maj":  (0, 0, 1, 2, 2, 0),   "F:maj":  (1, 1, 2, 3, 3, 1),
    "Gb:maj": (2, 2, 3, 4, 4, 2),   "G:maj":  (3, 3, 0, 0, 2, 3),
    "Ab:maj": (4, 4, 5, 6, 6, 4),   "A:maj":  (0, 2, 2, 2, 0,"x"),
    "Bb:maj": (1, 3, 3, 3, 1,"x"),  "B:maj":  (2, 4, 4, 4, 2,"x"),
    # ── Minor ──────────────────────────────────────────────────────────────
    "C:min":  ("x",4,5,5,3,"x"),    "Db:min": ("x",4,6,6,4,"x"),
    "D:min":  (1, 3, 2, 0,"x","x"),  "Eb:min": (2, 4, 3, 1,"x","x"),
    "E:min":  (0, 0, 0, 2, 2, 0),   "F:min":  (1, 1, 1, 3, 3, 1),
    "Gb:min": (2, 2, 2, 4, 4, 2),   "G:min":  (3, 3, 3, 5, 5, 3),
    "Ab:min": (4, 4, 4, 6, 6, 4),   "A:min":  (0, 1, 2, 2, 0,"x"),
    "Bb:min": (1, 2, 3, 3, 1,"x"),  "B:min":  (2, 3, 4, 4, 2,"x"),
    # ── Dominant 7 ─────────────────────────────────────────────────────────
    "C:7":    (0, 1, 3, 2, 3,"x"),  "D:7":    (2, 1, 2, 0,"x","x"),
    "E:7":    (0, 3, 1, 2, 2, 0),   "F:7":    (1, 1, 2, 3, 1, 1),
    "G:7":    (1, 3, 0, 0, 2, 3),   "A:7":    (0, 2, 0, 2, 0,"x"),
    "Bb:7":   (1, 3, 1, 3, 1,"x"),  "B:7":    (2, 0, 2, 1, 2,"x"),
    # ── Major 7 ────────────────────────────────────────────────────────────
    "C:maj7": (0, 0, 0, 2, 3,"x"),  "D:maj7": (2, 2, 2, 0,"x","x"),
    "E:maj7": (0, 0, 1, 1, 2, 0),   "F:maj7": (0, 1, 2, 3,"x","x"),
    "G:maj7": (2, 3, 0, 0, 2, 3),   "A:maj7": (0, 2, 1, 2, 0,"x"),
    "Bb:maj7":(1, 3, 2, 3, 1,"x"),  "B:maj7": (2, 4, 3, 4, 2,"x"),
    # ── Minor 7 ────────────────────────────────────────────────────────────
    "C:min7": (3, 4, 3, 5, 3,"x"),  "D:min7": (1, 1, 2, 0,"x","x"),
    "E:min7": (0, 3, 0, 2, 2, 0),   "F:min7": (1, 1, 1, 3, 1, 1),
    "Gb:min7":(2, 2, 2, 4, 2, 2),   "G:min7": (3, 3, 3, 5, 3, 3),
    "Ab:min7":(4, 4, 4, 6, 4, 4),   "A:min7": (0, 1, 0, 2, 0,"x"),
    "Bb:min7":(1, 2, 1, 3, 1,"x"),  "B:min7": (2, 3, 2, 4, 2,"x"),
    # ── Sus / Power ────────────────────────────────────────────────────────
    "D:sus4": (3, 3, 2, 0,"x","x"),  "A:sus4": (0, 3, 2, 2, 0,"x"),
    "E:sus4": (0, 0, 2, 2, 2, 0),    "G:sus4": (3, 3, 0, 0, 1, 3),
    "G:5":    ("x","x",5, 5, 3,"x"), "A:5":    ("x","x",2, 2, 0,"x"),
    "D:5":    ("x","x",7, 7, 5,"x"),
    # ── Sharp aliases (map to flat equivalents for voicing lookup) ──────────
    "C#:maj": ("x",4,6,6,6,4),  "C#:min": ("x",4,6,6,4,"x"),
    "D#:maj": (3, 4, 3, 1,"x","x"), "D#:min": (2, 4, 3, 1,"x","x"),
    "F#:maj": (2, 2, 3, 4, 4, 2),   "F#:min": (2, 2, 2, 4, 4, 2),
    "G#:maj": (4, 4, 5, 6, 6, 4),   "G#:min": (4, 4, 4, 6, 6, 4),
    "A#:maj": (1, 3, 3, 3, 1,"x"),  "A#:min": (1, 2, 3, 3, 1,"x"),
}

_STRING_NAMES = ["e", "B", "G", "D", "A", "E"]


def _get_voicing(label: str) -> tuple:
    """Look up voicing, with multi-level fallback."""
    clean = label.strip()
    candidates = [
        clean,
        canonicalize_chord_label(clean),
        f"{clean}:maj",
        f"{clean[:-1]}:min" if clean.endswith("m") else f"{clean}:min",
    ]
    for candidate in candidates:
        if candidate in _CHORD_VOICINGS:
            return _CHORD_VOICINGS[candidate]
    # Root-quality fallback
    if ":" in clean:
        root, qual = clean.split(":", 1)
        for q in (qual[:3], qual[0], "maj"):
            fb = f"{root}:{q}"
            if fb in _CHORD_VOICINGS:
                return _CHORD_VOICINGS[fb]
    return ("x","x","x","x","x","x")


def _fret(v) -> str:
    return "x" if v == "x" else str(int(v))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_tabs(fusion_result: dict) -> dict:
    """
    Build guitar tab from fused Chordify output.

    Returns dict with:
      "tab_list"   — per-chord-segment structured events (for UI sync display)
      "ascii_tab"  — measure-based ASCII 6-line tab (human readable)
      "sync_view"  — synchronized chord+lyric+timing list
    """
    chords   = fusion_result.get("chords", [])
    downbeats= fusion_result.get("downbeats", [])
    sections = fusion_result.get("sections", [])
    lyrics   = fusion_result.get("lyrics", [])
    meta     = fusion_result.get("metadata", {})
    bpm      = meta.get("bpm", 80.0)
    key      = meta.get("key", "?")
    duration = meta.get("duration", 0.0)

    # Build structured tab list
    tab_list = []
    for chord_seg in chords:
        tab_list.append({
            "start":   chord_seg["start"],
            "end":     chord_seg["end"],
            "chord":   chord_seg["label"],
            "voicing": _get_voicing(chord_seg["label"]),
            "section": _section_at(sections, chord_seg["start"]),
            "lyric":   _lyric_at(lyrics, chord_seg["start"]),
        })

    ascii_tab = _build_measure_tab(tab_list, downbeats, bpm, key, duration)
    sync_view = _build_sync_view(chords, lyrics, bpm)

    print(f"[Chordify] Tab generation complete — {len(tab_list)} chord events.")
    return {
        "tab_list":  tab_list,
        "ascii_tab": ascii_tab,
        "sync_view": sync_view,
    }


# ---------------------------------------------------------------------------
# Measure-based ASCII tab builder
# ---------------------------------------------------------------------------
def _build_measure_tab(
    tab_list: list,
    downbeats: list,
    bpm: float,
    key: str,
    duration: float,
) -> str:
    if not tab_list:
        return (
            "╔══════════════════════════════════════════╗\n"
            "║  Chordify — No chords detected yet.      ║\n"
            "║  Upload a real audio file to analyse.    ║\n"
            "╚══════════════════════════════════════════╝\n"
        )

    # Group chords into measures (4 beats each)
    seconds_per_beat = 60.0 / max(bpm, 40.0)
    seconds_per_bar  = seconds_per_beat * 4
    total_bars = max(1, int(np.ceil(duration / seconds_per_bar)))
    # Deduplicate downbeat times
    db_times = sorted({round(d["time"], 2) for d in downbeats})

    # Assign each chord to a bar
    bars: dict[int, list[dict]] = {i: [] for i in range(total_bars)}
    for ev in tab_list:
        bar_idx = int(ev["start"] / seconds_per_bar)
        bar_idx = min(bar_idx, total_bars - 1)
        if not bars[bar_idx] or bars[bar_idx][-1]["chord"] != ev["chord"]:
            bars[bar_idx].append(ev)

    # Build the 8-bars-per-line layout
    BARS_PER_LINE = 4
    COL_W = 10   # fixed column width for each bar

    header = (
        "╔══════════════════════════════════════════════════╗\n"
        "║           C H O R D I F Y  —  Tab Output         ║\n"
        "╚══════════════════════════════════════════════════╝\n"
        f"  Key: {key}   BPM: {bpm:.0f}   Duration: {duration:.0f}s\n\n"
    )

    lines_out = []
    for line_start in range(0, total_bars, BARS_PER_LINE):
        line_bars = list(range(line_start, min(line_start + BARS_PER_LINE, total_bars)))
        n = len(line_bars)

        # Chord name row
        chord_row = ""
        for bi in line_bars:
            evs = bars[bi]
            # Deduplicate consecutive same chord within bar
            unique = []
            for ev in evs:
                if not unique or unique[-1]["chord"] != ev["chord"]:
                    unique.append(ev)
            lbl = "  ".join(ev["chord"].replace(":maj","").replace(":min","m")
                            for ev in unique) if unique else "N"
            chord_row += lbl.center(COL_W) + " "

        # Lyric row (lyric at bar start)
        lyric_row = ""
        for bi in line_bars:
            t = bi * seconds_per_bar
            lyr = _lyric_near(tab_list, t, seconds_per_bar)
            lyric_row += lyr[:COL_W-1].center(COL_W) + " "

        # 6 string rows
        str_rows = {s: "" for s in _STRING_NAMES}
        for bi in line_bars:
            evs = bars[bi]
            if evs:
                voicing = evs[0]["voicing"]
                for si, sname in enumerate(_STRING_NAMES):
                    f = _fret(voicing[si])
                    str_rows[sname] += f.center(COL_W, "-") + "-"
            else:
                for sname in _STRING_NAMES:
                    str_rows[sname] += "-" * COL_W + "-"

        # Assemble block
        sep = "+" + (("-" * COL_W + "+") * n)
        block = []
        block.append("  " + chord_row)
        if any(c.strip() for c in lyric_row):
            block.append("  " + lyric_row)
        block.append(sep)
        for sname in _STRING_NAMES:
            block.append(f"{sname}|" + str_rows[sname] + "|")
        block.append(sep)
        block.append("")
        lines_out.append("\n".join(block))

    return header + "\n".join(lines_out)


# ---------------------------------------------------------------------------
# Synchronized view builder (for UI display)
# ---------------------------------------------------------------------------
def _build_sync_view(
    chords: list[dict],
    lyrics: list[dict],
    bpm: float,
) -> list[dict]:
    """
    Build a flat list of sync events sorted by time.
    Each event has: time, type ('chord'|'lyric'), label/text, end.
    Used by the UI to highlight the current chord/lyric during playback.
    """
    events = []
    for c in chords:
        events.append({
            "time":  c["start"],
            "end":   c["end"],
            "type":  "chord",
            "label": c["label"],
        })
    for l in lyrics:
        events.append({
            "time":  l["start"],
            "end":   l["end"],
            "type":  "lyric",
            "label": l["text"],
        })
    events.sort(key=lambda x: x["time"])
    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _section_at(sections: list, t: float) -> str:
    for s in sections:
        if s["start"] <= t < s["end"]:
            return s["label"]
    return ""


def _lyric_at(lyrics: list, t: float) -> str:
    for l in lyrics:
        if l["start"] <= t <= l["end"]:
            return l["text"]
    return ""


def _lyric_near(tab_list: list, t: float, window: float) -> str:
    """Find first lyric text near time t within window."""
    for ev in tab_list:
        if abs(ev["start"] - t) < window and ev.get("lyric"):
            return ev["lyric"]
    return ""


# Expose numpy for measure layout
import numpy as np  # noqa: E402  (imported after constants so it reads cleanly)
