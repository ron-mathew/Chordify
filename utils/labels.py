"""
Chordify label definitions.

122-class chord vocabulary and 10-class segment vocabulary,
preserved exactly from the Lamucal reference for checkpoint compatibility.
"""

# ---------------------------------------------------------------------------
# Chord vocabulary  (122 classes)
# ---------------------------------------------------------------------------
CHORD_LABELS = [
    "N", "X",
    # maj  (roots C … B)
    "C:maj", "C#:maj", "D:maj", "D#:maj", "E:maj", "F:maj",
    "F#:maj", "G:maj", "G#:maj", "A:maj", "A#:maj", "B:maj",
    # min
    "C:min", "C#:min", "D:min", "D#:min", "E:min", "F:min",
    "F#:min", "G:min", "G#:min", "A:min", "A#:min", "B:min",
    # 7
    "C:7", "C#:7", "D:7", "D#:7", "E:7", "F:7",
    "F#:7", "G:7", "G#:7", "A:7", "A#:7", "B:7",
    # maj7
    "C:maj7", "C#:maj7", "D:maj7", "D#:maj7", "E:maj7", "F:maj7",
    "F#:maj7", "G:maj7", "G#:maj7", "A:maj7", "A#:maj7", "B:maj7",
    # min7
    "C:min7", "C#:min7", "D:min7", "D#:min7", "E:min7", "F:min7",
    "F#:min7", "G:min7", "G#:min7", "A:min7", "A#:min7", "B:min7",
    # 6
    "C:6", "C#:6", "D:6", "D#:6", "E:6", "F:6",
    "F#:6", "G:6", "G#:6", "A:6", "A#:6", "B:6",
    # m6
    "C:m6", "C#:m6", "D:m6", "D#:m6", "E:m6", "F:m6",
    "F#:m6", "G:m6", "G#:m6", "A:m6", "A#:m6", "B:m6",
    # sus2
    "C:sus2", "C#:sus2", "D:sus2", "D#:sus2", "E:sus2", "F:sus2",
    "F#:sus2", "G:sus2", "G#:sus2", "A:sus2", "A#:sus2", "B:sus2",
    # sus4
    "C:sus4", "C#:sus4", "D:sus4", "D#:sus4", "E:sus4", "F:sus4",
    "F#:sus4", "G:sus4", "G#:sus4", "A:sus4", "A#:sus4", "B:sus4",
    # 5  (power chords)
    "C:5", "C#:5", "D:5", "D#:5", "E:5", "F:5",
    "F#:5", "G:5", "G#:5", "A:5", "A#:5", "B:5",
]

assert len(CHORD_LABELS) == 122, f"Expected 122 chord labels, got {len(CHORD_LABELS)}"

# ---------------------------------------------------------------------------
# Segment vocabulary  (10 classes)
# ---------------------------------------------------------------------------
SEGMENT_LABELS = [
    "start",
    "end",
    "intro",
    "outro",
    "verse",
    "chorus",
    "solo",
    "break",
    "bridge",
    "inst",
]

assert len(SEGMENT_LABELS) == 10, f"Expected 10 segment labels, got {len(SEGMENT_LABELS)}"

# ---------------------------------------------------------------------------
# Diatonic key tables (used by modulation / key detection)
# ---------------------------------------------------------------------------
MAJOR_CHORDS = [
    ["C",  "Dm",  "Em",  "F",  "G",  "Am",  "G7"],
    ["G",  "Am",  "Bm",  "C",  "D",  "Em",  "D7"],
    ["D",  "Em",  "F#m", "G",  "A",  "Bm",  "A7"],
    ["A",  "Bm",  "C#m", "D",  "E",  "F#m", "E7"],
    ["E",  "F#m", "G#m", "A",  "B",  "C#m", "B7"],
    ["F",  "Gm",  "Am",  "Bb", "C",  "Dm",  "C7"],
    ["B",  "C#m", "D#m", "E",  "F#", "G#m", "F#7"],
    ["Db", "Ebm", "Fm",  "Gb", "Ab", "Bbm", "Ab7"],
    ["Eb", "Fm",  "Gm",  "Ab", "Bb", "Cm",  "Bb7"],
    ["Gb", "Abm", "Bbm", "Cb", "Db", "Ebm", "Db7"],
    ["Ab", "Bbm", "Cm",  "Db", "Eb", "Fm",  "Eb7"],
    ["Bb", "Cm",  "Dm",  "Eb", "F",  "Gm",  "F7"],
    ["C#", "D#m", "E#m", "F#", "G#", "A#m", "G#7"],
    ["Cb", "Dbm", "Ebm", "Fb", "Gb", "Abm", "Gb7"],
    ["F#", "G#m", "A#m", "B",  "C#", "D#m", "C#7"],
]

MINOR_CHORDS = [
    ["Am",  "Bdim",  "C",  "Dm",  "Em",  "F",  "G"],
    ["Em",  "F#dim", "G",  "Am",  "Bm",  "C",  "D"],
    ["Bm",  "C#dim", "D",  "Em",  "F#m", "G",  "A"],
    ["F#m", "G#dim", "A",  "Bm",  "C#m", "D",  "E"],
    ["C#m", "D#dim", "E",  "F#m", "G#m", "A",  "B"],
    ["Dm",  "Edim",  "F",  "Gm",  "Am",  "Bb", "C"],
    ["G#m", "A#dim", "B",  "C#m", "D#m", "E",  "F#"],
    ["Bbm", "Cdim",  "Db", "Ebm", "Fm",  "Gb", "Ab"],
    ["Cm",  "Ddim",  "Eb", "Fm",  "Gm",  "Ab", "Bb"],
    ["Ebm", "Fdim",  "Gb", "Abm", "Bbm", "Cb", "Db"],
    ["Fm",  "Gdim",  "Ab", "Bbm", "Cm",  "Db", "Eb"],
    ["Gm",  "Adim",  "Bb", "Cm",  "Dm",  "Eb", "F"],
    ["A#m", "B#dim", "C#", "D#m", "E#m", "F#", "G#"],
    ["Abm", "Bbdim", "Cb", "Dbm", "Ebm", "Fb", "Gb"],
    ["D#m", "E#dim", "F#", "G#m", "A#m", "B",  "C#"],
]

MAJ2MIN_MAP = {
    "G": "Em", "C": "Am", "D": "Bm", "E": "C#m", "F": "Dm",
    "A": "F#m", "B": "G#m", "Db": "Bbm", "Eb": "Cm", "Gb": "Ebm",
    "Ab": "Fm", "Bb": "Gm", "C#": "A#m", "Cb": "Abm", "F#": "D#m",
}
MIN2MAJ_MAP = {v: k for k, v in MAJ2MIN_MAP.items()}

# Mapping from common shorthands and enharmonics to standard Chordify 122-class labels
CANONICAL_CHORD_MAP = {
    # Enharmonics / flats to sharps (Major)
    "Db": "C#:maj", "Eb": "D#:maj", "Gb": "F#:maj", "Ab": "G#:maj", "Bb": "A#:maj",
    "Db:maj": "C#:maj", "Eb:maj": "D#:maj", "Gb:maj": "F#:maj", "Ab:maj": "G#:maj", "Bb:maj": "A#:maj",
    # Enharmonics / flats to sharps (Minor)
    "Dbm": "C#:min", "Ebm": "D#:min", "Gbm": "F#:min", "Abm": "G#:min", "Bbm": "A#:min",
    "Db:min": "C#:min", "Eb:min": "D#:min", "Gb:min": "F#:min", "Ab:min": "G#:min", "Bb:min": "A#:min",
    # Shorthand minor (e.g. "Am" -> "A:min")
    "Cm": "C:min", "C#m": "C#:min", "Dm": "D:min", "D#m": "D#:min",
    "Em": "E:min", "Fm": "F:min", "F#m": "F#:min", "Gm": "G:min",
    "G#m": "G#:min", "Am": "A:min", "A#m": "A#:min", "Bm": "B:min",
    # Shorthand major (e.g. "C" -> "C:maj")
    "C": "C:maj", "C#": "C#:maj", "D": "D:maj", "D#": "D#:maj",
    "E": "E:maj", "F": "F:maj", "F#": "F#:maj", "G": "G:maj",
    "G#": "G#:maj", "A": "A:maj", "A#": "A#:maj", "B": "B:maj",
}


def canonicalize_chord_label(label: str) -> str:
    """Normalize any chord string to standard Chordify 122-class taxonomy."""
    if not label or label in ("N", "X"):
        return "N"
    if label in CHORD_LABELS:
        return label
    if label in CANONICAL_CHORD_MAP:
        return CANONICAL_CHORD_MAP[label]
    # Check if root is mapped
    if ":" in label:
        root, qual = label.split(":", 1)
        if root in CANONICAL_CHORD_MAP:
            mapped_root = CANONICAL_CHORD_MAP[root].split(":")[0]
            candidate = f"{mapped_root}:{qual}"
            if candidate in CHORD_LABELS:
                return candidate
    return label
