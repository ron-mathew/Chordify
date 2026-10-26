"""
Chordify checkpoint status checker.

Checks which model checkpoints are present and reports clearly.
If a checkpoint is missing it tells you exactly where to place it.

Usage
-----
  python scripts/download_checkpoints.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

# ---------------------------------------------------------------------------
# Load paths
# ---------------------------------------------------------------------------
paths_file = Path(__file__).parent.parent / "config" / "paths.yaml"
with open(paths_file) as f:
    paths_cfg = yaml.safe_load(f) or {}

project_root   = paths_cfg.get("project_root", "") or str(Path(__file__).parent.parent)
checkpoints_dir = paths_cfg.get("checkpoints_dir", "") or str(
    Path(project_root) / "checkpoints"
)
ckpt_names = paths_cfg.get("checkpoints", {})

# ---------------------------------------------------------------------------
# Checkpoint table
# ---------------------------------------------------------------------------
COMPONENTS = [
    {
        "component":    "U-Net (Source Separation)",
        "architecture": "UNets — 4 parallel encoder-decoder branches, ELU, BatchNorm",
        "config_key":   "unet",
        "filename":     ckpt_names.get("unet", "unet.pth"),
        "notes":        "Trained on MUSDB18 stereo mixture → 4 stems",
    },
    {
        "component":    "Beat-Net (Beat/Downbeat/BPM)",
        "architecture": "3-branch factorised axial transformer, 3-class output",
        "config_key":   "beat",
        "filename":     ckpt_names.get("beat", "beat_net.pth"),
        "notes":        "Input: drums + bass + other spectrograms",
    },
    {
        "component":    "Chord-Net (Chord Recognition / Key)",
        "architecture": "Single BaseTransformer, 122-class output",
        "config_key":   "chord",
        "filename":     ckpt_names.get("chord", "chord_net.pth"),
        "notes":        "Input: accompaniment spec, conditioned on beat logits",
    },
    {
        "component":    "Pitch-Net (Vocal Melody)",
        "architecture": "PitchEmbedding + PitchEncoder + lyric-conditioned decoder, 850-class",
        "config_key":   "pitch",
        "filename":     ckpt_names.get("pitch", "pitch_net.pth"),
        "notes":        "Input: vocal spec, optionally conditioned on Wav2Vec2 matrix",
    },
    {
        "component":    "Segment-Net (Music Structure)",
        "architecture": "SegmentEmbeddings + BaseTransformer, 10-class output",
        "config_key":   "segment",
        "filename":     ckpt_names.get("segment", "segment_net.pth"),
        "notes":        "Input: full-mix spec, conditioned on chord logits",
    },
    {
        "component":    "Whisper (Lyrics ASR)",
        "architecture": "OpenAI Whisper — downloads automatically via openai-whisper",
        "config_key":   "whisper",
        "filename":     "auto-downloaded by openai-whisper",
        "notes":        "pip install openai-whisper  (model downloads on first use)",
    },
    {
        "component":    "Wav2Vec2 (Lyric-Pitch Conditioning)",
        "architecture": "facebook/wav2vec2-base-960h — downloads automatically via HuggingFace",
        "config_key":   "wav2vec2",
        "filename":     "auto-downloaded by transformers",
        "notes":        "pip install transformers  (model downloads on first use)",
    },
]

print("=" * 80)
print("  CHORDIFY — Checkpoint Status")
print("=" * 80)
print(f"  Checkpoints directory: {checkpoints_dir}\n")

all_ok   = True
pth_keys = {"unet", "beat", "chord", "pitch", "segment"}

for c in COMPONENTS:
    key      = c["config_key"]
    filename = c["filename"]

    if key in pth_keys:
        full_path = Path(checkpoints_dir) / filename
        present   = full_path.exists()
        status    = "✓ PRESENT" if present else "✗ MISSING"
        if not present:
            all_ok = False
    else:
        # Whisper / Wav2Vec2 — not file-based checkpoints
        status = "– AUTO-DOWNLOAD"
        present = True

    print(f"  [{status:<14}]  {c['component']}")
    print(f"                       File : {filename}")
    if not present:
        print(f"                       Path : {full_path}")
        print(f"                       Note : {c['notes']}")
    print()

print("-" * 80)
if all_ok:
    print("  All PyTorch checkpoints are present.")
else:
    print("  Some checkpoints are MISSING.")
    print()
    print("  How to obtain checkpoints:")
    print("  ─────────────────────────────────────────────────────────────────")
    print("  The Lamucal/Chordify neural-network checkpoints are NOT distributed")
    print("  with this repository.  You have two options:")
    print()
    print("  Option A — Use the live Lamucal service to export checkpoints")
    print("             (if the project authors make them available):")
    print("             https://lamucal.ai")
    print()
    print("  Option B — Train your own checkpoints using the MUSDB18 dataset")
    print("             already present in your Chordify/train and Chordify/test")
    print("             directories.  Training scripts are a planned future")
    print("             addition to this project.")
    print()
    print("  In the meantime:")
    print("  • The pipeline will run with RANDOM WEIGHTS (for architecture demo).")
    print("  • Model outputs will be meaningless without trained checkpoints.")
    print("  • Tests 1–14 (structural tests) will still pass.")
    print("  ─────────────────────────────────────────────────────────────────")
print("=" * 80)
