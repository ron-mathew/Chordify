"""
Chordify CLI entry point.

Usage
-----
  python scripts/run_chordify.py --audio path/to/song.wav
  python scripts/run_chordify.py --audio path/to/song.wav --save-stems
  python scripts/run_chordify.py --audio path/to/song.wav --device cpu
"""
import argparse
import json
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from chordify import Chordify


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chordify — AI multimodal music analysis & tab generation"
    )
    parser.add_argument(
        "--audio", "-a", required=True,
        help="Path to input audio file (.wav, .mp3, .flac, etc.)"
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to config.yaml (default: config/config.yaml)"
    )
    parser.add_argument(
        "--device", "-d", default=None,
        help="Device override: 'cpu' or 'cuda' (default: auto)"
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Directory to save results (default: outputs/<audio_stem>/)"
    )
    parser.add_argument(
        "--save-stems", action="store_true",
        help="Save the 4 separated stems as individual .wav files"
    )
    parser.add_argument(
        "--print-tab", action="store_true",
        help="Print ASCII guitar tab to stdout"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    chordify = Chordify(config_path=args.config, device=args.device)
    result = chordify.analyze(
        audio_path=args.audio,
        save_stems=args.save_stems,
        output_dir=args.output_dir,
    )

    # Summary
    meta = result["metadata"]
    print("\n── Chordify Results ──────────────────────────")
    print(f"  Duration : {meta['duration']:.2f}s")
    print(f"  BPM      : {meta['bpm']:.1f}")
    print(f"  Key      : {meta['key']}")
    print(f"  Beats    : {len(result['beats'])}")
    print(f"  Chords   : {len(result['chords'])} segments")
    print(f"  Sections : {len(result['sections'])}")
    print(f"  Lyrics   : {len(result['lyrics'])} segments")
    print("──────────────────────────────────────────────")

    if args.print_tab:
        print("\n" + result["tabs"]["ascii_tab"])


if __name__ == "__main__":
    main()
