"""
Chordify dataset preprocessing script.

Reads MUSDB18 .stem.mp4 files, decodes the 5 streams using musdb/stempeg,
and saves each stem as a .wav file for direct use in training/evaluation.

Usage
-----
  pip install musdb stempeg
  python scripts/preprocess.py --dataset-dir data/musdb18 --output-dir data/stems

The script will never modify or delete the original .stem.mp4 files.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Chordify MUSDB18 preprocessor")
    p.add_argument(
        "--dataset-dir", "-d", required=True,
        help="Root directory of MUSDB18 (contains train/ and test/)"
    )
    p.add_argument(
        "--output-dir", "-o", required=True,
        help="Directory where extracted .wav stems will be saved"
    )
    p.add_argument(
        "--subset", choices=["train", "test", "all"], default="all",
        help="Which subset to process (default: all)"
    )
    p.add_argument(
        "--sample-rate", type=int, default=44100,
        help="Target sample rate for exported stems (default: 44100)"
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of tracks (useful for quick testing)"
    )
    return p.parse_args()


STEM_NAMES = ["mixture", "drums", "bass", "other", "vocals"]


def process_track(track, output_root: Path, sample_rate: int) -> None:
    """
    Decode one MUSDB track and save each stem as a WAV file.
    """
    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        raise ImportError(
            "[Chordify] Missing dependencies. Run:\n"
            "  pip install musdb stempeg soundfile numpy"
        )

    track_dir = output_root / track.name
    track_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(STEM_NAMES):
        out_path = track_dir / f"{name}.wav"
        if out_path.exists():
            print(f"  [skip] {track.name}/{name}.wav already exists")
            continue

        # musdb lazy-loads audio on attribute access
        if name == "mixture":
            audio = track.audio          # (samples, 2)
        else:
            audio = track.stems[i]       # (samples, 2)

        if track.rate != sample_rate:
            try:
                import resampy
                audio = resampy.resample(audio.T, track.rate, sample_rate).T
            except ImportError:
                print(
                    f"  [warn] resampy not installed; saving at native rate {track.rate} Hz"
                )

        sf.write(str(out_path), audio, sample_rate)
        print(f"  [saved] {track.name}/{name}.wav  shape={audio.shape}")


def main():
    args = parse_args()

    try:
        import musdb
    except ImportError:
        print("[Chordify] musdb not installed. Run: pip install musdb stempeg")
        sys.exit(1)

    subsets = ["train", "test"] if args.subset == "all" else [args.subset]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for subset in subsets:
        print(f"\n[Chordify] Processing subset: {subset}")
        db = musdb.DB(root=args.dataset_dir, subsets=subset, is_wav=False)
        tracks = list(db.tracks)
        if args.limit:
            tracks = tracks[: args.limit]

        for i, track in enumerate(tracks, 1):
            print(f"  [{i}/{len(tracks)}] {track.name}")
            try:
                process_track(track, output_root / subset, args.sample_rate)
            except Exception as e:
                print(f"  [ERROR] {track.name}: {e}")

    print(f"\n[Chordify] Preprocessing complete. Output: {output_root}")


if __name__ == "__main__":
    main()
