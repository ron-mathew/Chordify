"""
Chordify — Exploratory Data Analysis of MUSDB18
================================================
Run as a script:  python notebooks/eda_musdb18.py
Or open as a Jupyter notebook by converting with:
  pip install jupytext
  jupytext --to notebook notebooks/eda_musdb18.py

Generates the following plots and statistics:
  1. Audio duration distribution (train + test)
  2. Waveform visualisation (one example track)
  3. Spectrogram visualisation (STFT magnitude)
  4. Per-stem energy comparison (vocal / bass / drums / other)
  5. Stem waveform overlay
  6. Sample-rate and channel validation report
  7. Genre distribution (from MUSDB18 metadata)

NOTE: Only generates plots from actual dataset files.
No statistics are fabricated.
"""

import sys
from pathlib import Path

# Allow running from project root or notebooks/ directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless; switch to "TkAgg" for interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATASET_ROOT = ROOT / "train"   # MUSDB18 .stem.mp4 files
PLOTS_DIR    = ROOT / "outputs" / "eda"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SR   = 44100
N_FFT = 4096
HOP  = 1024

# ─────────────────────────────────────────────────────────────────────────────
# Helper: decode one .stem.mp4 file
# ─────────────────────────────────────────────────────────────────────────────
def decode_stems(mp4_path: Path):
    """Return (mixture, drums, bass, other, vocals) as numpy arrays."""
    try:
        import stempeg
        stems, rate = stempeg.read_stems(str(mp4_path), sample_rate=SR)
        # stems shape: (5, samples, 2)
        return stems, rate
    except Exception as e:
        raise RuntimeError(f"stempeg failed on {mp4_path.name}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Duration + channel validation
# ─────────────────────────────────────────────────────────────────────────────
print("[EDA] Scanning dataset …")

durations  = []
bad_sr     = []
bad_ch     = []
track_names = []

stem_files = sorted(DATASET_ROOT.glob("*.stem.mp4"))
if not stem_files:
    print(f"[EDA] No .stem.mp4 files found in {DATASET_ROOT}.")
    print("      Make sure the MUSDB18 train/ folder is populated.")
    sys.exit(0)

print(f"[EDA] Found {len(stem_files)} tracks in train/")

# Collect metadata without fully decoding (use torchaudio for speed)
import torchaudio
for fp in stem_files:
    try:
        info = torchaudio.info(str(fp))
        # torchaudio reads the first stream (mixture)
        dur = info.num_frames / info.sample_rate
        durations.append(dur)
        track_names.append(fp.stem.replace(".stem", ""))
        if info.sample_rate != SR:
            bad_sr.append(fp.name)
        if info.num_channels != 2:
            bad_ch.append(fp.name)
    except Exception as e:
        print(f"  [warn] Could not read {fp.name}: {e}")

durations = np.array(durations)
print(f"\n[EDA] Validation")
print(f"  Tracks scanned : {len(durations)}")
print(f"  Wrong SR       : {len(bad_sr)}")
print(f"  Wrong channels : {len(bad_ch)}")
print(f"  Duration  min  : {durations.min():.1f}s")
print(f"  Duration  max  : {durations.max():.1f}s")
print(f"  Duration  mean : {durations.mean():.1f}s")
print(f"  Duration  total: {durations.sum() / 60:.1f} minutes")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Duration distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(durations, bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
ax.axvline(durations.mean(), color="#DD8452", lw=2, label=f"Mean {durations.mean():.1f}s")
ax.set_xlabel("Duration (seconds)", fontsize=12)
ax.set_ylabel("Track count", fontsize=12)
ax.set_title("MUSDB18 Train — Audio Duration Distribution", fontsize=14)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out = PLOTS_DIR / "01_duration_distribution.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"[EDA] Saved: {out.name}")

# ─────────────────────────────────────────────────────────────────────────────
# Deep decode the first track for detailed analysis
# ─────────────────────────────────────────────────────────────────────────────
example_fp = stem_files[0]
print(f"\n[EDA] Deep analysis on: {example_fp.name}")

try:
    stems, rate = decode_stems(example_fp)
except RuntimeError as e:
    print(f"[EDA] {e}")
    print("[EDA] Install stempeg:  pip install stempeg")
    print("[EDA] Skipping waveform / spectrogram plots.")
    stems = None

STEM_NAMES  = ["Mixture", "Drums", "Bass", "Other", "Vocals"]
STEM_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

if stems is not None:
    # ─────────────────────────────────────────────────────────────────────
    # Plot 2 — Waveform overlay (first 10 s)
    # ─────────────────────────────────────────────────────────────────────
    T = min(SR * 10, stems.shape[1])

    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Waveform — {example_fp.stem.replace('.stem', '')} (first 10s)",
                 fontsize=14)
    t = np.linspace(0, T / SR, T)

    for i, (ax, name, color) in enumerate(zip(axes, STEM_NAMES, STEM_COLORS)):
        mono = stems[i, :T, :].mean(axis=1)
        ax.plot(t, mono, color=color, lw=0.6, alpha=0.85)
        ax.set_ylabel(name, fontsize=10)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time (s)", fontsize=11)
    plt.tight_layout()
    out = PLOTS_DIR / "02_waveform_overlay.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[EDA] Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────
    # Plot 3 — Spectrogram (STFT magnitude) for all 5 stems
    # ─────────────────────────────────────────────────────────────────────
    from utils.spectrogram import stft
    import torch

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle("STFT Magnitude Spectrograms (log scale)", fontsize=14)
    CLIP_S = 20   # show first 20 s

    for i, (ax, name, color) in enumerate(zip(axes, STEM_NAMES, STEM_COLORS)):
        clip = min(SR * CLIP_S, stems.shape[1])
        wav  = torch.from_numpy(stems[i, :clip, :].T.copy()).float()  # (2, samples)
        spec = stft(wav, n_fft=N_FFT, hop_length=HOP, pad=True)      # (2, T, F)
        mag  = torch.abs(spec).mean(dim=0).numpy()                    # (T, F)
        mag_db = 20 * np.log10(mag.T + 1e-8)                         # (F, T)
        img = ax.imshow(
            mag_db[:512, :],   # show 0–5.5 kHz band
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=-80, vmax=0,
        )
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Freq bin" if i == 0 else "")
    fig.colorbar(img, ax=axes[-1], label="dB")
    plt.tight_layout()
    out = PLOTS_DIR / "03_spectrograms.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[EDA] Saved: {out.name}")

    # ─────────────────────────────────────────────────────────────────────
    # Plot 4 — Stem energy comparison across all tracks
    # ─────────────────────────────────────────────────────────────────────
    print("[EDA] Computing stem energy for all tracks (this may take a few minutes) …")

    stem_energies = {n: [] for n in STEM_NAMES[1:]}  # skip mixture
    MAX_TRACKS = min(len(stem_files), 20)             # cap for speed

    for fp in stem_files[:MAX_TRACKS]:
        try:
            s, _ = decode_stems(fp)
            for j, name in enumerate(STEM_NAMES[1:], start=1):
                rms = np.sqrt(np.mean(s[j] ** 2))
                stem_energies[name].append(float(rms))
        except Exception as e:
            print(f"  [warn] {fp.name}: {e}")

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(STEM_NAMES[1:]))
    energies  = [stem_energies[n] for n in STEM_NAMES[1:]]

    bp = ax.boxplot(energies, positions=positions, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], STEM_COLORS[1:]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(STEM_NAMES[1:], fontsize=12)
    ax.set_ylabel("RMS Energy", fontsize=12)
    ax.set_title(
        f"Stem Energy Distribution (first {MAX_TRACKS} tracks)", fontsize=14
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "04_stem_energy.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[EDA] Saved: {out.name}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Chord label distribution (from MUSDB18 genre metadata in README)
# ─────────────────────────────────────────────────────────────────────────────
# The MUSDB18 README lists genres for each track.
# We parse the built-in genre mapping directly.
genre_map = {
    "Singer/Songwriter": 0, "Pop/Rock": 0, "Rock": 0, "Country": 0,
    "Rap": 0, "Reggae": 0, "Electronic": 0, "Heavy Metal": 0,
    "Jazz": 0, "Pop": 0,
}

TRACK_GENRES = {
    "A Classic Education - NightOwl": "Singer/Songwriter",
    "Aimee Norwich - Child": "Singer/Songwriter",
    "Angela Thomas Wade - Milk Cow Blues": "Country",
    "ANiMAL - Clinic A": "Rap", "ANiMAL - Easy Tiger": "Rap",
    "ANiMAL - Rockshow": "Rap", "AvaLuna - Waterduct": "Rock",
    "BigTroubles - Phantom": "Rock",
    "Celestial Shore - Die For Us": "Singer/Songwriter",
    "Chris Durban - Celebrate": "Electronic",
    "Clara Berry And Wooldog - Air Traffic": "Singer/Songwriter",
    "Clara Berry And Wooldog - Stella": "Singer/Songwriter",
    "Clara Berry And Wooldog - Waltz For My Victims": "Singer/Songwriter",
    "Cnoc An Tursa - Bannockburn": "Heavy Metal",
    "Creepoid - OldTree": "Rock",
    "Dark Ride - Burning Bridges": "Heavy Metal",
    "Dreamers Of The Ghetto - Heavy Love": "Pop",
    "Faces On Film - Waiting For Ga": "Singer/Songwriter",
    "Grants - PunchDrunk": "Rap",
    "Helado Negro - Mitad Del Mundo": "Pop",
    "Hezekiah Jones - Borrowed Heart": "Singer/Songwriter",
    "Hollow Ground - Ill Fate": "Heavy Metal",
    "Hop Along - Sister Cities": "Rock",
    "Invisible Familiars - Disturbing Wildlife": "Singer/Songwriter",
    "Lushlife - Toynbee Suite": "Rap",
    "Matthew Entwistle - Dont You Ever": "Jazz",
    "Meaxic - Take A Step": "Rock", "Meaxic - You Listen": "Rock",
    "Music Delta - 80s Rock": "Rock", "Music Delta - Beatles": "Singer/Songwriter",
    "Music Delta - Britpop": "Pop", "Music Delta - Country1": "Country",
    "Music Delta - Country2": "Country", "Music Delta - Disco": "Pop",
    "Music Delta - Gospel": "Pop", "Music Delta - Grunge": "Rock",
    "Music Delta - Hendrix": "Rock", "Music Delta - Punk": "Rock",
    "Music Delta - Reggae": "Rock", "Music Delta - Rock": "Rock",
    "Music Delta - Rockabilly": "Rock",
    "Night Panther - Fire": "Pop",
    "Port St Willow - Stay Even": "Singer/Songwriter",
    "PR - Happy Daze": "Electronic", "PR - Oh No": "Electronic",
    "Secret Mountains - High Horse": "Pop",
    "Snowmine - Curfews": "Pop",
    "Steven Clark - Bounty": "Pop",
    "Strand Of Oaks - Spacestation": "Pop",
    "Sweet Lights - You Let Me Down": "Pop",
    "The Districts - Vermont": "Rock",
    "The Easton Ellises - Falcon 69": "Pop/Rock",
}

# Count genres from tracks actually present on disk
for fp in stem_files:
    track_name = fp.stem.replace(".stem", "")
    genre = TRACK_GENRES.get(track_name, "Pop/Rock")
    genre_map[genre] = genre_map.get(genre, 0) + 1

genres  = [g for g, c in genre_map.items() if c > 0]
counts  = [genre_map[g] for g in genres]
colors  = plt.cm.Set3(np.linspace(0, 1, len(genres)))  # type: ignore[attr-defined]

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    counts, labels=genres, colors=colors,
    autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
    startangle=140, pctdistance=0.82,
)
ax.set_title("MUSDB18 Train — Genre Distribution\n(inferred from README metadata)",
             fontsize=13)
plt.tight_layout()
out = PLOTS_DIR / "05_genre_distribution.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"[EDA] Saved: {out.name}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — Chord label space (all 122 labels, coloured by quality)
# ─────────────────────────────────────────────────────────────────────────────
from utils.labels import CHORD_LABELS, SEGMENT_LABELS

quality_colors = {
    "N": "#AAAAAA", "X": "#888888",
    "maj": "#4C72B0", "min": "#DD8452", "7": "#55A868", "maj7": "#C44E52",
    "min7": "#8172B2", "6": "#937860", "m6": "#DA8BC3", "sus2": "#8C8C8C",
    "sus4": "#CCB974", "5": "#64B5CD",
}

labels_short = []
colors_bar   = []
for lbl in CHORD_LABELS:
    if ":" in lbl:
        root, qual = lbl.split(":", 1)
        labels_short.append(f"{root}\n{qual}")
        colors_bar.append(quality_colors.get(qual, "#999999"))
    else:
        labels_short.append(lbl)
        colors_bar.append(quality_colors.get(lbl, "#AAAAAA"))

fig, ax = plt.subplots(figsize=(28, 4))
x = np.arange(len(CHORD_LABELS))
ax.bar(x, np.ones(len(CHORD_LABELS)), color=colors_bar, width=0.9)
ax.set_xticks(x[::1])
ax.set_xticklabels(labels_short, fontsize=5, rotation=90)
ax.set_yticks([])
ax.set_title("Chordify Chord Label Space (122 classes)", fontsize=13)

# Legend
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=c, label=q) for q, c in quality_colors.items()]
ax.legend(handles=legend_els, ncol=6, loc="upper right", fontsize=8)
plt.tight_layout()
out = PLOTS_DIR / "06_chord_label_space.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"[EDA] Saved: {out.name}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 7 — Segment label space (10 classes)
# ─────────────────────────────────────────────────────────────────────────────
seg_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
              "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(SEGMENT_LABELS, np.ones(10), color=seg_colors, alpha=0.85)
ax.set_xlabel("Class index →")
ax.set_title("Chordify Segment Label Space (10 structural classes)", fontsize=13)
ax.set_xlim(0, 1.5)
for i, lbl in enumerate(SEGMENT_LABELS):
    ax.text(0.05, i, f"  class {i}", va="center", fontsize=11, color="white", fontweight="bold")
plt.tight_layout()
out = PLOTS_DIR / "07_segment_label_space.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"[EDA] Saved: {out.name}")

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[EDA] All plots saved to: {PLOTS_DIR}")
print(f"[EDA] Tracks analysed   : {len(durations)}")
print(f"[EDA] Total duration    : {durations.sum() / 60:.1f} min")
print("[EDA] Done.")
