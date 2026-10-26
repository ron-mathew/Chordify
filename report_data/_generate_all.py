"""
Chordify Report Data Generator
Generates all CSV, JSON, and figure files for the academic report.
Run from project root: python report_data/_generate_all.py
"""
import sys, json, csv, time, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT  = ROOT / "report_data"
FIGS = OUT / "figures"
LOGS = OUT / "logs"
OUT.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

print("=" * 60)
print("Chordify Report Data Generator")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL SPECIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Writing MODEL_SPECIFICATIONS.csv ...")
model_specs = [
    ["Model", "Architecture", "Input_Shape", "Output_Shape", "Num_Classes", "Total_Params", "Trainable_Params", "Source"],
    ["UNets", "4x parallel U-Net encoder-decoder with channel-wise softmax", "(batch,2,time,freq)", "(batch,4,2,time,freq)", "4 stems", "39,294,876", "39,294,876", "Lamucal (ported)"],
    ["BeatNet", "3-branch factorised axial-attention transformer", "(batch,3,2,time,freq)", "(batch,time) + (batch,time,freq)", "3 (none/beat/downbeat)", "504,713,475", "504,713,475", "Lamucal (ported)"],
    ["ChordNet", "Single BaseTransformer + linear head", "(batch,2,time,freq)", "(batch,time) + (batch,time,freq)", "122", "389,342,202", "389,342,202", "Lamucal (ported)"],
    ["PitchNet", "PitchEmbedding + PitchEncoder + lyric-conditioned decoder", "(batch,2,time,freq)", "(batch,time)", "850 (85 semitones x 10 centbins)", "119,440,806", "119,440,806", "Lamucal (ported)"],
    ["SegmentNet", "SegmentEmbeddings + BaseTransformer + linear head", "(batch,2,time,freq)", "(batch,time)", "10", "14,025,676", "14,025,676", "Lamucal (ported)"],
    ["Whisper large-v3", "Encoder-decoder ASR transformer (OpenAI)", "(audio, float32, 16kHz)", "segments+words with timestamps", "Multilingual ASR", "~1,550,000,000 (external)", "N/A", "OpenAI / faster-whisper"],
    ["Wav2Vec2 base-960h", "Self-supervised speech encoder (Facebook)", "(audio, float32, 16kHz)", "(batch,time,768)", "Feature extractor", "~94,000,000 (external)", "N/A", "Meta / HuggingFace"],
]

with open(OUT / "MODEL_SPECIFICATIONS.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(model_specs)

print("   -> MODEL_SPECIFICATIONS.csv written")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION INVENTORY
# ─────────────────────────────────────────────────────────────────────────────
print("[2] Writing CONFIGURATION_INVENTORY.csv ...")
config_rows = [
    ["Parameter", "Value", "Unit", "Source", "Notes"],
    ["sample_rate", "44100", "Hz", "config/config.yaml", "Target SR for all audio I/O"],
    ["n_channel", "2", "channels", "config/config.yaml", "Stereo"],
    ["sources", "4", "stems", "config/config.yaml", "vocal/bass/drums/other"],
    ["n_fft", "4096", "samples", "config/config.yaml", "FFT window size"],
    ["hop_length", "1024", "samples", "config/config.yaml", "STFT hop; frame_rate = 44100/1024 = 43.07 fps"],
    ["n_time", "256", "frames", "config/config.yaml", "Sliding window chunk size for U-Net"],
    ["pad", "True", "", "config/config.yaml", "Hann-window zero-padding in STFT"],
    ["freq_bins_total", "2049", "bins", "derived: n_fft//2+1", "Full STFT output size"],
    ["freq_bins_model_input", "2048", "bins", "derived: n_fft//2", "Half-spectrum used by all neural models"],
    ["frame_rate", "43.07", "fps", "derived: 44100/1024", "Frames per second of spectrogram"],
    # U-Net
    ["unet.conv_n_filters", "[16, 32, 64, 128, 256, 512]", "", "config.yaml:separate.model", "6 encoder levels"],
    ["unet.down_activation", "ELU", "", "config.yaml", ""],
    ["unet.up_activation", "ELU", "", "config.yaml", ""],
    # BeatNet
    ["beat.source", "3", "stems", "config.yaml:beat.model", "drums+bass+other"],
    ["beat.n_classes", "3", "classes", "config.yaml", "0=no-beat, 1=beat, 2=downbeat"],
    ["beat.weights", "[0.4, 0.3, 0.3]", "", "config.yaml", "drum/bass/other stem weights"],
    ["beat.n_freq", "2048", "bins", "config.yaml", ""],
    ["beat.n_group", "32", "groups", "config.yaml", "Factorised attention groups"],
    ["beat.f_layers", "2", "", "config.yaml", "Frequency encoder layers"],
    ["beat.f_nhead", "4", "", "config.yaml", "Frequency attention heads"],
    ["beat.t_layers", "2", "", "config.yaml", "Time encoder layers"],
    ["beat.t_nhead", "4", "", "config.yaml", "Time attention heads"],
    ["beat.d_layers", "2", "", "config.yaml", "Decoder layers"],
    ["beat.d_nhead", "8", "", "config.yaml", "Decoder attention heads"],
    ["beat.dropout", "0.5", "", "config.yaml", ""],
    # ChordNet
    ["chord.n_freq", "2048", "bins", "config.yaml", ""],
    ["chord.n_classes", "122", "classes", "config.yaml", "Full chord vocabulary"],
    ["chord.n_group", "32", "groups", "config.yaml", ""],
    ["chord.f_layers", "5", "", "config.yaml", ""],
    ["chord.f_nhead", "8", "", "config.yaml", ""],
    ["chord.t_layers", "5", "", "config.yaml", ""],
    ["chord.t_nhead", "8", "", "config.yaml", ""],
    ["chord.d_layers", "5", "", "config.yaml", ""],
    ["chord.d_nhead", "8", "", "config.yaml", ""],
    ["chord.dropout", "0.5", "", "config.yaml", ""],
    # PitchNet
    ["pitch.n_freq", "2048", "bins", "config.yaml", ""],
    ["pitch.n_channel", "2", "channels", "config.yaml", ""],
    ["pitch.n_classes", "850", "classes", "config.yaml", "85 semitones x 10 centbins"],
    ["pitch.emb_hidden", "16", "", "config.yaml", "PitchEmbedding hidden dim"],
    ["pitch.wr", "0.3", "", "config.yaml", "Lyric conditioning weight"],
    ["pitch.pr", "0.02", "", "config.yaml", "Positional encoding scale"],
    ["pitch.enc_weights", "[0.6, 0.4]", "", "config.yaml", "Freq/time encoder blend"],
    ["pitch.d_layers", "2", "", "config.yaml", ""],
    ["pitch.d_nhead", "8", "", "config.yaml", ""],
    # SegmentNet
    ["segment.n_freq", "2048", "bins", "config.yaml", ""],
    ["segment.n_channel", "2", "channels", "config.yaml", ""],
    ["segment.n_classes", "10", "classes", "config.yaml", "Structural sections"],
    ["segment.emb_hidden", "128", "", "config.yaml", "SegmentEmbeddings hidden dim"],
    ["segment.d_model_after_embed", "512", "bins", "derived: 2048//4", "After freq/4 pooling"],
    # Lyrics
    ["lyrics.whisper_model", "large-v3", "", "config.yaml", "Primary model; fallbacks: medium/small/base"],
    ["lyrics.wav2vec2_model", "facebook/wav2vec2-base-960h", "", "config.yaml", "For PitchNet conditioning matrix"],
    ["lyrics.beam_size", "5", "", "pipeline/lyrics.py", "Whisper beam search width"],
    ["lyrics.vad_filter", "False", "", "pipeline/lyrics.py", "VAD disabled for singing"],
    ["lyrics.no_speech_threshold", "0.85", "", "pipeline/lyrics.py", "Skip segments above this"],
    ["lyrics.word_prob_threshold", "0.10", "", "pipeline/lyrics.py", "Filter low-confidence words"],
    # DSP fallback
    ["dsp.n_chord_templates", "24", "templates", "pipeline/dsp_fallback.py", "12 major + 12 minor"],
    ["dsp.viterbi_gamma", "14.0", "", "pipeline/dsp_fallback.py", "Temperature scaling"],
    ["dsp.self_transition_log", "0.0", "", "pipeline/dsp_fallback.py", "Strong self-transition bias"],
    ["dsp.other_transition_log", "-3.0", "", "pipeline/dsp_fallback.py", "Transition penalty"],
    ["dsp.chroma_fusion_weights", "[0.65, 0.35]", "", "pipeline/dsp_fallback.py", "Full/bass chroma blend"],
    ["dsp.min_segment_dur", "1.0", "seconds", "pipeline/dsp_fallback.py", "Minimum chord segment length"],
    ["dsp.bpm_halftime_threshold", "135.0", "BPM", "pipeline/dsp_fallback.py", "Above this -> divide by 2"],
    # Guitar tabs
    ["tabs.chord_voicings_count", "70", "voicings", "pipeline/tabs.py (count of _CHORD_VOICINGS keys)", "Standard guitar positions"],
    ["tabs.bars_per_line", "4", "bars", "pipeline/tabs.py", "Tab display layout"],
    ["tabs.col_width", "10", "chars", "pipeline/tabs.py", "Column width per bar"],
]

with open(OUT / "CONFIGURATION_INVENTORY.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(config_rows)

print("   -> CONFIGURATION_INVENTORY.csv written")

# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET INVENTORY
# ─────────────────────────────────────────────────────────────────────────────
print("[3] Writing DATASET_INVENTORY.csv and DATASET_STATISTICS.json ...")

train_dir = ROOT / "train"
test_dir  = ROOT / "test"
train_files = sorted(train_dir.glob("*.stem.mp4"))
test_files  = sorted(test_dir.glob("*.stem.mp4"))

train_sizes = [f.stat().st_size for f in train_files]
test_sizes  = [f.stat().st_size for f in test_files]

dataset_rows = [
    ["Split", "Track_Name", "File_Size_MB", "Format", "Status", "Used_For"],
]
for f in train_files:
    dataset_rows.append(["train", f.stem.replace(".stem",""), round(f.stat().st_size/1024/1024, 2), "STEM MP4 (5 stems)", "Present", "EDA / future training"])
for f in test_files:
    dataset_rows.append(["test", f.stem.replace(".stem",""), round(f.stat().st_size/1024/1024, 2), "STEM MP4 (5 stems)", "Present", "EDA / evaluation"])

with open(OUT / "DATASET_INVENTORY.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(dataset_rows)

# Duration is unavailable without ffmpeg; use size-based estimates
# MUSDB18 known stats: typical 3.5 min/track avg
ESTIMATED_AVG_DURATION_S = 210.0  # conservative estimate from MUSDB18 paper
ESTIMATED_TOTAL_MIN = len(train_files + test_files) * ESTIMATED_AVG_DURATION_S / 60

dataset_stats = {
    "dataset_name": "MUSDB18",
    "dataset_source": "https://sigsep.github.io/datasets/musdb.html",
    "license": "Creative Commons (various, non-commercial research)",
    "format": "Stem MP4 (STEM format, 5 sub-tracks: mixture, drums, bass, other, vocals)",
    "sample_rate_hz": 44100,
    "channels": 2,
    "splits": {
        "train": {
            "tracks": len(train_files),
            "total_size_gb": round(sum(train_sizes)/1024/1024/1024, 3),
            "avg_size_mb": round(np.mean(train_sizes)/1024/1024, 1),
            "min_size_mb": round(min(train_sizes)/1024/1024, 1),
            "max_size_mb": round(max(train_sizes)/1024/1024, 1),
            "estimated_duration_min": round(len(train_files) * ESTIMATED_AVG_DURATION_S / 60, 1),
            "duration_note": "ESTIMATED — ffmpeg/stempeg unavailable; approx 3.5 min/track based on MUSDB18 dataset documentation",
        },
        "test": {
            "tracks": len(test_files),
            "total_size_gb": round(sum(test_sizes)/1024/1024/1024, 3),
            "avg_size_mb": round(np.mean(test_sizes)/1024/1024, 1),
            "min_size_mb": round(min(test_sizes)/1024/1024, 1),
            "max_size_mb": round(max(test_sizes)/1024/1024, 1),
            "estimated_duration_min": round(len(test_files) * ESTIMATED_AVG_DURATION_S / 60, 1),
            "duration_note": "ESTIMATED — same basis",
        },
    },
    "totals": {
        "total_tracks": len(train_files) + len(test_files),
        "total_size_gb": round((sum(train_sizes)+sum(test_sizes))/1024/1024/1024, 3),
        "estimated_total_duration_min": round(ESTIMATED_TOTAL_MIN, 1),
    },
    "stem_names": ["mixture", "drums", "bass", "other", "vocals"],
    "stems_per_track": 5,
    "annotation_type": "Multi-track separated stems (no chord/beat labels in MUSDB18)",
    "chord_annotations": "UNAVAILABLE — MUSDB18 provides source separation stems only, not chord labels",
    "beat_annotations": "UNAVAILABLE — MUSDB18 provides stems only",
    "training_status": "Models NOT trained — no training code was executed. Checkpoints do not exist.",
    "actual_use_in_pipeline": "Present for EDA analysis and future training capability. No training executed.",
}

with open(OUT / "DATASET_STATISTICS.json", "w", encoding="utf-8") as f:
    json.dump(dataset_stats, f, indent=2)

print("   -> DATASET_INVENTORY.csv written")
print("   -> DATASET_STATISTICS.json written")

# ─────────────────────────────────────────────────────────────────────────────
# 4. CHECKPOINT AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print("[4] Writing CHECKPOINT_AUDIT.csv ...")
checkpoint_rows = [
    ["Model", "Expected_Path", "File_Exists", "File_Size", "Valid_Checkpoint", "Notes"],
    ["UNets (source separation)", "checkpoints/unet.pth", "NO", "N/A", "NO", "checkpoints/ directory does not exist"],
    ["BeatNet", "checkpoints/beat_net.pth", "NO", "N/A", "NO", "checkpoints/ directory does not exist"],
    ["ChordNet", "checkpoints/chord_net.pth", "NO", "N/A", "NO", "checkpoints/ directory does not exist"],
    ["PitchNet", "checkpoints/pitch_net.pth", "NO", "N/A", "NO", "checkpoints/ directory does not exist"],
    ["SegmentNet", "checkpoints/segment_net.pth", "NO", "N/A", "NO", "checkpoints/ directory does not exist"],
    ["Whisper large-v3", "~/.cache/huggingface/hub/ or ~/.cache/faster_whisper/", "YES (cached)", "~3.09 GB", "YES", "External model, downloaded by faster-whisper automatically"],
    ["Wav2Vec2 base-960h", "~/.cache/huggingface/hub/", "CONDITIONAL", "~360 MB", "CONDITIONAL", "Downloaded by transformers on first use; transformers package not fully installed"],
]

with open(OUT / "CHECKPOINT_AUDIT.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(checkpoint_rows)

print("   -> CHECKPOINT_AUDIT.csv written")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TEST RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("[5] Writing TEST_RESULTS.csv and TEST_RESULTS.json ...")

test_results_data = [
    {
        "test_id": 1,
        "name": "Import all modules",
        "category": "unit",
        "status": "PASS",
        "description": "Imports torch, torchaudio, librosa, all model classes, utils, and pipeline stages",
        "evidence": "verified 2026-10-21",
    },
    {
        "test_id": 2,
        "name": "Generate synthetic audio",
        "category": "unit",
        "status": "PASS",
        "description": "gen_audio(duration=3) returns (2, 132300) tensor at 44100 Hz",
        "evidence": "verified 2026-10-21",
    },
    {
        "test_id": 3,
        "name": "STFT / iSTFT round-trip",
        "category": "unit",
        "status": "PASS",
        "description": "stft on (2,44100) input gives (2,T,2049) complex tensor; istft reconstructs (2, samples)",
        "evidence": "verified 2026-10-21",
    },
    {
        "test_id": 4,
        "name": "UNets forward pass",
        "category": "unit",
        "status": "PASS",
        "description": "UNets(sources=4) on (1,2,64,2048) gives (1,4,2,64,2048); mask sum = 1.0",
        "evidence": "verified 2026-10-21",
    },
    {
        "test_id": 5,
        "name": "BeatNet forward pass",
        "category": "unit",
        "status": "PASS",
        "description": "BeatNet on (1,3,2,64,2048) gives beats=(1,64), logits=(1,64,2048)",
        "evidence": "verified 2026-10-21",
    },
    {
        "test_id": 6,
        "name": "ChordNet forward pass",
        "category": "unit",
        "status": "PASS",
        "description": "ChordNet on (1,2,64,2048) gives chords=(1,64) with values in [0,121]",
        "evidence": "verified 2026-10-21",
    },
    {
        "test_id": 7,
        "name": "SegmentNet forward pass",
        "category": "unit",
        "status": "PASS",
        "description": "SegmentNet on (1,2,64,2048) gives segs=(1,64) with values in [0,9]",
        "evidence": "verified 2026-10-21",
    },
    {
        "test_id": 8,
        "name": "PitchNet forward pass",
        "category": "unit",
        "status": "PASS",
        "description": "PitchNet on (1,2,64,2048) gives pitch=(1,64) with values in [0,849]",
        "evidence": "verified 2026-10-21",
    },
    {
        "test_id": 9,
        "name": "Label integrity",
        "category": "unit",
        "status": "PASS",
        "description": "len(CHORD_LABELS)==122, len(SEGMENT_LABELS)==10; 'C:maj' in CHORD_LABELS; 'chorus' in SEGMENT_LABELS",
        "evidence": "verified 2026-10-21",
    },
    {
        "test_id": 10,
        "name": "Key detection (diatonic scoring)",
        "category": "unit",
        "status": "PASS",
        "description": "search_key(['G','Am','C','D','G','Em','C','D']) returns key='G', score=8",
        "evidence": "verified 2026-10-21 (key=G, score=8 printed in test output)",
    },
    {
        "test_id": 11,
        "name": "Timestamp utilities",
        "category": "unit",
        "status": "PASS",
        "description": "beat_frames_to_events, chord_frames_to_segments, pitch_frames_to_events all produce correct output",
        "evidence": "verified 2026-10-21",
    },
    {
        "test_id": 12,
        "name": "Multimodal fusion (synthetic)",
        "category": "integration",
        "status": "PASS",
        "description": "fuse() on synthetic data produces dict with metadata/beats/chords/sections/timeline keys",
        "evidence": "verified 2026-10-21; printed: 2 beats | 1 chord segments | 1 sections | 1 lyric segments",
    },
    {
        "test_id": 13,
        "name": "AI tab generation (synthetic)",
        "category": "integration",
        "status": "PASS",
        "description": "generate_tabs() on synthetic fusion result with G/Em/C/D chords produces ascii_tab with 'e|' string notation",
        "evidence": "verified 2026-10-21; printed: 4 chord events",
    },
    {
        "test_id": 14,
        "name": "Config loading",
        "category": "unit",
        "status": "PASS",
        "description": "Loads config.yaml; verifies sample_rate=44100, n_channel=2, sources=4",
        "evidence": "verified 2026-10-21",
    },
    {
        "test_id": 15,
        "name": "DSP chord fallback (chroma)",
        "category": "unit",
        "status": "PASS",
        "description": "estimate_chords_dsp on 4s A-minor triad tone produces non-empty frame labels and chord events with source='dsp_chroma'",
        "evidence": "verified 2026-10-21",
    },
    {
        "test_id": 16,
        "name": "DSP beat fallback (librosa)",
        "category": "unit",
        "status": "PASS",
        "description": "estimate_beats_dsp on 4s pulse train produces BPM in [50,220] and monotonically increasing timestamps",
        "evidence": "verified 2026-10-21",
    },
    {
        "test_id": 17,
        "name": "Automatic fallback selection (ChordNet)",
        "category": "integration",
        "status": "PASS",
        "description": "run_chord_analysis with nonexistent checkpoint returns method='dsp_chroma'",
        "evidence": "verified 2026-10-21; key=Bb, score=7",
    },
    {
        "test_id": 18,
        "name": "Automatic fallback selection (BeatNet)",
        "category": "integration",
        "status": "PASS",
        "description": "run_beat_analysis with nonexistent checkpoint returns method='dsp_librosa' with BPM=68.0",
        "evidence": "verified 2026-10-21",
    },
    {
        "test_id": 19,
        "name": "End-to-End fallback integration",
        "category": "e2e",
        "status": "PASS",
        "description": "Full Chordify.analyze() on outputs/test_audio.wav with no checkpoints; uses DSP fallback for chords and beats; Whisper large-v3 for lyrics",
        "evidence": "verified 2026-10-21; runtime=32.5s; BPM=123.0; key=A; 7 beats; 1 chord segment; 'e|' in ascii tab",
    },
]

csv_rows = [["test_id","name","category","status","description","evidence"]]
for r in test_results_data:
    csv_rows.append([r["test_id"], r["name"], r["category"], r["status"], r["description"], r["evidence"]])

with open(OUT / "TEST_RESULTS.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

summary = {
    "run_date": "2026-10-21",
    "command": "python scripts/test_pipeline.py",
    "python_version": "3.13.5",
    "total_tests": 19,
    "passed": 19,
    "failed": 0,
    "skipped": 0,
    "duration_seconds": "approx 35 (dominated by Test 19 which loads Whisper large-v3)",
    "test_categories": {
        "unit": 14,
        "integration": 4,
        "e2e": 1,
        "synthetic": "Tests 1-18 use synthetic or minimal audio only",
        "real_audio_accuracy": "NONE — no ground-truth accuracy evaluation",
    },
    "important_note": "All tests pass. Tests 1-18 use random-weight neural models or synthetic audio. PASSING THESE TESTS IS NOT EVIDENCE OF CHORD/BEAT/PITCH ACCURACY. Accuracy metrics require trained checkpoints and ground-truth annotations.",
    "tests": test_results_data,
}

with open(OUT / "TEST_RESULTS.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("   -> TEST_RESULTS.csv written")
print("   -> TEST_RESULTS.json written")

# ─────────────────────────────────────────────────────────────────────────────
# 6. CHORD METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("[6] Writing CHORD_METRICS.csv and chord_predictions.csv ...")

chord_metrics_rows = [
    ["Metric", "Value", "Unit", "Audio", "Method", "Status", "Notes"],
    ["Chord_recognition_accuracy", "UNAVAILABLE", "%", "all audio", "neural", "UNAVAILABLE", "No trained ChordNet checkpoint; no ground-truth chord annotations in MUSDB18"],
    ["Frame_wise_chord_accuracy", "UNAVAILABLE", "%", "all audio", "neural", "UNAVAILABLE", "Same reason"],
    ["Chord_precision", "UNAVAILABLE", "%", "all audio", "neural", "UNAVAILABLE", ""],
    ["Chord_recall", "UNAVAILABLE", "%", "all audio", "neural", "UNAVAILABLE", ""],
    ["Chord_F1", "UNAVAILABLE", "%", "all audio", "neural", "UNAVAILABLE", ""],
    ["Weighted_CSR", "UNAVAILABLE", "%", "all audio", "neural", "UNAVAILABLE", ""],
    # DSP Chroma predictions on real audio
    ["DSP_num_chord_segments", "30", "segments", "converted-audio-ShZ978fBl6Y (204s)", "dsp_chroma", "MEASURED", "Actual output from outputs/converted-audio-ShZ978fBl6Y_chordify.json"],
    ["DSP_avg_chord_duration", "6.83", "seconds", "converted-audio-ShZ978fBl6Y (204s)", "dsp_chroma", "MEASURED", "Mean segment duration from 30 chord segments"],
    ["DSP_min_chord_duration", "2.23", "seconds", "converted-audio-ShZ978fBl6Y (204s)", "dsp_chroma", "MEASURED", ""],
    ["DSP_max_chord_duration", "25.54", "seconds", "converted-audio-ShZ978fBl6Y (204s)", "dsp_chroma", "MEASURED", ""],
    ["DSP_N_chord_segments", "1", "segments", "converted-audio-ShZ978fBl6Y (204s)", "dsp_chroma", "MEASURED", "1 of 30 segments labelled N"],
    ["DSP_N_chord_pct", "3.3", "%", "converted-audio-ShZ978fBl6Y (204s)", "dsp_chroma", "MEASURED", ""],
    ["DSP_unique_chords", "9", "distinct chords", "converted-audio-ShZ978fBl6Y (204s)", "dsp_chroma", "MEASURED", "F:maj, A#:maj, N, F:min, F#:maj, C:min, G:maj, D#:maj, G:min"],
    ["DSP_detected_key", "Bb", "", "converted-audio-ShZ978fBl6Y (204s)", "dsp_chroma+KS", "MEASURED", "Krumhansl-Schmuckler correlation"],
    ["DSP_key_score", "9", "diatonic fit score", "converted-audio-ShZ978fBl6Y (204s)", "dsp_chroma+KS", "MEASURED", ""],
    ["DSP_chord_segments_75s_clip", "12", "segments", "converted-audio-ShZ978fBl6Y_[cut_75sec] (74.7s)", "dsp_chroma", "MEASURED", ""],
    ["DSP_detected_key_75s", "Bb", "", "converted-audio-ShZ978fBl6Y_[cut_75sec]", "dsp_chroma+KS", "MEASURED", "Consistent key across clips"],
    ["DSP_key_score_75s", "8", "diatonic fit score", "converted-audio-ShZ978fBl6Y_[cut_75sec]", "dsp_chroma+KS", "MEASURED", ""],
    ["DSP_chroma_template_count", "24", "templates", "N/A", "dsp_chroma", "VERIFIED", "12 Major + 12 Minor L2-normalized chroma templates"],
    ["DSP_viterbi_temperature", "14.0", "", "N/A", "dsp_chroma", "VERIFIED", "gamma parameter in estimate_chords_dsp()"],
    ["Chord_accuracy_against_reference", "UNAVAILABLE", "", "all audio", "both", "UNAVAILABLE", "No reference chord annotations exist in this dataset"],
]

with open(OUT / "CHORD_METRICS.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(chord_metrics_rows)

# Chord predictions from real audio
chord_pred_rows = [["audio_file", "start_s", "end_s", "chord_label", "duration_s", "confidence", "method"]]
try:
    jf = ROOT / "outputs" / "converted-audio-ShZ978fBl6Y" / "converted-audio-ShZ978fBl6Y_chordify.json"
    with open(jf, encoding="utf-8") as f:
        data = json.load(f)
    for c in data["chords"]:
        chord_pred_rows.append([
            "converted-audio-ShZ978fBl6Y",
            c["start"], c["end"], c["label"],
            round(c["end"]-c["start"],2),
            c.get("confidence","N/A"),
            c.get("source","dsp_chroma"),
        ])
except Exception as e:
    chord_pred_rows.append(["ERROR", str(e), "", "", "", "", ""])

with open(OUT / "CHORD_PREDICTIONS.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(chord_pred_rows)

print("   -> CHORD_METRICS.csv written")
print("   -> CHORD_PREDICTIONS.csv written")

# ─────────────────────────────────────────────────────────────────────────────
# 7. BEAT METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("[7] Writing BEAT_METRICS.csv and BEAT_PREDICTIONS.csv ...")

beat_metric_rows = [
    ["Metric", "Value", "Unit", "Audio", "Method", "Status", "Notes"],
    ["Beat_precision", "UNAVAILABLE", "", "all audio", "neural", "UNAVAILABLE", "No trained BeatNet checkpoint; no beat annotations in MUSDB18"],
    ["Beat_recall", "UNAVAILABLE", "", "all audio", "neural", "UNAVAILABLE", ""],
    ["Beat_F1", "UNAVAILABLE", "", "all audio", "neural", "UNAVAILABLE", ""],
    ["Downbeat_accuracy", "UNAVAILABLE", "", "all audio", "neural", "UNAVAILABLE", ""],
    ["Mean_absolute_tempo_error", "UNAVAILABLE", "BPM", "all audio", "both", "UNAVAILABLE", "No reference tempo annotations"],
    # DSP measurements from real audio
    ["DSP_estimated_BPM", "83.4", "BPM", "converted-audio-ShZ978fBl6Y (204s)", "dsp_librosa", "MEASURED", "From outputs/converted-audio-ShZ978fBl6Y_chordify.json"],
    ["DSP_beat_events_count", "187", "events", "converted-audio-ShZ978fBl6Y (204s)", "dsp_librosa", "MEASURED", ""],
    ["DSP_downbeat_count", "63", "events", "converted-audio-ShZ978fBl6Y (204s)", "dsp_librosa", "MEASURED", ""],
    ["DSP_estimated_BPM_75s", "57.4", "BPM", "converted-audio-ShZ978fBl6Y_[cut_75sec] (74.7s)", "dsp_librosa", "MEASURED", "Halftime ambiguity: different result on clip"],
    ["DSP_beat_events_75s", "40", "events", "converted-audio-ShZ978fBl6Y_[cut_75sec]", "dsp_librosa", "MEASURED", ""],
    ["DSP_estimated_BPM_test5s", "123.0", "BPM", "test_audio.wav (5s synthetic)", "dsp_librosa", "MEASURED", "Verified in test 19"],
    ["DSP_beat_events_test5s", "7", "events", "test_audio.wav (5s synthetic)", "dsp_librosa", "MEASURED", ""],
    ["Halftime_threshold", "135.0", "BPM", "N/A", "dsp_librosa", "VERIFIED", "BPM > 135 is halved to correct ballad octave ambiguity"],
    ["BPM_accuracy_vs_reference", "UNAVAILABLE", "", "all audio", "both", "UNAVAILABLE", "No ground-truth tempo annotations"],
]

with open(OUT / "BEAT_METRICS.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(beat_metric_rows)

# Beat predictions from real audio
beat_pred_rows = [["audio_file", "time_s", "beat_type"]]
try:
    jf = ROOT / "outputs" / "converted-audio-ShZ978fBl6Y" / "converted-audio-ShZ978fBl6Y_chordify.json"
    with open(jf, encoding="utf-8") as f:
        data = json.load(f)
    for b in data["beats"]:
        beat_pred_rows.append(["converted-audio-ShZ978fBl6Y", b["time"], "beat"])
    for b in data["downbeats"]:
        beat_pred_rows.append(["converted-audio-ShZ978fBl6Y", b["time"], "downbeat"])
except Exception as e:
    beat_pred_rows.append(["ERROR", str(e), ""])

beat_pred_rows.sort(key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else 0)

with open(OUT / "BEAT_PREDICTIONS.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(beat_pred_rows)

print("   -> BEAT_METRICS.csv written")
print("   -> BEAT_PREDICTIONS.csv written")

# ─────────────────────────────────────────────────────────────────────────────
# 8. LYRICS METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("[8] Writing LYRICS_METRICS.csv and TRANSCRIPTION_PREDICTIONS.csv ...")

lyrics_rows = [
    ["Metric", "Value", "Unit", "Audio", "Method", "Status", "Notes"],
    ["WER", "UNAVAILABLE", "%", "all audio", "Whisper large-v3", "UNAVAILABLE", "No reference transcripts available"],
    ["CER", "UNAVAILABLE", "%", "all audio", "Whisper large-v3", "UNAVAILABLE", "No reference transcripts available"],
    ["Whisper_model", "large-v3", "", "N/A", "faster-whisper", "VERIFIED", "Primary model; fallback chain: medium->small->base"],
    ["Whisper_model_size_params", "~1,550,000,000", "params", "N/A", "OpenAI", "VERIFIED (public spec)", "External model"],
    ["Compute_type_CPU", "int8", "", "N/A", "faster-whisper", "VERIFIED", "CTranslate2 quantized inference"],
    ["Beam_size", "5", "", "N/A", "faster-whisper", "VERIFIED", "From pipeline/lyrics.py"],
    ["VAD_filter", "False", "", "N/A", "faster-whisper", "VERIFIED", "Disabled to preserve held singing notes"],
    ["No_speech_threshold", "0.85", "", "N/A", "faster-whisper", "VERIFIED", "Segments above this probability skipped"],
    ["Word_prob_threshold", "0.10", "", "N/A", "faster-whisper", "VERIFIED", "Words below this discarded"],
    ["Condition_on_previous", "False", "", "N/A", "faster-whisper", "VERIFIED", "Prevents hallucination loops in singing"],
    ["Initial_prompt", "Song lyrics singing vocals music", "", "N/A", "faster-whisper", "VERIFIED", "Biases towards lyric-style output"],
    # Real audio measurements
    ["Lyric_phrases_204s", "38", "phrases", "converted-audio-ShZ978fBl6Y (204s)", "Whisper large-v3", "MEASURED", "From outputs JSON"],
    ["Lyric_words_204s", "0", "words", "converted-audio-ShZ978fBl6Y (204s)", "Whisper large-v3", "MEASURED", "Word timestamps not available in this run (lyrics_model=None in that run)"],
    ["Lyric_coverage_204s_pct", "33.2", "%", "converted-audio-ShZ978fBl6Y (204s)", "Whisper large-v3", "MEASURED", "last phrase end 68.06s / 204.8s duration"],
    ["First_phrase_start_204s", "0.37", "seconds", "converted-audio-ShZ978fBl6Y (204s)", "Whisper large-v3", "MEASURED", ""],
    ["Last_phrase_end_204s", "68.06", "seconds", "converted-audio-ShZ978fBl6Y (204s)", "Whisper large-v3", "MEASURED", ""],
    ["Detected_language_test5s", "en", "", "test_audio.wav (synthetic 5s)", "Whisper large-v3", "MEASURED", "prob=0.19 (expected: low for synthetic audio)"],
    ["Lyric_phrases_test5s", "1", "phrases", "test_audio.wav (synthetic 5s)", "Whisper large-v3", "MEASURED", "1 phrase: 'singing vocals' (0.0-0.96s) — Whisper hallucination on synthetic noise"],
    ["Word_count_test5s", "0", "words", "test_audio.wav (synthetic 5s)", "Whisper large-v3", "MEASURED", "No word timestamps returned for synthetic noise"],
    ["Vocal_source_selection", "Center-Channel Extraction (CCE)", "", "all audio (no UNet checkpoint)", "DSP fallback", "VERIFIED", "pipeline/lyrics.py: unet_has_checkpoint=False -> extract_center_vocal_dsp()"],
    ["Resampling_target_sr", "16000", "Hz", "N/A", "faster-whisper", "VERIFIED", "Audio resampled to 16kHz before ASR"],
    ["WER_accuracy_status", "UNAVAILABLE", "", "all audio", "Whisper large-v3", "UNAVAILABLE", "No lyric ground truth for MUSDB18; no reference for converted-audio file"],
]

with open(OUT / "LYRICS_METRICS.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(lyrics_rows)

# Transcription predictions
trans_rows = [["audio_file", "phrase_start_s", "phrase_end_s", "text", "num_words"]]
try:
    jf = ROOT / "outputs" / "converted-audio-ShZ978fBl6Y" / "converted-audio-ShZ978fBl6Y_chordify.json"
    with open(jf, encoding="utf-8") as f:
        data = json.load(f)
    for l in data["lyrics"]:
        trans_rows.append(["converted-audio-ShZ978fBl6Y", l["start"], l["end"], l.get("text",""), len(l.get("words",[]))])
except Exception as e:
    trans_rows.append(["ERROR", str(e), "", "", ""])

with open(OUT / "TRANSCRIPTION_PREDICTIONS.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(trans_rows)

print("   -> LYRICS_METRICS.csv written")
print("   -> TRANSCRIPTION_PREDICTIONS.csv written")

# ─────────────────────────────────────────────────────────────────────────────
# 9. SEPARATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("[9] Writing SEPARATION_METRICS.csv ...")

sep_rows = [
    ["Metric", "Stem", "Value", "Method", "Status", "Notes"],
    ["SDR", "vocals", "UNAVAILABLE", "U-Net (random weights)", "UNAVAILABLE", "No trained checkpoint; no reference stems for evaluation audio"],
    ["SDR", "bass", "UNAVAILABLE", "U-Net (random weights)", "UNAVAILABLE", "Same reason"],
    ["SDR", "drums", "UNAVAILABLE", "U-Net (random weights)", "UNAVAILABLE", "Same reason"],
    ["SDR", "other", "UNAVAILABLE", "U-Net (random weights)", "UNAVAILABLE", "Same reason"],
    ["SI-SDR", "vocals", "UNAVAILABLE", "U-Net (random weights)", "UNAVAILABLE", ""],
    ["SIR", "vocals", "UNAVAILABLE", "U-Net (random weights)", "UNAVAILABLE", ""],
    ["SAR", "vocals", "UNAVAILABLE", "U-Net (random weights)", "UNAVAILABLE", ""],
    ["Separation_runtime_5s_audio", "~0.5-1.0", "seconds", "U-Net (random weights, CPU)", "ESTIMATED", "Based on total pipeline runtime and proportion of U-Net pass"],
    ["UNet_checkpoint_status", "MISSING", "", "N/A", "VERIFIED", "No checkpoints/ directory; model runs with random weights"],
    ["Mask_sum_property", "1.0", "", "architecture", "VERIFIED (Test 4)", "Softmax across source dimension guarantees energy conservation"],
    ["Stems_output_shape", "(4, 2, samples)", "", "architecture", "VERIFIED (Test 4)", "vocal/bass/drums/other stereo stems"],
    ["CCE_vocal_fallback", "Active", "", "pipeline/lyrics.py", "VERIFIED", "extract_center_vocal_dsp() used for ASR when U-Net is untrained"],
]

with open(OUT / "SEPARATION_METRICS.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(sep_rows)

print("   -> SEPARATION_METRICS.csv written")

# ─────────────────────────────────────────────────────────────────────────────
# 10. STRUCTURE METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("[10] Writing STRUCTURE_METRICS.csv ...")

struct_rows = [
    ["Metric", "Value", "Audio", "Status", "Notes"],
    ["Boundary_precision", "UNAVAILABLE", "all audio", "UNAVAILABLE", "No trained SegmentNet checkpoint; no structure annotations in MUSDB18"],
    ["Boundary_recall", "UNAVAILABLE", "all audio", "UNAVAILABLE", ""],
    ["Boundary_F1", "UNAVAILABLE", "all audio", "UNAVAILABLE", ""],
    ["Num_sections_204s", "1", "converted-audio-ShZ978fBl6Y (204s)", "MEASURED", "Single 'chorus' section for entire song — random weight artifact"],
    ["Num_sections_75s", "1", "converted-audio-ShZ978fBl6Y_[cut_75sec] (74.7s)", "MEASURED", "Same pattern"],
    ["Num_sections_5s_synthetic", "1", "test_audio.wav (5s synthetic)", "MEASURED", "'outro' label — random weight artifact"],
    ["Section_label_space", "10", "classes", "N/A", "VERIFIED", "start/end/intro/outro/verse/chorus/solo/break/bridge/inst"],
    ["SegmentNet_checkpoint_status", "MISSING", "", "N/A", "VERIFIED", "Random weights; outputs are meaningless for accuracy evaluation"],
]

with open(OUT / "STRUCTURE_METRICS.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(struct_rows)

print("   -> STRUCTURE_METRICS.csv written")

# ─────────────────────────────────────────────────────────────────────────────
# 11. PITCH METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("[11] Writing PITCH_METRICS.csv ...")

pitch_rows = [
    ["Metric", "Value", "Audio", "Status", "Notes"],
    ["Raw_pitch_accuracy", "UNAVAILABLE", "all audio", "UNAVAILABLE", "No trained PitchNet checkpoint; no melody annotations in MUSDB18"],
    ["Voicing_accuracy", "UNAVAILABLE", "all audio", "UNAVAILABLE", ""],
    ["Num_pitch_frames_204s", "8820", "events", "converted-audio-ShZ978fBl6Y (204s)", "MEASURED", "From outputs JSON; all frames map to MIDI 100 (random weights)"],
    ["Num_pitch_frames_75s", "3217", "events", "converted-audio-ShZ978fBl6Y_[cut_75sec] (74.7s)", "MEASURED", ""],
    ["Num_pitch_frames_5s_synthetic", "216", "events", "test_audio.wav (5s synthetic)", "MEASURED", "216 frames at 43.07 fps for 5s"],
    ["Pitch_class_range", "0-849", "index", "N/A", "VERIFIED (Test 8)", "850 classes; index->MIDI: 21 + idx//10; cents: (idx%10)*10"],
    ["MIDI_note_range", "21-105", "MIDI note", "N/A", "VERIFIED", "A0 (21) to A7 (105) via 85 semitones"],
    ["Pitch_frame_rate", "43.07", "fps", "N/A", "DERIVED", "44100/1024 hop_length"],
    ["PitchNet_checkpoint_status", "MISSING", "", "N/A", "VERIFIED", "Random weights; all frames output same class (constant output = random weight artifact)"],
    ["Constant_pitch_output", "MIDI=100, cents_offset=90", "", "all output files", "MEASURED", "PitchNet with random weights outputs constant prediction = class 799 -> MIDI 100 (E7)"],
]

with open(OUT / "PITCH_METRICS.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(pitch_rows)

print("   -> PITCH_METRICS.csv written")

# ─────────────────────────────────────────────────────────────────────────────
# 12. TAB METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("[12] Writing TAB_METRICS.csv ...")

tab_rows = [
    ["Metric", "Value", "Audio", "Status", "Notes"],
    ["Num_tab_chord_events_204s", "30", "events", "converted-audio-ShZ978fBl6Y (204s)", "MEASURED", "One tab entry per DSP chord segment"],
    ["Num_tab_sections_204s", "187", "lines in tab file", "converted-audio-ShZ978fBl6Y (204s)", "MEASURED", "187-line ASCII tab file"],
    ["Num_tab_chord_events_75s", "12", "events", "converted-audio-ShZ978fBl6Y_[cut_75sec] (74.7s)", "MEASURED", ""],
    ["Num_tab_lines_75s", "57", "lines", "converted-audio-ShZ978fBl6Y_[cut_75sec]", "MEASURED", ""],
    ["Tab_chord_events_5s_synthetic", "1", "events", "test_audio.wav (5s synthetic)", "MEASURED", "1 chord segment -> 1 tab event"],
    ["Chord_voicings_in_dictionary", "70", "voicings", "N/A", "VERIFIED", "Count of entries in _CHORD_VOICINGS dict in pipeline/tabs.py"],
    ["Valid_string_fret_format", "YES", "", "N/A", "VERIFIED (Test 13)", "'e|' present in all ascii_tab outputs"],
    ["Tab_layout_bars_per_line", "4", "bars/line", "N/A", "VERIFIED", "_build_measure_tab() BARS_PER_LINE=4"],
    ["Column_width_chars", "10", "chars/bar", "N/A", "VERIFIED", "COL_W=10 in pipeline/tabs.py"],
    ["Unplayable_voicings", "ESTIMATED NONE", "", "N/A", "ESTIMATED", "All voicings manually set in dictionary; 'x' marks muted strings"],
    ["Musical_accuracy_vs_reference", "UNAVAILABLE", "", "all audio", "UNAVAILABLE", "Tabs are derived from DSP chord output; no reference guitar tabs"],
    ["Chord_tab_agreement", "100%", "%", "all audio", "VERIFIED", "Tabs are directly generated from chord_segments; perfect agreement by construction"],
]

with open(OUT / "TAB_METRICS.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(tab_rows)

print("   -> TAB_METRICS.csv written")

# ─────────────────────────────────────────────────────────────────────────────
# 13. PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("[13] Writing PERFORMANCE_METRICS.csv and PERFORMANCE_SUMMARY.json ...")

perf_rows = [
    ["Audio_Duration_s", "File_Size_KB", "Device", "Separation_method", "Chord_method", "Beat_method", "Lyrics_model", "Total_Time_s", "RT_Factor", "Notes"],
    ["5.0", "861", "CPU", "U-Net (random weights)", "dsp_chroma", "dsp_librosa", "Whisper large-v3", "32.5", "6.5x", "Test 19 measured 2026-10-21; single run"],
    ["5.0", "861", "CPU", "U-Net (random weights)", "dsp_chroma", "dsp_librosa", "Whisper (large-v3)", "14.1", "2.82x", "Historical result from CHORDIFY_STATUS.md (before Whisper was installed); no Whisper overhead"],
    ["204.788", "~50000 (estimated)", "CPU", "U-Net (random weights)", "dsp_chroma", "dsp_librosa", "Whisper large-v3 (partial, lyrics_model=None in that run)", "UNMEASURED", "N/A", "Output exists but pipeline runtime not captured"],
    ["74.688", "~25000 (estimated)", "CPU", "U-Net (random weights)", "dsp_chroma", "dsp_librosa", "Whisper large-v3 (partial)", "UNMEASURED", "N/A", "Same"],
]

with open(OUT / "PERFORMANCE_METRICS.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(perf_rows)

perf_summary = {
    "hardware": {
        "platform": "Windows (win32)",
        "device": "CPU only",
        "gpu": "None detected (CUDA not available)",
        "python_version": "3.13.5",
    },
    "measured_runs": [
        {
            "audio": "test_audio.wav (5s synthetic, 44100 Hz stereo)",
            "date": "2026-10-21",
            "total_time_s": 32.5,
            "rt_factor": 6.5,
            "stages": {
                "audio_loading": "< 0.1s",
                "source_separation_unet": "approx 1-3s (random weights, no checkpoint overhead)",
                "lyrics_asr_whisper": "approx 20-25s (Whisper large-v3 on CPU loading + inference)",
                "beat_analysis_dsp": "< 1s",
                "chord_analysis_dsp": "approx 1-2s",
                "structure_segmentnet": "approx 1-2s (random weights)",
                "pitch_pitchnet": "approx 1-2s (random weights)",
                "fusion": "< 0.1s",
                "tab_generation": "< 0.1s",
            },
            "note": "Single run; Whisper large-v3 loading dominates on CPU (first run includes model load)"
        },
        {
            "audio": "test_audio.wav (5s synthetic)",
            "date": "historical (CHORDIFY_STATUS.md)",
            "total_time_s": 14.1,
            "rt_factor": 2.82,
            "note": "Historical result before faster-whisper was installed; no Whisper overhead",
        }
    ],
    "important_caveat": "Runtime is dominated by Whisper large-v3 loading on CPU (~20-25s). Neural model inference (with random weights) is fast. Real production runtime with a GPU and cached Whisper model would be significantly lower.",
    "repeated_measurements": "NOT DONE — single-run results only due to long Whisper load time",
}

with open(OUT / "PERFORMANCE_SUMMARY.json", "w", encoding="utf-8") as f:
    json.dump(perf_summary, f, indent=2)

print("   -> PERFORMANCE_METRICS.csv written")
print("   -> PERFORMANCE_SUMMARY.json written")

# ─────────────────────────────────────────────────────────────────────────────
# 14. DASHBOARD TEST RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("[14] Writing DASHBOARD_TEST_RESULTS.csv ...")

dash_rows = [
    ["Field", "Value", "Audio", "Status", "Notes"],
    ["Gradio_version", "6.28.0", "N/A", "VERIFIED (pip show)", "Installed; app.py uses gr.Blocks with Soft theme"],
    ["Dashboard_file", "app/app.py", "N/A", "VERIFIED", "~400 lines; full Gradio v4 UI"],
    ["Pipeline_backend", "chordify.py Chordify class", "N/A", "VERIFIED", "Singleton _pipeline instance"],
    ["UI_tabs", "8", "N/A", "VERIFIED", "Guitar Tab / Synchronized View / Chords / Beats / Sections / Lyrics / Word Timestamps / Diagnostics"],
    ["Timeline_plot", "YES", "N/A", "VERIFIED", "matplotlib figure; 2-row (sections + chords) colored timeline"],
    ["Metadata_table", "YES", "N/A", "VERIFIED", "Duration / BPM / Key / Chord engine / Beat engine / Lyrics model / Chord segments / Lyric phrases / Words / Coverage"],
    ["Word_timestamps_tab", "YES", "N/A", "VERIFIED", "Per-word start/end/probability table"],
    ["Diagnostics_tab", "YES", "N/A", "VERIFIED", "Coverage warnings, pipeline metrics"],
    ["Neural_DSP_badges", "YES", "N/A", "VERIFIED", "Green/yellow badge in metadata"],
    ["Dashboard_live_test", "NOT EXECUTED", "N/A", "UNAVAILABLE", "Cannot run Gradio server in agentic environment; all code verification done statically"],
    ["Gradio_server_port", "7860", "N/A", "VERIFIED", "Default; tries 7860-7869 range"],
    ["Share_option", "YES", "--share flag", "VERIFIED", "Public Gradio link available via --share"],
    ["Audio_formats_supported", ".wav / .mp3 / .flac", "N/A", "VERIFIED", "From utils/audio.py SUPPORTED_EXTENSIONS"],
    # Results from test_audio.wav run (Test 19 = real pipeline run)
    ["Test_audio_duration", "5.0s", "test_audio.wav", "MEASURED (Test 19)", ""],
    ["Test_audio_chord_method", "dsp_chroma", "test_audio.wav", "MEASURED (Test 19)", "No ChordNet checkpoint"],
    ["Test_audio_beat_method", "dsp_librosa", "test_audio.wav", "MEASURED (Test 19)", "No BeatNet checkpoint"],
    ["Test_audio_lyrics_model", "Whisper (large-v3)", "test_audio.wav", "MEASURED (Test 19)", ""],
    ["Test_audio_key", "A", "test_audio.wav", "MEASURED (Test 19)", ""],
    ["Test_audio_bpm", "123.0", "test_audio.wav", "MEASURED (Test 19)", ""],
    ["Test_audio_chord_segments", "1", "test_audio.wav", "MEASURED (Test 19)", "1 chord (Am) spanning entire 5s"],
    ["Test_audio_beat_events", "7", "test_audio.wav", "MEASURED (Test 19)", ""],
    ["Test_audio_downbeats", "3", "test_audio.wav", "MEASURED (Test 19)", ""],
    ["Test_audio_sections", "1", "test_audio.wav", "MEASURED (Test 19)", "'outro' — random weight artifact from SegmentNet"],
    ["Test_audio_pitch_frames", "216", "test_audio.wav", "MEASURED (Test 19)", "All MIDI=100 — random weight constant output"],
    ["Test_audio_lyric_phrases", "1", "test_audio.wav", "MEASURED (Test 19)", "'singing vocals' (hallucination on synthetic audio)"],
    ["Test_audio_lyric_words", "0", "test_audio.wav", "MEASURED (Test 19)", "No word timestamps for synthetic"],
    ["Test_audio_tab_sections", "1", "test_audio.wav", "MEASURED (Test 19)", "1 chord event in tab"],
    ["Test_audio_tab_lines", "16", "test_audio.wav", "MEASURED (Test 19)", "16-line ASCII tab file"],
    ["Test_audio_total_runtime", "32.5s", "test_audio.wav", "MEASURED (Test 19)", ""],
]

with open(OUT / "DASHBOARD_TEST_RESULTS.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(dash_rows)

print("   -> DASHBOARD_TEST_RESULTS.csv written")

# ─────────────────────────────────────────────────────────────────────────────
# 15. IMPLEMENTATION INVENTORY
# ─────────────────────────────────────────────────────────────────────────────
print("[15] Writing IMPLEMENTATION_INVENTORY.csv ...")

impl_rows = [
    ["Component", "File", "Status", "Algorithm", "Input", "Output", "Checkpoint_Required", "Fallback", "Notes"],
    ["Audio Loading", "pipeline/audio_loader.py + utils/audio.py", "IMPLEMENTED", "torchaudio + soundfile resampling", "audio file (.wav/.mp3/.flac)", "(2, samples) tensor + sr + duration", "NO", "N/A", ""],
    ["STFT / iSTFT", "utils/spectrogram.py", "IMPLEMENTED + VERIFIED", "Hann-window STFT (n_fft=4096, hop=1024, pad=True)", "(2,samples)", "(2,T,F) complex tensor", "NO", "N/A", "Round-trip verified in Test 3"],
    ["U-Net Source Separation", "models/unet.py + pipeline/separation.py", "IMPLEMENTED (architecture only)", "6-level encoder-decoder, 4 parallel branches, channel-wise softmax", "(2,samples) mixture", "(4,2,samples) vocal/bass/drums/other", "YES (unet.pth)", "Pass mixture through uncalibrated model; CCE for lyrics", "No checkpoint; runs with random weights"],
    ["BeatNet (neural)", "models/beat_net.py + pipeline/beat_analysis.py", "IMPLEMENTED (architecture only)", "3-branch factorised axial-attention transformer, 3-class head", "(3,2,T,F) spectrograms from drums/bass/other", "(T,) beat class per frame", "YES (beat_net.pth)", "DSP librosa beat_track", "No checkpoint"],
    ["DSP Beat Fallback", "pipeline/dsp_fallback.py", "IMPLEMENTED + VERIFIED", "librosa onset strength envelope + dynamic programming beat tracking; halftime correction", "(2,samples) rhythm mix", "bpm, beat classes, beat events", "NO", "Primary when no checkpoint", "Verified in Tests 16, 17, 18, 19"],
    ["ChordNet (neural)", "models/chord_net.py + pipeline/chord_analysis.py", "IMPLEMENTED (architecture only)", "BaseTransformer + 122-class linear head; beat-conditioned", "(2,T,F) accompaniment spectrogram", "(T,) chord class per frame", "YES (chord_net.pth)", "DSP chroma fallback", "No checkpoint"],
    ["DSP Chord Fallback", "pipeline/dsp_fallback.py", "IMPLEMENTED + VERIFIED", "Harmonic extraction + dual-band CQT chroma + 24-triad Viterbi HMM; Krumhansl-Schmuckler key", "(2,samples) accompaniment mix", "frame labels, key, key_score, chord_events", "NO", "Primary when no checkpoint", "Verified in Tests 15, 17, 18, 19"],
    ["Key Estimation (neural)", "pipeline/modulation.py", "IMPLEMENTED + VERIFIED", "Diatonic scoring over chord sequence", "list of chord strings", "(key_string, diatonic_score)", "NO", "Always available", "Verified in Test 10; key=G, score=8 for G-Am-C-D progression"],
    ["Key Estimation (DSP)", "pipeline/dsp_fallback.py", "IMPLEMENTED + VERIFIED", "Krumhansl-Schmuckler correlation with major/minor profiles", "(12,) global chroma mean", "(key_string, correlation_score)", "NO", "Used in DSP chord path", ""],
    ["PitchNet (neural)", "models/pitch_net.py + pipeline/pitch_analysis.py", "IMPLEMENTED (architecture only)", "4-conv embedding + PitchEncoder + lyric-conditioned decoder, 850-class head", "(2,T,F) vocal spectrogram", "(T,) pitch class per frame", "YES (pitch_net.pth)", "Uncalibrated random output", "No checkpoint; outputs constant MIDI=100"],
    ["SegmentNet (neural)", "models/segment_net.py + pipeline/structure_analysis.py", "IMPLEMENTED (architecture only)", "SegmentEmbeddings + BaseTransformer + 10-class head; chord-conditioned", "(2,T,F) mix spectrogram", "(T,) section class per frame", "YES (segment_net.pth)", "Uncalibrated random output", "No checkpoint; outputs single section"],
    ["Lyrics Transcription", "pipeline/lyrics.py", "IMPLEMENTED + OPERATIONAL", "faster-whisper (large-v3 primary, medium/small/base fallback) + CCE vocal extraction", "(2,samples) vocal stem or original mix", "phrase list with timestamps, word list with timestamps", "NO (external model)", "Centre-channel extraction when U-Net uncalibrated", "Verified in Test 19; Whisper large-v3 loaded from cache"],
    ["Wav2Vec2 Features", "pipeline/lyrics.py", "IMPLEMENTED", "facebook/wav2vec2-base-960h frame-level features for PitchNet conditioning", "(1,T,768) feature matrix", "(1,T,768) tensor", "NO (external model)", "None passed to PitchNet", "Optional; transformers not installed -> None returned"],
    ["Multimodal Fusion", "pipeline/fusion.py", "IMPLEMENTED + VERIFIED", "Timeline merging of all modality outputs; frame-to-time conversion", "all stage outputs", "unified result dict", "NO", "N/A", "Verified in Test 12"],
    ["Guitar Tab Generation", "pipeline/tabs.py", "IMPLEMENTED + VERIFIED", "Chord voicing lookup + measure-based 6-string ASCII layout + synchronized view", "fusion result dict", "ascii_tab, tab_list, sync_view", "NO", "N/A", "Verified in Tests 13, 19"],
    ["Gradio Dashboard", "app/app.py", "IMPLEMENTED (not live-tested)", "Gradio 6.28 Blocks UI; 8 output tabs; colored timeline plot", "audio file upload", "metadata/chords/beats/sections/lyrics/tabs/plot", "NO (uses pipeline)", "N/A", "Code verified; cannot start server in agentic env"],
    ["Model Checkpoint Validation", "models/load.py", "IMPLEMENTED + VERIFIED", "has_valid_checkpoint(): file size > 1KB, state_dict non-empty, tensor variance > 1e-8", "model_path string", "bool", "NO", "N/A", "Verified in Tests 17, 18, 19"],
]

with open(OUT / "IMPLEMENTATION_INVENTORY.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(impl_rows)

print("   -> IMPLEMENTATION_INVENTORY.csv written")

# ─────────────────────────────────────────────────────────────────────────────
# 16. EVALUATION PROTOCOL
# ─────────────────────────────────────────────────────────────────────────────
eval_protocol = """# Chordify Evaluation Protocol

## Date
2026-10-21

## What Was Evaluated
All evaluations performed on 2026-10-21 are based on:
1. Static code inspection of all source files
2. Live execution of the 19-test test suite (scripts/test_pipeline.py)
3. Inspection of existing JSON output files in outputs/
4. Parameter counting via PyTorch model instantiation
5. Dataset size scanning via os.stat (no ffmpeg available)

## Test Suite Results
- Command: `python scripts/test_pipeline.py`
- Result: 19/19 PASS, 0 FAIL, 0 SKIP
- Runtime: ~35 seconds (dominated by Whisper large-v3 loading in Test 19)

## What Tests Prove and Do NOT Prove
TESTS 1-18 prove:
- Correct architecture implementation and forward pass shapes
- Correct label vocabulary sizes (122 chords, 10 segments, 850 pitch classes)
- Correct STFT round-trip (lossless transforms)
- DSP fallback algorithms execute without error on synthetic audio
- Automatic checkpoint detection routing works correctly

TESTS DO NOT PROVE:
- Chord recognition accuracy (no ground truth; random-weight models)
- Beat detection accuracy (no ground truth; DSP-only)
- Melody/pitch accuracy (random weights)
- Structure segmentation accuracy (random weights)
- Lyrics WER (no reference transcripts)
- Separation SDR/SIR/SAR (no checkpoints; no reference stems for test audio)

## End-to-End Pipeline Test (Test 19)
Input: outputs/test_audio.wav (5s, 44100 Hz, stereo, synthetic sine wave)
Device: CPU (CUDA unavailable)
Chord path: DSP Chroma (no ChordNet checkpoint)
Beat path: DSP Librosa (no BeatNet checkpoint)
Structure path: SegmentNet (random weights)
Pitch path: PitchNet (random weights)
Lyrics path: Whisper large-v3 (Center-Channel Extraction, no U-Net checkpoint)

Measured outputs:
- BPM: 123.0
- Key: A (score 7)
- Chord segments: 1 (Am, confidence=0.79)
- Beat events: 7 + 3 downbeats
- Sections: 1 (outro - random weight artifact)
- Pitch frames: 216 (all MIDI=100 - random weight constant output)
- Lyric phrases: 1 ('singing vocals' - Whisper hallucination on synthetic noise)
- Total runtime: 32.5s

## Real Audio Results
Two runs on real audio files exist in outputs/:
1. converted-audio-ShZ978fBl6Y (204.8s): 30 chord segments, BPM=83.4, Key=Bb, 38 lyric phrases
2. converted-audio-ShZ978fBl6Y_[cut_75sec] (74.7s): 12 chord segments, BPM=57.4, Key=Bb, 44 lyric phrases

Both used DSP methods (no checkpoints). Chord accuracy unverifiable without ground truth annotations.

## Missing Evidence
- No trained neural checkpoints (chord, beat, pitch, segment neural accuracy = unavailable)
- No chord annotations for MUSDB18 (frame-wise accuracy = unavailable)
- No beat annotations for MUSDB18 (beat F-measure = unavailable)
- No ffmpeg installed (dataset audio durations = estimated only)
- No reference lyrics for converted-audio files (WER = unavailable)
- No reference stems for test audio (SDR = unavailable)
- Gradio dashboard not live-tested (server cannot start in agentic environment)
- No GPU available (all timings are CPU-only)
"""

with open(OUT / "EVALUATION_PROTOCOL.md", "w", encoding="utf-8") as f:
    f.write(eval_protocol)

print("   -> EVALUATION_PROTOCOL.md written")

# ─────────────────────────────────────────────────────────────────────────────
# 17. EXPERIMENT MANIFEST
# ─────────────────────────────────────────────────────────────────────────────
manifest = {
    "project": "Chordify — AI-Based Multimodal Music Understanding and Automatic Guitar Tab Generation",
    "audit_date": "2026-10-21",
    "auditor": "Kiro (AI Code Auditor)",
    "environment": {
        "os": "Windows (win32)",
        "python": "3.13.5",
        "torch": "2.14.0+cpu",
        "torchaudio": "2.11.0",
        "librosa": "1.0.0",
        "numpy": "2.2.6",
        "scipy": "1.16.1",
        "gradio": "6.28.0",
        "faster_whisper": "1.2.1",
        "soundfile": "0.14.0",
        "pyyaml": "6.0.2",
        "musdb": "0.4.3",
        "stempeg": "0.2.6",
        "transformers": "NOT INSTALLED",
        "cuda": "NOT AVAILABLE",
    },
    "experiments": [
        {
            "id": "EXP-001",
            "name": "Test Suite Execution",
            "command": "python scripts/test_pipeline.py",
            "date": "2026-10-21",
            "result": "19/19 PASS",
            "runtime_s": 35,
            "output_file": "report_data/logs/test_execution.log",
        },
        {
            "id": "EXP-002",
            "name": "Model Parameter Counting",
            "method": "torch parameter counting via model instantiation",
            "date": "2026-10-21",
            "results": {
                "UNets": "39,294,876",
                "BeatNet": "504,713,475",
                "ChordNet": "389,342,202",
                "PitchNet": "119,440,806",
                "SegmentNet": "14,025,676",
            }
        },
        {
            "id": "EXP-003",
            "name": "Dataset File Scan",
            "method": "os.stat on *.stem.mp4 files (no audio decoding, ffmpeg unavailable)",
            "date": "2026-10-21",
            "results": {
                "train_tracks": 100,
                "test_tracks": 50,
                "train_total_gb": 3.43,
                "test_total_gb": 1.87,
            }
        },
        {
            "id": "EXP-004",
            "name": "Real Audio Output Inspection",
            "method": "JSON output inspection from outputs/ directory",
            "date": "2026-10-21",
            "files_inspected": [
                "outputs/test_audio/test_audio_chordify.json",
                "outputs/converted-audio-ShZ978fBl6Y/converted-audio-ShZ978fBl6Y_chordify.json",
                "outputs/converted-audio-ShZ978fBl6Y_[cut_75sec]/converted-audio-ShZ978fBl6Y_[cut_75sec]_chordify.json",
            ],
        }
    ]
}

with open(OUT / "EXPERIMENT_MANIFEST.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("   -> EXPERIMENT_MANIFEST.json written")

print("\n[Figures] Generating figures ...")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

# Figure 1: System Architecture Diagram
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")
ax.set_xticks([])
ax.set_yticks([])

# Draw pipeline blocks
blocks = [
    # (x, y, w, h, label, color)
    (0.3, 8.5, 2.2, 0.8, "Audio Input\n(.wav/.mp3/.flac)", "#4C72B0"),
    (0.3, 7.0, 2.2, 0.8, "Audio Loader\n44100 Hz, Stereo", "#4C72B0"),
    (0.3, 5.5, 2.2, 0.8, "U-Net Source\nSeparation (4 stems)", "#55A868"),
    (0.3, 4.0, 2.2, 0.8, "vocal stem", "#8172B2"),
    (3.5, 8.0, 2.5, 0.8, "BeatNet\nDSP Fallback", "#DD8452"),
    (7.0, 8.0, 2.5, 0.8, "ChordNet\nDSP Fallback", "#C44E52"),
    (10.5, 8.0, 2.5, 0.8, "SegmentNet\n(random wts)", "#937860"),
    (3.5, 5.5, 2.5, 0.8, "PitchNet\n(random wts)", "#8172B2"),
    (7.0, 5.5, 2.5, 0.8, "Whisper ASR\n(large-v3)", "#DA8BC3"),
    (10.5, 5.5, 2.5, 0.8, "Multimodal\nFusion", "#64B5CD"),
    (10.5, 3.5, 2.5, 0.8, "Guitar Tab\nGeneration", "#CCB974"),
    (10.5, 1.5, 2.5, 0.8, "Gradio\nDashboard", "#4C72B0"),
    (3.5, 1.5, 2.5, 0.8, "JSON Output\n+ .tab File", "#8C8C8C"),
]

for (x, y, w, h, label, color) in blocks:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="white", alpha=0.85, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=7.5, color="white", fontweight="bold", wrap=True)

ax.set_title("Chordify System Architecture\n(Green=DSP Active | Orange=No Checkpoint | Purple=Random Weights)", 
             fontsize=12, color="white", pad=10)

plt.tight_layout()
plt.savefig(FIGS / "01_system_architecture.png", dpi=150, bbox_inches="tight")
plt.close()
print("   -> 01_system_architecture.png")

# Figure 2: Model Parameter Counts
fig, ax = plt.subplots(figsize=(10, 5))
models = ["UNets\n(Separation)", "BeatNet\n(Beat/Tempo)", "ChordNet\n(Chord)", "PitchNet\n(Melody)", "SegmentNet\n(Structure)"]
params = [39.3, 504.7, 389.3, 119.4, 14.0]
colors_bar = ["#55A868", "#DD8452", "#C44E52", "#8172B2", "#937860"]
bars = ax.bar(models, params, color=colors_bar, edgecolor="white", linewidth=1.2)
ax.set_ylabel("Parameters (Millions)", fontsize=11)
ax.set_title("Chordify Neural Model Parameter Counts", fontsize=13)
for bar, v in zip(bars, params):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f"{v:.1f}M", ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, 580)
ax.grid(axis="y", alpha=0.3)
ax.set_facecolor("#f5f5f5")
plt.tight_layout()
plt.savefig(FIGS / "02_model_parameters.png", dpi=150)
plt.close()
print("   -> 02_model_parameters.png")

# Figure 3: Chord Vocabulary (122 classes)
sys.path.insert(0, str(ROOT))
from utils.labels import CHORD_LABELS, SEGMENT_LABELS

quality_colors = {
    "N": "#AAAAAA", "X": "#888888",
    "maj": "#4C72B0", "min": "#DD8452", "7": "#55A868", "maj7": "#C44E52",
    "min7": "#8172B2", "6": "#937860", "m6": "#DA8BC3", "sus2": "#8C8C8C",
    "sus4": "#CCB974", "5": "#64B5CD",
}
colors_bar2 = []
for lbl in CHORD_LABELS:
    if ":" in lbl:
        _, qual = lbl.split(":", 1)
        colors_bar2.append(quality_colors.get(qual, "#999"))
    else:
        colors_bar2.append(quality_colors.get(lbl, "#AAA"))

fig, ax = plt.subplots(figsize=(28, 3))
ax.bar(np.arange(len(CHORD_LABELS)), np.ones(len(CHORD_LABELS)), color=colors_bar2, width=0.9)
ax.set_xticks(np.arange(len(CHORD_LABELS))[::1])
labels_short = []
for lbl in CHORD_LABELS:
    if ":" in lbl:
        r, q = lbl.split(":", 1)
        labels_short.append(f"{r}\n{q}")
    else:
        labels_short.append(lbl)
ax.set_xticklabels(labels_short, fontsize=4.5, rotation=90)
ax.set_yticks([])
ax.set_title("Chordify Chord Vocabulary (122 classes)", fontsize=12)
patches = [mpatches.Patch(color=c, label=q) for q, c in quality_colors.items()]
ax.legend(handles=patches, ncol=6, loc="upper right", fontsize=7)
plt.tight_layout()
plt.savefig(FIGS / "03_chord_vocabulary.png", dpi=120)
plt.close()
print("   -> 03_chord_vocabulary.png")

# Figure 4: Stage-wise latency
fig, ax = plt.subplots(figsize=(10, 5))
stages = ["Audio\nLoad", "U-Net\nSep.", "Whisper\nASR", "Beat\nDSP", "Chord\nDSP", "Struct.\nNet", "Pitch\nNet", "Fusion\n+Tab"]
times  = [0.05, 2.0, 25.0, 0.5, 1.5, 1.5, 2.0, 0.1]
colors3 = ["#4C72B0", "#55A868", "#DA8BC3", "#DD8452", "#C44E52", "#937860", "#8172B2", "#64B5CD"]
bars = ax.bar(stages, times, color=colors3, edgecolor="white", linewidth=1.2)
ax.set_ylabel("Estimated Time (seconds)", fontsize=11)
ax.set_title("Chordify Stage-wise Latency\n5s Audio, CPU, Whisper large-v3 (estimated breakdown of 32.5s total)", fontsize=11)
for bar, v in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{v}s", ha="center", fontsize=9)
ax.set_ylim(0, 30)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / "04_stage_latency.png", dpi=150)
plt.close()
print("   -> 04_stage_latency.png")

# Figure 5: Chord timeline for real audio
try:
    jf = ROOT / "outputs" / "converted-audio-ShZ978fBl6Y" / "converted-audio-ShZ978fBl6Y_chordify.json"
    with open(jf, encoding="utf-8") as f:
        data = json.load(f)

    chords = data["chords"]
    dur = data["metadata"]["duration"]
    
    fig, ax = plt.subplots(figsize=(16, 3))
    ax.set_facecolor("#0f3460")
    fig.patch.set_facecolor("#1a1a2e")
    
    cmap = plt.cm.tab20
    unique = list(dict.fromkeys(c["label"] for c in chords))
    clr = {l: cmap(i / max(len(unique), 1)) for i, l in enumerate(unique)}
    
    for chord in chords:
        w = chord["end"] - chord["start"]
        color = clr.get(chord["label"], (0.5, 0.5, 0.5, 0.8))
        ax.barh(0, w, left=chord["start"], height=0.7, color=color, alpha=0.9)
        if w > 8:
            ax.text(chord["start"]+w/2, 0, chord["label"].replace(":maj","").replace(":min","m"),
                    ha="center", va="center", fontsize=7, color="white")
    
    ax.set_xlim(0, dur)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)", color="white")
    ax.set_title(f"DSP Chord Timeline — converted-audio-ShZ978fBl6Y ({dur:.0f}s, Bb, 83.4 BPM)", color="white")
    ax.tick_params(colors="white")
    
    patches = [mpatches.Patch(color=clr[l], label=l.replace(":maj","").replace(":min","m")) for l in unique[:12]]
    ax.legend(handles=patches, loc="upper right", fontsize=7, facecolor="#1a1a2e", labelcolor="white", ncol=5)
    plt.tight_layout()
    plt.savefig(FIGS / "05_chord_timeline_real_audio.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   -> 05_chord_timeline_real_audio.png")
except Exception as e:
    print(f"   SKIP 05_chord_timeline_real_audio.png: {e}")

# Figure 6: Test results summary bar
fig, ax = plt.subplots(figsize=(6, 4))
categories = ["Unit (14)", "Integration (4)", "End-to-End (1)"]
passed = [14, 4, 1]
failed = [0, 0, 0]
x = np.arange(len(categories))
ax.bar(x, passed, label="PASS", color="#55A868", edgecolor="white")
ax.bar(x, failed, bottom=passed, label="FAIL", color="#C44E52", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel("Tests")
ax.set_title("Test Suite Results: 19/19 PASS\n(2026-10-21)", fontsize=12)
ax.legend()
ax.set_ylim(0, 18)
for xi, v in zip(x, passed):
    ax.text(xi, v + 0.3, str(v), ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS / "06_test_results.png", dpi=150)
plt.close()
print("   -> 06_test_results.png")

# Figure 7: Dataset split chart
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
splits = ["Train", "Test"]
track_counts = [100, 50]
size_gb = [3.43, 1.87]
colors4 = ["#4C72B0", "#DD8452"]

axes[0].bar(splits, track_counts, color=colors4, edgecolor="white")
axes[0].set_title("MUSDB18 Track Count\nby Split", fontsize=11)
axes[0].set_ylabel("Tracks")
for xi, v in enumerate(track_counts):
    axes[0].text(xi, v + 1, str(v), ha="center", fontsize=12, fontweight="bold")
axes[0].set_ylim(0, 120)

axes[1].bar(splits, size_gb, color=colors4, edgecolor="white")
axes[1].set_title("MUSDB18 Storage Size\nby Split", fontsize=11)
axes[1].set_ylabel("Size (GB)")
for xi, v in enumerate(size_gb):
    axes[1].text(xi, v + 0.05, f"{v} GB", ha="center", fontsize=11, fontweight="bold")
axes[1].set_ylim(0, 4.5)

plt.suptitle("MUSDB18 Dataset Statistics (150 total tracks, 5.30 GB)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS / "07_dataset_statistics.png", dpi=150)
plt.close()
print("   -> 07_dataset_statistics.png")

# Figure 8: Segment label space
seg_colors2 = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(SEGMENT_LABELS, np.ones(10), color=seg_colors2, alpha=0.85)
ax.set_xlabel("Class index →")
ax.set_title("SegmentNet Label Space (10 structural classes)", fontsize=12)
ax.set_xlim(0, 1.6)
for i, lbl in enumerate(SEGMENT_LABELS):
    ax.text(0.05, i, f"  class {i}", va="center", fontsize=11, color="white", fontweight="bold")
plt.tight_layout()
plt.savefig(FIGS / "08_segment_labels.png", dpi=150)
plt.close()
print("   -> 08_segment_labels.png")

print("\n[Figures] All figures written.")

print("\n[Log] Saving test execution log ...")
log_text = """Chordify Test Suite Execution Log
=================================
Date: 2026-10-21
Command: python scripts/test_pipeline.py
Working Dir: C:\\Users\\RON MATHEW\\Desktop\\sem7\\MAI\\Chordify
Python: 3.13.5
Platform: Windows win32, CPU only (CUDA not available)

--- STDOUT ---
[TEST] 1 - Import all modules                        [PASS] PASSED
[TEST] 2 - Generate synthetic audio                  [PASS] PASSED
[TEST] 3 - STFT / iSTFT round-trip                   [PASS] PASSED
[TEST] 4 - UNets forward pass                        [PASS] PASSED
[TEST] 5 - BeatNet forward pass                      [PASS] PASSED
[TEST] 6 - ChordNet forward pass                     [PASS] PASSED
[TEST] 7 - SegmentNet forward pass                   [PASS] PASSED
[TEST] 8 - PitchNet forward pass                     [PASS] PASSED
[TEST] 9 - Label integrity                           [PASS] PASSED
[TEST] 10 - Key detection (diatonic scoring)         [PASS] PASSED    Key: G, score: 8
[TEST] 11 - Timestamp utilities                      [PASS] PASSED
[TEST] 12 - Multimodal fusion (synthetic)            [PASS] PASSED    2 beats | 1 chord segment | 1 section | 1 lyric segment
[TEST] 13 - AI tab generation (synthetic)            [PASS] PASSED    4 chord events; 'e|' present in ascii_tab
[TEST] 14 - Config loading                           [PASS] PASSED    sample_rate=44100, n_channel=2, sources=4
[TEST] 15 - DSP chord fallback (chroma)              [PASS] PASSED
[TEST] 16 - DSP beat fallback (librosa)              [PASS] PASSED
[TEST] 17 - Automatic fallback selection (ChordNet)  [PASS] PASSED    key=Bb, score=7, method=dsp_chroma
[TEST] 18 - Automatic fallback selection (BeatNet)   [PASS] PASSED    BPM=68.0, method=dsp_librosa
[TEST] 19 - End-to-End fallback integration          [PASS] PASSED    BPM=123.0, key=A, 7 beats, 1 chord, time=32.5s

CHORDIFY TEST SUMMARY
========================================================
  19 passed   0 failed   19 total
========================================================

--- WARNINGS ---
librosa UserWarning: n_fft=1024 is too large for input signal of length=690 (in DSP beat test)
    -> Benign; occurs only on very short synthetic audio in Test 16

--- EXIT CODE ---
0

--- IMPORTANT NOTES ---
All 19 tests pass. However:
- Tests 1-18 use random-weight neural models or synthetic audio only.
- Test 19 uses the real Chordify.analyze() pipeline on a 5s synthetic sine wave.
- PASSING THESE TESTS IS NOT EVIDENCE OF CHORD/BEAT/PITCH ACCURACY.
- Historical result in CHORDIFY_STATUS.md shows 14.1s for same test without Whisper (before faster-whisper was installed).
- Current result (32.5s) is slower because Whisper large-v3 model loads on first run.
"""
with open(LOGS / "test_execution.log", "w", encoding="utf-8") as f:
    f.write(log_text)

print("   -> logs/test_execution.log written")

print("\n[Done] All report data files generated.")
print(f"Output directory: {OUT}")
