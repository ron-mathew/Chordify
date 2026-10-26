# Chordify — Executive Summary

**Project:** AI-Based Multimodal Music Understanding and Automatic Guitar Tab Generation  
**Audit Date:** 2026-10-21  
**Auditor:** Kiro AI Code Auditor

---

## What Chordify Is

Chordify is a fully implemented Python pipeline that performs multimodal music analysis on audio files. It combines five custom neural network models (ported from the Lamucal open-source research codebase), two DSP signal processing fallback algorithms, and the OpenAI Whisper speech recognition model to produce chord charts, beat maps, structural section labels, melody estimates, lyric transcriptions, and guitar tab output — all from a single audio file.

The system includes:
- A complete Python source codebase (~3,500 lines across 30 files)
- A Gradio v4 web dashboard (app/app.py) with 8 output tabs
- A 19-test progressive test suite that passes completely
- DSP fallback algorithms that produce real, verifiable musical analysis without neural checkpoints
- The MUSDB18 dataset (150 tracks, 5.30 GB) loaded in the dataset directories

---

## Implementation Status

### Fully Operational (with DSP fallbacks)
| Component | Status | Method Active |
|-----------|--------|---------------|
| Audio loading and STFT | ✅ Complete | Always active |
| Beat detection and BPM estimation | ✅ Complete | DSP librosa fallback (no BeatNet checkpoint) |
| Chord recognition and key estimation | ✅ Complete | DSP chroma + Viterbi (no ChordNet checkpoint) |
| Lyrics transcription + word timestamps | ✅ Complete | Whisper large-v3 (operational) |
| Guitar tab generation (6-string ASCII) | ✅ Complete | Always active (derived from chord output) |
| Multimodal timeline fusion | ✅ Complete | Always active |
| Gradio dashboard UI | ✅ Implemented | Code verified; not live-tested in this audit |

### Architecture Implemented, Checkpoints Missing
| Component | Status | Impact |
|-----------|--------|--------|
| U-Net source separation | Architecture only | Runs with random weights; CCE fallback for lyrics |
| BeatNet neural beat detection | Architecture only | Falls back to DSP librosa (transparent) |
| ChordNet neural chord recognition | Architecture only | Falls back to DSP chroma (transparent) |
| PitchNet melody estimation | Architecture only | Random weight constant output (MIDI=100) |
| SegmentNet structure segmentation | Architecture only | Random weight single-section output |

---

## Key Numerical Findings

### Models
- 5 custom neural architectures totalling **1,066,816,534 parameters**
- Largest model: BeatNet with **504.7 million parameters**
- All 5 architectures smoke-tested and shape-verified with random weights

### Audio Processing
- Sample rate: **44,100 Hz**, stereo, n_fft=**4,096**, hop_length=**1,024**
- Frame rate: **43.07 fps**, 256-frame sliding window for U-Net
- Chord vocabulary: **122 classes** (12 roots × 10 qualities + N + X)
- Pitch classes: **850** (85 semitones × 10 cent bins, MIDI 21–105)
- Segment classes: **10** (intro/verse/chorus/bridge/solo/etc.)

### Dataset
- MUSDB18: **150 tracks** (100 train + 50 test), **5.30 GB** on disk
- Format: STEM MP4 with 5 sub-tracks (mixture + 4 stems)
- No chord/beat/melody annotations — source separation stems only
- ffmpeg unavailable; audio durations estimated at ~3.5 min/track

### Tests
- **19 / 19 tests pass** (run 2026-10-21 on Python 3.13.5, CPU)
- Runtime: ~35 seconds (including Whisper large-v3 load in Test 19)

### Real-Audio Pipeline Run (204-second audio, DSP methods)
- BPM: **83.4**, Key: **Bb** (diatonic score 9/10)
- Chord segments: **30** (avg duration 6.83s, 9 unique chords)
- Beat events: **187** regular + **63** downbeats
- Lyric phrases: **38** (33.2% coverage, first word at 0.37s, last at 68.06s)
- Tabs: 187-line ASCII file with valid 6-string format

### End-to-End Performance
- 5-second synthetic audio, CPU only: **32.5 seconds** total runtime (6.5× real-time)
- Historical result without Whisper: **14.1 seconds** (2.82× real-time)
- Whisper large-v3 loading dominates (~20–25s on first run)

---

## What Cannot Be Reported

The following metrics are **unavailable** and cannot be fabricated:

| Metric | Reason |
|--------|--------|
| Chord recognition accuracy (frame-wise, WCS) | No trained ChordNet checkpoint; no chord annotations in MUSDB18 |
| Beat F-measure | No reference beat annotations |
| Source separation SDR/SIR/SAR | No trained U-Net checkpoint; no reference stems for test audio |
| Pitch estimation accuracy | No trained PitchNet checkpoint |
| Structure boundary F-measure | No trained SegmentNet checkpoint; no annotations |
| WER / CER (lyrics) | No reference transcripts available |
| Gradient descent training metrics | No training was executed |

---

## Files in This Evidence Package

| File | Contents |
|------|----------|
| REPORT_NUMBERS.md | **Master numerical reference** — all verified values with evidence |
| IMPLEMENTATION_INVENTORY.csv | Per-component status, algorithm, I/O, checkpoint requirement |
| MODEL_SPECIFICATIONS.csv | Architecture details for all 7 models |
| CONFIGURATION_INVENTORY.csv | All config.yaml and pipeline constants |
| DATASET_INVENTORY.csv | All 150 MUSDB18 tracks with sizes |
| DATASET_STATISTICS.json | Dataset summary statistics |
| CHECKPOINT_AUDIT.csv | Checkpoint existence and validity for each model |
| TEST_RESULTS.csv + .json | All 19 test results with evidence |
| EVALUATION_PROTOCOL.md | What was evaluated, how, and what was not |
| CHORD_METRICS.csv | DSP chord measurements + accuracy status |
| BEAT_METRICS.csv | DSP beat measurements + accuracy status |
| LYRICS_METRICS.csv | Whisper configuration + real audio measurements |
| SEPARATION_METRICS.csv | U-Net architecture verification + accuracy status |
| STRUCTURE_METRICS.csv | SegmentNet measurements + accuracy status |
| PITCH_METRICS.csv | PitchNet measurements + accuracy status |
| TAB_METRICS.csv | Tab generation measurements |
| PERFORMANCE_METRICS.csv | Runtime benchmarks |
| DASHBOARD_TEST_RESULTS.csv | UI verification results |
| EXPERIMENT_MANIFEST.json | Environment and experiment log |
| LIMITATIONS_AND_FUTURE_WORK.md | Academic limitations section |
| figures/ | 8 report-ready figures |
| logs/test_execution.log | Complete test run transcript |
