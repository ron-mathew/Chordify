# Chordify Evaluation Protocol

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
