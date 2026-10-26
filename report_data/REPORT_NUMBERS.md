# Chordify — Complete Numerical Reference for Academic Report

**Project:** Chordify — AI-Based Multimodal Music Understanding and Automatic Guitar Tab Generation  
**Audit Date:** 2026-10-21  
**Status key:** ✅ VERIFIED | ⚠️ MEASURED (real audio, no reference) | 📊 HISTORICAL | 🔴 UNAVAILABLE | 🔵 ESTIMATED

---

## 1. Project Specifications

| Value | Number | Unit | Source | Status |
|-------|--------|------|--------|--------|
| Project version | 1.0.0 | — | config/config.yaml | ✅ VERIFIED |
| Number of pipeline stages | 10 | stages | chordify.py analyze() | ✅ VERIFIED |
| Number of neural models | 5 | models | UNets, BeatNet, ChordNet, PitchNet, SegmentNet | ✅ VERIFIED |
| Number of DSP fallback methods | 2 | methods | dsp_fallback.py (chord + beat) | ✅ VERIFIED |
| Number of external ASR models | 2 | models | Whisper large-v3 + Wav2Vec2 base-960h | ✅ VERIFIED |
| Lines of code (approx.) | ~3500 | LOC | static inspection | 🔵 ESTIMATED |

---

## 2. Model Architecture

### UNets (Source Separation)
| Parameter | Value | Source | Status |
|-----------|-------|--------|--------|
| Architecture | 4 parallel U-Net encoder-decoders | models/unet.py | ✅ VERIFIED |
| Number of encoder levels | 6 | config.yaml: conv_n_filters | ✅ VERIFIED |
| Filter sizes (encoder) | [16, 32, 64, 128, 256, 512] | config.yaml | ✅ VERIFIED |
| Activation function | ELU (both encoder and decoder) | config.yaml | ✅ VERIFIED |
| Output softmax | Channel-wise (energy-conserving) | models/unet.py | ✅ VERIFIED |
| Input shape | (batch, 2, time, 2048) | models/unet.py smoke test | ✅ VERIFIED |
| Output shape | (batch, 4, 2, time, 2048) | models/unet.py smoke test | ✅ VERIFIED |
| Total parameters | 39,294,876 | Torch parameter count 2026-10-21 | ✅ VERIFIED |

### BeatNet (Beat & Tempo Detection)
| Parameter | Value | Source | Status |
|-----------|-------|--------|--------|
| Architecture | 3 parallel BaseTransformer branches | models/beat_net.py | ✅ VERIFIED |
| Input sources | 3 (drums=0.4, bass=0.3, other=0.3) | config.yaml | ✅ VERIFIED |
| Output classes | 3 (0=no-beat, 1=beat, 2=downbeat) | config.yaml / models/beat_net.py | ✅ VERIFIED |
| Frequency encoder layers (f_layers) | 2 | config.yaml | ✅ VERIFIED |
| Frequency attention heads (f_nhead) | 4 | config.yaml | ✅ VERIFIED |
| Time encoder layers (t_layers) | 2 | config.yaml | ✅ VERIFIED |
| Time attention heads (t_nhead) | 4 | config.yaml | ✅ VERIFIED |
| Decoder layers (d_layers) | 2 | config.yaml | ✅ VERIFIED |
| Decoder attention heads (d_nhead) | 8 | config.yaml | ✅ VERIFIED |
| Dropout | 0.5 | config.yaml | ✅ VERIFIED |
| n_group (factorised attention) | 32 | config.yaml | ✅ VERIFIED |
| n_freq | 2048 | config.yaml | ✅ VERIFIED |
| Total parameters | 504,713,475 | Torch parameter count 2026-10-21 | ✅ VERIFIED |

### ChordNet (Chord Recognition)
| Parameter | Value | Source | Status |
|-----------|-------|--------|--------|
| Architecture | Single BaseTransformer + linear head | models/chord_net.py | ✅ VERIFIED |
| Number of chord classes | 122 | config.yaml / utils/labels.py | ✅ VERIFIED |
| Chord qualities in vocabulary | 10 (maj, min, 7, maj7, min7, 6, m6, sus2, sus4, 5) + N + X | utils/labels.py | ✅ VERIFIED |
| Frequency encoder layers | 5 | config.yaml | ✅ VERIFIED |
| Frequency attention heads | 8 | config.yaml | ✅ VERIFIED |
| Time encoder layers | 5 | config.yaml | ✅ VERIFIED |
| Time attention heads | 8 | config.yaml | ✅ VERIFIED |
| Decoder layers | 5 | config.yaml | ✅ VERIFIED |
| Decoder attention heads | 8 | config.yaml | ✅ VERIFIED |
| Dropout | 0.5 | config.yaml | ✅ VERIFIED |
| Total parameters | 389,342,202 | Torch parameter count 2026-10-21 | ✅ VERIFIED |

### PitchNet (Melody / Pitch Estimation)
| Parameter | Value | Source | Status |
|-----------|-------|--------|--------|
| Architecture | PitchEmbedding (4-conv) + PitchEncoder + lyric-conditioned decoder | models/pitch_net.py | ✅ VERIFIED |
| Number of pitch classes | 850 | config.yaml / models/pitch_net.py | ✅ VERIFIED |
| Pitch range | 85 semitones (A0–A7, MIDI 21–105) | models/pitch_net.py | ✅ VERIFIED |
| Sub-semitone resolution | 10 bins/semitone = 10 cents | models/pitch_net.py | ✅ VERIFIED |
| Embedding hidden dim | 16 | config.yaml | ✅ VERIFIED |
| Lyric conditioning weight (wr) | 0.3 | config.yaml | ✅ VERIFIED |
| Freq/time encoder blend weights | [0.6, 0.4] | config.yaml | ✅ VERIFIED |
| Positional encoding scale (pr) | 0.02 | config.yaml | ✅ VERIFIED |
| PitchEmbedding freq reduction | ÷16 (4× MaxPool2d stride 2) | models/pitch_net.py | ✅ VERIFIED |
| Total parameters | 119,440,806 | Torch parameter count 2026-10-21 | ✅ VERIFIED |

### SegmentNet (Structure Segmentation)
| Parameter | Value | Source | Status |
|-----------|-------|--------|--------|
| Architecture | SegmentEmbeddings + BaseTransformer + linear head | models/segment_net.py | ✅ VERIFIED |
| Number of structural classes | 10 | config.yaml / utils/labels.py | ✅ VERIFIED |
| Class names | start, end, intro, outro, verse, chorus, solo, break, bridge, inst | utils/labels.py | ✅ VERIFIED |
| Embedding hidden dim | 128 | config.yaml | ✅ VERIFIED |
| Freq reduction after embedding | ÷4 (2× MaxPool2d) | models/segment_net.py | ✅ VERIFIED |
| d_model after embedding | 512 (= 2048÷4) | derived | ✅ VERIFIED |
| Total parameters | 14,025,676 | Torch parameter count 2026-10-21 | ✅ VERIFIED |

### Combined Neural Model Summary
| Model | Parameters | Checkpoint | Status |
|-------|-----------|------------|--------|
| UNets | 39,294,876 | MISSING | ❌ No checkpoint |
| BeatNet | 504,713,475 | MISSING | ❌ No checkpoint |
| ChordNet | 389,342,202 | MISSING | ❌ No checkpoint |
| PitchNet | 119,440,806 | MISSING | ❌ No checkpoint |
| SegmentNet | 14,025,676 | MISSING | ❌ No checkpoint |
| **TOTAL** | **1,066,816,534** | — | — |

---

## 3. Audio Preprocessing

| Parameter | Value | Unit | Source | Status |
|-----------|-------|------|--------|--------|
| Sample rate | 44,100 | Hz | config/config.yaml | ✅ VERIFIED |
| Channels | 2 | (stereo) | config/config.yaml | ✅ VERIFIED |
| FFT window size (n_fft) | 4,096 | samples | config/config.yaml | ✅ VERIFIED |
| FFT hop length | 1,024 | samples | config/config.yaml | ✅ VERIFIED |
| STFT frequency bins (total) | 2,049 | bins (n_fft/2 + 1) | derived | ✅ VERIFIED |
| STFT frequency bins (model input) | 2,048 | bins (n_fft/2) | pipeline code | ✅ VERIFIED |
| Frame rate | 43.07 | fps (44100/1024) | derived | ✅ VERIFIED |
| STFT window | Hann window | — | utils/spectrogram.py | ✅ VERIFIED |
| STFT padding | True (zero-pad) | — | config/config.yaml | ✅ VERIFIED |
| U-Net sliding window size | 256 | frames | config/config.yaml (n_time) | ✅ VERIFIED |
| U-Net chunk size in samples | (256-5)×1024 + 4096 = 259,076 | samples | derived | ✅ VERIFIED |
| ASR input sample rate | 16,000 | Hz | pipeline/lyrics.py | ✅ VERIFIED |
| Source stems output | (4, 2, samples) | — | pipeline/separation.py | ✅ VERIFIED |

---

## 4. Dataset Statistics

| Statistic | Value | Unit | Source | Status |
|-----------|-------|------|--------|--------|
| Dataset name | MUSDB18 | — | train/ and test/ directories | ✅ VERIFIED |
| Total tracks (train) | 100 | tracks | ls train/*.stem.mp4 | ✅ VERIFIED |
| Total tracks (test) | 50 | tracks | ls test/*.stem.mp4 | ✅ VERIFIED |
| Total tracks | 150 | tracks | — | ✅ VERIFIED |
| Train set size | 3.43 | GB | os.stat | ✅ VERIFIED |
| Test set size | 1.87 | GB | os.stat | ✅ VERIFIED |
| Total size | 5.30 | GB | os.stat | ✅ VERIFIED |
| Avg track size (train) | 35.1 | MB | os.stat | ✅ VERIFIED |
| Avg track size (test) | 38.3 | MB | os.stat | ✅ VERIFIED |
| Min track size (train) | 2.0 | MB | os.stat | ✅ VERIFIED |
| Max track size (train) | 96.4 | MB | os.stat | ✅ VERIFIED |
| Min track size (test) | 11.7 | MB | os.stat | ✅ VERIFIED |
| Max track size (test) | 66.0 | MB | os.stat | ✅ VERIFIED |
| Estimated avg duration | ~3.5 | min/track | MUSDB18 documentation reference | 🔵 ESTIMATED |
| Estimated total duration (train) | ~350 | min | estimated | 🔵 ESTIMATED |
| Estimated total duration (test) | ~175 | min | estimated | 🔵 ESTIMATED |
| Format | STEM MP4 (5 sub-tracks) | — | file inspection | ✅ VERIFIED |
| Stems per track | 5 | (mixture, drums, bass, other, vocals) | stempeg format | ✅ VERIFIED |
| Chord annotations in MUSDB18 | NONE | — | dataset documentation | 🔴 UNAVAILABLE |
| Beat annotations in MUSDB18 | NONE | — | dataset documentation | 🔴 UNAVAILABLE |
| ffmpeg available for decoding | NO | — | tested in environment | ✅ VERIFIED |
| Used for training | NO | — | no training code executed | ✅ VERIFIED |

---

## 5. Model / Checkpoint Status

| Model | Expected Path | Exists | Valid | Notes |
|-------|---------------|--------|-------|-------|
| UNets | checkpoints/unet.pth | ❌ NO | ❌ NO | checkpoints/ dir does not exist |
| BeatNet | checkpoints/beat_net.pth | ❌ NO | ❌ NO | same |
| ChordNet | checkpoints/chord_net.pth | ❌ NO | ❌ NO | same |
| PitchNet | checkpoints/pitch_net.pth | ❌ NO | ❌ NO | same |
| SegmentNet | checkpoints/segment_net.pth | ❌ NO | ❌ NO | same |
| Whisper large-v3 | cached by faster-whisper | ✅ YES | ✅ YES | ~3.09 GB; successfully loaded in Test 19 |
| Wav2Vec2 base-960h | cached by transformers | ❌ UNKNOWN | ❌ UNKNOWN | transformers not installed |

---

## 6. Test Results

| Metric | Value | Source | Status |
|--------|-------|--------|--------|
| Total tests | 19 | scripts/test_pipeline.py | ✅ VERIFIED |
| Tests passed | 19 | run 2026-10-21 | ✅ VERIFIED |
| Tests failed | 0 | run 2026-10-21 | ✅ VERIFIED |
| Tests skipped | 0 | run 2026-10-21 | ✅ VERIFIED |
| Unit tests | 14 | test categories | ✅ VERIFIED |
| Integration tests | 4 | test categories | ✅ VERIFIED |
| End-to-end tests | 1 | test categories | ✅ VERIFIED |
| Total test runtime | ~35 | seconds | measured 2026-10-21 | ✅ VERIFIED |
| Key detection result (G-Am-C-D) | key=G, score=8 | Test 10 | ✅ VERIFIED |
| Previously reported 14-test result | 14/14 PASS | CHORDIFY_STATUS.md | 📊 HISTORICAL |
| Previously reported 19-test result | 19/19 PASS | CHORDIFY_STATUS.md | 📊 HISTORICAL |
| **⚠️ Important** | None of these tests measure recognition accuracy | — | — |

---

## 7. Chord Recognition

### DSP Chroma Fallback — Real Audio Measurements
| Metric | Value | Audio | Source | Status |
|--------|-------|-------|--------|--------|
| Chord segments detected | 30 | converted-audio (204s) | outputs JSON | ⚠️ MEASURED |
| Average chord duration | 6.83 | s | outputs JSON | ⚠️ MEASURED |
| Minimum chord duration | 2.23 | s | outputs JSON | ⚠️ MEASURED |
| Maximum chord duration | 25.54 | s | outputs JSON | ⚠️ MEASURED |
| N (no-chord) segments | 1 of 30 (3.3%) | converted-audio (204s) | outputs JSON | ⚠️ MEASURED |
| Unique chord labels detected | 9 | converted-audio (204s) | outputs JSON | ⚠️ MEASURED |
| Detected key | Bb | converted-audio (204s) | outputs JSON | ⚠️ MEASURED |
| Key score (diatonic fit) | 9 /10 | converted-audio (204s) | outputs JSON | ⚠️ MEASURED |
| Chord segments (75s clip) | 12 | 75s clip | outputs JSON | ⚠️ MEASURED |
| Key (75s clip) | Bb | 75s clip | outputs JSON | ⚠️ MEASURED |
| Chord template count | 24 | — | dsp_fallback.py | ✅ VERIFIED |
| Viterbi temperature γ | 14.0 | — | dsp_fallback.py | ✅ VERIFIED |
| Chroma fusion weights | 65% full-band + 35% bass | — | dsp_fallback.py | ✅ VERIFIED |
| Minimum segment duration | 1.0 | s | dsp_fallback.py | ✅ VERIFIED |
| Halftime BPM threshold | 135.0 | BPM | dsp_fallback.py | ✅ VERIFIED |
| **Frame-wise chord accuracy** | **UNAVAILABLE** | — | no ground truth | 🔴 UNAVAILABLE |
| **Weighted CSR** | **UNAVAILABLE** | — | no ground truth | 🔴 UNAVAILABLE |

---

## 8. Beat Tracking

### DSP Librosa Fallback — Real Audio Measurements
| Metric | Value | Audio | Source | Status |
|--------|-------|-------|--------|--------|
| Estimated BPM | 83.4 | converted-audio (204s) | outputs JSON | ⚠️ MEASURED |
| Beat events detected | 187 | converted-audio (204s) | outputs JSON | ⚠️ MEASURED |
| Downbeat events detected | 63 | converted-audio (204s) | outputs JSON | ⚠️ MEASURED |
| Estimated BPM (75s clip) | 57.4 | 75s clip | outputs JSON | ⚠️ MEASURED |
| Beat events (75s clip) | 40 | 75s clip | outputs JSON | ⚠️ MEASURED |
| Estimated BPM (5s synthetic) | 123.0 | test_audio.wav | Test 19 | ⚠️ MEASURED |
| Beat events (5s synthetic) | 7 + 3 downbeats | test_audio.wav | Test 19 | ⚠️ MEASURED |
| **Beat F-measure** | **UNAVAILABLE** | — | no reference | 🔴 UNAVAILABLE |
| **Tempo error (BPM)** | **UNAVAILABLE** | — | no reference | 🔴 UNAVAILABLE |
| Halftime correction threshold | 135.0 | BPM | dsp_fallback.py | ✅ VERIFIED |

---

## 9. Lyrics Transcription

### Configuration
| Parameter | Value | Source | Status |
|-----------|-------|--------|--------|
| Primary ASR model | Whisper large-v3 | config.yaml | ✅ VERIFIED |
| Inference engine | faster-whisper (CTranslate2 backend) | pipeline/lyrics.py | ✅ VERIFIED |
| Compute type (CPU) | int8 (quantized) | pipeline/lyrics.py | ✅ VERIFIED |
| Beam size | 5 | pipeline/lyrics.py | ✅ VERIFIED |
| VAD filter | False (disabled) | pipeline/lyrics.py | ✅ VERIFIED |
| No-speech threshold | 0.85 | pipeline/lyrics.py | ✅ VERIFIED |
| Word probability threshold | 0.10 | pipeline/lyrics.py | ✅ VERIFIED |
| Condition on previous text | False | pipeline/lyrics.py | ✅ VERIFIED |
| Vocal source (no U-Net ckpt) | Center-Channel Extraction (CCE) | pipeline/lyrics.py | ✅ VERIFIED |
| ASR input sample rate | 16,000 Hz | pipeline/lyrics.py | ✅ VERIFIED |
| Word timestamps | Yes | pipeline/lyrics.py | ✅ VERIFIED |

### Real Audio Measurements
| Metric | Value | Audio | Source | Status |
|--------|-------|-------|--------|--------|
| Lyric phrases detected | 38 | converted-audio (204s) | outputs JSON | ⚠️ MEASURED |
| Lyric coverage (% of audio duration) | 33.2% | converted-audio (204s) | outputs JSON | ⚠️ MEASURED |
| First phrase start | 0.37 | s | outputs JSON | ⚠️ MEASURED |
| Last phrase end | 68.06 | s | outputs JSON | ⚠️ MEASURED |
| Lyric phrases (75s clip) | 44 | 75s clip | outputs JSON | ⚠️ MEASURED |
| Lyric phrases (5s synthetic) | 1 (hallucination) | test_audio.wav | Test 19 | ⚠️ MEASURED |
| Language detected (5s synthetic) | en | test_audio.wav | Test 19 | ⚠️ MEASURED |
| Language probability (5s synthetic) | 0.19 | test_audio.wav | Test 19 | ⚠️ MEASURED |
| **Word Error Rate (WER)** | **UNAVAILABLE** | — | no reference transcripts | 🔴 UNAVAILABLE |

---

## 10. Source Separation

| Metric | Value | Source | Status |
|--------|-------|--------|--------|
| Architecture | 4 parallel U-Nets with channel-wise softmax | models/unet.py | ✅ VERIFIED |
| Mask energy conservation | Verified (sum = 1.0 ± 1e-5) | Test 4 | ✅ VERIFIED |
| Output stems | vocal / bass / drums / other | models/unet.py | ✅ VERIFIED |
| Checkpoint status | MISSING | audit 2026-10-21 | ✅ VERIFIED |
| **SDR (vocals)** | **UNAVAILABLE** | no checkpoint, no reference | 🔴 UNAVAILABLE |
| **SDR (bass)** | **UNAVAILABLE** | no checkpoint, no reference | 🔴 UNAVAILABLE |
| **SDR (drums)** | **UNAVAILABLE** | no checkpoint, no reference | 🔴 UNAVAILABLE |
| **SDR (other)** | **UNAVAILABLE** | no checkpoint, no reference | 🔴 UNAVAILABLE |
| CCE fallback active | Yes | pipeline/lyrics.py | ✅ VERIFIED |

---

## 11. Structural Segmentation

| Metric | Value | Audio | Source | Status |
|--------|-------|-------|--------|--------|
| Segment label count | 10 | — | utils/labels.py | ✅ VERIFIED |
| Sections detected (204s audio) | 1 ('chorus') | converted-audio | outputs JSON | ⚠️ MEASURED |
| Sections detected (5s synthetic) | 1 ('outro') | test_audio.wav | Test 19 | ⚠️ MEASURED |
| **Boundary F-measure** | **UNAVAILABLE** | — | no checkpoint, no reference | 🔴 UNAVAILABLE |
| Note | Single-section output expected with random weights | — | — | — |

---

## 12. Pitch Estimation

| Metric | Value | Audio | Source | Status |
|--------|-------|-------|--------|--------|
| Pitch class count | 850 | — | config.yaml / Test 8 | ✅ VERIFIED |
| Pitch range | MIDI 21–105 (A0–A7) | — | models/pitch_net.py | ✅ VERIFIED |
| Sub-semitone resolution | 10 cents | — | models/pitch_net.py | ✅ VERIFIED |
| Pitch frames (204s audio) | 8,820 | converted-audio | outputs JSON | ⚠️ MEASURED |
| Pitch frames (5s synthetic) | 216 | test_audio.wav | Test 19 | ⚠️ MEASURED |
| Pitch output (random weights) | All MIDI=100 (E7, cents_offset=90) | all audio | outputs JSON | ⚠️ MEASURED |
| **Raw pitch accuracy** | **UNAVAILABLE** | — | no checkpoint, no reference | 🔴 UNAVAILABLE |

---

## 13. Guitar Tab Generation

| Metric | Value | Audio | Source | Status |
|--------|-------|-------|--------|--------|
| Chord voicings in dictionary | 70 | — | pipeline/tabs.py | ✅ VERIFIED |
| Tab chord events (204s audio) | 30 | converted-audio | outputs JSON | ⚠️ MEASURED |
| Tab file lines (204s audio) | 187 | converted-audio | .tab file | ⚠️ MEASURED |
| Tab chord events (75s audio) | 12 | 75s clip | outputs JSON | ⚠️ MEASURED |
| Tab file lines (75s audio) | 57 | 75s clip | .tab file | ⚠️ MEASURED |
| Tab chord events (5s synthetic) | 1 | test_audio.wav | Test 19 | ⚠️ MEASURED |
| Tab file lines (5s synthetic) | 16 | test_audio.wav | .tab file | ⚠️ MEASURED |
| Valid 6-string ASCII format | YES ('e\|' present) | all outputs | Test 13, 19 | ✅ VERIFIED |
| Bars per line | 4 | — | pipeline/tabs.py | ✅ VERIFIED |
| Chord-tab agreement | 100% (by construction) | — | code inspection | ✅ VERIFIED |
| Musical accuracy vs reference | UNAVAILABLE | — | no reference tabs | 🔴 UNAVAILABLE |

---

## 14. Runtime and Resource Consumption

| Metric | Value | Unit | Audio | Source | Status |
|--------|-------|------|-------|--------|--------|
| Total pipeline runtime (5s, with Whisper) | 32.5 | s | test_audio.wav | Test 19, 2026-10-21 | ✅ VERIFIED |
| Real-time factor (5s audio) | 6.5× | — | test_audio.wav | Test 19 | ✅ VERIFIED |
| Historical runtime (5s, no Whisper) | 14.1 | s | test_audio.wav | CHORDIFY_STATUS.md | 📊 HISTORICAL |
| Historical RT factor (no Whisper) | 2.82× | — | CHORDIFY_STATUS.md | 📊 HISTORICAL |
| Whisper loading overhead (est.) | ~20–25 | s | first run, CPU | estimated | 🔵 ESTIMATED |
| Device | CPU | — | test_audio.wav | Test 19 | ✅ VERIFIED |
| GPU | None (CUDA unavailable) | — | test_audio.wav | Test 19 | ✅ VERIFIED |
| Peak RAM | NOT MEASURED | — | — | — | 🔴 UNAVAILABLE |
| GPU memory | NOT APPLICABLE | — | — | no GPU | 🔴 UNAVAILABLE |
| Repeated measurements | NOT DONE | — | — | — | 🔴 UNAVAILABLE |

---

## 15. End-to-End Dashboard Testing

| Field | Value | Source | Status |
|-------|-------|--------|--------|
| Gradio version installed | 6.28.0 | pip show | ✅ VERIFIED |
| Dashboard UI tabs | 8 | app/app.py | ✅ VERIFIED |
| Live server test | NOT EXECUTED | cannot start server in agentic env | 🔴 UNAVAILABLE |
| Pipeline backend verified | YES (all stages) | Test 19 | ✅ VERIFIED |
| Test audio BPM | 123.0 | Test 19 outputs | ✅ VERIFIED |
| Test audio key | A | Test 19 outputs | ✅ VERIFIED |
| Test audio chord method | dsp_chroma | Test 19 outputs | ✅ VERIFIED |
| Test audio beat method | dsp_librosa | Test 19 outputs | ✅ VERIFIED |
| Test audio lyrics model | Whisper (large-v3) | Test 19 outputs | ✅ VERIFIED |
| Test audio chords detected | 1 (Am, 5.0s, conf=0.79) | Test 19 outputs | ✅ VERIFIED |
| Test audio beats detected | 7 beats + 3 downbeats | Test 19 outputs | ✅ VERIFIED |
| Test audio sections detected | 1 ('outro') | Test 19 outputs | ✅ VERIFIED |
| Test audio pitch frames | 216 (all MIDI=100) | Test 19 outputs | ✅ VERIFIED |
| Test audio lyric phrases | 1 ('singing vocals') | Test 19 outputs | ✅ VERIFIED |
| Test audio tab lines | 16 | Test 19 outputs | ✅ VERIFIED |
| Total runtime (5s audio) | 32.5 s | Test 19 | ✅ VERIFIED |
| Real audio BPM (204s) | 83.4 | outputs inspection | ⚠️ MEASURED |
| Real audio key (204s) | Bb (score 9) | outputs inspection | ⚠️ MEASURED |
| Real audio chords (204s) | 30 segments | outputs inspection | ⚠️ MEASURED |
| Real audio lyrics phrases (204s) | 38 phrases, 33.2% coverage | outputs inspection | ⚠️ MEASURED |

---

## Appendix: Derived Quantities

| Quantity | Formula | Value |
|----------|---------|-------|
| U-Net sliding window (samples) | (256-5) × 1024 + 4096 | 259,076 samples |
| Frame duration | 1024 / 44100 | 23.22 ms |
| Frame rate | 44100 / 1024 | 43.07 fps |
| Pitch MIDI note from class index | 21 + class_index ÷ 10 | MIDI 21 (A0) to 105 (A7) |
| Pitch cents offset from class index | (class_index mod 10) × 10 | 0–90 cents |
| SegmentNet d_model after embedding | 2048 ÷ 4 | 512 |
| PitchEmbedding freq after 4 pools | 2048 ÷ 16 | 128 |
| U-Net mask sum (verified) | softmax across source dim | 1.0 ± 1e-5 |
| Total neural parameters (all 5 models) | sum | 1,066,816,534 (~1.07 billion) |
