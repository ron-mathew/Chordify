# Chordify — Limitations and Future Work

**Date:** 2026-10-21

---

## Current Limitations

### 1. No Trained Neural Checkpoints
The five custom neural models (UNets, BeatNet, ChordNet, PitchNet, SegmentNet) have their architectures fully implemented and shape-verified, but no trained `.pth` checkpoint files exist. Consequently:
- Source separation runs with random weights (outputs are acoustically meaningless)
- Neural beat detection is bypassed in favour of the DSP librosa fallback
- Neural chord recognition is bypassed in favour of the DSP chroma fallback
- Pitch/melody estimation outputs a constant MIDI=100 for every frame
- Structure segmentation outputs a single section label for the entire track

This is the single most significant limitation for reporting accuracy metrics.

### 2. No Ground-Truth Annotations for Accuracy Evaluation
MUSDB18 provides only multi-track separated stems, not chord labels, beat positions, melody annotations, or structural segmentation labels. Without a chord-annotated dataset such as Billboard, RWC, or CASD, frame-wise chord accuracy (MIREX standard), Weighted Chord Symbol Recall (WCSR), and beat F-measure cannot be calculated.

### 3. Chord DSP Fallback Covers Only 24 Chord Classes
The DSP chroma fallback recognises 24 classes (12 Major + 12 Minor triads). The neural ChordNet is designed for 122 chord classes including dominant 7th, major 7th, minor 7th, 6th, m6, sus2, sus4, and power chords. Extended and suspended chord recognition requires trained neural weights.

### 4. Structure Segmentation is Non-Functional
Without a trained SegmentNet checkpoint, the system always outputs a single section ('chorus' or 'outro') spanning the entire audio file. Real-world structural boundaries (verse/chorus/bridge transitions) are not detected.

### 5. Melody/Pitch Estimation is Non-Functional
Without a trained PitchNet checkpoint, the pitch output is a constant index (MIDI 100, E7) for every frame. Melody tracking requires trained weights.

### 6. Lyrics Coverage is Partial
On the 204-second real audio test, lyrics coverage was 33.2% of the audio duration (words found only in the first 68 seconds). This is partly because the vocal source is derived via Center-Channel Extraction (CCE) rather than a trained U-Net, and partly because Whisper's no-speech detection may filter out quiet or melismatic singing.

### 7. Tempo Ambiguity
The DSP beat tracker applies halftime correction for BPM > 135, but this is a heuristic. Complex meters (3/4, 6/8, 5/4) and tempo changes within a song are not handled.

### 8. No GPU Acceleration in Test Environment
All pipeline timing measurements were taken on CPU only (CUDA unavailable). Real-time performance with a GPU would be significantly faster, particularly for the Whisper ASR stage.

### 9. Source Audio for Lyrics Uses CCE, Not Neural Separation
Because the U-Net has no checkpoint, the CCE (Centre-Channel Extraction) method is used as the vocal source for Whisper. CCE preserves centred vocals but also retains centred instruments, which can degrade transcription quality for songs with strong centred non-vocal instruments.

### 10. MUSDB18 Cannot Be Decoded Without ffmpeg
The stempeg/torchaudio libraries require ffmpeg to decode `.stem.mp4` files. ffmpeg is not installed in the current environment, preventing direct audio duration measurement, waveform visualisation, and stem extraction from the dataset.

### 11. Dataset Has No Chord Vocabulary Alignment
MUSDB18 does not use the Lamucal/Chordify 122-class chord vocabulary. Any future evaluation would require mapping between annotation systems.

### 12. Single-Run Performance Measurements
Due to the long Whisper model loading time (~20–25 seconds on CPU), only one reliable timing measurement was taken (32.5 seconds for 5-second audio). Statistical confidence (mean ± std) cannot be reported without multiple runs.

---

## Future Work

### Priority 1: Train or Obtain Neural Checkpoints
The highest-impact improvement is acquiring trained weights for all five models:
- Fine-tune on MUSDB18 stems (source separation) and RWC/Billboard (chord recognition)
- This would unlock neural chord accuracy, beat accuracy, melody tracking, and structure detection
- Original Lamucal checkpoints (if released) would be compatible with this architecture

### Priority 2: Chord Evaluation on Annotated Dataset
- Evaluate the DSP chroma fallback on a chord-annotated dataset (McGill Billboard, RWC-Pop, CASD)
- Compute frame-wise accuracy, WCSR, and chord change boundary F-measure
- Compare DSP fallback against ChordNet with trained weights

### Priority 3: Beat Evaluation
- Evaluate DSP librosa fallback on a beat-annotated dataset (BALLROOM, SMC, GTZAN)
- Measure beat precision, recall, F-measure at 70ms and 500ms tolerance windows

### Priority 4: WER Measurement for Lyrics
- Obtain lyrics ground truth for the test audio files
- Compute WER, CER, insertions, deletions, and substitutions

### Priority 5: Structure Segmentation
- Obtain structure annotations (SALAMI dataset)
- Evaluate boundary detection precision, recall, and F-measure with 3-second tolerance

### Priority 6: GPU Deployment
- Deploy on CUDA-capable hardware
- Re-measure all pipeline timings with GPU acceleration
- Report real-time factor and GPU memory consumption

### Priority 7: Install ffmpeg
- Install ffmpeg to enable MUSDB18 stem decoding
- Measure actual audio duration statistics from the dataset
- Run full EDA notebook (notebooks/eda_musdb18.py) to generate verified duration plots

### Priority 8: Extended Chord Vocabulary in DSP Path
- Add 7th and sus chord templates to the DSP chroma fallback
- Evaluate whether extended templates improve accuracy without neural weights

### Priority 9: Vocals-Only Evaluation
- With U-Net checkpoint, evaluate separation quality using BSS Eval metrics (SDR, SIR, SAR) on MUSDB18 test set
- Compare CCE (current fallback) against neural U-Net separation for ASR quality

### Priority 10: Training Infrastructure
- Implement training loops for all five models
- Use MUSDB18 train split for source separation training
- Use chord-annotated datasets for ChordNet training
- Document batch size, learning rate, optimizer, and epochs
