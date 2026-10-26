# 🎸 Chordify
### AI-Based Multimodal Music Understanding and Automatic Guitar Tab Generation

> **Academic project — Semester 7, MAI**  
> Based on the open-source [Lamucal](https://github.com/Lamucal/Lamucal) project.

---

## Overview

Chordify is a multimodal music intelligence pipeline that takes a single audio file as input and produces a unified, time-synchronised analysis including:

| Output | Description |
|---|---|
| **BPM / Tempo** | Estimated beats per minute |
| **Beats & Downbeats** | Frame-level beat events with timestamps |
| **Key** | Estimated tonal centre via diatonic scoring |
| **Chords** | 122-class chord recognition with timestamps |
| **Song Sections** | 10-class structural segmentation (verse, chorus, …) |
| **Vocal Pitch** | 850-class sub-semitone pitch tracking |
| **Lyrics** | Whisper ASR transcription with word-level timestamps |
| **Guitar Tab** | AI-generated 6-string ASCII tablature |

All outputs are fused onto a **common timeline** — the key research contribution that distinguishes Chordify from single-task MIR systems.

---

## Architecture

```
Input Audio (.wav / .mp3 / .flac)
        │
        ▼
  Audio Loader  (44.1 kHz stereo)
        │
        ▼
     S T F T  (n_fft=4096, hop=1024, Hann window)
        │
        ▼
   U-Net Source Separation
        │
   ┌────┴────────────────────────────────┐
   │ vocal  │  bass  │  drums  │  other  │
   ▼        ▼        ▼         ▼
Pitch-Net  Lyrics   Beat-Net (drums+bass+other)
           (ASR)        │
                        ▼
                   Chord-Net  ← beat logits
                        │
                        ▼
                  Segment-Net ← chord logits
                        │
                        ▼
             ┌──────────────────────────┐
             │  Multimodal Fusion       │
             │  common timestamp base   │
             └──────────────────────────┘
                        │
                        ▼
               AI Tab Generation
                        │
                        ▼
            Chordify Output (JSON + ASCII Tab)
```

### Models

| Model | Architecture | Input | Output |
|---|---|---|---|
| **U-Net** | 4 parallel encoder-decoders, ELU, BN, softmax masks | Mix spectrogram | 4 stem waveforms |
| **Beat-Net** | 3-branch factorised axial transformer, weighted sum | Drums+bass+other specs | Beat class per frame |
| **Chord-Net** | Single BaseTransformer + 122-class head | Accomp spec + beat logits | Chord class per frame |
| **Segment-Net** | Conv embedding + BaseTransformer + 10-class head | Mix spec + chord logits | Section class per frame |
| **Pitch-Net** | Conv embedding + PitchEncoder + lyric-attn decoder | Vocal spec + lyrics matrix | Pitch class per frame |
| **Whisper** | OpenAI Whisper (base) | Vocal mono 16 kHz | Text + word timestamps |
| **Wav2Vec2** | facebook/wav2vec2-base-960h | Vocal mono 16 kHz | Frame features for PitchNet |

---

## Dataset

The project uses **MUSDB18** — 150 multi-track songs in Native Instruments Stems format (`.stem.mp4`).

```
Chordify/
├── train/   100 songs  (100 × .stem.mp4)
└── test/    50  songs  (50  × .stem.mp4)
```

Each `.stem.mp4` contains 5 stereo AAC streams at 44.1 kHz:
- Stream 0 — Mixture
- Stream 1 — Drums
- Stream 2 — Bass
- Stream 3 — Other
- Stream 4 — Vocals

---

## Project Structure

```
Chordify/
│
├── chordify.py              ← Main pipeline orchestrator
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── config.yaml          ← All model + audio settings
│   └── paths.yaml           ← Checkpoint paths (edit this)
│
├── models/                  ← Neural network architectures
│   ├── transformers.py      ← Shared factorised axial transformer
│   ├── unet.py
│   ├── beat_net.py
│   ├── chord_net.py
│   ├── pitch_net.py
│   ├── segment_net.py
│   └── load.py
│
├── pipeline/                ← Per-stage pipeline modules
│   ├── audio_loader.py
│   ├── separation.py
│   ├── beat_analysis.py
│   ├── chord_analysis.py
│   ├── structure_analysis.py
│   ├── pitch_analysis.py
│   ├── lyrics.py            ← Whisper + Wav2Vec2 (implemented)
│   ├── fusion.py            ← Multimodal timeline merger
│   ├── modulation.py        ← Key detection
│   └── tabs.py              ← AI guitar tab generation
│
├── utils/
│   ├── audio.py
│   ├── spectrogram.py
│   ├── timestamps.py
│   └── labels.py            ← 122 chord + 10 segment labels
│
├── scripts/
│   ├── run_chordify.py      ← CLI entry point
│   ├── test_pipeline.py     ← Progressive test suite (14 tests)
│   ├── preprocess.py        ← MUSDB18 stem extractor
│   └── download_checkpoints.py ← Checkpoint status checker
│
├── notebooks/
│   └── eda_musdb18.py       ← EDA: duration, waveform, spectrograms, energy
│
├── app/
│   └── app.py               ← Gradio web UI
│
├── checkpoints/             ← Place .pth files here
├── data/                    ← Extracted stems (after preprocessing)
├── outputs/                 ← Pipeline results
└── CHORDIFY_STATUS.md       ← Current implementation status
```

---

## Setup

### 1. Install dependencies

```bash
pip install torch==2.2.2 torchaudio==2.2.2
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA from https://pytorch.org/get-started/locally/

### 2. Verify checkpoints

```bash
python scripts/download_checkpoints.py
```

> **Note:** Pretrained checkpoints for U-Net, Beat-Net, Chord-Net, Pitch-Net, and Segment-Net are **not included** in this repository. The pipeline will run with random weights for architecture demonstration. See CHORDIFY_STATUS.md for details.

### 3. Run tests

```bash
python scripts/test_pipeline.py
```

All 14 structural tests should pass without checkpoints.

### 4. Run analysis (CLI)

```bash
python scripts/run_chordify.py --audio path/to/song.wav --print-tab
```

### 5. Run web UI

```bash
python app/app.py
# then open http://localhost:7860
```

### 6. Run EDA

```bash
python notebooks/eda_musdb18.py
# plots saved to outputs/eda/
```

---

## Output Format

```json
{
  "metadata": {
    "duration": 215.3,
    "sample_rate": 44100,
    "bpm": 127.5,
    "key": "G",
    "key_score": 8
  },
  "beats":     [{"time": 0.023, "type": "beat"}, ...],
  "downbeats": [{"time": 0.023}, ...],
  "chords":    [{"start": 0.0, "end": 2.4, "label": "G:maj"}, ...],
  "sections":  [{"start": 0.0, "end": 16.2, "label": "intro"}, ...],
  "pitch":     [{"time": 0.023, "midi_note": 67, "cents_offset": 20}, ...],
  "lyrics":    [{"start": 12.4, "end": 13.1, "text": "hello"}, ...],
  "tabs": {
    "tab_list":  [...],
    "ascii_tab": "╔══... e|--0--... B|--1--..."
  },
  "timeline":  [...]
}
```

---

## Checkpoint Status

| Component | Architecture | Checkpoint | Status |
|---|---|---|---|
| U-Net | UNets (4 sources) | `checkpoints/unet.pth` | ❌ Missing |
| Beat-Net | BeatNet (3-branch transformer) | `checkpoints/beat_net.pth` | ❌ Missing |
| Chord-Net | ChordNet (122-class) | `checkpoints/chord_net.pth` | ❌ Missing |
| Pitch-Net | PitchNet (850-class) | `checkpoints/pitch_net.pth` | ❌ Missing |
| Segment-Net | SegmentNet (10-class) | `checkpoints/segment_net.pth` | ❌ Missing |
| Whisper | openai-whisper (base) | auto-download | ✅ Available |
| Wav2Vec2 | wav2vec2-base-960h | auto-download | ✅ Available |

Pipeline runs with random weights (architecture demo). Meaningful outputs require trained checkpoints.

---

## Attribution and License

Chordify is an **academic adaptation** of the open-source [Lamucal](https://github.com/Lamucal/Lamucal) project.

- The neural network architectures (U-Net, BaseTransformer, BeatNet, ChordNet, PitchNet, SegmentNet), STFT utilities, chord label vocabulary, key detection algorithm, and pipeline orchestration logic are **adapted from Lamucal**. Original authors retain copyright.
- The MUSDB18 dataset is used under its original terms (DSD100 Restricted + MedleyDB CC BY-NC-SA 4.0). See `data/README.md` and the [MUSDB18 paper](https://zenodo.org/record/1117372).
- OpenAI Whisper is used under the MIT License.
- HuggingFace Wav2Vec2 models are used under their respective model licences.

**Chordify-specific contributions:**
- Complete `get_lyrics()` implementation (Whisper + Wav2Vec2) replacing the original stub
- Multimodal fusion module (`pipeline/fusion.py`)
- AI guitar tab generation with chord voicing dictionary (`pipeline/tabs.py`)
- Timestamp utilities (`utils/timestamps.py`)
- Modular pipeline architecture with config-driven paths
- Gradio web UI (`app/app.py`)
- EDA notebook (`notebooks/eda_musdb18.py`)
- Progressive test suite (`scripts/test_pipeline.py`)
- Checkpoint status checker (`scripts/download_checkpoints.py`)

---

## Known Limitations

1. **No pretrained checkpoints** — model outputs are random without them.
2. **Lyrics** requires `openai-whisper` and `transformers` (pip install).
3. **Stem decoding** from `.stem.mp4` requires `stempeg` (pip install).
4. **Training code** is not included in this version — inference only.
5. **Tab voicing dictionary** covers open-position and common barre voicings; extended chords fall back to simplified voicings.
6. **Wav2Vec2 / PitchNet conditioning** — the lyrics_matrix shape from Wav2Vec2 may not exactly match PitchNet's d_model dimension without a learned projection layer (planned future work).

---

*Chordify v1.0.0 — Academic project, MAI Semester 7*
