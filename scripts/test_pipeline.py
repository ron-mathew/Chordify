"""
Chordify progressive test suite.

Runs each pipeline stage in isolation so failures are easy to pinpoint.
No checkpoints are needed — smoke tests use random weights.

Run from project root:
  python scripts/test_pipeline.py
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import sys
import io
# Force UTF-8 on Windows console if supported, otherwise use ASCII
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "  –"

results = []


def test(name, fn):
    print(f"\n[TEST] {name}")
    try:
        fn()
        print(f"{PASS} PASSED")
        results.append((name, "PASS"))
    except Exception as e:
        print(f"{FAIL} FAILED: {e}")
        traceback.print_exc()
        results.append((name, f"FAIL: {e}"))


# ============================================================
# TEST 1 — Imports
# ============================================================
def t1_imports():
    import torch                         # noqa
    import torchaudio                    # noqa
    import librosa                       # noqa
    from models import BeatNet, ChordNet, PitchNet, SegmentNet, UNets  # noqa
    from utils.labels      import CHORD_LABELS, SEGMENT_LABELS         # noqa
    from utils.spectrogram import stft, istft                          # noqa
    from utils.audio       import read_audio, gen_audio                # noqa
    from pipeline.modulation import search_key                         # noqa


test("1 — Import all modules", t1_imports)


# ============================================================
# TEST 2 — Synthetic audio
# ============================================================
def t2_gen_audio():
    from utils.audio import gen_audio
    wav, sr = gen_audio(duration=3)
    assert wav.shape == (2, 44100 * 3), f"Unexpected shape {wav.shape}"
    assert sr == 44100


test("2 — Generate synthetic audio", t2_gen_audio)


# ============================================================
# TEST 3 — STFT / iSTFT round-trip
# ============================================================
def t3_stft():
    import torch
    from utils.spectrogram import stft, istft
    wav = torch.randn(2, 44100)
    cfg = {"n_fft": 4096, "hop_length": 1024, "pad": True}
    spec = stft(wav, **{k: v for k, v in cfg.items() if k != "pad"}, pad=cfg["pad"])
    assert spec.shape[0] == 2, "Channel dimension wrong"
    assert spec.shape[2] == 4096 // 2 + 1
    wav2 = istft(spec, n_fft=4096, hop_length=1024, pad=True)
    assert wav2.shape[0] == 2


test("3 — STFT / iSTFT round-trip", t3_stft)


# ============================================================
# TEST 4 — UNets forward pass
# ============================================================
def t4_unet():
    import torch
    from models import UNets
    net = UNets(sources=4, n_channel=2)
    net.eval()
    # U-Net operates on n_fft//2 = 2048 freq bins (magnitude spectrum half)
    x = torch.randn(1, 2, 64, 2048)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (1, 4, 2, 64, 2048), f"Got {y.shape}"
    mask_sum = y.sum(dim=1)
    assert (mask_sum - 1.0).abs().max() < 1e-5, "Masks do not sum to 1"


test("4 — UNets forward pass", t4_unet)


# ============================================================
# TEST 5 — BeatNet forward pass
# ============================================================
def t5_beatnet():
    import torch
    from models import BeatNet
    net = BeatNet()
    net.eval()
    x = torch.randn(1, 3, 2, 64, 2048)
    with torch.no_grad():
        beats, logits = net(x)
    assert beats.shape == (1, 64)
    assert logits.shape[0] == 1


test("5 — BeatNet forward pass", t5_beatnet)


# ============================================================
# TEST 6 — ChordNet forward pass
# ============================================================
def t6_chordnet():
    import torch
    from models import ChordNet
    net = ChordNet()
    net.eval()
    x = torch.randn(1, 2, 64, 2048)
    with torch.no_grad():
        chords, logits = net(x)
    assert chords.shape == (1, 64)
    assert chords.min() >= 0 and chords.max() < 122


test("6 — ChordNet forward pass", t6_chordnet)


# ============================================================
# TEST 7 — SegmentNet forward pass
# ============================================================
def t7_segmentnet():
    import torch
    from models import SegmentNet
    net = SegmentNet()
    net.eval()
    x = torch.randn(1, 2, 64, 2048)
    with torch.no_grad():
        segs = net(x)
    assert segs.shape == (1, 64)
    assert segs.min() >= 0 and segs.max() < 10


test("7 — SegmentNet forward pass", t7_segmentnet)


# ============================================================
# TEST 8 — PitchNet forward pass
# ============================================================
def t8_pitchnet():
    import torch
    from models import PitchNet
    net = PitchNet()
    net.eval()
    x = torch.randn(1, 2, 64, 2048)
    with torch.no_grad():
        pitch, logits = net(x, None)
    assert pitch.shape == (1, 64)
    assert pitch.min() >= 0 and pitch.max() < 850


test("8 — PitchNet forward pass", t8_pitchnet)


# ============================================================
# TEST 9 — Chord / key labels
# ============================================================
def t9_labels():
    from utils.labels import CHORD_LABELS, SEGMENT_LABELS
    assert len(CHORD_LABELS) == 122
    assert len(SEGMENT_LABELS) == 10
    assert "C:maj" in CHORD_LABELS
    assert "chorus" in SEGMENT_LABELS


test("9 — Label integrity", t9_labels)


# ============================================================
# TEST 10 — Key detection
# ============================================================
def t10_key():
    from pipeline.modulation import search_key
    chords = ["G", "Am", "C", "D", "G", "Em", "C", "D"]
    key, score = search_key(chords)
    assert isinstance(key, str) and len(key) > 0
    assert isinstance(score, int)
    print(f"   Key: {key}, score: {score}", end="")


test("10 — Key detection (diatonic scoring)", t10_key)


# ============================================================
# TEST 11 — Timestamp utilities
# ============================================================
def t11_timestamps():
    from utils.timestamps import (
        beat_frames_to_events,
        chord_frames_to_segments,
        pitch_frames_to_events,
    )
    beats = beat_frames_to_events([0, 1, 2, 1, 0, 1], 1024, 44100)
    assert any(e["type"] == "beat" for e in beats)
    assert any(e["type"] == "downbeat" for e in beats)

    chords = chord_frames_to_segments(["C:maj"] * 5 + ["A:min"] * 3, 1024, 44100)
    assert len(chords) == 2

    pitch = pitch_frames_to_events([0, 120, 240], 1024, 44100)
    assert pitch[0]["midi_note"] == 21


test("11 — Timestamp utilities", t11_timestamps)


# ============================================================
# TEST 12 — Fusion (synthetic data)
# ============================================================
def t12_fusion():
    from pipeline.fusion import fuse
    result = fuse(
        beat_preds=[0, 1, 2, 1, 0],
        bpm=120.0,
        chord_labels=["C:maj"] * 5,
        song_key="C",
        key_score=7,
        segment_labels=["intro"] * 5,
        pitch_preds=[100] * 5,
        lyrics=[{"start": 0.0, "end": 1.0, "text": "hello"}],
        duration=5.0,
        sample_rate=44100,
        hop_length=1024,
    )
    assert "metadata" in result
    assert "beats" in result
    assert "chords" in result
    assert "sections" in result
    assert "timeline" in result


test("12 — Multimodal fusion (synthetic)", t12_fusion)


# ============================================================
# TEST 13 — Tab generation (synthetic)
# ============================================================
def t13_tabs():
    from pipeline.fusion import fuse
    from pipeline.tabs import generate_tabs

    result = fuse(
        beat_preds=[0, 2, 1, 1, 2, 1],
        bpm=130.0,
        chord_labels=["G:maj", "G:maj", "E:min", "E:min", "C:maj", "D:maj"],
        song_key="G",
        key_score=6,
        segment_labels=["verse"] * 6,
        pitch_preds=[200] * 6,
        lyrics=[],
        duration=6.0,
        sample_rate=44100,
        hop_length=1024,
    )
    tabs = generate_tabs(result)
    assert "ascii_tab" in tabs
    assert "tab_list" in tabs
    assert len(tabs["tab_list"]) > 0
    assert "e|" in tabs["ascii_tab"]


test("13 — AI tab generation (synthetic)", t13_tabs)


# ============================================================
# TEST 14 — Config loading
# ============================================================
def t14_config():
    from chordify import _load_config
    cfg = _load_config()
    assert cfg["audio"]["sample_rate"] == 44100
    assert cfg["audio"]["n_channel"] == 2
    assert cfg["audio"]["sources"] == 4


test("14 — Config loading", t14_config)


# ============================================================
# TEST 15 — DSP Chord Fallback
# ============================================================
def t15_dsp_chords():
    import numpy as np
    from pipeline.dsp_fallback import estimate_chords_dsp

    sr = 44100
    dur = 4.0
    t = np.linspace(0, dur, int(sr * dur))
    # Synthesize A-minor triad (A=440, C=523.25, E=659.25)
    harm = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 523.25 * t) + 0.3 * np.sin(2 * np.pi * 659.25 * t)
    stereo = np.stack([harm, harm], axis=0)

    frame_labels, key, key_score, events = estimate_chords_dsp(stereo, sr=sr)
    assert len(frame_labels) > 0, "Expected non-empty frame labels"
    assert len(events) > 0, "Expected non-empty chord events"
    assert events[0]["start"] == 0.0, "Start timestamp must be 0.0"
    for ev in events:
        assert ev["end"] > ev["start"], f"Invalid segment interval: {ev}"
        assert ev["source"] == "dsp_chroma"


test("15 — DSP chord fallback (chroma)", t15_dsp_chords)


# ============================================================
# TEST 16 — DSP Beat Fallback
# ============================================================
def t16_dsp_beats():
    import numpy as np
    from pipeline.dsp_fallback import estimate_beats_dsp

    sr = 44100
    dur = 4.0
    t = np.linspace(0, dur, int(sr * dur))
    # Pulse train at 2 Hz (120 BPM)
    audio = np.zeros_like(t)
    step = int(sr * 0.5)
    for idx in range(0, len(t), step):
        audio[idx:min(idx + 500, len(t))] = 0.8

    bpm, beat_classes, beat_events, beat_frames = estimate_beats_dsp(audio, sr=sr)
    assert 50 <= bpm <= 220, f"Unexpected BPM: {bpm}"
    assert len(beat_classes) > 0, "Expected non-empty beat classes"
    if beat_events:
        for i in range(1, len(beat_events)):
            assert beat_events[i]["time"] >= beat_events[i-1]["time"], "Timestamps must increase"


test("16 — DSP beat fallback (librosa)", t16_dsp_beats)


# ============================================================
# TEST 17 — Automatic Fallback Selection (Chord)
# ============================================================
def t17_auto_chord_selection():
    import torch
    from chordify import _load_config
    from pipeline.chord_analysis import run_chord_analysis

    cfg = _load_config()
    cfg["chord"]["model_path"] = "non_existent_chord_net.pth"
    dummy_audio = torch.randn(4, 2, 44100 * 2)

    labels, key, score, logits, method, chord_events = run_chord_analysis(
        dummy_audio, None, cfg, device="cpu", sr=44100
    )
    assert method == "dsp_chroma", f"Expected 'dsp_chroma', got {method}"
    assert len(labels) > 0


test("17 — Automatic fallback selection (ChordNet)", t17_auto_chord_selection)


# ============================================================
# TEST 18 — Automatic Fallback Selection (Beat)
# ============================================================
def t18_auto_beat_selection():
    import torch
    from chordify import _load_config
    from pipeline.beat_analysis import run_beat_analysis

    cfg = _load_config()
    cfg["beat"]["model_path"] = "non_existent_beat_net.pth"
    dummy_audio = torch.randn(4, 2, 44100 * 2)

    beat_preds, bpm, beat_logits, method = run_beat_analysis(
        dummy_audio, cfg, device="cpu", sr=44100
    )
    assert method == "dsp_librosa", f"Expected 'dsp_librosa', got {method}"
    assert len(beat_preds) > 0


test("18 — Automatic fallback selection (BeatNet)", t18_auto_beat_selection)


# ============================================================
# TEST 19 — End-to-End Pipeline with DSP Fallback
# ============================================================
def t19_e2e_fallback():
    from chordify import Chordify
    c = Chordify(device="cpu")
    res = c.analyze("outputs/test_audio.wav")

    assert "metadata" in res
    assert res["metadata"]["chord_method"] == "dsp_chroma"
    assert res["metadata"]["beat_method"] == "dsp_librosa"
    assert "chords" in res and len(res["chords"]) > 0
    assert "tabs" in res and "ascii_tab" in res["tabs"]
    assert "e|" in res["tabs"]["ascii_tab"]


test("19 — End-to-End fallback integration", t19_e2e_fallback)


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 56)
print("  CHORDIFY TEST SUMMARY")
print("=" * 56)
passed = sum(1 for _, s in results if s == "PASS")
failed = sum(1 for _, s in results if s.startswith("FAIL"))
for name, status in results:
    icon = "[OK]" if status == "PASS" else "[X]"
    print(f"  {icon}  {name}")
print("-" * 56)
print(f"  {passed} passed   {failed} failed   {len(results)} total")
print("=" * 56)

if failed:
    sys.exit(1)
