"""
Chordify lyrics / ASR pipeline stage.

Uses faster-whisper (CTranslate2 backend) with whisper-large-v3 support,
word-level timestamp tracking, and singing-vocal optimization.

Features:
- Primary model: whisper-large-v3 (with graceful fallback to medium/small/base on CPU memory limits)
- Vocal stem routing: uses neural vocal stem if U-Net checkpoint exists;
  uses Center-Channel Vocal Extraction (CCE) if U-Net is uncalibrated,
  preventing random-weight noise from destroying sung vocals
- Singing-optimized inference: conservative VAD / no-speech threshold,
  condition_on_previous_text=False to prevent hallucination loops
- Word-level timestamps: preserves exact per-word start/end timing throughout pipeline
"""
from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import torch

from models.load import has_valid_checkpoint

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
_fw_model_cache: dict = {}
_wav2vec2_cache: dict = {}
_active_whisper_name: str = "whisper-large-v3"


def get_active_whisper_model_name() -> str:
    global _active_whisper_name
    return _active_whisper_name


def _load_faster_whisper(model_size: str, device: str = "cpu"):
    global _active_whisper_name
    key = f"{model_size}_{device}"
    if key in _fw_model_cache:
        return _fw_model_cache[key]

    from faster_whisper import WhisperModel
    compute_type = "float16" if device == "cuda" else "int8"

    # Attempt primary model (e.g. large-v3), fallback if CPU/memory/network fails
    models_to_try = [model_size]
    if model_size == "large-v3":
        models_to_try.extend(["medium", "small", "base"])
    elif model_size not in ["base", "tiny"]:
        models_to_try.append("base")

    last_err = None
    for m_name in models_to_try:
        try:
            print(f"[Chordify] Attempting to load faster-whisper '{m_name}' on {device} ({compute_type}) …")
            model = WhisperModel(m_name, device=device, compute_type=compute_type)
            _fw_model_cache[key] = model
            _active_whisper_name = f"Whisper ({m_name})"
            print(f"[Chordify] Successfully loaded faster-whisper '{m_name}'.")
            return model
        except Exception as e:
            last_err = e
            print(f"[Chordify] Notice: Could not load '{m_name}' ({e}). Trying fallback …")

    raise RuntimeError(f"[Chordify] Failed to load any faster-whisper model. Last error: {last_err}")


def _load_wav2vec2(model_name: str):
    if model_name in _wav2vec2_cache:
        return _wav2vec2_cache[model_name]
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        print(f"[Chordify] Loading Wav2Vec2 '{model_name}' …")
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)
        model.eval()
        _wav2vec2_cache[model_name] = (processor, model)
        return processor, model
    except ImportError:
        raise ImportError(
            "[Chordify] transformers not installed.\n"
            "  Run:  pip install transformers"
        )


# ---------------------------------------------------------------------------
# Audio preparation helper
# ---------------------------------------------------------------------------
def _resample_to_16k(wav_mono: np.ndarray, src_sr: int) -> np.ndarray:
    target_sr = 16000
    if src_sr == target_sr:
        return wav_mono.astype(np.float32)
    try:
        import soxr
        return soxr.resample(wav_mono, src_sr, target_sr).astype(np.float32)
    except ImportError:
        pass
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(src_sr, target_sr)
        return resample_poly(wav_mono, target_sr // g, src_sr // g).astype(np.float32)
    except ImportError:
        warnings.warn("[Chordify] No resampler found (install soxr). ASR quality may suffer.")
        return wav_mono.astype(np.float32)


def extract_center_vocal_dsp(waveform: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Extract centered vocals using Mid-Side Center Channel Extraction (CCE).
    Removes panned stereo instruments while preserving centered lead vocals.
    """
    if isinstance(waveform, torch.Tensor):
        wav = waveform.detach().cpu().numpy()
    else:
        wav = np.asarray(waveform)

    if wav.ndim == 1 or wav.shape[0] < 2:
        return wav.squeeze()

    L = wav[0]
    R = wav[1]
    mid = (L + R) * 0.5
    side = (L - R) * 0.5
    # Center-focused signal
    center = mid - 0.5 * np.abs(side) * np.sign(mid)
    return center.astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def get_lyrics(
    waveform: torch.Tensor | np.ndarray,
    sample_rate: int,
    cfg: dict,
    device: str = "cpu",
    orig_waveform: torch.Tensor | np.ndarray | None = None,
    unet_has_checkpoint: bool = False,
) -> tuple[list[dict], torch.Tensor | None, list[dict]]:
    """
    Transcribe lyrics from vocal audio with word-level timestamps.

    Parameters
    ----------
    waveform            : vocal stem (from U-Net separation)
    sample_rate         : audio sample rate
    cfg                 : lyrics config dictionary
    device              : 'cpu' or 'cuda'
    orig_waveform       : original audio mixture (fallback when U-Net is untrained)
    unet_has_checkpoint : whether U-Net has genuine trained weights

    Returns
    -------
    lyrics_phrases : list of {"start", "end", "text", "words"} phrase-level dicts
    lyrics_matrix  : (1, T, hidden_size) tensor for PitchNet conditioning, or None
    all_words      : flat list of {"start", "end", "word", "probability"}
    """
    if not cfg.get("enabled", True):
        print("[Chordify] Lyrics disabled in config.")
        return [], None, []

    # ── Select audio source: genuine vocal stem vs center-channel DSP extraction
    if unet_has_checkpoint:
        print("[Chordify] Lyrics input: Neural isolated vocal stem.")
        if isinstance(waveform, torch.Tensor):
            wav_np = waveform.detach().cpu().numpy()
        else:
            wav_np = np.asarray(waveform)
        wav_mono = wav_np.mean(axis=0) if wav_np.ndim == 2 else wav_np
    else:
        print("[Chordify] Lyrics input: Center-Channel Vocal Extraction (bypassing uncalibrated U-Net noise).")
        source_audio = orig_waveform if orig_waveform is not None else waveform
        wav_mono = extract_center_vocal_dsp(source_audio)

    wav_16k = _resample_to_16k(wav_mono, sample_rate)

    # ── faster-whisper transcription ──────────────────────────────────────
    lyrics_phrases: list[dict] = []
    all_words: list[dict] = []

    try:
        model_size = cfg.get("whisper_model", "large-v3")
        fw_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        fw_model = _load_faster_whisper(model_size, fw_device)

        language = cfg.get("language", None)
        vad_filter = cfg.get("vad_filter", False)

        segments, info = fw_model.transcribe(
            wav_16k,
            language=language,
            word_timestamps=True,
            beam_size=5,
            vad_filter=vad_filter,
            condition_on_previous_text=False,  # Essential for singing to avoid repeating loops
            initial_prompt="Song lyrics singing vocals music",
        )

        for seg in segments:
            # Skip hallucinations in long instrumental silences
            if getattr(seg, "no_speech_prob", 0.0) > 0.85:
                continue

            seg_text = (seg.text or "").strip()
            if not seg_text:
                continue

            seg_words = getattr(seg, "words", None)
            words_in_seg = []

            if seg_words:
                for w in seg_words:
                    w_text = (w.word if hasattr(w, "word") else str(w)).strip()
                    prob = float(getattr(w, "probability", 0.9))
                    # Filter low-probability noise artifacts
                    if w_text and prob > 0.10:
                        word_dict = {
                            "word": w_text,
                            "start": round(float(w.start), 3),
                            "end": round(float(w.end), 3),
                            "probability": round(prob, 2),
                        }
                        words_in_seg.append(word_dict)
                        all_words.append(word_dict)

            if words_in_seg:
                lyrics_phrases.append({
                    "start": words_in_seg[0]["start"],
                    "end": words_in_seg[-1]["end"],
                    "text": seg_text,
                    "words": words_in_seg,
                })
            elif seg_text:
                lyrics_phrases.append({
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "text": seg_text,
                    "words": [],
                })

        print(
            f"[Chordify] Whisper ({get_active_whisper_model_name()}) found {len(lyrics_phrases)} phrases, "
            f"{len(all_words)} words (lang={info.language}, prob={info.language_probability:.2f})"
        )

    except Exception as e:
        print(f"[Chordify] WARNING — Whisper failed: {type(e).__name__}: {e}")
        print("[Chordify] Continuing without lyrics.")

    # ── Optional Wav2Vec2 frame-level matrix ──────────────────────────────
    lyrics_matrix: torch.Tensor | None = None
    try:
        w2v_name = cfg.get("wav2vec2_model", "facebook/wav2vec2-base-960h")
        processor, w2v_model = _load_wav2vec2(w2v_name)
        inputs = processor(wav_16k, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = w2v_model(**inputs)
        lyrics_matrix = out.last_hidden_state
    except Exception:
        pass

    return lyrics_phrases, lyrics_matrix, all_words
