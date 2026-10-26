"""
Chordify — AI-Based Multimodal Music Understanding and
           Automatic Guitar Tab Generation

Main pipeline orchestrator.

Usage
-----
  from chordify import Chordify

  chordify = Chordify()
  result = chordify.analyze("path/to/song.wav")

Or via CLI:
  python scripts/run_chordify.py --audio path/to/song.wav

Based on the open-source Lamucal project.
Lamucal: https://github.com/Lamucal/Lamucal
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from pipeline.audio_loader   import load_audio
from pipeline.separation     import separate_sources
from pipeline.beat_analysis  import run_beat_analysis
from pipeline.chord_analysis import run_chord_analysis
from pipeline.structure_analysis import run_structure_analysis
from pipeline.pitch_analysis import run_pitch_analysis
from pipeline.lyrics         import get_lyrics
from pipeline.fusion         import fuse
from pipeline.tabs           import generate_tabs
from utils.audio             import write_audio


class Chordify:
    """
    Chordify multimodal music analysis pipeline.

    Parameters
    ----------
    config_path : path to config/config.yaml (auto-detected if None)
    device      : 'cuda', 'cpu', or None for auto-detect
    """

    def __init__(
        self,
        config_path: str | None = None,
        device: str | None = None,
    ):
        self.config  = _load_config(config_path)
        self.device  = _resolve_device(device)
        self._print_banner()

    # ------------------------------------------------------------------
    def analyze(
        self,
        audio_path: str,
        save_stems: bool = False,
        output_dir: str | None = None,
    ) -> dict:
        """
        Run the complete Chordify pipeline on an audio file.

        Parameters
        ----------
        audio_path : path to the input audio file
        save_stems : if True, write the 4 separated stems as .wav files
        output_dir : where to save results (defaults to outputs/)

        Returns
        -------
        Unified Chordify output dict  (see pipeline/fusion.py for schema)
        plus a "tabs" key containing ASCII tab and structured tab list.
        """
        t0 = time.time()

        audio_cfg = self.config["audio"]
        spec_cfg  = self.config["spec"]

        # ---------------------------------------------------------------- 1. Load audio
        waveform, sr, duration = load_audio(
            audio_path,
            sample_rate=audio_cfg["sample_rate"],
            n_channel=audio_cfg["n_channel"],
            device=self.device,
        )

        # ---------------------------------------------------------------- 2. Source separation
        inst_waveforms = separate_sources(waveform, sr, self.config, self.device)
        # inst_waveforms: (4, 2, samples) → vocal=0, bass=1, drums=2, other=3

        if save_stems:
            _save_stems(inst_waveforms, audio_path, output_dir, sr)

        # ---------------------------------------------------------------- 3. Lyrics / ASR
        # Check if genuine U-Net weights are available; otherwise use CCE to avoid random noise
        from models.load import has_valid_checkpoint
        from pipeline.lyrics import get_active_whisper_model_name

        unet_has_cp = has_valid_checkpoint(self.config["separate"].get("model_path"))
        lyrics_cfg = self.config["lyrics"]
        print("[Chordify] Running Lyrics / ASR …")
        lyrics_res = get_lyrics(
            inst_waveforms[0],   # vocal stem
            sr,
            lyrics_cfg,
            device=self.device,
            orig_waveform=waveform,
            unet_has_checkpoint=unet_has_cp,
        )
        if len(lyrics_res) == 3:
            lyrics, lyrics_matrix, all_words = lyrics_res
        else:
            lyrics, lyrics_matrix = lyrics_res[:2]
            all_words = []

        # ---------------------------------------------------------------- 4. Beat detection
        beat_res = run_beat_analysis(
            inst_waveforms, self.config, self.device, sr=sr
        )
        if len(beat_res) == 4:
            beat_preds, bpm, beat_logits, beat_method = beat_res
        else:
            beat_preds, bpm, beat_logits = beat_res[:3]
            beat_method = "neural"

        # ---------------------------------------------------------------- 5. Chord recognition
        chord_res = run_chord_analysis(
            inst_waveforms, beat_logits, self.config, self.device, sr=sr, beat_preds=beat_preds
        )
        chord_labels = chord_res[0]
        song_key = chord_res[1]
        key_score = chord_res[2]
        chord_logits = chord_res[3]
        chord_method = chord_res[4] if len(chord_res) > 4 else "neural"
        dsp_chord_events = chord_res[5] if len(chord_res) > 5 else None

        # ---------------------------------------------------------------- 6. Structure analysis
        segment_labels = run_structure_analysis(
            waveform, chord_logits, self.config, self.device
        )

        # ---------------------------------------------------------------- 7. Pitch / melody
        pitch_preds, _ = run_pitch_analysis(
            inst_waveforms[0],   # vocal stem
            lyrics_matrix,
            self.config,
            self.device,
        )

        # ---------------------------------------------------------------- 8. Multimodal fusion
        result = fuse(
            beat_preds=beat_preds,
            bpm=bpm,
            chord_labels=chord_labels,
            song_key=song_key,
            key_score=key_score,
            segment_labels=segment_labels,
            pitch_preds=pitch_preds,
            lyrics=lyrics,
            duration=duration,
            sample_rate=sr,
            hop_length=spec_cfg["hop_length"],
            metadata_extra={
                "chord_method": chord_method,
                "beat_method":  beat_method,
                "lyrics_model": get_active_whisper_model_name(),
            },
            override_chord_segments=dsp_chord_events,
            words=all_words,
        )

        # ---------------------------------------------------------------- 9. AI tab generation
        tabs = generate_tabs(result)
        result["tabs"] = tabs

        # ---------------------------------------------------------------- 10. Save output
        out_path = _save_result(result, audio_path, output_dir)

        elapsed = time.time() - t0
        print(f"\n[Chordify] ✓ Analysis complete in {elapsed:.1f}s")
        print(f"[Chordify]   BPM     : {result['metadata']['bpm']}")
        print(f"[Chordify]   Key     : {result['metadata']['key']}")
        print(f"[Chordify]   Results : {out_path}")

        return result

    # ------------------------------------------------------------------
    def separate(self, audio_path: str, output_dir: str | None = None) -> None:
        """
        Run only source separation and save the 4 stems.
        Useful for inspecting separation quality independently.
        """
        audio_cfg = self.config["audio"]
        waveform, sr, _ = load_audio(
            audio_path,
            sample_rate=audio_cfg["sample_rate"],
            n_channel=audio_cfg["n_channel"],
            device=self.device,
        )
        inst_waveforms = separate_sources(waveform, sr, self.config, self.device)
        _save_stems(inst_waveforms, audio_path, output_dir, sr)

    # ------------------------------------------------------------------
    def _print_banner(self) -> None:
        print("=" * 56)
        print("  C H O R D I F Y")
        print("  AI-Based Multimodal Music Understanding")
        print("  & Automatic Guitar Tab Generation")
        print("=" * 56)
        print(f"  Device : {self.device}")
        if self.device.startswith("cuda"):
            print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  SR     : {self.config['audio']['sample_rate']} Hz")
        print("=" * 56)


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------
def _load_config(config_path: str | None = None) -> dict:
    """
    Load YAML config, resolving checkpoint paths from paths.yaml.
    """
    import yaml

    if config_path is None:
        config_path = Path(__file__).parent / "config" / "config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load paths override
    paths_file = Path(__file__).parent / "config" / "paths.yaml"
    if paths_file.exists():
        with open(paths_file, "r") as f:
            paths = yaml.safe_load(f) or {}

        # Resolve project root
        project_root = paths.get("project_root", "") or str(Path(__file__).parent)
        checkpoints_dir = paths.get("checkpoints_dir", "") or str(
            Path(project_root) / "checkpoints"
        )

        # Inject checkpoint paths into config
        ckpt_map = paths.get("checkpoints", {})
        for stage in ("separate", "beat", "chord", "pitch", "segment"):
            model_key = {"separate": "unet"}.get(stage, stage)
            ckpt_file = ckpt_map.get(model_key, "")
            if ckpt_file:
                full_path = str(Path(checkpoints_dir) / ckpt_file)
                if Path(full_path).exists():
                    cfg[stage]["model_path"] = full_path
                else:
                    cfg[stage]["model_path"] = ""
            else:
                cfg[stage]["model_path"] = cfg.get(stage, {}).get("model_path", "")

    # Inject spec config into separate block for sliding-window separation
    if "spec" not in cfg["separate"]:
        cfg["separate"]["spec"] = dict(cfg["spec"])
        cfg["separate"]["spec"]["n_time"] = cfg["spec"].get("n_time", 256)

    return cfg


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        dev = "cuda"
        print(f"[Chordify] CUDA available — using GPU.")
    else:
        dev = "cpu"
        print(f"[Chordify] CUDA not available — using CPU.")
    return dev


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
_STEM_NAMES = ["vocal", "bass", "drums", "other"]


def _save_stems(inst_waveforms, audio_path, output_dir, sr):
    base = Path(audio_path).stem
    out  = Path(output_dir) if output_dir else Path(__file__).parent / "outputs" / base
    out.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(_STEM_NAMES):
        stem_path = out / f"{base}_{name}.wav"
        write_audio(stem_path, inst_waveforms[i].cpu(), sr)
        print(f"[Chordify] Saved stem: {stem_path}")


def _save_result(result: dict, audio_path: str, output_dir: str | None) -> Path:
    base = Path(audio_path).stem
    out  = Path(output_dir) if output_dir else Path(__file__).parent / "outputs" / base
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"{base}_chordify.json"

    # Serialise — remove non-serialisable items (tensors inside tabs)
    safe = _make_serialisable(result)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, ensure_ascii=False)

    # Save ASCII tab separately
    ascii_tab = result.get("tabs", {}).get("ascii_tab", "")
    if ascii_tab:
        tab_path = out / f"{base}_chordify.tab"
        tab_path.write_text(ascii_tab, encoding="utf-8")

    return json_path


def _make_serialisable(obj):
    """Recursively convert tensors/numpy arrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serialisable(v) for v in obj]
    if hasattr(obj, "tolist"):           # tensor / ndarray
        return obj.tolist()
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)
