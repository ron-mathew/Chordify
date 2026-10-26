"""
Chordify music structure segmentation pipeline stage.

SegmentNet accepts the full-mix spectrogram conditioned on ChordNet
logits and predicts per-frame structural section class (0–9).

Segment label space:
  0=start  1=end  2=intro  3=outro  4=verse
  5=chorus 6=solo 7=break  8=bridge 9=inst
"""
import torch

from utils.spectrogram import get_spec
from utils.labels import SEGMENT_LABELS
from models import get_model


def run_structure_analysis(
    waveform: torch.Tensor,
    chord_logits: torch.Tensor,
    config: dict,
    device: str = "cpu",
) -> list[str]:
    """
    Run SegmentNet on the original mix.

    Parameters
    ----------
    waveform      : (2, samples) full-mix stereo waveform
    chord_logits  : (batch, time, freq) conditioning from ChordNet
    config        : full Chordify config dict
    device        : compute device

    Returns
    -------
    segment_labels : list[str] — per-frame section label
    """
    spec_cfg    = config["spec"]
    segment_cfg = config["segment"]

    print("[Chordify] Running Segment-Net …")
    segment_net = get_model(
        segment_cfg["model_name"],
        segment_cfg["model"],
        model_path=segment_cfg.get("model_path") or None,
        is_train=False,
        device=device,
    )

    orig_spec = get_spec(waveform, spec_cfg)   # (2, time, freq)

    with torch.no_grad():
        n_fft = spec_cfg["n_fft"]
        seg_features = orig_spec[:, :, : n_fft // 2].unsqueeze(0)  # (1,2,T,F)
        seg_features = seg_features.to(device)
        seg_mag = torch.abs(seg_features)

        segment_pred = segment_net(seg_mag, chord_logits)

    seg_idx = segment_pred.squeeze(0).cpu().tolist()
    segment_labels = [SEGMENT_LABELS[int(i)] for i in seg_idx]

    print(f"[Chordify] Structure analysis complete ({len(segment_labels)} frames).")
    return segment_labels
