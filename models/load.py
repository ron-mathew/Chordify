"""
Chordify model factory.

Instantiates a model class by name, optionally loading a checkpoint.
Raises clear, user-facing errors when checkpoints are missing or corrupt
instead of silently using random weights.
"""
import os
import torch

from models.unet import UNets
from models.beat_net import BeatNet
from models.chord_net import ChordNet
from models.pitch_net import PitchNet
from models.segment_net import SegmentNet


_MODEL_REGISTRY = {
    "unet": UNets,
    "beat": BeatNet,
    "chord": ChordNet,
    "pitch": PitchNet,
    "segment": SegmentNet,
}


def get_model_cls(name: str):
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"[Chordify] Unknown model name '{name}'. "
            f"Valid options: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name]


def has_valid_checkpoint(model_path: str | None) -> bool:
    """
    Check if a checkpoint exists, is readable, and contains valid non-random weights.
    """
    if not model_path or not os.path.isfile(model_path):
        return False
    try:
        if os.path.getsize(model_path) < 1024:
            return False
        state = torch.load(model_path, map_location="cpu")
        if not isinstance(state, dict) or len(state) == 0:
            return False
        # Verify that state contains non-trivial tensors
        for k, v in state.items():
            if torch.is_tensor(v) and v.numel() > 10:
                var = float(torch.var(v.float()).item())
                if var > 1e-8:
                    return True
        return False
    except Exception:
        return False


def get_model(
    model,
    config: dict,
    model_path: str | None = None,
    is_train: bool = True,
    device: str = "cpu",
    require_checkpoint: bool = False,
):
    """
    Build a model from config and optionally load a checkpoint.

    Parameters
    ----------
    model          : model class or string name
    config         : kwargs dict for the model constructor
    model_path     : path to a .pth state-dict file (optional)
    is_train       : if True sets model.train(), else model.eval()
    device         : 'cpu' or 'cuda' etc.
    require_checkpoint : if True, raises an error when model_path is missing
    """
    if isinstance(model, str):
        model = get_model_cls(model)

    net = model(**config)

    if model_path:
        if not os.path.isfile(model_path):
            msg = (
                f"[Chordify] Checkpoint not found: '{model_path}'\n"
                f"  Model: {model.__name__}\n"
                f"  Please provide a valid checkpoint path in config/paths.yaml."
            )
            if require_checkpoint:
                raise FileNotFoundError(msg)
            else:
                print(f"[Chordify] WARNING — {msg}\n"
                      f"  Running with random weights (inference will be meaningless).")
        else:
            print(f"[Chordify] Loading checkpoint: {model_path}")
            state = torch.load(model_path, map_location=device)
            net.load_state_dict(state)
    elif require_checkpoint:
        raise FileNotFoundError(
            f"[Chordify] No checkpoint path provided for {model.__name__}.\n"
            f"  Set the path in config/paths.yaml."
        )
    else:
        print(
            f"[Chordify] WARNING — No checkpoint for {model.__name__}. "
            "Running with random weights."
        )

    net.to(device)
    if is_train:
        net.train()
    else:
        net.eval()
    return net
