import json
from pathlib import Path

from restoration.io import atomic_json


def effective_lora_strength(requested_strength: float, rank: int, alpha: int) -> float:
    if requested_strength < 0:
        raise ValueError("LoRA strength must be non-negative")
    if rank <= 0 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    return requested_strength * alpha / rank


def save_lora_metadata(
    checkpoint_path: Path,
    rank: int,
    alpha: int,
    target_modules: list[str],
    base_model: str,
) -> Path:
    path = checkpoint_path.with_suffix(".metadata.json")
    atomic_json(
        path,
        {
            "checkpoint": str(checkpoint_path),
            "rank": rank,
            "alpha": alpha,
            "training_scale": alpha / rank,
            "target_modules": target_modules,
            "base_model": base_model,
        },
    )
    return path


def load_lora_metadata(checkpoint_path: Path) -> dict | None:
    path = checkpoint_path.with_suffix(".metadata.json")
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_lora_base(checkpoint_path: Path, base_model: Path) -> dict | None:
    metadata = load_lora_metadata(checkpoint_path)
    if metadata is None:
        return None
    trained_base = Path(metadata["base_model"]).resolve()
    requested_base = base_model.resolve()
    if trained_base != requested_base:
        raise RuntimeError(
            f"LoRA was trained on {trained_base}, but inference requested {requested_base}"
        )
    return metadata
