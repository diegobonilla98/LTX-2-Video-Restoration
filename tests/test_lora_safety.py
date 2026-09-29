from pathlib import Path

import pytest

from restoration.lora import effective_lora_strength, save_lora_metadata, validate_lora_base


def test_effective_lora_strength_preserves_training_scale() -> None:
    assert effective_lora_strength(1.0, 64, 64) == 1.0
    assert effective_lora_strength(1.0, 64, 32) == 0.5
    assert effective_lora_strength(0.75, 64, 32) == 0.375


def test_effective_lora_strength_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        effective_lora_strength(-1.0, 64, 64)
    with pytest.raises(ValueError):
        effective_lora_strength(1.0, 0, 64)


def test_lora_metadata_records_scaling(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapter.safetensors"
    path = save_lora_metadata(checkpoint, 64, 32, ["attn1.to_q"], "base.safetensors")
    payload = path.read_text(encoding="utf-8")
    assert '"training_scale": 0.5' in payload
    assert '"attn1.to_q"' in payload


def test_lora_base_metadata_matches(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapter.safetensors"
    base = tmp_path / "base.safetensors"
    save_lora_metadata(checkpoint, 64, 32, ["attn1.to_q"], str(base))
    metadata = validate_lora_base(checkpoint, base)
    assert metadata is not None
    assert metadata["base_model"] == str(base)


def test_lora_base_metadata_rejects_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapter.safetensors"
    base = tmp_path / "base.safetensors"
    save_lora_metadata(checkpoint, 64, 32, ["attn1.to_q"], str(base))
    with pytest.raises(RuntimeError, match="trained on"):
        validate_lora_base(checkpoint, tmp_path / "wrong.safetensors")
