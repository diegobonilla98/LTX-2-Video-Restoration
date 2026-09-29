from data_gen.build_ffhq_aggressive_dataset import CURRICULUM_ANCHORS, TIER_FACTORS, TOTAL_STEPS
from train import train_ffhq_aggressive


def test_aggressive_factors_extend_moderate_range() -> None:
    assert TIER_FACTORS["easy"] == (8, 12, 16)
    assert max(TIER_FACTORS["medium"]) == 24
    assert max(TIER_FACTORS["hard"]) == 32


def test_aggressive_curriculum_retains_bridge_examples() -> None:
    assert CURRICULUM_ANCHORS[0]["step"] == 1
    assert CURRICULUM_ANCHORS[-1]["step"] == TOTAL_STEPS
    assert CURRICULUM_ANCHORS[-1]["weights"]["easy"] == 0.15
    assert CURRICULUM_ANCHORS[-1]["weights"]["hard"] == 0.55
    assert all(abs(sum(anchor["weights"].values()) - 1.0) < 1e-9 for anchor in CURRICULUM_ANCHORS)


def test_aggressive_checkpoint_selection_uses_initialization_then_latest(tmp_path, monkeypatch) -> None:
    output = tmp_path / "output"
    initialization = tmp_path / "moderate.safetensors"
    monkeypatch.setattr(train_ffhq_aggressive, "OUTPUT_DIR", output)
    monkeypatch.setattr(train_ffhq_aggressive, "INITIAL_CHECKPOINT", initialization)
    assert train_ffhq_aggressive.select_checkpoint() == initialization
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    first = checkpoint_root / "lora_weights_step_00500.safetensors"
    latest = checkpoint_root / "lora_weights_step_01000.safetensors"
    first.touch()
    latest.touch()
    assert train_ffhq_aggressive.select_checkpoint() == latest
