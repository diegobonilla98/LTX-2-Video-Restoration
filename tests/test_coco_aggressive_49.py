from pathlib import Path

import yaml
import numpy as np
from PIL import Image

from data_gen.build_coco_aggressive_49_dataset import FRAME_COUNT, TIER_FACTORS, TRAIN_IDENTITIES
from data_gen.trajectory import build_frames, cosine_severity
from readiness.validate_coco_aggressive_49_launch import configuration_checks
from restoration.schema import DegradationSpec
from train.ltx_setup import load_training_config


CONFIG_PATH = Path("configs/coco_pixel_aggressive_49.yaml")


def test_49_frame_trajectory_endpoints_and_severity() -> None:
    grid_y, grid_x = np.indices((512, 512))
    clean = Image.fromarray(
        np.stack((grid_x % 256, grid_y % 256, (grid_x + grid_y) % 256), axis=-1).astype(np.uint8),
        mode="RGB",
    )
    spec = DegradationSpec("pixelation", None, 32, 17)
    frames = build_frames(clean, spec, "progressive", FRAME_COUNT)
    severities = [cosine_severity(index, FRAME_COUNT) for index in range(FRAME_COUNT)]
    assert len(frames) == 49
    assert severities[0] == 1.0
    assert severities[-1] == 0.0
    assert all(right <= left for left, right in zip(severities, severities[1:]))
    assert frames[-1].tobytes() == clean.tobytes()
    assert frames[0].tobytes() != clean.tobytes()


def test_coco_curriculum_is_balanced_and_aggressive() -> None:
    assert TRAIN_IDENTITIES % len(TIER_FACTORS) == 0
    assert min(TIER_FACTORS["easy"]) >= 8
    assert max(TIER_FACTORS["hard"]) == 32


def test_coco_training_config_uses_49_frames() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["validation"]["video_dims"] == [512, 512, 49]
    assert config["optimization"]["steps"] == 8000
    assert config["model"]["load_checkpoint"].endswith("lora_weights_step_05000.safetensors")


def test_loaded_coco_configuration_passes_launch_gate() -> None:
    config = load_training_config(CONFIG_PATH)
    assert isinstance(config.validation.video_dims, tuple)
    assert all(configuration_checks(config).values())
