from pathlib import Path

import yaml

from project_config import DISTILLED_SIGMAS, PROJECT_ROOT, VIDEO_LORA_TARGETS


CONFIG_NAMES = ("ffhq_pixel_curriculum", "ffhq_pixel_aggressive", "coco_pixel_aggressive_49")


def test_distilled_schedule() -> None:
    assert len(DISTILLED_SIGMAS) == 9
    assert DISTILLED_SIGMAS[0] == 1.0
    assert DISTILLED_SIGMAS[-1] == 0.0
    assert all(left > right for left, right in zip(DISTILLED_SIGMAS, DISTILLED_SIGMAS[1:]))


def test_lora_targets_are_video_specific() -> None:
    assert len(VIDEO_LORA_TARGETS) == 10
    assert all("audio" not in target for target in VIDEO_LORA_TARGETS)


def test_retained_training_configs_exist_and_parse() -> None:
    for name in CONFIG_NAMES:
        path = PROJECT_ROOT / "configs" / f"{name}.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert Path(config["model"]["model_path"]).is_file()
        assert config["training_strategy"]["video"]["is_generated"] is True
        assert config["optimization"]["batch_size"] == 1
