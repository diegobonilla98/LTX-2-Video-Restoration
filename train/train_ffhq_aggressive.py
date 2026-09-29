import json
from pathlib import Path

from project_config import OUTPUT_ROOT, PROJECT_ROOT
from train.train_supervised import run_training

CONFIG_PATH = PROJECT_ROOT / "configs/ffhq_pixel_aggressive.yaml"
OUTPUT_DIR = OUTPUT_ROOT / "ffhq_pixel_aggressive"
INITIAL_CHECKPOINT = (
    OUTPUT_ROOT / "releases/ffhq_moderate_v1/adapter_init/ffhq_moderate_v1.safetensors"
)
TELEGRAM_ENABLED = True
TELEGRAM_UPDATE_INTERVAL = 100
DISABLE_PROGRESS_BARS = True


def select_checkpoint() -> Path:
    checkpoints = sorted((OUTPUT_DIR / "checkpoints").glob("lora_weights_step_*.safetensors"))
    return checkpoints[-1] if checkpoints else INITIAL_CHECKPOINT


def main() -> None:
    checkpoint, stats = run_training(
        config_path=CONFIG_PATH,
        telegram_enabled=TELEGRAM_ENABLED,
        load_checkpoint_override=select_checkpoint(),
        telegram_update_interval=TELEGRAM_UPDATE_INTERVAL,
        disable_progress_bars=DISABLE_PROGRESS_BARS,
    )
    print(json.dumps({"checkpoint": str(checkpoint), "stats": str(stats)}, indent=2))


if __name__ == "__main__":
    main()
