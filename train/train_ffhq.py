import json

from project_config import PROJECT_ROOT
from train.train_supervised import run_training

CONFIG_PATH = PROJECT_ROOT / "configs/ffhq_pixel_curriculum.yaml"
TELEGRAM_ENABLED = True
TELEGRAM_UPDATE_INTERVAL = 100
DISABLE_PROGRESS_BARS = True


def main() -> None:
    checkpoint, stats = run_training(
        config_path=CONFIG_PATH,
        telegram_enabled=TELEGRAM_ENABLED,
        telegram_update_interval=TELEGRAM_UPDATE_INTERVAL,
        disable_progress_bars=DISABLE_PROGRESS_BARS,
    )
    print(json.dumps({"checkpoint": str(checkpoint), "stats": str(stats)}, indent=2))


if __name__ == "__main__":
    main()
