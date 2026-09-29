import json
import shutil
import subprocess
from pathlib import Path

import psutil
from telegram_training_helper import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

from project_config import OUTPUT_ROOT, PRECOMPUTED_ROOT, PROJECT_ROOT
from restoration.io import atomic_json
from train.ltx_setup import load_training_config


ENVIRONMENT_PATH = OUTPUT_ROOT / "readiness/environment.json"
DATA_VALIDATION_PATH = PRECOMPUTED_ROOT / "coco_pixel_aggressive_49/validation.json"
SMOKE_PATH = OUTPUT_ROOT / "readiness/coco_aggressive_49/summary.json"
SOURCE_STATE_PATH = OUTPUT_ROOT / "ffhq_pixel_aggressive/run_state.json"
SOURCE_CHECKPOINT = OUTPUT_ROOT / "ffhq_pixel_aggressive/checkpoints/lora_weights_step_05000.safetensors"
CONFIG_PATH = PROJECT_ROOT / "configs/coco_pixel_aggressive_49.yaml"
OUTPUT_PATH = OUTPUT_ROOT / "readiness/coco_aggressive_49_launch.json"
EXPECTED_TRAIN_SAMPLES = 24576


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configuration_checks(config: object) -> dict[str, bool]:
    return {
        "steps": config.optimization.steps == 8000,
        "video_dims": tuple(config.validation.video_dims) == (512, 512, 49),
        "validation_interval": config.validation.interval == 500,
        "validation_samples": len(config.validation.samples) == 3,
        "source_checkpoint": Path(config.model.load_checkpoint) == SOURCE_CHECKPOINT,
    }


def main() -> None:
    environment = read(ENVIRONMENT_PATH)
    data_validation = read(DATA_VALIDATION_PATH)
    smoke = read(SMOKE_PATH)
    source_state = read(SOURCE_STATE_PATH)
    config = load_training_config(CONFIG_PATH)
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    disk = shutil.disk_usage(PROJECT_ROOT)
    memory = psutil.virtual_memory()
    config_checks = configuration_checks(config)
    gates = {
        "environment": bool(environment["passed"]),
        "telegram": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "aggressive_source_complete": source_state.get("status") == "complete" and source_state.get("current_step") == 5000,
        "source_checkpoint": SOURCE_CHECKPOINT.is_file(),
        "coco_data": bool(data_validation["passed"]),
        "precomputed_count": data_validation["precomputed"].get("latents") == EXPECTED_TRAIN_SAMPLES
        and data_validation["precomputed"].get("conditions") == EXPECTED_TRAIN_SAMPLES,
        "frame_49_smoke": bool(smoke["passed"]) and smoke["pixel_frames"] == 49,
        "latent_shape": all(shape == [128, 7, 16, 16] for shape in smoke["latent_shapes"].values()),
        "sample_video": bool(smoke["sample_videos"]),
        "checkpoint_present": Path(smoke["checkpoint"]).is_file(),
        "gpu_idle": gpu_query.returncode == 0 and not gpu_query.stdout.strip(),
        "memory_headroom": memory.available / 2**30 >= 12.0,
        "disk_headroom": disk.free / 2**30 >= 100.0,
        "configuration": all(config_checks.values()),
    }
    payload = {
        "status": "ready" if all(gates.values()) else "blocked",
        "gates": gates,
        "available_memory_gib": memory.available / 2**30,
        "free_disk_gib": disk.free / 2**30,
        "training_steps": config.optimization.steps,
        "pixel_frames": config.validation.video_dims[2],
        "validation_interval": config.validation.interval,
        "validation_samples": len(config.validation.samples),
        "configuration_checks": config_checks,
        "resolved_video_dims": list(config.validation.video_dims),
        "long_run_launched": False,
    }
    atomic_json(OUTPUT_PATH, payload)
    print(json.dumps(payload, indent=2))
    if not all(gates.values()):
        raise RuntimeError("COCO aggressive 49-frame launch gate failed")


if __name__ == "__main__":
    main()
