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
DATA_VALIDATION_PATH = PRECOMPUTED_ROOT / "ffhq_pixel_curriculum/validation.json"
SMOKE_PATH = OUTPUT_ROOT / "readiness/ffhq_curriculum_v2/summary.json"
CONFIG_PATH = PROJECT_ROOT / "configs/ffhq_pixel_curriculum.yaml"
OUTPUT_PATH = OUTPUT_ROOT / "readiness/ffhq_launch.json"
EXPECTED_TRAIN_SAMPLES = 12288


def read(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    environment = read(ENVIRONMENT_PATH)
    data_validation = read(DATA_VALIDATION_PATH)
    smoke = read(SMOKE_PATH)
    config = load_training_config(CONFIG_PATH)
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    disk = shutil.disk_usage(PROJECT_ROOT)
    memory = psutil.virtual_memory()
    gates = {
        "environment": bool(environment["passed"]),
        "telegram": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "ffhq_data": bool(data_validation["passed"]),
        "precomputed_count": data_validation["precomputed"].get("latents") == EXPECTED_TRAIN_SAMPLES
        and data_validation["precomputed"].get("conditions") == EXPECTED_TRAIN_SAMPLES,
        "curriculum_smoke": bool(smoke["passed"]),
        "sample_video": bool(smoke["sample_videos"]),
        "checkpoint_present": Path(smoke["checkpoint"]).is_file(),
        "gpu_idle": gpu_query.returncode == 0 and not gpu_query.stdout.strip(),
        "memory_headroom": memory.available / 2**30 >= 12.0,
        "disk_headroom": disk.free / 2**30 >= 50.0,
        "configuration": config.optimization.steps == 6000
        and config.validation.interval == 500
        and len(config.validation.samples) == 3,
    }
    payload = {
        "status": "ready" if all(gates.values()) else "blocked",
        "gates": gates,
        "available_memory_gib": memory.available / 2**30,
        "free_disk_gib": disk.free / 2**30,
        "training_steps": config.optimization.steps,
        "validation_interval": config.validation.interval,
        "validation_samples": len(config.validation.samples),
        "long_run_launched": False,
    }
    atomic_json(OUTPUT_PATH, payload)
    print(json.dumps(payload, indent=2))
    if not all(gates.values()):
        raise RuntimeError("FFHQ overnight launch gate failed")


if __name__ == "__main__":
    main()
