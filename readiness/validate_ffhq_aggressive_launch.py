import json
import shutil
import subprocess
from pathlib import Path

import psutil
from telegram_training_helper import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

from project_config import OUTPUT_ROOT, PRECOMPUTED_ROOT, PROJECT_ROOT
from restoration.io import atomic_json, sha256_file
from train.ltx_setup import load_training_config

ENVIRONMENT_PATH = OUTPUT_ROOT / "readiness/environment.json"
DATA_VALIDATION_PATH = PRECOMPUTED_ROOT / "ffhq_pixel_aggressive/validation.json"
SMOKE_PATH = OUTPUT_ROOT / "readiness/ffhq_aggressive_v2/summary.json"
MODERATE_STATE_PATH = OUTPUT_ROOT / "ffhq_pixel_curriculum/run_state.json"
RELEASE_PATH = OUTPUT_ROOT / "releases/ffhq_moderate_v1/release.json"
CONFIG_PATH = PROJECT_ROOT / "configs/ffhq_pixel_aggressive.yaml"
OUTPUT_PATH = OUTPUT_ROOT / "readiness/ffhq_aggressive_launch.json"
EXPECTED_TRAIN_SAMPLES = 12288


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    environment = read(ENVIRONMENT_PATH)
    data_validation = read(DATA_VALIDATION_PATH)
    smoke = read(SMOKE_PATH)
    moderate_state = read(MODERATE_STATE_PATH)
    release = read(RELEASE_PATH)
    config = load_training_config(CONFIG_PATH)
    checkpoint = Path(release["checkpoint"])
    adapter_init = Path(release["adapter_init"])
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
        "moderate_complete": moderate_state.get("status") == "complete"
        and moderate_state.get("current_step") == 6000,
        "moderate_release": checkpoint.is_file()
        and sha256_file(checkpoint) == release["checkpoint_sha256"],
        "adapter_initialization": adapter_init.is_file()
        and sha256_file(adapter_init) == release["adapter_init_sha256"],
        "aggressive_data": bool(data_validation["passed"]),
        "precomputed_count": data_validation["precomputed"].get("latents") == EXPECTED_TRAIN_SAMPLES
        and data_validation["precomputed"].get("conditions") == EXPECTED_TRAIN_SAMPLES,
        "continuation_smoke": bool(smoke["passed"]),
        "sample_video": bool(smoke["sample_videos"]),
        "checkpoint_present": Path(smoke["checkpoint"]).is_file(),
        "gpu_idle": gpu_query.returncode == 0 and not gpu_query.stdout.strip(),
        "memory_headroom": memory.available / 2**30 >= 12.0,
        "disk_headroom": disk.free / 2**30 >= 50.0,
        "configuration": config.optimization.steps == 5000
        and config.validation.interval == 500
        and len(config.validation.samples) == 3
        and Path(config.model.load_checkpoint) == adapter_init,
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
        raise RuntimeError("Aggressive FFHQ launch gate failed")


if __name__ == "__main__":
    main()
