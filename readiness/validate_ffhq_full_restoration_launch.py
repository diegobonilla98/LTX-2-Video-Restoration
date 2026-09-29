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
MANIFEST_VALIDATION_PATH = PROJECT_ROOT / "data/manifests/ffhq_full_restoration/validation.json"
CACHE_VALIDATION_PATH = PRECOMPUTED_ROOT / "ffhq_full_restoration/validation.json"
SMOKE_PATH = OUTPUT_ROOT / "readiness/ffhq_full_restoration/summary.json"
SOURCE_CHECKPOINT = OUTPUT_ROOT / "ffhq_pixel_aggressive/checkpoints/lora_weights_step_05000.safetensors"
CONFIG_PATH = PROJECT_ROOT / "configs/ffhq_full_restoration.yaml"
OUTPUT_PATH = OUTPUT_ROOT / "readiness/ffhq_full_restoration_launch.json"
EXPECTED_TRAIN_SAMPLES = 12288


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configuration_checks(config: object) -> dict[str, bool]:
    conditions = config.training_strategy.video.conditions
    return {
        "distilled_base": "distilled" in Path(config.model.model_path).name,
        "source_checkpoint": Path(config.model.load_checkpoint) == SOURCE_CHECKPOINT,
        "rank_64": config.lora.rank == 64 and config.lora.alpha == 64,
        "flow_matching": config.flow_matching.timestep_sampling_mode == "shifted_logit_normal",
        "first_frame_only": len(conditions) == 1
        and conditions[0].type == "first_frame"
        and conditions[0].probability == 1.0,
        "optimization": config.optimization.steps == 5000
        and config.optimization.batch_size == 1
        and config.optimization.learning_rate == 0.000015,
        "video_dims": tuple(config.validation.video_dims) == (512, 512, 49),
        "validation": config.validation.inference_steps == 8
        and config.validation.interval == 500
        and len(config.validation.samples) == 5,
        "checkpoints": config.checkpoints.interval == 500 and config.checkpoints.keep_last_n == 10,
    }


def main() -> None:
    environment = read(ENVIRONMENT_PATH)
    manifest = read(MANIFEST_VALIDATION_PATH)
    cache = read(CACHE_VALIDATION_PATH)
    smoke = read(SMOKE_PATH)
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
        "source_checkpoint": SOURCE_CHECKPOINT.is_file(),
        "manifest": bool(manifest["passed"]),
        "cache": bool(cache["passed"]),
        "precomputed_count": cache["precomputed"].get("latents") == EXPECTED_TRAIN_SAMPLES
        and cache["precomputed"].get("conditions") == EXPECTED_TRAIN_SAMPLES,
        "real_optimizer_smoke": bool(smoke["passed"]) and smoke.get("protocol_version") == 2,
        "all_adaptive_shapes_smoked": bool(smoke["gates"]["adaptive_latent_shapes"]),
        "lora_updated": bool(smoke["gates"]["nonzero_lora_update"]),
        "finite_losses": bool(smoke["gates"]["finite_positive_losses"]),
        "sample_video": bool(smoke["gates"]["validation_video"]),
        "nonstatic_validation": bool(smoke["gates"]["nonstatic_validation"]),
        "gpu_idle": gpu_query.returncode == 0 and not gpu_query.stdout.strip(),
        "memory_headroom": memory.available / 2**30 >= 20.0,
        "disk_headroom": disk.free / 2**30 >= 100.0,
        "configuration": all(config_checks.values()),
    }
    payload = {
        "status": "ready" if all(gates.values()) else "blocked",
        "gates": gates,
        "configuration_checks": config_checks,
        "available_memory_gib": memory.available / 2**30,
        "free_disk_gib": disk.free / 2**30,
        "training_steps": config.optimization.steps,
        "validation_interval": config.validation.interval,
        "long_run_launched": False,
    }
    atomic_json(OUTPUT_PATH, payload)
    print(json.dumps(payload, indent=2))
    if not all(gates.values()):
        raise RuntimeError("FFHQ full restoration launch gate failed")


if __name__ == "__main__":
    main()
