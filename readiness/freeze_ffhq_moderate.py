import datetime
import json
import os
from pathlib import Path

from project_config import OUTPUT_ROOT, PROJECT_ROOT
from restoration.io import atomic_json, atomic_text, sha256_file

SOURCE_ROOT = OUTPUT_ROOT / "ffhq_pixel_curriculum"
RELEASE_ROOT = OUTPUT_ROOT / "releases/ffhq_moderate_v1"
ADAPTER_INIT_ROOT = RELEASE_ROOT / "adapter_init"
CHECKPOINT_NAME = "lora_weights_step_06000.safetensors"
TRAINING_STATE_NAME = "training_state_step_06000.pt"
ADAPTER_INIT_NAME = "ffhq_moderate_v1.safetensors"


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and os.path.samefile(source, destination):
        return
    temporary = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    temporary.replace(destination)


def main() -> None:
    state_path = SOURCE_ROOT / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "complete" or state.get("current_step") != 6000:
        raise RuntimeError("The moderate FFHQ run is not a completed 6000-step run")
    checkpoint = SOURCE_ROOT / "checkpoints" / CHECKPOINT_NAME
    training_state = SOURCE_ROOT / "checkpoints" / TRAINING_STATE_NAME
    if not checkpoint.is_file() or not training_state.is_file():
        raise FileNotFoundError("The moderate final checkpoint or training state is missing")
    RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
    released_checkpoint = RELEASE_ROOT / CHECKPOINT_NAME
    released_training_state = RELEASE_ROOT / TRAINING_STATE_NAME
    adapter_init = ADAPTER_INIT_ROOT / ADAPTER_INIT_NAME
    hardlink(checkpoint, released_checkpoint)
    hardlink(training_state, released_training_state)
    hardlink(checkpoint, adapter_init)
    config_source = PROJECT_ROOT / "configs/ffhq_pixel_curriculum.yaml"
    atomic_text(RELEASE_ROOT / config_source.name, config_source.read_text(encoding="utf-8"))
    atomic_text(RELEASE_ROOT / state_path.name, state_path.read_text(encoding="utf-8"))
    manifest = {
        "release": "ffhq_moderate_v1",
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "source_output": str(SOURCE_ROOT),
        "completed_steps": 6000,
        "checkpoint": str(released_checkpoint),
        "checkpoint_sha256": sha256_file(released_checkpoint),
        "adapter_init": str(adapter_init),
        "adapter_init_sha256": sha256_file(adapter_init),
        "training_state": str(released_training_state),
        "training_state_sha256": sha256_file(released_training_state),
        "config": str(RELEASE_ROOT / config_source.name),
        "config_sha256": sha256_file(RELEASE_ROOT / config_source.name),
    }
    atomic_json(RELEASE_ROOT / "release.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
