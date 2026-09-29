import json
import os
from pathlib import Path

import yaml

from project_config import MANIFEST_ROOT, OUTPUT_ROOT, PRECOMPUTED_ROOT, PROJECT_ROOT
from restoration.io import atomic_json, atomic_text, read_jsonl
from train.train_supervised import run_training

SOURCE_ROOT = PRECOMPUTED_ROOT / "ffhq_pixel_aggressive/train/.precomputed"
SMOKE_ROOT = PRECOMPUTED_ROOT / "ffhq_pixel_aggressive_smoke/train"
OUTPUT_DIR = OUTPUT_ROOT / "readiness/ffhq_aggressive_v2"
CONFIG_PATH = PROJECT_ROOT / "readiness/ffhq_aggressive_smoke_v2.yaml"


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    temporary.replace(destination)


def prepare_smoke_data() -> None:
    records = list(read_jsonl(MANIFEST_ROOT / "ffhq_pixel_aggressive/train.jsonl"))[:3]
    tiers = {record["render"]["curriculum_tier"]: record["sample_id"] for record in records}
    if set(tiers) != {"easy", "medium", "hard"}:
        raise RuntimeError("The first aggressive FFHQ identity does not cover all tiers")
    for sample_id in tiers.values():
        for directory in ("latents", "conditions"):
            hardlink(
                SOURCE_ROOT / directory / f"{sample_id}.pt",
                SMOKE_ROOT / ".precomputed" / directory / f"{sample_id}.pt",
            )
    atomic_json(
        SMOKE_ROOT / "curriculum.json",
        {
            "seed": 20260831,
            "total_steps": 3,
            "anchors": [
                {"step": 1, "weights": {"easy": 0.7, "medium": 0.25, "hard": 0.05}},
                {"step": 2, "weights": {"easy": 0.4, "medium": 0.4, "hard": 0.2}},
                {"step": 3, "weights": {"easy": 0.15, "medium": 0.3, "hard": 0.55}},
            ],
            "tiers": {tier: [sample_id] for tier, sample_id in tiers.items()},
        },
    )


def prepare_config() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs/ffhq_pixel_aggressive.yaml").read_text(encoding="utf-8"))
    config["optimization"]["steps"] = 3
    config["data"]["preprocessed_data_root"] = str(SMOKE_ROOT / ".precomputed")
    config["validation"]["samples"] = config["validation"]["samples"][-1:]
    config["validation"]["interval"] = 3
    config["validation"]["skip_initial_validation"] = True
    config["checkpoints"]["interval"] = 3
    config["checkpoints"]["keep_last_n"] = 2
    config["checkpoints"]["no_resume"] = False
    config["output_dir"] = str(OUTPUT_DIR)
    atomic_text(CONFIG_PATH, yaml.safe_dump(config, sort_keys=False))


def main() -> None:
    prepare_smoke_data()
    prepare_config()
    checkpoint, stats = run_training(
        CONFIG_PATH,
        telegram_enabled=False,
        disable_progress_bars=True,
    )
    trace_path = OUTPUT_DIR / "curriculum_trace.jsonl"
    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    videos = sorted((OUTPUT_DIR / "samples").glob("step_000003_*.mp4"))
    state = json.loads((OUTPUT_DIR / "run_state.json").read_text(encoding="utf-8"))
    payload = {
        "checkpoint": str(checkpoint),
        "initialized_from": str(
            PROJECT_ROOT
            / "outputs/releases/ffhq_moderate_v1/adapter_init/ffhq_moderate_v1.safetensors"
        ),
        "stats": str(stats),
        "curriculum_steps": len(trace),
        "sample_videos": [str(path) for path in videos],
        "memory": state["memory"],
        "passed": (
            len(trace) == 3
            and len(videos) == 1
            and Path(checkpoint).is_file()
            and state["memory"]["headroom_passed"]
        ),
    }
    atomic_json(OUTPUT_DIR / "summary.json", payload)
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise RuntimeError("Aggressive FFHQ continuation smoke failed")


if __name__ == "__main__":
    main()
