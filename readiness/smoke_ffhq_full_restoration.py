import json
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from safetensors import safe_open

from project_config import MANIFEST_ROOT, OUTPUT_ROOT, PRECOMPUTED_ROOT, PROJECT_ROOT
from restoration.io import atomic_json, atomic_text, read_jsonl
from train.train_supervised import run_training

SOURCE_ROOT = PRECOMPUTED_ROOT / "ffhq_full_restoration/train/.precomputed"
SMOKE_ROOT = PRECOMPUTED_ROOT / "ffhq_full_restoration_smoke/train"
OUTPUT_DIR = OUTPUT_ROOT / "readiness/ffhq_full_restoration"
CONFIG_PATH = PROJECT_ROOT / "readiness/ffhq_full_restoration_smoke.yaml"
SOURCE_CHECKPOINT = OUTPUT_ROOT / "ffhq_pixel_aggressive/checkpoints/lora_weights_step_05000.safetensors"
SELECTED_INDICES = (0, 1, 2, 3, 19)


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    temporary.replace(destination)


def selected_records() -> list[dict]:
    records = list(read_jsonl(MANIFEST_ROOT / "ffhq_full_restoration/train.jsonl"))
    return [records[index] for index in SELECTED_INDICES]


def prepare_smoke_data(records: list[dict]) -> None:
    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)
    for record in records:
        sample_id = record["sample_id"]
        for directory in ("latents", "conditions"):
            hardlink(
                SOURCE_ROOT / directory / f"{sample_id}.pt",
                SMOKE_ROOT / ".precomputed" / directory / f"{sample_id}.pt",
            )
    case_names = [f"case_{index}" for index in range(len(records))]
    anchors = []
    for step, active in enumerate(case_names, start=1):
        anchors.append(
            {
                "step": step,
                "weights": {name: float(name == active) for name in case_names},
            }
        )
    atomic_json(
        SMOKE_ROOT / "curriculum.json",
        {
            "seed": 20260903,
            "total_steps": len(records),
            "anchors": anchors,
            "tiers": {name: [record["sample_id"]] for name, record in zip(case_names, records)},
        },
    )


def prepare_config(step_count: int) -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs/ffhq_full_restoration.yaml").read_text(encoding="utf-8"))
    config["optimization"]["steps"] = step_count
    config["data"]["preprocessed_data_root"] = str(SMOKE_ROOT / ".precomputed")
    config["data"]["num_dataloader_workers"] = 0
    config["validation"]["samples"] = config["validation"]["samples"][-1:]
    config["validation"]["interval"] = step_count
    config["validation"]["skip_initial_validation"] = True
    config["checkpoints"]["interval"] = step_count
    config["checkpoints"]["keep_last_n"] = 2
    config["checkpoints"]["no_resume"] = True
    config["output_dir"] = str(OUTPUT_DIR)
    atomic_text(CONFIG_PATH, yaml.safe_dump(config, sort_keys=False))


def checkpoint_delta(candidate: Path) -> dict:
    with safe_open(SOURCE_CHECKPOINT, framework="pt", device="cpu") as source_file:
        source_keys = set(source_file.keys())
        with safe_open(candidate, framework="pt", device="cpu") as candidate_file:
            candidate_keys = set(candidate_file.keys())
            common = sorted(source_keys & candidate_keys)
            maximum = 0.0
            changed = 0
            for key in common:
                source = source_file.get_tensor(key)
                trained = candidate_file.get_tensor(key)
                delta = float(torch.max(torch.abs(trained.float() - source.float())).item())
                maximum = max(maximum, delta)
                changed += int(delta > 0)
    return {
        "source_keys": len(source_keys),
        "candidate_keys": len(candidate_keys),
        "common_keys": len(common),
        "changed_tensors": changed,
        "maximum_absolute_delta": maximum,
    }


def video_motion(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        valid, frame = capture.read()
        if not valid:
            break
        frames.append(frame.astype(np.float32) / 255.0)
    capture.release()
    transition_mse = [float(np.mean((right - left) ** 2)) for left, right in zip(frames, frames[1:])]
    return {
        "frames": len(frames),
        "first_last_mse": float(np.mean((frames[-1] - frames[0]) ** 2)) if frames else 0.0,
        "mean_transition_mse": float(np.mean(transition_mse)) if transition_mse else 0.0,
        "nonidentical_transitions": sum(value > 1e-8 for value in transition_mse),
    }


def main() -> None:
    records = selected_records()
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    prepare_smoke_data(records)
    prepare_config(len(records))
    checkpoint, stats = run_training(
        CONFIG_PATH,
        telegram_enabled=False,
        disable_progress_bars=True,
    )
    checkpoint = Path(checkpoint)
    trace = [
        json.loads(line)
        for line in (OUTPUT_DIR / "curriculum_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    metrics = [
        json.loads(line)
        for line in (OUTPUT_DIR / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    videos = sorted((OUTPUT_DIR / "samples").glob("step_000005_*.mp4"))
    state = json.loads((OUTPUT_DIR / "run_state.json").read_text(encoding="utf-8"))
    shapes = {}
    for record in records:
        latent = torch.load(
            SOURCE_ROOT / "latents" / f"{record['sample_id']}.pt",
            map_location="cpu",
            weights_only=True,
        )
        shapes[record["sample_id"]] = list(latent["latents"].shape)
    delta = checkpoint_delta(checkpoint)
    motion = video_motion(videos[0]) if len(videos) == 1 else {
        "frames": 0,
        "first_last_mse": 0.0,
        "mean_transition_mse": 0.0,
        "nonidentical_transitions": 0,
    }
    expected_shapes = {
        record["sample_id"]: [128, record["render"]["latent_frame_count"], 16, 16]
        for record in records
    }
    losses = [row.get("train/loss") for row in metrics]
    payload = {
        "protocol_version": 2,
        "checkpoint": str(checkpoint),
        "initialized_from": str(SOURCE_CHECKPOINT),
        "stats": str(stats),
        "samples": [
            {
                "sample_id": record["sample_id"],
                "family": record["render"]["degradation_family"],
                "tier": record["render"]["curriculum_tier"],
                "frames": record["render"]["frame_count"],
            }
            for record in records
        ],
        "latent_shapes": shapes,
        "optimizer_steps": len(metrics),
        "curriculum_steps": len(trace),
        "losses": losses,
        "checkpoint_delta": delta,
        "sample_videos": [str(path) for path in videos],
        "video_motion": motion,
        "memory": state["memory"],
    }
    gates = {
        "optimizer_steps": len(metrics) == len(records),
        "curriculum_steps": len(trace) == len(records),
        "all_cases_sampled": {row["tier"] for row in trace} == {f"case_{index}" for index in range(len(records))},
        "finite_positive_losses": len(losses) == len(records)
        and all(value is not None and math.isfinite(value) and value > 0 for value in losses),
        "adaptive_latent_shapes": shapes == expected_shapes,
        "checkpoint": checkpoint.is_file(),
        "nonzero_lora_update": delta["common_keys"] > 0 and delta["changed_tensors"] > 0 and delta["maximum_absolute_delta"] > 0,
        "validation_video": len(videos) == 1 and videos[0].stat().st_size > 0,
        "nonstatic_validation": motion["frames"] == 49
        and motion["nonidentical_transitions"] >= 40
        and motion["first_last_mse"] > 1e-6,
        "memory_headroom": bool(state["memory"]["headroom_passed"]),
        "process_swap": state["memory"]["process_peak_swap_bytes"] == 0,
        "cgroup_swap": state["memory"]["cgroup_swap_peak_bytes"] in {None, 0},
    }
    payload["gates"] = gates
    payload["passed"] = all(gates.values())
    atomic_json(OUTPUT_DIR / "summary.json", payload)
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise RuntimeError("FFHQ full restoration real optimizer smoke failed")


if __name__ == "__main__":
    main()
