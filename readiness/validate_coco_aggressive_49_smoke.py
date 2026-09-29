import json
from pathlib import Path

from project_config import OUTPUT_ROOT
from restoration.io import atomic_json


SUMMARY_PATH = OUTPUT_ROOT / "readiness/coco_aggressive_49/summary.json"
OUTPUT_PATH = OUTPUT_ROOT / "readiness/coco_aggressive_49_smoke_validation.json"


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    checkpoint = Path(summary["checkpoint"])
    videos = [Path(path) for path in summary["sample_videos"]]
    gates = {
        "smoke_passed": bool(summary["passed"]),
        "pixel_frames": summary["pixel_frames"] == 49,
        "latent_shapes": all(shape == [128, 7, 16, 16] for shape in summary["latent_shapes"].values()),
        "checkpoint": checkpoint.is_file(),
        "sample_video": bool(videos) and all(path.is_file() for path in videos),
        "optimizer_steps": summary["curriculum_steps"] == 3,
        "memory_headroom": bool(summary["memory"]["headroom_passed"]),
        "process_swap": summary["memory"]["process_peak_swap_bytes"] == 0,
        "cgroup_swap": summary["memory"]["cgroup_swap_peak_bytes"] == 0,
    }
    payload = {"status": "ready" if all(gates.values()) else "blocked", "gates": gates}
    atomic_json(OUTPUT_PATH, payload)
    print(json.dumps(payload, indent=2))
    if not all(gates.values()):
        raise RuntimeError("COCO aggressive 49-frame smoke validation failed")


if __name__ == "__main__":
    main()
