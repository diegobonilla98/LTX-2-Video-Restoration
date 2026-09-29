import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
from PIL import Image

from data_gen.precompute_ltx import load_clean, record_spec
from data_gen.trajectory import build_frames
from inference.restoration_pipeline import PersistentRestorationPipeline, atomic_save_png
from project_config import MANIFEST_ROOT, OUTPUT_ROOT
from restoration.io import atomic_json, read_jsonl, stable_int


CHECKPOINT = OUTPUT_ROOT / "coco_pixel_aggressive_49/checkpoints/lora_weights_step_08000.safetensors"
MANIFEST_PATH = MANIFEST_ROOT / "coco_pixel_aggressive_49/test.jsonl"
OUTPUT_DIR = OUTPUT_ROOT / "coco_pixel_aggressive_49/heldout_gallery"
FRAME_COUNT = 49
EXAMPLES_PER_TIER = 4
SELECTION_SEED = 20260902
TIERS = ("easy", "medium", "hard")
PROMPT = "Progressively depixelate the image while preserving object identity, geometry, texture, lighting, and scene composition."


def select_records() -> list[dict]:
    rows = list(read_jsonl(MANIFEST_PATH))
    by_identity = {}
    for row in rows:
        by_identity.setdefault(row["source_relpath"], {})[row["render"]["curriculum_tier"]] = row
    identities = sorted(by_identity, key=lambda value: stable_int(value, SELECTION_SEED))
    selected = []
    for tier_index, tier in enumerate(TIERS):
        for offset in range(EXAMPLES_PER_TIER):
            identity = identities[tier_index * EXAMPLES_PER_TIER + offset]
            selected.append(by_identity[identity][tier])
    return selected


def metrics(prediction: Image.Image, target: Image.Image) -> dict[str, float]:
    pred = torch.from_numpy(np.asarray(prediction, dtype=np.uint8).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    clean = torch.from_numpy(np.asarray(target, dtype=np.uint8).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    mse = torch.mean((pred - clean) ** 2)
    return {
        "mse": float(mse.item()),
        "psnr": float((-10.0 * torch.log10(mse.clamp_min(1e-12))).item()),
        "ms_ssim": float(piq.multi_scale_ssim(pred, clean, data_range=1.0).item()),
    }


def render_sheets(rows: list[dict]) -> None:
    for tier in TIERS:
        tier_rows = [row for row in rows if row["tier"] == tier]
        figure, axes = plt.subplots(len(tier_rows), 3, figsize=(12, 16), constrained_layout=True)
        for row_index, row in enumerate(tier_rows):
            images = [Image.open(row["degraded_path"]), Image.open(row["restored_path"]), Image.open(row["clean_path"])]
            titles = [f"Degraded · factor {row['pixel_factor']}", f"Restored · PSNR {row['restored']['psnr']:.2f}", "Clean target"]
            for column_index, (image, title) in enumerate(zip(images, titles)):
                axes[row_index, column_index].imshow(image)
                axes[row_index, column_index].set_title(title)
                axes[row_index, column_index].axis("off")
        figure.suptitle(f"COCO held-out {tier} examples · step 8000", fontsize=18)
        figure.savefig(OUTPUT_DIR / f"heldout_{tier}_gallery.png", dpi=160)
        plt.close(figure)


def main() -> None:
    selected = select_records()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, record in enumerate(selected):
        tier = record["render"]["curriculum_tier"]
        sample_dir = OUTPUT_DIR / f"{index + 1:02d}_{tier}_{record['sample_id']}"
        clean = load_clean(record)
        degraded = build_frames(clean, record_spec(record), record["trajectory"], FRAME_COUNT)[0]
        atomic_save_png(sample_dir / "clean.png", clean)
        atomic_save_png(sample_dir / "degraded.png", degraded)
        prepared.append((record, sample_dir))
    pipeline = PersistentRestorationPipeline(CHECKPOINT, frame_count=FRAME_COUNT)
    try:
        pipeline.prepare(PROMPT)
        for index, (record, sample_dir) in enumerate(prepared, start=1):
            restored_path = sample_dir / "restored.png"
            if not restored_path.is_file():
                seed = stable_int(record["sample_id"], SELECTION_SEED) % (2**31)
                result = pipeline.restore_final(sample_dir / "degraded.png", PROMPT, seed)
                atomic_save_png(restored_path, result.final_image)
                atomic_json(sample_dir / "metadata.json", result.metadata)
            print(f"Generated held-out example {index}/{len(prepared)}", flush=True)
    finally:
        pipeline.close()
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
    rows = []
    for record, sample_dir in prepared:
        clean = Image.open(sample_dir / "clean.png").convert("RGB")
        degraded = Image.open(sample_dir / "degraded.png").convert("RGB")
        restored = Image.open(sample_dir / "restored.png").convert("RGB")
        rows.append(
            {
                "sample_id": record["sample_id"],
                "source_relpath": record["source_relpath"],
                "tier": record["render"]["curriculum_tier"],
                "pixel_factor": record["degradation"]["pixel_factor0"],
                "degraded_path": str(sample_dir / "degraded.png"),
                "restored_path": str(sample_dir / "restored.png"),
                "clean_path": str(sample_dir / "clean.png"),
                "degraded": metrics(degraded, clean),
                "restored": metrics(restored, clean),
            }
        )
    render_sheets(rows)
    summary = {
        "checkpoint": str(CHECKPOINT),
        "selection_seed": SELECTION_SEED,
        "examples": len(rows),
        "identities": len({row["source_relpath"] for row in rows}),
        "tiers": {tier: sum(row["tier"] == tier for row in rows) for tier in TIERS},
        "rows": rows,
    }
    atomic_json(OUTPUT_DIR / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
