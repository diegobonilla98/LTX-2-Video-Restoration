import gc
import json
from pathlib import Path

import lpips
import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
from PIL import Image

from inference.restoration_pipeline import PersistentRestorationPipeline, atomic_save_png
from project_config import OUTPUT_ROOT
from restoration.io import atomic_json, stable_int


STEP_5000_CHECKPOINT = OUTPUT_ROOT / "ffhq_pixel_aggressive/checkpoints/lora_weights_step_05000.safetensors"
STEP_8000_CHECKPOINT = OUTPUT_ROOT / "coco_pixel_aggressive_49/checkpoints/lora_weights_step_08000.safetensors"
GALLERY_ROOT = OUTPUT_ROOT / "coco_pixel_aggressive_49/heldout_gallery"
OUTPUT_DIR = OUTPUT_ROOT / "coco_pixel_aggressive_49/lora_strength_sweep"
FRAME_COUNT = 49
EXAMPLES_PER_TIER = 2
TIERS = ("easy", "medium", "hard")
STRENGTHS = (0.25, 0.5, 0.75, 1.0)
SEED = 20260903
PROMPT = "Progressively depixelate the image while preserving object identity, geometry, texture, lighting, and scene composition."


def selected_rows() -> list[dict]:
    rows = json.loads((GALLERY_ROOT / "summary.json").read_text(encoding="utf-8"))["rows"]
    return [row for tier in TIERS for row in [item for item in rows if item["tier"] == tier][:EXAMPLES_PER_TIER]]


def generate_configuration(name: str, checkpoint: Path, strength: float, rows: list[dict]) -> None:
    pipeline = PersistentRestorationPipeline(checkpoint, frame_count=FRAME_COUNT, lora_strength=strength)
    try:
        pipeline.prepare(PROMPT)
        for index, row in enumerate(rows, start=1):
            destination = OUTPUT_DIR / row["sample_id"] / f"{name}.png"
            if not destination.is_file():
                seed = stable_int(row["sample_id"], SEED) % (2**31)
                result = pipeline.restore_final(Path(row["degraded_path"]), PROMPT, seed)
                atomic_save_png(destination, result.final_image)
            print(f"{name}: {index}/{len(rows)}", flush=True)
    finally:
        pipeline.close()
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


def tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0


def measure(prediction: Image.Image, target: Image.Image, perceptual: lpips.LPIPS, device: torch.device) -> dict[str, float]:
    pred = tensor(prediction, device)
    clean = tensor(target, device)
    mse = torch.mean((pred - clean) ** 2)
    return {
        "mse": float(mse.item()),
        "psnr": float((-10.0 * torch.log10(mse.clamp_min(1e-12))).item()),
        "ms_ssim": float(piq.multi_scale_ssim(pred, clean, data_range=1.0).item()),
        "lpips": float(perceptual(pred * 2.0 - 1.0, clean * 2.0 - 1.0).mean().item()),
    }


def evaluate(rows: list[dict], names: list[str]) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perceptual = lpips.LPIPS(net="alex").to(device).eval()
    results = []
    with torch.inference_mode():
        for row in rows:
            clean = Image.open(row["clean_path"]).convert("RGB")
            for name in names:
                prediction = Image.open(OUTPUT_DIR / row["sample_id"] / f"{name}.png").convert("RGB")
                results.append({"sample_id": row["sample_id"], "tier": row["tier"], "configuration": name, **measure(prediction, clean, perceptual, device)})
    return results


def plot(rows: list[dict], names: list[str]) -> None:
    for tier in TIERS:
        tier_rows = [row for row in rows if row["tier"] == tier]
        columns = ["degraded", *names, "clean"]
        figure, axes = plt.subplots(len(tier_rows), len(columns), figsize=(24, 8), constrained_layout=True)
        for row_index, row in enumerate(tier_rows):
            paths = [Path(row["degraded_path"])] + [OUTPUT_DIR / row["sample_id"] / f"{name}.png" for name in names] + [Path(row["clean_path"])]
            titles = ["Degraded", "Step 5000 · 1.0", *[f"Step 8000 · {strength:.2f}" for strength in STRENGTHS], "Clean"]
            for column_index, (path, title) in enumerate(zip(paths, titles)):
                axes[row_index, column_index].imshow(Image.open(path).convert("RGB"))
                axes[row_index, column_index].set_title(title)
                axes[row_index, column_index].axis("off")
        figure.suptitle(f"LoRA inference-strength sweep · {tier} · identical seeds", fontsize=18)
        figure.savefig(OUTPUT_DIR / f"strength_sweep_{tier}.png", dpi=160)
        plt.close(figure)


def main() -> None:
    rows = selected_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configurations = [("step5000_1.00", STEP_5000_CHECKPOINT, 1.0)] + [
        (f"step8000_{strength:.2f}", STEP_8000_CHECKPOINT, strength) for strength in STRENGTHS
    ]
    for name, checkpoint, strength in configurations:
        generate_configuration(name, checkpoint, strength, rows)
    names = [name for name, _, _ in configurations]
    results = evaluate(rows, names)
    means = {
        name: {
            metric: float(np.mean([row[metric] for row in results if row["configuration"] == name]))
            for metric in ("mse", "psnr", "ms_ssim", "lpips")
        }
        for name in names
    }
    plot(rows, names)
    payload = {
        "examples": len(rows),
        "identities": [row["sample_id"] for row in rows],
        "seed": SEED,
        "configurations": [{"name": name, "checkpoint": str(checkpoint), "strength": strength} for name, checkpoint, strength in configurations],
        "means": means,
        "best_lpips": min(names, key=lambda name: means[name]["lpips"]),
        "best_psnr": max(names, key=lambda name: means[name]["psnr"]),
        "rows": results,
    }
    atomic_json(OUTPUT_DIR / "summary.json", payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
