import csv
import gc
import json
from pathlib import Path

import lpips
import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
from PIL import Image

from data_gen.precompute_ltx import load_clean, record_spec
from data_gen.restoration_trajectory import build_restoration_frames
from inference.restoration_pipeline import PersistentRestorationPipeline, atomic_save_png
from project_config import MANIFEST_ROOT, OUTPUT_ROOT
from restoration.io import atomic_json, read_jsonl, stable_int

RUN_ROOT = OUTPUT_ROOT / "ffhq_full_restoration"
PAPER_ROOT = RUN_ROOT / "paper_package_20260904"
SAMPLE_ROOT = PAPER_ROOT / "severity_samples"
FIGURE_ROOT = PAPER_ROOT / "figures"
TABLE_ROOT = PAPER_ROOT / "tables"
MANIFEST_PATH = MANIFEST_ROOT / "ffhq_full_restoration/test.jsonl"
CHECKPOINT = RUN_ROOT / "checkpoints/lora_weights_step_05000.safetensors"
FAMILIES = ("jpeg", "blur", "mosaic", "low_resolution", "mixed")
TIERS = ("easy", "medium", "hard")
FAMILY_LABELS = {"jpeg": "JPEG", "blur": "Gaussian blur", "mosaic": "Mosaic", "low_resolution": "Low resolution", "mixed": "Mixed"}
FRAME_COUNTS = {"easy": 17, "medium": 25, "hard": 33}
SELECTION_SEED = 20260904
GENERATION_SEED = 20260904
PROMPT = "Progressively restore the degraded face while preserving identity, facial geometry, expression, lighting, texture, and background."


def select_records() -> list[dict]:
    rows = list(read_jsonl(MANIFEST_PATH))
    index = {(row["source_relpath"], row["render"]["curriculum_tier"], row["render"]["degradation_family"]): row for row in rows}
    identities = sorted({row["source_relpath"] for row in rows}, key=lambda value: stable_int(value, SELECTION_SEED))
    selected = []
    for family_index, family in enumerate(FAMILIES):
        identity = identities[family_index]
        for tier in TIERS:
            selected.append(index[(identity, tier, family)])
    return selected


def prepare(records: list[dict]) -> list[tuple[dict, Path]]:
    prepared = []
    for record in records:
        tier = record["render"]["curriculum_tier"]
        family = record["render"]["degradation_family"]
        directory = SAMPLE_ROOT / family / tier
        clean = load_clean(record)
        degraded = build_restoration_frames(clean, record_spec(record), FRAME_COUNTS[tier])[0]
        atomic_save_png(directory / "clean.png", clean)
        atomic_save_png(directory / "degraded.png", degraded)
        prepared.append((record, directory))
    return prepared


def generate(prepared: list[tuple[dict, Path]]) -> None:
    completed = 0
    for tier in TIERS:
        pipeline = PersistentRestorationPipeline(CHECKPOINT, frame_count=FRAME_COUNTS[tier])
        try:
            pipeline.prepare(PROMPT)
            for record, directory in prepared:
                if record["render"]["curriculum_tier"] != tier:
                    continue
                seed = stable_int(record["sample_id"], GENERATION_SEED) % (2**31)
                result = pipeline.restore_final(directory / "degraded.png", PROMPT, seed)
                atomic_save_png(directory / "restored.png", result.final_image)
                metadata = dict(result.metadata)
                metadata.update({"sample_id": record["sample_id"], "source_relpath": record["source_relpath"], "degradation_family": record["render"]["degradation_family"], "curriculum_tier": tier})
                atomic_json(directory / "metadata.json", metadata)
                completed += 1
                print(f"severity inference: {completed}/{len(prepared)}", flush=True)
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
    return {"mse": float(mse), "psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))), "ms_ssim": float(piq.multi_scale_ssim(pred, clean, data_range=1.0)), "lpips": float(perceptual(pred * 2 - 1, clean * 2 - 1).mean())}


def evaluate(prepared: list[tuple[dict, Path]]) -> list[dict]:
    device = torch.device("cuda")
    perceptual = lpips.LPIPS(net="alex").to(device).eval()
    rows = []
    with torch.inference_mode():
        for record, directory in prepared:
            clean = Image.open(directory / "clean.png").convert("RGB")
            for configuration in ("degraded", "restored"):
                image = Image.open(directory / f"{configuration}.png").convert("RGB")
                rows.append({"sample_id": record["sample_id"], "family": record["render"]["degradation_family"], "tier": record["render"]["curriculum_tier"], "configuration": configuration, **measure(image, clean, perceptual, device)})
    del perceptual
    torch.cuda.empty_cache()
    return rows


def render_grid(prepared: list[tuple[dict, Path]]) -> None:
    lookup = {(record["render"]["degradation_family"], record["render"]["curriculum_tier"]): directory for record, directory in prepared}
    figure, axes = plt.subplots(len(FAMILIES), 7, figsize=(20, 15), constrained_layout=True)
    titles = ("Clean target", "Easy degraded", "Easy restored", "Medium degraded", "Medium restored", "Hard degraded", "Hard restored")
    for row, family in enumerate(FAMILIES):
        easy = lookup[(family, "easy")]
        paths = [easy / "clean.png"]
        for tier in TIERS:
            directory = lookup[(family, tier)]
            paths.extend((directory / "degraded.png", directory / "restored.png"))
        for column, path in enumerate(paths):
            axes[row, column].imshow(Image.open(path).convert("RGB"))
            axes[row, column].set_title(titles[column] if row == 0 else "")
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(FAMILY_LABELS[family], fontsize=13)
    figure.suptitle("FFHQ restoration across easy, medium, and hard corruption", fontsize=19)
    figure.savefig(FIGURE_ROOT / "severity_restoration_grid.png", dpi=180)
    plt.close(figure)


def main() -> None:
    records = select_records()
    prepared = prepare(records)
    generate(prepared)
    rows = evaluate(prepared)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    with (TABLE_ROOT / "severity_sample_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for tier in TIERS:
        summary[tier] = {}
        for configuration in ("degraded", "restored"):
            subset = [row for row in rows if row["tier"] == tier and row["configuration"] == configuration]
            summary[tier][configuration] = {metric: float(np.mean([row[metric] for row in subset])) for metric in ("mse", "psnr", "ms_ssim", "lpips")}
    atomic_json(PAPER_ROOT / "severity_samples_summary.json", {"scope": "One deterministic held-out identity per degradation family at easy, medium, and hard severity.", "checkpoint": str(CHECKPOINT), "frame_counts": FRAME_COUNTS, "means_by_tier": summary})
    render_grid(prepared)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
