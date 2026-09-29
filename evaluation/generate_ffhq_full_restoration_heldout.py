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
HELDOUT_ROOT = PAPER_ROOT / "heldout"
FIGURE_ROOT = PAPER_ROOT / "figures"
TABLE_ROOT = PAPER_ROOT / "tables"
SUMMARY_PATH = PAPER_ROOT / "fixed_validation_summary.json"
MANIFEST_PATH = MANIFEST_ROOT / "ffhq_full_restoration/test.jsonl"
FINAL_CHECKPOINT = RUN_ROOT / "checkpoints/lora_weights_step_05000.safetensors"
FRAME_COUNT = 49
FAMILIES = ("jpeg", "blur", "mosaic", "low_resolution", "mixed")
FAMILY_LABELS = {
    "jpeg": "JPEG",
    "blur": "Gaussian blur",
    "mosaic": "Mosaic",
    "low_resolution": "Low resolution",
    "mixed": "Mixed",
}
EXAMPLES_PER_FAMILY = 2
SELECTION_SEED = 20260904
GENERATION_SEED = 20260904
PROMPT = "Progressively restore the degraded face while preserving identity, facial geometry, expression, lighting, texture, and background."


def configuration_paths() -> list[tuple[str, Path, int]]:
    fixed = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    best_step = int(fixed["exploratory_best_lpips_step"])
    best_checkpoint = (
        OUTPUT_ROOT / "ffhq_pixel_aggressive/checkpoints/lora_weights_step_05000.safetensors"
        if best_step == 0
        else RUN_ROOT / f"checkpoints/lora_weights_step_{best_step:05d}.safetensors"
    )
    configurations = [(f"step_{best_step:05d}", best_checkpoint, best_step)]
    if best_checkpoint != FINAL_CHECKPOINT:
        configurations.append(("step_05000", FINAL_CHECKPOINT, 5000))
    return configurations


def select_records() -> list[dict]:
    rows = list(read_jsonl(MANIFEST_PATH))
    by_identity = {}
    for row in rows:
        if row["render"]["curriculum_tier"] == "extreme":
            by_identity.setdefault(row["source_relpath"], {})[row["render"]["degradation_family"]] = row
    identities = sorted(by_identity, key=lambda value: stable_int(value, SELECTION_SEED))
    selected = []
    for family_index, family in enumerate(FAMILIES):
        for offset in range(EXAMPLES_PER_FAMILY):
            identity = identities[family_index * EXAMPLES_PER_FAMILY + offset]
            selected.append(by_identity[identity][family])
    if len({record["source_relpath"] for record in selected}) != len(selected):
        raise RuntimeError("Held-out selection reused an identity")
    return selected


def prepare_inputs(records: list[dict]) -> list[tuple[dict, Path]]:
    prepared = []
    for index, record in enumerate(records, start=1):
        family = record["render"]["degradation_family"]
        directory = HELDOUT_ROOT / f"{index:02d}_{family}_{record['sample_id']}"
        clean = load_clean(record)
        degraded = build_restoration_frames(clean, record_spec(record), FRAME_COUNT)[0]
        atomic_save_png(directory / "clean.png", clean)
        atomic_save_png(directory / "degraded.png", degraded)
        prepared.append((record, directory))
    return prepared


def generate(name: str, checkpoint: Path, prepared: list[tuple[dict, Path]]) -> None:
    pipeline = PersistentRestorationPipeline(checkpoint, frame_count=FRAME_COUNT)
    try:
        pipeline.prepare(PROMPT)
        for index, (record, directory) in enumerate(prepared, start=1):
            destination = directory / f"{name}.png"
            seed = stable_int(record["sample_id"], GENERATION_SEED) % (2**31)
            result = pipeline.restore_final(directory / "degraded.png", PROMPT, seed)
            atomic_save_png(destination, result.final_image)
            metadata = dict(result.metadata)
            metadata["sample_id"] = record["sample_id"]
            metadata["source_relpath"] = record["source_relpath"]
            metadata["degradation_family"] = record["render"]["degradation_family"]
            metadata["curriculum_tier"] = record["render"]["curriculum_tier"]
            atomic_json(directory / f"{name}.metadata.json", metadata)
            print(f"{name}: {index}/{len(prepared)}", flush=True)
    finally:
        pipeline.close()
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


def tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0


def metrics(
    prediction: Image.Image,
    target: Image.Image,
    perceptual: lpips.LPIPS,
    device: torch.device,
) -> dict[str, float]:
    pred = tensor(prediction, device)
    clean = tensor(target, device)
    mse = torch.mean((pred - clean) ** 2)
    return {
        "mse": float(mse.item()),
        "psnr": float((-10.0 * torch.log10(mse.clamp_min(1e-12))).item()),
        "ms_ssim": float(piq.multi_scale_ssim(pred, clean, data_range=1.0).item()),
        "lpips": float(perceptual(pred * 2.0 - 1.0, clean * 2.0 - 1.0).mean().item()),
    }


def evaluate(prepared: list[tuple[dict, Path]], names: list[str]) -> list[dict]:
    device = torch.device("cuda")
    perceptual = lpips.LPIPS(net="alex").to(device).eval()
    rows = []
    with torch.inference_mode():
        for record, directory in prepared:
            clean = Image.open(directory / "clean.png").convert("RGB")
            degraded = Image.open(directory / "degraded.png").convert("RGB")
            family = record["render"]["degradation_family"]
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "source_relpath": record["source_relpath"],
                    "family": family,
                    "configuration": "degraded",
                    **metrics(degraded, clean, perceptual, device),
                }
            )
            for name in names:
                restored = Image.open(directory / f"{name}.png").convert("RGB")
                rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "source_relpath": record["source_relpath"],
                        "family": family,
                        "configuration": name,
                        **metrics(restored, clean, perceptual, device),
                    }
                )
    del perceptual
    torch.cuda.empty_cache()
    return rows


def paired_bootstrap(rows: list[dict], configuration: str, metric: str) -> dict:
    baseline = {row["sample_id"]: row[metric] for row in rows if row["configuration"] == "degraded"}
    candidate = {row["sample_id"]: row[metric] for row in rows if row["configuration"] == configuration}
    sample_ids = sorted(baseline)
    deltas = np.asarray([candidate[sample_id] - baseline[sample_id] for sample_id in sample_ids])
    rng = np.random.default_rng(SELECTION_SEED)
    indices = rng.integers(0, len(deltas), size=(10000, len(deltas)))
    bootstrapped = deltas[indices].mean(axis=1)
    return {
        "mean_delta": float(deltas.mean()),
        "ci95_low": float(np.percentile(bootstrapped, 2.5)),
        "ci95_high": float(np.percentile(bootstrapped, 97.5)),
        "improved_fraction": float(np.mean(deltas < 0 if metric in {"mse", "lpips"} else deltas > 0)),
    }


def aggregate(rows: list[dict], names: list[str]) -> dict:
    configurations = ("degraded", *names)
    means = {
        name: {
            metric: float(np.mean([row[metric] for row in rows if row["configuration"] == name]))
            for metric in ("mse", "psnr", "ms_ssim", "lpips")
        }
        for name in configurations
    }
    by_family = {
        family: {
            name: {
                metric: float(
                    np.mean(
                        [
                            row[metric]
                            for row in rows
                            if row["family"] == family and row["configuration"] == name
                        ]
                    )
                )
                for metric in ("mse", "psnr", "ms_ssim", "lpips")
            }
            for name in configurations
        }
        for family in FAMILIES
    }
    uncertainty = {
        name: {
            metric: paired_bootstrap(rows, name, metric)
            for metric in ("mse", "psnr", "ms_ssim", "lpips")
        }
        for name in names
    }
    return {"means": means, "by_family": by_family, "paired_bootstrap_vs_degraded": uncertainty}


def render_galleries(prepared: list[tuple[dict, Path]], names: list[str]) -> None:
    columns = ("degraded", *names, "clean")
    titles = ["Degraded", *[name.replace("step_", "Step ") for name in names], "Clean target"]
    for page, start in enumerate(range(0, len(prepared), 5), start=1):
        page_rows = prepared[start : start + 5]
        figure, axes = plt.subplots(len(page_rows), len(columns), figsize=(4 * len(columns), 19), constrained_layout=True)
        for row_index, (record, directory) in enumerate(page_rows):
            paths = [directory / "degraded.png", *[directory / f"{name}.png" for name in names], directory / "clean.png"]
            for column_index, (path, title) in enumerate(zip(paths, titles)):
                axes[row_index, column_index].imshow(Image.open(path).convert("RGB"))
                axes[row_index, column_index].set_title(title if row_index == 0 else "")
                axes[row_index, column_index].axis("off")
            axes[row_index, 0].set_ylabel(FAMILY_LABELS[record["render"]["degradation_family"]], fontsize=12)
        figure.suptitle(f"Held-out FFHQ extreme restoration examples · page {page}", fontsize=17)
        figure.savefig(FIGURE_ROOT / f"heldout_gallery_{page}.png", dpi=180)
        plt.close(figure)


def plot_heldout_metrics(summary: dict, names: list[str]) -> None:
    configurations = ("degraded", *names)
    labels = ["Degraded", *[name.replace("step_", "Step ") for name in names]]
    colors = ("#9ca3af", "#376795", "#d18b2c")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, metric, title in zip(axes, ("psnr", "lpips"), ("PSNR (dB)", "LPIPS")):
        values = [summary["means"][name][metric] for name in configurations]
        bars = axis.bar(labels, values, color=colors[: len(values)], edgecolor="#374151", linewidth=0.7)
        axis.bar_label(bars, fmt="%.3f", padding=3)
        axis.set(title=title, ylabel=title)
        axis.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Held-out FFHQ final-frame fidelity", fontsize=16)
    figure.savefig(FIGURE_ROOT / "heldout_metric_comparison.png", dpi=200)
    figure.savefig(FIGURE_ROOT / "heldout_metric_comparison.pdf")
    plt.close(figure)


def write_rows(rows: list[dict]) -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    with (TABLE_ROOT / "heldout_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["sample_id", "source_relpath", "family", "configuration", "mse", "psnr", "ms_ssim", "lpips"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    configurations = configuration_paths()
    records = select_records()
    prepared = prepare_inputs(records)
    for name, checkpoint, _ in configurations:
        generate(name, checkpoint, prepared)
    names = [name for name, _, _ in configurations]
    rows = evaluate(prepared, names)
    summary = aggregate(rows, names)
    summary.update(
        {
            "scope": "Ten deterministic, identity-disjoint FFHQ test images at extreme degradation; two images per family.",
            "selection_seed": SELECTION_SEED,
            "generation_seed_namespace": GENERATION_SEED,
            "examples": len(prepared),
            "configurations": [
                {"name": name, "checkpoint": str(checkpoint), "step": step}
                for name, checkpoint, step in configurations
            ],
            "sample_ids": [record["sample_id"] for record, _ in prepared],
        }
    )
    write_rows(rows)
    render_galleries(prepared, names)
    plot_heldout_metrics(summary, names)
    atomic_json(PAPER_ROOT / "heldout_summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"by_family"}}, indent=2))


if __name__ == "__main__":
    main()
