import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import lpips
import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
from PIL import Image

from project_config import OUTPUT_ROOT, PROJECT_ROOT
from restoration.io import atomic_json

RUN_ROOT = OUTPUT_ROOT / "ffhq_full_restoration"
SAMPLES_ROOT = RUN_ROOT / "samples"
VALIDATION_ROOT = PROJECT_ROOT / "data/ffhq_full_restoration_validation"
PAPER_ROOT = RUN_ROOT / "paper_package_20260904"
FIGURE_ROOT = PAPER_ROOT / "figures"
TABLE_ROOT = PAPER_ROOT / "tables"
PROVENANCE_ROOT = PAPER_ROOT / "provenance"
VIDEO_ROOT = PAPER_ROOT / "validation_videos"
METRICS_PATH = RUN_ROOT / "metrics.jsonl"
STEPS = tuple(range(0, 5001, 500))
FAMILIES = ("jpeg", "blur", "mosaic", "low_resolution", "mixed")
FAMILY_LABELS = {
    "jpeg": "JPEG",
    "blur": "Gaussian blur",
    "mosaic": "Mosaic",
    "low_resolution": "Low resolution",
    "mixed": "Mixed",
}
COLORS = {
    "jpeg": "#376795",
    "blur": "#d18b2c",
    "mosaic": "#9a4771",
    "low_resolution": "#687d45",
    "mixed": "#d65f5f",
}
ROLLING_WINDOW = 100
SAMPLED_WINDOW = 200


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#374151",
            "axes.labelcolor": "#1f2937",
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
            "text.color": "#111827",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def video_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        valid, frame = capture.read()
        if not valid:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != 49:
        raise RuntimeError(f"{path} contains {len(frames)} frames, expected 49")
    return frames


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0


def metrics(
    prediction: np.ndarray,
    target: np.ndarray,
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


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    cumulative = np.cumsum(np.where(finite, values, 0.0))
    counts = np.cumsum(finite.astype(np.int64))
    for index in range(len(values)):
        left = max(0, index - window + 1)
        total = cumulative[index] - (cumulative[left - 1] if left else 0.0)
        count = counts[index] - (counts[left - 1] if left else 0)
        if count:
            result[index] = total / count
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_training_rows() -> list[dict]:
    return [json.loads(line) for line in METRICS_PATH.read_text(encoding="utf-8").splitlines() if line]


def evaluate_validation() -> tuple[list[dict], dict[tuple[int, str], list[np.ndarray]]]:
    device = torch.device("cuda")
    perceptual = lpips.LPIPS(net="alex").to(device).eval()
    rows = []
    videos = {}
    with torch.inference_mode():
        for family_index, family in enumerate(FAMILIES, start=1):
            clean = load_rgb(VALIDATION_ROOT / f"{family}_clean.png")
            degraded = load_rgb(VALIDATION_ROOT / f"{family}_extreme.png")
            rows.append({"step": -1, "family": family, "source": "degraded", **metrics(degraded, clean, perceptual, device)})
            for step in STEPS:
                frames = video_frames(SAMPLES_ROOT / f"step_{step:06d}_{family_index}.mp4")
                videos[(step, family)] = frames
                rows.append({"step": step, "family": family, "source": "restored", **metrics(frames[-1], clean, perceptual, device)})
    del perceptual
    torch.cuda.empty_cache()
    return rows, videos


def plot_training(training_rows: list[dict]) -> dict:
    steps = np.asarray([row["global_step"] for row in training_rows], dtype=np.int64)
    losses = np.asarray([row["train/loss"] for row in training_rows], dtype=np.float64)
    times = np.asarray([row["train/step_time"] for row in training_rows], dtype=np.float64)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].plot(steps, losses, color="#c7ced6", linewidth=0.45, alpha=0.5, label="Per step")
    axes[0, 0].plot(steps, moving_average(losses, ROLLING_WINDOW), color="#376795", linewidth=2.2, label="100-step mean")
    axes[0, 0].set(title="Flow-matching training loss", xlabel="Optimizer step", ylabel="Loss")
    axes[0, 0].legend(frameon=False)
    sigma_keys = (
        "train/loss_sigma_0.00-0.25",
        "train/loss_sigma_0.25-0.50",
        "train/loss_sigma_0.50-0.75",
        "train/loss_sigma_0.75-1.00",
    )
    sigma_colors = ("#6f8fa8", "#376795", "#d18b2c", "#9a4771")
    for key, color in zip(sigma_keys, sigma_colors):
        values = np.asarray([row.get(key, math.nan) for row in training_rows], dtype=np.float64)
        axes[0, 1].plot(
            steps,
            moving_average(values, ROLLING_WINDOW),
            color=color,
            linewidth=1.8,
            label=key.removeprefix("train/loss_sigma_"),
        )
    axes[0, 1].set(title="Loss by sampled diffusion sigma", xlabel="Optimizer step", ylabel="100-step conditional mean")
    axes[0, 1].legend(frameon=False, ncol=2)
    tier_colors = {"easy": "#376795", "medium": "#d18b2c", "hard": "#9a4771", "extreme": "#687d45"}
    for tier, color in tier_colors.items():
        target = np.asarray([row[f"curriculum/weight_{tier}"] for row in training_rows], dtype=np.float64)
        realized = np.asarray([row[f"curriculum/sampled_{tier}"] for row in training_rows], dtype=np.float64)
        axes[1, 0].plot(steps, target, color=color, linewidth=2.0, label=f"{tier} target")
        axes[1, 0].plot(
            steps,
            moving_average(realized, SAMPLED_WINDOW),
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
            label=f"{tier} realized",
        )
    axes[1, 0].set(title="Adaptive-frame curriculum", xlabel="Optimizer step", ylabel="Sampling fraction", ylim=(0, 0.9))
    axes[1, 0].legend(frameon=False, ncol=2)
    axes[1, 1].plot(steps, times, color="#c7ced6", linewidth=0.5, alpha=0.55)
    axes[1, 1].plot(steps, moving_average(times, 100), color="#376795", linewidth=2.0)
    axes[1, 1].set(title="Step duration", xlabel="Optimizer step", ylabel="Seconds")
    axes[1, 1].set_ylim(0, min(20, float(np.percentile(times, 99))))
    for axis in axes.flat:
        axis.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("FFHQ full-restoration training diagnostics", fontsize=16)
    figure.savefig(FIGURE_ROOT / "training_diagnostics.png", dpi=200)
    figure.savefig(FIGURE_ROOT / "training_diagnostics.pdf")
    plt.close(figure)
    steady = times[times < 20]
    return {
        "loss_first_200_mean": float(losses[:200].mean()),
        "loss_last_200_mean": float(losses[-200:].mean()),
        "loss_change_percent": float(100 * (losses[-200:].mean() / losses[:200].mean() - 1)),
        "steady_step_median_seconds": float(np.median(steady)),
        "steady_step_p95_seconds": float(np.percentile(steady, 95)),
        "long_steps": int(np.sum(times >= 20)),
    }


def checkpoint_means(validation_rows: list[dict]) -> dict[int, dict[str, float]]:
    restored = [row for row in validation_rows if row["source"] == "restored"]
    return {
        step: {
            metric: float(np.mean([row[metric] for row in restored if row["step"] == step]))
            for metric in ("mse", "psnr", "ms_ssim", "lpips")
        }
        for step in STEPS
    }


def plot_checkpoint_metrics(validation_rows: list[dict], means: dict[int, dict[str, float]]) -> None:
    restored = [row for row in validation_rows if row["source"] == "restored"]
    metric_specs = (
        ("psnr", "PSNR (dB)", True),
        ("lpips", "LPIPS", False),
        ("ms_ssim", "MS-SSIM", True),
        ("mse", "MSE", False),
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for axis, (metric, label, _) in zip(axes.flat, metric_specs):
        for family in FAMILIES:
            rows = [row for row in restored if row["family"] == family]
            axis.plot(
                [row["step"] for row in rows],
                [row[metric] for row in rows],
                color=COLORS[family],
                linewidth=1.1,
                marker="o",
                markersize=3,
                alpha=0.55,
                label=FAMILY_LABELS[family],
            )
        axis.plot(STEPS, [means[step][metric] for step in STEPS], color="#111827", linewidth=2.7, marker="o", label="Mean")
        axis.set(title=label, xlabel="Checkpoint step", ylabel=label)
        axis.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=3)
    figure.suptitle("Fixed-validation final-frame fidelity across checkpoints", fontsize=16)
    figure.savefig(FIGURE_ROOT / "checkpoint_metric_curves.png", dpi=200)
    figure.savefig(FIGURE_ROOT / "checkpoint_metric_curves.pdf")
    plt.close(figure)


def plot_checkpoint_gallery(videos: dict[tuple[int, str], list[np.ndarray]], best_step: int) -> None:
    chosen = tuple(dict.fromkeys((0, 2500, best_step, 5000)))
    columns = ("Degraded", *[f"Step {step}" for step in chosen], "Clean target")
    figure, axes = plt.subplots(len(FAMILIES), len(columns), figsize=(4 * len(columns), 19), constrained_layout=True)
    for row_index, family in enumerate(FAMILIES):
        clean = load_rgb(VALIDATION_ROOT / f"{family}_clean.png")
        degraded = load_rgb(VALIDATION_ROOT / f"{family}_extreme.png")
        images = (degraded, *[videos[(step, family)][-1] for step in chosen], clean)
        for column_index, (image, title) in enumerate(zip(images, columns)):
            axes[row_index, column_index].imshow(image)
            axes[row_index, column_index].set_title(title if row_index == 0 else "")
            axes[row_index, column_index].axis("off")
        axes[row_index, 0].set_ylabel(FAMILY_LABELS[family], fontsize=12)
    figure.suptitle("FFHQ fixed-validation checkpoint comparison", fontsize=17)
    figure.savefig(FIGURE_ROOT / "checkpoint_final_frame_gallery.png", dpi=170)
    plt.close(figure)


def plot_final_trajectories(videos: dict[tuple[int, str], list[np.ndarray]]) -> list[dict]:
    indices = (0, 12, 24, 36, 48)
    figure, axes = plt.subplots(len(FAMILIES), len(indices), figsize=(18, 18), constrained_layout=True)
    rows = []
    for row_index, family in enumerate(FAMILIES):
        frames = videos[(5000, family)]
        clean = load_rgb(VALIDATION_ROOT / f"{family}_clean.png").astype(np.float32) / 255.0
        previous = None
        for frame_index, frame in enumerate(frames):
            normalized = frame.astype(np.float32) / 255.0
            rows.append(
                {
                    "family": family,
                    "frame": frame_index + 1,
                    "mse_to_clean": float(np.mean((normalized - clean) ** 2)),
                    "mse_to_previous": 0.0 if previous is None else float(np.mean((normalized - previous) ** 2)),
                }
            )
            previous = normalized
        for column_index, frame_index in enumerate(indices):
            axes[row_index, column_index].imshow(frames[frame_index])
            axes[row_index, column_index].set_title(f"Frame {frame_index + 1}" if row_index == 0 else "")
            axes[row_index, column_index].axis("off")
        axes[row_index, 0].set_ylabel(FAMILY_LABELS[family], fontsize=12)
    figure.suptitle("Step-5000 generated restoration trajectories", fontsize=17)
    figure.savefig(FIGURE_ROOT / "final_checkpoint_trajectories.png", dpi=170)
    plt.close(figure)
    curve, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        axis.plot(
            [row["frame"] for row in family_rows],
            [row["mse_to_clean"] for row in family_rows],
            color=COLORS[family],
            linewidth=2.0,
            label=FAMILY_LABELS[family],
        )
    axis.set(title="Generated trajectory distance to clean target", xlabel="Video frame", ylabel="MSE to paired clean image")
    axis.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=3)
    curve.savefig(FIGURE_ROOT / "trajectory_fidelity_curves.png", dpi=200)
    curve.savefig(FIGURE_ROOT / "trajectory_fidelity_curves.pdf")
    plt.close(curve)
    return rows


def copy_provenance(best_step: int) -> None:
    PROVENANCE_ROOT.mkdir(parents=True, exist_ok=True)
    sources = (
        PROJECT_ROOT / "configs/ffhq_full_restoration.yaml",
        RUN_ROOT / "run_state.json",
        RUN_ROOT / "training_config.yaml",
        RUN_ROOT / "lora_targets.json",
        PROJECT_ROOT / "data/manifests/ffhq_full_restoration/summary.json",
        PROJECT_ROOT / "data/manifests/ffhq_full_restoration/validation.json",
        OUTPUT_ROOT / "readiness/ffhq_full_restoration/summary.json",
        OUTPUT_ROOT / "readiness/environment.json",
    )
    for source in sources:
        shutil.copy2(source, PROVENANCE_ROOT / source.name)
    checkpoint = (
        OUTPUT_ROOT / "ffhq_pixel_aggressive/checkpoints/lora_weights_step_05000.safetensors"
        if best_step == 0
        else RUN_ROOT / f"checkpoints/lora_weights_step_{best_step:05d}.safetensors"
    )
    shutil.copy2(checkpoint.with_suffix(".metadata.json"), PROVENANCE_ROOT / "exploratory_best_checkpoint.metadata.json")


def copy_videos(best_step: int) -> None:
    VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    for step in tuple(dict.fromkeys((best_step, 5000))):
        for family_index, family in enumerate(FAMILIES, start=1):
            shutil.copy2(
                SAMPLES_ROOT / f"step_{step:06d}_{family_index}.mp4",
                VIDEO_ROOT / f"{family}_step_{step:05d}.mp4",
            )


def main() -> None:
    configure_plotting()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    training_rows = load_training_rows()
    if len(training_rows) != 5000:
        raise RuntimeError(f"Expected 5000 training rows, received {len(training_rows)}")
    validation_rows, videos = evaluate_validation()
    means = checkpoint_means(validation_rows)
    best_lpips_step = min(STEPS, key=lambda step: means[step]["lpips"])
    best_psnr_step = max(STEPS, key=lambda step: means[step]["psnr"])
    training_summary = plot_training(training_rows)
    plot_checkpoint_metrics(validation_rows, means)
    plot_checkpoint_gallery(videos, best_lpips_step)
    trajectory_rows = plot_final_trajectories(videos)
    write_csv(TABLE_ROOT / "training_metrics.csv", training_rows, sorted({key for row in training_rows for key in row}))
    write_csv(
        TABLE_ROOT / "validation_checkpoint_metrics.csv",
        validation_rows,
        ["step", "family", "source", "mse", "psnr", "ms_ssim", "lpips"],
    )
    write_csv(
        TABLE_ROOT / "trajectory_metrics_step5000.csv",
        trajectory_rows,
        ["family", "frame", "mse_to_clean", "mse_to_previous"],
    )
    baselines = [row for row in validation_rows if row["source"] == "degraded"]
    final_rows = [row for row in validation_rows if row["source"] == "restored" and row["step"] == 5000]
    final_improved_both = sum(
        final["mse"] < baseline["mse"] and final["lpips"] < baseline["lpips"]
        for baseline, final in zip(baselines, final_rows)
    )
    payload = {
        "scope": "Five fixed extreme validation identities, one per degradation family; checkpoint metrics use MP4-decoded final frames.",
        "steps": list(STEPS),
        "training": training_summary,
        "degraded_baseline_mean": {
            metric: float(np.mean([row[metric] for row in baselines]))
            for metric in ("mse", "psnr", "ms_ssim", "lpips")
        },
        "checkpoint_means": {str(step): values for step, values in means.items()},
        "exploratory_best_lpips_step": best_lpips_step,
        "exploratory_best_psnr_step": best_psnr_step,
        "final_improved_both_families": final_improved_both,
        "final_family_count": len(FAMILIES),
    }
    atomic_json(PAPER_ROOT / "fixed_validation_summary.json", payload)
    copy_provenance(best_lpips_step)
    copy_videos(best_lpips_step)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
