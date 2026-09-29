import csv
import json
import math
from pathlib import Path

import cv2
import lpips
import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
from PIL import Image


PROJECT_ROOT = Path("/home/boni/projects/video_training/degradation_undoing")
RUN_ROOT = PROJECT_ROOT / "outputs/coco_pixel_aggressive_49"
SAMPLES_ROOT = RUN_ROOT / "samples"
VALIDATION_ROOT = PROJECT_ROOT / "data/coco_aggressive_49_validation"
RESULTS_ROOT = RUN_ROOT / "analysis"
METRICS_PATH = RUN_ROOT / "metrics.jsonl"
STEPS = [5000, 5500, 6000, 6500, 7000, 7500, 8000]
TIERS = ["easy", "medium", "hard"]
COLORS = {"easy": "#3b6ea8", "medium": "#d18b2c", "hard": "#9a4771"}
ROLLING_WINDOW = 100
SAMPLED_WINDOW = 200


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def video_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != 49:
        raise RuntimeError(f"{path} contains {len(frames)} frames, expected 49")
    return frames


def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0


def image_metrics(prediction: np.ndarray, target: np.ndarray, perceptual: lpips.LPIPS, device: torch.device) -> dict[str, float]:
    pred = to_tensor(prediction, device)
    clean = to_tensor(target, device)
    mse = torch.mean((pred - clean) ** 2)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    ms_ssim = piq.multi_scale_ssim(pred, clean, data_range=1.0)
    lpips_value = perceptual(pred * 2.0 - 1.0, clean * 2.0 - 1.0)
    return {
        "mse": float(mse.item()),
        "psnr": float(psnr.item()),
        "ms_ssim": float(ms_ssim.item()),
        "lpips": float(lpips_value.mean().item()),
    }


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(len(values)):
        segment = values[max(0, index - window + 1) : index + 1]
        finite = segment[np.isfinite(segment)]
        if finite.size:
            result[index] = finite.mean()
    return result


def load_training_rows() -> list[dict]:
    return [json.loads(line) for line in METRICS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_samples() -> tuple[list[dict], dict[tuple[int, str], list[np.ndarray]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perceptual = lpips.LPIPS(net="alex").to(device).eval()
    clean = load_rgb(VALIDATION_ROOT / "clean.png")
    rows = []
    videos = {}
    with torch.inference_mode():
        for tier in TIERS:
            baseline = image_metrics(load_rgb(VALIDATION_ROOT / f"{tier}.png"), clean, perceptual, device)
            rows.append({"step": 0, "tier": tier, "source": "degraded", **baseline})
        for step in STEPS:
            for sample_index, tier in enumerate(TIERS, start=1):
                frames = video_frames(SAMPLES_ROOT / f"step_{step:06d}_{sample_index}.mp4")
                videos[(step, tier)] = frames
                measured = image_metrics(frames[-1], clean, perceptual, device)
                rows.append({"step": step, "tier": tier, "source": "restored", **measured})
    return rows, videos


def save_metric_table(rows: list[dict]) -> None:
    path = RESULTS_ROOT / "validation_final_frame_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "tier", "source", "mse", "psnr", "ms_ssim", "lpips"])
        writer.writeheader()
        writer.writerows(rows)


def plot_training(rows: list[dict], validation_rows: list[dict]) -> None:
    steps = np.asarray([row["global_step"] for row in rows], dtype=np.int64)
    losses = np.asarray([row["train/loss"] for row in rows], dtype=np.float64)
    times = np.asarray([row["train/step_time"] for row in rows], dtype=np.float64)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].plot(steps, losses, color="#b8c2cc", linewidth=0.5, alpha=0.45, label="Per step")
    axes[0, 0].plot(steps, moving_average(losses, ROLLING_WINDOW), color="#274c77", linewidth=2.2, label=f"{ROLLING_WINDOW}-step mean")
    axes[0, 0].set(title="Training loss\nFlow-matching objective; lower is better", xlabel="Global step", ylabel="Loss")
    axes[0, 0].legend(frameon=False)
    sigma_keys = ["train/loss_sigma_0.00-0.25", "train/loss_sigma_0.25-0.50", "train/loss_sigma_0.50-0.75", "train/loss_sigma_0.75-1.00"]
    sigma_colors = ["#7798ab", "#3b6ea8", "#d18b2c", "#9a4771"]
    for key, color in zip(sigma_keys, sigma_colors):
        values = np.asarray([row.get(key, math.nan) for row in rows], dtype=np.float64)
        axes[0, 1].plot(steps, moving_average(values, ROLLING_WINDOW), color=color, linewidth=1.8, label=key.removeprefix("train/loss_sigma_"))
    axes[0, 1].set(title=f"Loss by sampled sigma range\nRolling {ROLLING_WINDOW}-step mean within each sparse bucket", xlabel="Global step", ylabel="Loss")
    axes[0, 1].legend(frameon=False, ncol=2)
    for tier in TIERS:
        weights = np.asarray([row[f"curriculum/weight_{tier}"] for row in rows], dtype=np.float64)
        sampled = np.asarray([row[f"curriculum/sampled_{tier}"] for row in rows], dtype=np.float64)
        axes[1, 0].plot(steps, weights, color=COLORS[tier], linewidth=2.0, label=f"{tier} target")
        axes[1, 0].plot(steps, moving_average(sampled, SAMPLED_WINDOW), color=COLORS[tier], linewidth=1.0, linestyle="--", alpha=0.75, label=f"{tier} sampled")
    axes[1, 0].set(title=f"Curriculum mixture\nTarget weights and rolling {SAMPLED_WINDOW}-step realized fractions", xlabel="Global step", ylabel="Fraction", ylim=(0, 0.65))
    axes[1, 0].legend(frameon=False, ncol=2)
    restored = [row for row in validation_rows if row["source"] == "restored"]
    for tier in TIERS:
        tier_rows = [row for row in restored if row["tier"] == tier]
        axes[1, 1].plot([row["step"] for row in tier_rows], [row["lpips"] for row in tier_rows], color=COLORS[tier], marker="o", linewidth=2.0, label=tier)
    axes[1, 1].set(title="Held-out final-frame LPIPS\nOne fixed validation identity per severity; lower is better", xlabel="Checkpoint step", ylabel="LPIPS")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(axis="y", color="#d9dde2", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(RESULTS_ROOT / "training_diagnostics.png", dpi=180)
    plt.close(figure)
    steady = times[times < 20.0]
    timing = {"median_steady_step_seconds": float(np.median(steady)), "p95_steady_step_seconds": float(np.percentile(steady, 95)), "validation_or_checkpoint_steps": int(np.sum(times >= 20.0))}
    (RESULTS_ROOT / "timing_summary.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")


def plot_checkpoint_sheet(videos: dict[tuple[int, str], list[np.ndarray]]) -> None:
    clean = load_rgb(VALIDATION_ROOT / "clean.png")
    selected_steps = [5000, 6500, 8000]
    figure, axes = plt.subplots(len(TIERS), 5, figsize=(18, 11), constrained_layout=True)
    for row_index, tier in enumerate(TIERS):
        images = [load_rgb(VALIDATION_ROOT / f"{tier}.png")] + [videos[(step, tier)][-1] for step in selected_steps] + [clean]
        titles = [f"{tier.title()} degraded", "Step 5000", "Step 6500", "Step 8000", "Clean target"]
        for column_index, (image, title) in enumerate(zip(images, titles)):
            axes[row_index, column_index].imshow(image)
            axes[row_index, column_index].set_title(title)
            axes[row_index, column_index].axis("off")
    figure.savefig(RESULTS_ROOT / "checkpoint_final_frame_comparison.png", dpi=160)
    plt.close(figure)


def plot_trajectory_sheet(videos: dict[tuple[int, str], list[np.ndarray]]) -> None:
    indices = [0, 12, 24, 36, 48]
    figure, axes = plt.subplots(len(TIERS), len(indices), figsize=(18, 11), constrained_layout=True)
    for row_index, tier in enumerate(TIERS):
        frames = videos[(8000, tier)]
        for column_index, frame_index in enumerate(indices):
            axes[row_index, column_index].imshow(frames[frame_index])
            axes[row_index, column_index].set_title(f"{tier.title()} · frame {frame_index + 1}")
            axes[row_index, column_index].axis("off")
    figure.savefig(RESULTS_ROOT / "step_8000_trajectory.png", dpi=160)
    plt.close(figure)


def save_summary(training_rows: list[dict], validation_rows: list[dict]) -> None:
    restored = [row for row in validation_rows if row["source"] == "restored"]
    by_step = {}
    for step in STEPS:
        rows = [row for row in restored if row["step"] == step]
        by_step[str(step)] = {metric: float(np.mean([row[metric] for row in rows])) for metric in ("mse", "psnr", "ms_ssim", "lpips")}
    baseline_rows = [row for row in validation_rows if row["source"] == "degraded"]
    baseline = {metric: float(np.mean([row[metric] for row in baseline_rows])) for metric in ("mse", "psnr", "ms_ssim", "lpips")}
    losses = np.asarray([row["train/loss"] for row in training_rows], dtype=np.float64)
    payload = {
        "metric_rows": len(training_rows),
        "first_step": training_rows[0]["global_step"],
        "last_step": training_rows[-1]["global_step"],
        "loss_first_200_mean": float(losses[:200].mean()),
        "loss_last_200_mean": float(losses[-200:].mean()),
        "degraded_baseline_mean": baseline,
        "checkpoint_means": by_step,
        "best_mean_lpips_step": min(STEPS, key=lambda step: by_step[str(step)]["lpips"]),
        "best_mean_psnr_step": max(STEPS, key=lambda step: by_step[str(step)]["psnr"]),
        "validation_scope": "one fixed clean COCO identity rendered at easy, medium, and hard degradation",
    }
    (RESULTS_ROOT / "analysis_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    training_rows = load_training_rows()
    validation_rows, videos = evaluate_samples()
    save_metric_table(validation_rows)
    plot_training(training_rows, validation_rows)
    plot_checkpoint_sheet(videos)
    plot_trajectory_sheet(videos)
    save_summary(training_rows, validation_rows)
    print((RESULTS_ROOT / "analysis_summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
