import csv
import gc
import hashlib
import json
import os
from pathlib import Path

import lpips
import numpy as np
import piq
import torch
from PIL import Image

from data_gen.restoration_trajectory import build_restoration_frames
from inference.restoration_pipeline import PersistentRestorationPipeline, atomic_save_png
from project_config import OUTPUT_ROOT
from restoration.io import atomic_json, stable_int
from restoration.schema import DegradationSpec


SOURCE_ROOT = Path("/home/boni/projects/video_training/degradation_undoing/data/somfaces_aligned_v2")
RUN_ROOT = OUTPUT_ROOT / "somfaces_aligned_degradation_evaluation"
SAMPLE_ROOT = RUN_ROOT / "samples"
CHECKPOINT = OUTPUT_ROOT / "ffhq_full_restoration/checkpoints/lora_weights_step_05000.safetensors"
FAMILIES = ("jpeg", "blur", "mosaic", "low_resolution", "mixed")
FAMILY_PARAMETERS = {
    "jpeg": ({"jpeg_quality0": 18}, {"jpeg_quality0": 8}),
    "blur": ({"sigma0": 4.5}, {"sigma0": 7.0}),
    "mosaic": ({"pixel_factor0": 16}, {"pixel_factor0": 32}),
    "low_resolution": ({"lowres_factor0": 8.0}, {"lowres_factor0": 16.0}),
    "mixed": (
        {"jpeg_quality0": 35, "sigma0": 2.0, "pixel_factor0": 5, "lowres_factor0": 3.0},
        {"jpeg_quality0": 20, "sigma0": 3.0, "pixel_factor0": 8, "lowres_factor0": 5.0},
    ),
}
FRAME_COUNT = 49
SEED = 20260906
PROMPT = "Progressively restore the degraded face while preserving identity, facial geometry, expression, lighting, texture, and background."


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def degradation_spec(family: str, variant: int, seed: int) -> DegradationSpec:
    values = FAMILY_PARAMETERS[family][variant]
    orders = {
        "jpeg": ("jpeg",),
        "blur": ("gaussian",),
        "mosaic": ("pixelation",),
        "low_resolution": ("low_resolution",),
        "mixed": ("low_resolution", "pixelation", "gaussian", "jpeg"),
    }
    kinds = {
        "jpeg": "jpeg",
        "blur": "gaussian",
        "mosaic": "pixelation",
        "low_resolution": "low_resolution",
        "mixed": "mixed_restoration",
    }
    return DegradationSpec(
        kind=kinds[family],
        sigma0=values.get("sigma0"),
        pixel_factor0=values.get("pixel_factor0"),
        seed=seed,
        jpeg_quality0=values.get("jpeg_quality0"),
        lowres_factor0=values.get("lowres_factor0"),
        operator_order=orders[family],
    )


def source_images() -> list[Path]:
    paths = sorted(
        path
        for path in SOURCE_ROOT.glob("[0-9][0-9][0-9].png")
        if path.is_file()
    )
    if len(paths) != 15:
        raise RuntimeError(f"Expected 15 somfaces images, found {len(paths)}")
    digests = [file_digest(path) for path in paths]
    if len(set(digests)) != len(paths):
        raise RuntimeError("Duplicate source files detected")
    return paths


def prepare() -> list[dict]:
    records = []
    for index, source in enumerate(source_images()):
        family = FAMILIES[index % len(FAMILIES)]
        variant = (index // len(FAMILIES)) % 2
        sample_id = f"{index + 1:02d}_{family}"
        seed = stable_int(f"somfaces:{source.name}:{family}:{variant}", SEED) % (2**31)
        spec = degradation_spec(family, variant, seed)
        with Image.open(source) as image:
            clean = image.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        frames = build_restoration_frames(clean, spec, FRAME_COUNT)
        directory = SAMPLE_ROOT / sample_id
        atomic_save_png(directory / "clean.png", clean)
        atomic_save_png(directory / "degraded.png", frames[0])
        digest = file_digest(source)
        record = {
            "sample_id": sample_id,
            "source_name": source.name,
            "aligned_source_name": source.name,
            "source_sha256": digest,
            "alignment": "PULSE FFHQ 68-point landmark transform",
            "family": family,
            "degradation": spec.to_dict(),
            "frame_count": FRAME_COUNT,
            "seed": seed,
            "clean_path": str(directory / "clean.png"),
            "degraded_path": str(directory / "degraded.png"),
            "ltx_path": str(directory / "ltx.png"),
        }
        atomic_json(directory / "metadata.json", record)
        records.append(record)
    atomic_json(RUN_ROOT / "manifest.json", {"dataset": "somfaces", "label": "degradation", "records": records})
    return records


def generate(records: list[dict]) -> None:
    pending = [record for record in records if not Path(record["ltx_path"]).is_file()]
    if not pending:
        print("All LTX outputs already exist", flush=True)
        return
    pipeline = PersistentRestorationPipeline(CHECKPOINT, frame_count=FRAME_COUNT)
    try:
        pipeline.prepare(PROMPT)
        for index, record in enumerate(pending, start=1):
            result = pipeline.restore_final(Path(record["degraded_path"]), PROMPT, record["seed"])
            atomic_save_png(Path(record["ltx_path"]), result.final_image)
            atomic_json(Path(record["ltx_path"]).with_name("ltx_metadata.json"), result.metadata)
            print(f"somfaces restoration: {index}/{len(pending)}", flush=True)
    finally:
        pipeline.close()
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


def tensor(path: Path, device: torch.device) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0


def metrics(candidate: torch.Tensor, target: torch.Tensor, perceptual: lpips.LPIPS) -> dict[str, float]:
    mse = torch.mean((candidate - target) ** 2)
    return {
        "mse": float(mse),
        "psnr": float(-10.0 * torch.log10(mse.clamp_min(1e-12))),
        "ms_ssim": float(piq.multi_scale_ssim(candidate, target, data_range=1.0)),
        "lpips": float(perceptual(candidate * 2.0 - 1.0, target * 2.0 - 1.0).mean()),
    }


def evaluate(records: list[dict]) -> None:
    device = torch.device("cuda")
    perceptual = lpips.LPIPS(net="alex").to(device).eval()
    rows = []
    with torch.inference_mode():
        for record in records:
            target = tensor(Path(record["clean_path"]), device)
            for configuration, path_key in (("degraded", "degraded_path"), ("ltx", "ltx_path")):
                candidate = tensor(Path(record[path_key]), device)
                rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "source_name": record["source_name"],
                        "family": record["family"],
                        "configuration": configuration,
                        **metrics(candidate, target, perceptual),
                    }
                )
    del perceptual
    torch.cuda.empty_cache()
    with (RUN_ROOT / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        configuration: {
            metric: float(np.mean([row[metric] for row in rows if row["configuration"] == configuration]))
            for metric in ("mse", "psnr", "ms_ssim", "lpips")
        }
        for configuration in ("degraded", "ltx")
    }
    atomic_json(RUN_ROOT / "summary.json", {"count": len(records), "label": "degradation", "checkpoint": str(CHECKPOINT), "means": summary})
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    records = prepare()
    sample_id = os.environ.get("SOMFACES_SAMPLE_ID")
    evaluate_only = os.environ.get("SOMFACES_EVALUATE_ONLY") == "1"
    if sample_id:
        selected = [record for record in records if record["sample_id"] == sample_id]
        if len(selected) != 1:
            raise RuntimeError(f"Unknown sample id: {sample_id}")
        generate(selected)
        return
    if not evaluate_only:
        generate(records)
    missing = [record["sample_id"] for record in records if not Path(record["ltx_path"]).is_file()]
    if missing:
        raise RuntimeError(f"Cannot evaluate with missing LTX outputs: {missing}")
    evaluate(records)


if __name__ == "__main__":
    main()
