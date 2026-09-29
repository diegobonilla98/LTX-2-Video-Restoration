import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image

from data_gen.restoration_trajectory import build_restoration_frames
from data_gen.trajectory import center_crop_512
from project_config import FFHQ_ROOT, MANIFEST_ROOT, PRECOMPUTED_ROOT, PROJECT_ROOT
from restoration.io import atomic_json, atomic_jsonl, read_jsonl, sha256_file, stable_int
from restoration.schema import DegradationSpec, RestorationRecord

DATASET_NAME = "ffhq_full_restoration"
PROMPT = "Progressively restore the degraded face while preserving identity, facial geometry, expression, lighting, texture, and background."
SEED = 20260903
TRAIN_IDENTITIES = 12288
VALIDATION_IDENTITIES = 128
TEST_IDENTITIES = 512
TOTAL_STEPS = 5000
FAMILIES = ("jpeg", "blur", "mosaic", "low_resolution", "mixed")
TIERS = ("easy", "medium", "hard", "extreme")
FRAME_COUNTS = {"easy": 17, "medium": 25, "hard": 33, "extreme": 49}
DISTANCE_UNITS = {"easy": 2, "medium": 3, "hard": 4, "extreme": 6}
CURRICULUM_ANCHORS = (
    {"step": 1, "weights": {"easy": 0.80, "medium": 0.15, "hard": 0.05, "extreme": 0.00}},
    {"step": 1000, "weights": {"easy": 0.55, "medium": 0.30, "hard": 0.12, "extreme": 0.03}},
    {"step": 2500, "weights": {"easy": 0.35, "medium": 0.35, "hard": 0.22, "extreme": 0.08}},
    {"step": 4000, "weights": {"easy": 0.22, "medium": 0.30, "hard": 0.30, "extreme": 0.18}},
    {"step": 5000, "weights": {"easy": 0.15, "medium": 0.25, "hard": 0.32, "extreme": 0.28}},
)
STRENGTHS = {
    "jpeg": {
        "easy": {"jpeg_quality0": 60},
        "medium": {"jpeg_quality0": 35},
        "hard": {"jpeg_quality0": 18},
        "extreme": {"jpeg_quality0": 8},
    },
    "blur": {
        "easy": {"sigma0": 1.25},
        "medium": {"sigma0": 2.5},
        "hard": {"sigma0": 4.5},
        "extreme": {"sigma0": 7.0},
    },
    "mosaic": {
        "easy": {"pixel_factor0": 4},
        "medium": {"pixel_factor0": 8},
        "hard": {"pixel_factor0": 16},
        "extreme": {"pixel_factor0": 32},
    },
    "low_resolution": {
        "easy": {"lowres_factor0": 2.0},
        "medium": {"lowres_factor0": 4.0},
        "hard": {"lowres_factor0": 8.0},
        "extreme": {"lowres_factor0": 16.0},
    },
    "mixed": {
        "easy": {"jpeg_quality0": 75, "sigma0": 0.6, "pixel_factor0": 2, "lowres_factor0": 1.5},
        "medium": {"jpeg_quality0": 55, "sigma0": 1.2, "pixel_factor0": 3, "lowres_factor0": 2.0},
        "hard": {"jpeg_quality0": 35, "sigma0": 2.0, "pixel_factor0": 5, "lowres_factor0": 3.0},
        "extreme": {"jpeg_quality0": 20, "sigma0": 3.0, "pixel_factor0": 8, "lowres_factor0": 5.0},
    },
}
VALIDATION_ROOT = PROJECT_ROOT / "data/ffhq_full_restoration_validation"


def atomic_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def ranked_images() -> list[Path]:
    images = sorted(FFHQ_ROOT.glob("*.png"))
    return sorted(images, key=lambda path: stable_int(path.name, 20260830))


def split_images(images: list[Path]) -> dict[str, list[Path]]:
    required = TRAIN_IDENTITIES + VALIDATION_IDENTITIES + TEST_IDENTITIES
    if len(images) < required:
        raise RuntimeError(f"FFHQ has {len(images)} PNG files, expected at least {required}")
    train_end = TRAIN_IDENTITIES
    validation_end = train_end + VALIDATION_IDENTITIES
    return {
        "train": images[:train_end],
        "validation": images[train_end:validation_end],
        "test": images[validation_end:required],
    }


def cell_for_index(index: int) -> tuple[str, str]:
    return FAMILIES[index % len(FAMILIES)], TIERS[index % len(TIERS)]


def spec_for(family: str, tier: str, seed: int) -> DegradationSpec:
    values = STRENGTHS[family][tier]
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


def record_for(path: Path, split: str, family: str, tier: str, digest: str) -> dict:
    seed = stable_int(f"ffhq-full-restoration:{split}:{path.name}:{family}:{tier}", SEED) % (2**31)
    sample_id = hashlib.sha256(
        f"ffhq-full-restoration:{split}:{path.name}:{family}:{tier}".encode("utf-8")
    ).hexdigest()[:20]
    spec = spec_for(family, tier, seed)
    frame_count = FRAME_COUNTS[tier]
    return RestorationRecord(
        sample_id=sample_id,
        split=split,
        source_root="ffhq",
        source_relpath=path.name,
        source_sha256=digest,
        prompt=PROMPT,
        trajectory="progressive",
        degradation=spec,
        clean_kind="natural",
        render={
            "curriculum_tier": tier,
            "degradation_family": family,
            "strength": STRENGTHS[family][tier],
            "severity_range": [1.0, 0.0],
            "severity_schedule": "cosine_squared",
            "trajectory_builder": "fixed_endpoint_homotopy_v1",
            "frame_count": frame_count,
            "latent_frame_count": 1 + (frame_count - 1) // 8,
            "restoration_distance_units": DISTANCE_UNITS[tier],
            "crop": "identity",
            "source_size": [512, 512],
        },
    ).to_dict()


def records_for_split(paths: list[Path], split: str, digests: dict[Path, str]) -> list[dict]:
    rows = []
    for index, path in enumerate(paths):
        cells = (cell_for_index(index),) if split == "train" else tuple(
            (family, tier) for tier in TIERS for family in FAMILIES
        )
        for family, tier in cells:
            rows.append(record_for(path, split, family, tier, digests[path]))
    return rows


def save_validation_media(paths: list[Path]) -> None:
    rows = []
    for family_index, family in enumerate(FAMILIES):
        clean = center_crop_512(Image.open(paths[family_index]).convert("RGB"))
        tier = "extreme"
        spec = spec_for(family, tier, stable_int(f"validation:{family}", SEED) % (2**31))
        frames = build_restoration_frames(clean, spec, FRAME_COUNTS[tier])
        atomic_png(VALIDATION_ROOT / f"{family}_extreme.png", frames[0])
        atomic_png(VALIDATION_ROOT / f"{family}_clean.png", clean)
        selected = [frames[index] for index in (0, 12, 24, 36, 48)]
        row = Image.new("RGB", (256 * len(selected), 256))
        for column, frame in enumerate(selected):
            row.paste(frame.resize((256, 256), Image.Resampling.LANCZOS), (column * 256, 0))
        rows.append(row)
    sheet = Image.new("RGB", (256 * 5, 256 * len(rows)))
    for row_index, row in enumerate(rows):
        sheet.paste(row, (0, row_index * 256))
    atomic_png(VALIDATION_ROOT / "trajectory_contact_sheet.png", sheet)


def source_digests(paths: list[Path], destination: Path) -> dict[Path, str]:
    cache_path = destination / "source_hashes.json"
    cached = {}
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    if not cached:
        for split in ("train", "validation", "test"):
            manifest = destination / f"{split}.jsonl"
            if manifest.is_file():
                for record in read_jsonl(manifest):
                    cached.setdefault(record["source_relpath"], {"sha256": record["source_sha256"]})
    digests = {}
    refreshed = {}
    for path in paths:
        stat = path.stat()
        entry = cached.get(path.name, {})
        reusable = entry.get("sha256") and (
            "size" not in entry
            or (entry.get("size") == stat.st_size and entry.get("mtime_ns") == stat.st_mtime_ns)
        )
        digest = entry["sha256"] if reusable else sha256_file(path)
        digests[path] = digest
        refreshed[path.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}
    atomic_json(cache_path, refreshed)
    return digests


def main() -> None:
    images = ranked_images()
    splits = split_images(images)
    selected_paths = [path for paths in splits.values() for path in paths]
    destination = MANIFEST_ROOT / DATASET_NAME
    digests = source_digests(selected_paths, destination)
    counts = {}
    curriculum_tiers = {tier: [] for tier in TIERS}
    train_cells = Counter()
    for split, paths in splits.items():
        rows = records_for_split(paths, split, digests)
        counts[split] = atomic_jsonl(destination / f"{split}.jsonl", rows)
        if split == "train":
            for row in rows:
                tier = row["render"]["curriculum_tier"]
                family = row["render"]["degradation_family"]
                curriculum_tiers[tier].append(row["sample_id"])
                train_cells[f"{family}:{tier}"] += 1
    curriculum = {
        "dataset": DATASET_NAME,
        "seed": SEED,
        "total_steps": TOTAL_STEPS,
        "anchors": list(CURRICULUM_ANCHORS),
        "tiers": curriculum_tiers,
    }
    atomic_json(destination / "curriculum.json", curriculum)
    precomputed_train = PRECOMPUTED_ROOT / DATASET_NAME / "train"
    atomic_json(precomputed_train / "curriculum.json", curriculum)
    (precomputed_train / ".precomputed/latents").mkdir(parents=True, exist_ok=True)
    (precomputed_train / ".precomputed/conditions").mkdir(parents=True, exist_ok=True)
    atomic_json(
        destination / "summary.json",
        {
            "source_root": "ffhq",
            "source_files_found": len(images),
            "selected_identities": {name: len(paths) for name, paths in splits.items()},
            "samples": counts,
            "families": list(FAMILIES),
            "tiers": list(TIERS),
            "frame_counts": FRAME_COUNTS,
            "latent_frame_counts": {tier: 1 + (count - 1) // 8 for tier, count in FRAME_COUNTS.items()},
            "restoration_distance_units": DISTANCE_UNITS,
            "train_cell_counts": dict(sorted(train_cells.items())),
            "precomputed_curriculum_path": str(precomputed_train / "curriculum.json"),
        },
    )
    save_validation_media(splits["validation"])


if __name__ == "__main__":
    main()
