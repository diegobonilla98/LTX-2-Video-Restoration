import hashlib
import os
from pathlib import Path

from PIL import Image

from data_gen.trajectory import build_frames
from project_config import COCO_ROOT, MANIFEST_ROOT, PRECOMPUTED_ROOT, PROJECT_ROOT
from restoration.io import atomic_json, atomic_jsonl, sha256_file, stable_int
from restoration.schema import DegradationSpec, RestorationRecord


DATASET_NAME = "coco_pixel_aggressive_49"
PROMPT = "Progressively depixelate the image while preserving object identity, geometry, texture, lighting, and scene composition."
SEED = 20260901
FRAME_COUNT = 49
TRAIN_IDENTITIES = 24576
VALIDATION_IDENTITIES = 512
TEST_IDENTITIES = 5000
TOTAL_STEPS = 8000
TIERS = ("easy", "medium", "hard")
TIER_FACTORS = {
    "easy": (8, 12, 16),
    "medium": (12, 16, 24),
    "hard": (16, 24, 32),
}
CURRICULUM_ANCHORS = (
    {"step": 1, "weights": {"easy": 0.70, "medium": 0.25, "hard": 0.05}},
    {"step": 1500, "weights": {"easy": 0.50, "medium": 0.35, "hard": 0.15}},
    {"step": 3500, "weights": {"easy": 0.35, "medium": 0.40, "hard": 0.25}},
    {"step": 6000, "weights": {"easy": 0.20, "medium": 0.35, "hard": 0.45}},
    {"step": 8000, "weights": {"easy": 0.15, "medium": 0.30, "hard": 0.55}},
)
VALIDATION_ROOT = PROJECT_ROOT / "data/coco_aggressive_49_validation"


def atomic_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def ranked_images(directory: Path, seed: int) -> list[Path]:
    images = sorted(directory.glob("*.jpg"))
    return sorted(images, key=lambda path: stable_int(path.name, seed))


def split_images() -> dict[str, list[Path]]:
    training_pool = ranked_images(COCO_ROOT / "train2017", SEED)
    test_pool = ranked_images(COCO_ROOT / "val2017", SEED)
    required = TRAIN_IDENTITIES + VALIDATION_IDENTITIES
    if len(training_pool) < required:
        raise RuntimeError(f"COCO train2017 has {len(training_pool)} images, expected at least {required}")
    if len(test_pool) < TEST_IDENTITIES:
        raise RuntimeError(f"COCO val2017 has {len(test_pool)} images, expected at least {TEST_IDENTITIES}")
    return {
        "train": training_pool[:TRAIN_IDENTITIES],
        "validation": training_pool[TRAIN_IDENTITIES:required],
        "test": test_pool[:TEST_IDENTITIES],
    }


def relative_path(path: Path) -> str:
    return path.relative_to(COCO_ROOT).as_posix()


def factor_for(path: Path, tier: str) -> int:
    factors = TIER_FACTORS[tier]
    return factors[stable_int(f"{path.name}:{tier}", SEED) % len(factors)]


def record_for(path: Path, split: str, tier: str, digest: str) -> dict:
    factor = factor_for(path, tier)
    seed = stable_int(f"coco-aggressive-49:{split}:{path.name}:{tier}", SEED) % (2**31)
    sample_id = hashlib.sha256(
        f"coco-aggressive-49:{split}:{path.name}:{tier}".encode("utf-8")
    ).hexdigest()[:20]
    return RestorationRecord(
        sample_id=sample_id,
        split=split,
        source_root="coco2017",
        source_relpath=relative_path(path),
        source_sha256=digest,
        prompt=PROMPT,
        trajectory="progressive",
        degradation=DegradationSpec("pixelation", None, factor, seed),
        clean_kind="natural",
        render={
            "curriculum_tier": tier,
            "pixel_factor": factor,
            "crop": "resize_short_side_then_center_512",
            "pixel_frame_count": FRAME_COUNT,
        },
    ).to_dict()


def save_validation_media(path: Path) -> None:
    clean = Image.open(path).convert("RGB")
    rows = []
    selected_indices = (0, 12, 24, 36, 48)
    for tier in TIERS:
        factor = max(TIER_FACTORS[tier])
        spec = DegradationSpec("pixelation", None, factor, stable_int(tier, SEED) % (2**31))
        frames = build_frames(clean, spec, "progressive", FRAME_COUNT)
        atomic_png(VALIDATION_ROOT / f"{tier}.png", frames[0])
        row = Image.new("RGB", (512 * len(selected_indices), 512))
        for column, frame_index in enumerate(selected_indices):
            row.paste(frames[frame_index], (column * 512, 0))
        rows.append(row)
    atomic_png(VALIDATION_ROOT / "clean.png", build_frames(clean, spec, "progressive", FRAME_COUNT)[-1])
    sheet = Image.new("RGB", (512 * len(selected_indices), 512 * len(rows)))
    for row_index, row in enumerate(rows):
        sheet.paste(row, (0, row_index * 512))
    atomic_png(VALIDATION_ROOT / "curriculum_contact_sheet.png", sheet)


def main() -> None:
    splits = split_images()
    selected_paths = [path for paths in splits.values() for path in paths]
    digests = {path: sha256_file(path) for path in sorted(selected_paths)}
    destination = MANIFEST_ROOT / DATASET_NAME
    curriculum_tiers = {tier: [] for tier in TIERS}
    sample_counts = {}
    for split, paths in splits.items():
        rows = []
        for index, path in enumerate(paths):
            tiers = (TIERS[index % len(TIERS)],) if split == "train" else TIERS
            for tier in tiers:
                row = record_for(path, split, tier, digests[path])
                rows.append(row)
                if split == "train":
                    curriculum_tiers[tier].append(row["sample_id"])
        sample_counts[split] = atomic_jsonl(destination / f"{split}.jsonl", rows)
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
            "source_root": "coco2017",
            "source_files_found": {
                "train2017": len(list((COCO_ROOT / "train2017").glob("*.jpg"))),
                "val2017": len(list((COCO_ROOT / "val2017").glob("*.jpg"))),
            },
            "selected_identities": {split: len(paths) for split, paths in splits.items()},
            "samples": sample_counts,
            "train_tier_counts": {tier: len(values) for tier, values in curriculum_tiers.items()},
            "tiers": {tier: list(factors) for tier, factors in TIER_FACTORS.items()},
            "pixel_frame_count": FRAME_COUNT,
            "latent_frame_count": 1 + (FRAME_COUNT - 1) // 8,
            "precomputed_curriculum_path": str(precomputed_train / "curriculum.json"),
        },
    )
    save_validation_media(splits["validation"][0])


if __name__ == "__main__":
    main()
