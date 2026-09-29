import hashlib
import os
from pathlib import Path

from PIL import Image, ImageOps

from data_gen.trajectory import build_frames
from project_config import FFHQ_PROMPT, FFHQ_ROOT, MANIFEST_ROOT, PRECOMPUTED_ROOT, PROJECT_ROOT
from restoration.io import atomic_json, atomic_jsonl, sha256_file, stable_int
from restoration.schema import DegradationSpec, RestorationRecord

SEED = 20260830
DATASET_NAME = "ffhq_pixel_curriculum"
TRAIN_IDENTITIES = 4096
VALIDATION_IDENTITIES = 128
TEST_IDENTITIES = 512
TOTAL_STEPS = 6000
TIER_FACTORS = {
    "easy": (2, 3, 4),
    "medium": (4, 6, 8),
    "hard": (8, 12, 16),
}
CURRICULUM_ANCHORS = (
    {"step": 1, "weights": {"easy": 1.00, "medium": 0.00, "hard": 0.00}},
    {"step": 1000, "weights": {"easy": 0.65, "medium": 0.30, "hard": 0.05}},
    {"step": 2500, "weights": {"easy": 0.40, "medium": 0.45, "hard": 0.15}},
    {"step": 4000, "weights": {"easy": 0.25, "medium": 0.40, "hard": 0.35}},
    {"step": 6000, "weights": {"easy": 0.15, "medium": 0.30, "hard": 0.55}},
)
VALIDATION_ROOT = PROJECT_ROOT / "data/ffhq_validation"


def atomic_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def ranked_images() -> list[Path]:
    images = sorted(FFHQ_ROOT.glob("*.png"))
    return sorted(images, key=lambda path: stable_int(path.name, SEED))


def split_images(images: list[Path]) -> dict[str, list[Path]]:
    required = TRAIN_IDENTITIES + VALIDATION_IDENTITIES + TEST_IDENTITIES
    if len(images) < required:
        raise RuntimeError(f"FFHQ has {len(images)} PNG files, expected at least {required}")
    train_end = TRAIN_IDENTITIES
    validation_end = train_end + VALIDATION_IDENTITIES
    test_end = validation_end + TEST_IDENTITIES
    return {
        "train": images[:train_end],
        "validation": images[train_end:validation_end],
        "test": images[validation_end:test_end],
    }


def factor_for(path: Path, tier: str) -> int:
    factors = TIER_FACTORS[tier]
    return factors[stable_int(f"{path.name}:{tier}", SEED) % len(factors)]


def record_for(path: Path, split: str, tier: str, digest: str) -> dict:
    factor = factor_for(path, tier)
    seed = stable_int(f"ffhq:{split}:{path.name}:{tier}", SEED) % (2**31)
    sample_id = hashlib.sha256(f"ffhq:{split}:{path.name}:{tier}".encode("utf-8")).hexdigest()[:20]
    record = RestorationRecord(
        sample_id=sample_id,
        split=split,
        source_root="ffhq",
        source_relpath=path.name,
        source_sha256=digest,
        prompt=FFHQ_PROMPT,
        trajectory="progressive",
        degradation=DegradationSpec(kind="pixelation", sigma0=None, pixel_factor0=factor, seed=seed),
        clean_kind="natural",
        render={
            "curriculum_tier": tier,
            "pixel_factor": factor,
            "source_size": [512, 512],
            "crop": "identity",
        },
    )
    return record.to_dict()


def save_validation_media(validation_paths: list[Path]) -> None:
    clean = Image.open(validation_paths[0]).convert("RGB")
    rows = []
    for tier in TIER_FACTORS:
        factor = max(TIER_FACTORS[tier])
        spec = DegradationSpec("pixelation", None, factor, stable_int(tier, SEED) % (2**31))
        frames = build_frames(clean, spec, "progressive")
        atomic_png(VALIDATION_ROOT / f"{tier}.png", frames[0])
        selected = [frames[index] for index in (0, 6, 12, 18, 24)]
        rows.append(ImageOps.expand(Image.new("RGB", (512 * 5, 512)), border=0))
        for index, frame in enumerate(selected):
            rows[-1].paste(frame, (index * 512, 0))
    atomic_png(VALIDATION_ROOT / "clean.png", clean)
    sheet = Image.new("RGB", (512 * 5, 512 * len(rows)))
    for index, row in enumerate(rows):
        sheet.paste(row, (0, index * 512))
    atomic_png(VALIDATION_ROOT / "curriculum_contact_sheet.png", sheet)


def main() -> None:
    images = ranked_images()
    splits = split_images(images)
    selected = [path for paths in splits.values() for path in paths]
    digests = {path: sha256_file(path) for path in selected}
    destination = MANIFEST_ROOT / DATASET_NAME
    curriculum_tiers = {tier: [] for tier in TIER_FACTORS}
    counts = {}
    for split, paths in splits.items():
        rows = []
        for path in paths:
            for tier in TIER_FACTORS:
                row = record_for(path, split, tier, digests[path])
                rows.append(row)
                if split == "train":
                    curriculum_tiers[tier].append(row["sample_id"])
        counts[split] = atomic_jsonl(destination / f"{split}.jsonl", rows)
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
            "tiers": {name: list(factors) for name, factors in TIER_FACTORS.items()},
            "unused_source_images": len(images) - len(selected),
            "precomputed_curriculum_path": str(
                PRECOMPUTED_ROOT / DATASET_NAME / "train/curriculum.json"
            ),
        },
    )
    save_validation_media(splits["validation"])


if __name__ == "__main__":
    main()
