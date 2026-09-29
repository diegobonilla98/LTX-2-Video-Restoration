import json
from collections import Counter, defaultdict

import torch
from PIL import Image

from data_gen.precompute_ltx import load_clean, record_spec
from data_gen.trajectory import build_frames, cosine_severity
from project_config import FFHQ_ROOT, MANIFEST_ROOT, PRECOMPUTED_ROOT
from restoration.io import atomic_json, read_jsonl

DATASET_NAME = "ffhq_pixel_curriculum"
EXPECTED_IDENTITIES = {"train": 4096, "validation": 128, "test": 512}
EXPECTED_TIERS = {
    "easy": {2, 3, 4},
    "medium": {4, 6, 8},
    "hard": {8, 12, 16},
}
REQUIRE_PRECOMPUTED = True


def main() -> None:
    root = MANIFEST_ROOT / DATASET_NAME
    split_rows = {split: list(read_jsonl(root / f"{split}.jsonl")) for split in EXPECTED_IDENTITIES}
    errors = []
    source_sets = {}
    for split, rows in split_rows.items():
        expected_samples = EXPECTED_IDENTITIES[split] * len(EXPECTED_TIERS)
        if len(rows) != expected_samples:
            errors.append(f"{split} has {len(rows)} samples, expected {expected_samples}")
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["source_relpath"]].append(row)
            path = FFHQ_ROOT / row["source_relpath"]
            if not path.is_file():
                errors.append(f"missing source {path}")
                continue
            with Image.open(path) as image:
                if image.mode != "RGB" or image.size != (512, 512):
                    errors.append(f"invalid FFHQ image {path}: {image.mode} {image.size}")
            tier = row["render"]["curriculum_tier"]
            factor = row["degradation"]["pixel_factor0"]
            if tier not in EXPECTED_TIERS or factor not in EXPECTED_TIERS[tier]:
                errors.append(f"invalid tier/factor for {row['sample_id']}: {tier}/{factor}")
            if row["degradation"]["kind"] != "pixelation" or row["trajectory"] != "progressive":
                errors.append(f"invalid degradation or trajectory for {row['sample_id']}")
        for relpath, identity_rows in grouped.items():
            tiers = Counter(row["render"]["curriculum_tier"] for row in identity_rows)
            if tiers != Counter({tier: 1 for tier in EXPECTED_TIERS}):
                errors.append(f"identity {relpath} does not have one sample per tier")
            if len({row["source_sha256"] for row in identity_rows}) != 1:
                errors.append(f"identity {relpath} has inconsistent source hashes")
        source_sets[split] = set(grouped)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = source_sets[left] & source_sets[right]
        if overlap:
            errors.append(f"{left}/{right} identity leakage: {len(overlap)}")
    curriculum = json.loads((root / "curriculum.json").read_text(encoding="utf-8"))
    hard_weights = []
    easy_weights = []
    for anchor in curriculum["anchors"]:
        weights = anchor["weights"]
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            errors.append(f"weights do not sum to one at step {anchor['step']}")
        hard_weights.append(weights["hard"])
        easy_weights.append(weights["easy"])
    if hard_weights != sorted(hard_weights):
        errors.append("hard curriculum weight is not monotonic")
    if easy_weights != sorted(easy_weights, reverse=True):
        errors.append("easy curriculum weight is not monotonic decreasing")
    severities = [cosine_severity(index) for index in range(25)]
    if any(right > left for left, right in zip(severities, severities[1:])) or severities[-1] != 0.0:
        errors.append("trajectory severity is not monotonic with a clean endpoint")
    for tier in EXPECTED_TIERS:
        row = next(
            row for row in split_rows["validation"] if row["render"]["curriculum_tier"] == tier
        )
        clean = load_clean(row)
        frames = build_frames(clean, record_spec(row), row["trajectory"])
        if frames[-1].tobytes() != clean.tobytes():
            errors.append(f"{tier} trajectory clean endpoint mismatch")
        if frames[0].tobytes() == clean.tobytes():
            errors.append(f"{tier} degraded endpoint equals clean input")
    precomputed = PRECOMPUTED_ROOT / DATASET_NAME / "train/.precomputed"
    precomputed_counts = {}
    if REQUIRE_PRECOMPUTED:
        for directory in ("latents", "conditions"):
            paths = sorted((precomputed / directory).glob("*.pt"))
            precomputed_counts[directory] = len(paths)
            if len(paths) != EXPECTED_IDENTITIES["train"] * len(EXPECTED_TIERS):
                errors.append(f"precomputed {directory} count is {len(paths)}")
        condition_paths = sorted((precomputed / "conditions").glob("*.pt"))
        if condition_paths and len({path.stat().st_ino for path in condition_paths}) != 1:
            errors.append("identical FFHQ prompts were not hardlink-deduplicated")
        for tier, sample_ids in curriculum["tiers"].items():
            path = precomputed / "latents" / f"{sample_ids[0]}.pt"
            if path.is_file():
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if tuple(payload["latents"].shape) != (128, 4, 16, 16):
                    errors.append(f"invalid {tier} latent shape {tuple(payload['latents'].shape)}")
    payload = {
        "dataset": DATASET_NAME,
        "identities": {split: len(values) for split, values in source_sets.items()},
        "samples": {split: len(rows) for split, rows in split_rows.items()},
        "precomputed": precomputed_counts,
        "errors": errors,
        "passed": not errors,
    }
    atomic_json(PRECOMPUTED_ROOT / DATASET_NAME / "validation.json", payload)
    print(json.dumps(payload, indent=2))
    if errors:
        raise RuntimeError(f"FFHQ validation failed with {len(errors)} errors")


if __name__ == "__main__":
    main()
