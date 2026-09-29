import json
from collections import Counter, defaultdict

import torch
from PIL import Image

from data_gen.precompute_ltx import load_clean, record_spec
from data_gen.trajectory import build_frames, cosine_severity
from project_config import COCO_ROOT, MANIFEST_ROOT, PRECOMPUTED_ROOT, PROJECT_ROOT
from restoration.io import atomic_json, read_jsonl


DATASET_NAME = "coco_pixel_aggressive_49"
FRAME_COUNT = 49
EXPECTED_IDENTITIES = {"train": 24576, "validation": 512, "test": 5000}
EXPECTED_SAMPLES = {"train": 24576, "validation": 1536, "test": 15000}
EXPECTED_TIERS = {
    "easy": {8, 12, 16},
    "medium": {12, 16, 24},
    "hard": {16, 24, 32},
}
REQUIRE_PRECOMPUTED = True


def main() -> None:
    root = MANIFEST_ROOT / DATASET_NAME
    split_rows = {split: list(read_jsonl(root / f"{split}.jsonl")) for split in EXPECTED_IDENTITIES}
    errors = []
    source_sets = {}
    tier_counts = {}
    for split, rows in split_rows.items():
        if len(rows) != EXPECTED_SAMPLES[split]:
            errors.append(f"{split} has {len(rows)} samples, expected {EXPECTED_SAMPLES[split]}")
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["source_relpath"]].append(row)
            tier = row["render"]["curriculum_tier"]
            factor = row["degradation"]["pixel_factor0"]
            if tier not in EXPECTED_TIERS or factor not in EXPECTED_TIERS[tier]:
                errors.append(f"invalid tier/factor for {row['sample_id']}: {tier}/{factor}")
            if row["degradation"]["kind"] != "pixelation" or row["trajectory"] != "progressive":
                errors.append(f"invalid degradation or trajectory for {row['sample_id']}")
            if row["render"].get("pixel_frame_count") != FRAME_COUNT:
                errors.append(f"invalid frame count for {row['sample_id']}")
        if len(grouped) != EXPECTED_IDENTITIES[split]:
            errors.append(f"{split} has {len(grouped)} identities, expected {EXPECTED_IDENTITIES[split]}")
        for relpath, identity_rows in grouped.items():
            path = COCO_ROOT / relpath
            if not path.is_file():
                errors.append(f"missing source {path}")
                continue
            with Image.open(path) as image:
                if image.width < 1 or image.height < 1:
                    errors.append(f"invalid COCO dimensions {path}: {image.size}")
            tiers = Counter(row["render"]["curriculum_tier"] for row in identity_rows)
            expected = Counter({identity_rows[0]["render"]["curriculum_tier"]: 1}) if split == "train" else Counter({tier: 1 for tier in EXPECTED_TIERS})
            if tiers != expected:
                errors.append(f"identity {relpath} has invalid tier coverage: {dict(tiers)}")
            if len({row["source_sha256"] for row in identity_rows}) != 1:
                errors.append(f"identity {relpath} has inconsistent source hashes")
        source_sets[split] = set(grouped)
        tier_counts[split] = dict(Counter(row["render"]["curriculum_tier"] for row in rows))
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = source_sets[left] & source_sets[right]
        if overlap:
            errors.append(f"{left}/{right} identity leakage: {len(overlap)}")
    expected_train_per_tier = EXPECTED_IDENTITIES["train"] // len(EXPECTED_TIERS)
    if tier_counts["train"] != {tier: expected_train_per_tier for tier in EXPECTED_TIERS}:
        errors.append(f"unbalanced training tiers: {tier_counts['train']}")
    curriculum = json.loads((root / "curriculum.json").read_text(encoding="utf-8"))
    if any(len(curriculum["tiers"][tier]) != expected_train_per_tier for tier in EXPECTED_TIERS):
        errors.append("curriculum tier pools do not match balanced training tiers")
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
    severities = [cosine_severity(index, FRAME_COUNT) for index in range(FRAME_COUNT)]
    if any(right > left for left, right in zip(severities, severities[1:])) or severities[-1] != 0.0:
        errors.append("49-frame trajectory severity is not monotonic with a clean endpoint")
    for tier in EXPECTED_TIERS:
        row = next(row for row in split_rows["validation"] if row["render"]["curriculum_tier"] == tier)
        clean = load_clean(row)
        frames = build_frames(clean, record_spec(row), row["trajectory"], FRAME_COUNT)
        if len(frames) != FRAME_COUNT:
            errors.append(f"{tier} trajectory does not contain {FRAME_COUNT} frames")
        if frames[-1].tobytes() != clean.tobytes():
            errors.append(f"{tier} trajectory clean endpoint mismatch")
        if frames[0].tobytes() == clean.tobytes():
            errors.append(f"{tier} degraded endpoint equals clean input")
    contact_sheet = PROJECT_ROOT / "data/coco_aggressive_49_validation/curriculum_contact_sheet.png"
    if not contact_sheet.is_file():
        errors.append("curriculum contact sheet is missing")
    precomputed = PRECOMPUTED_ROOT / DATASET_NAME / "train/.precomputed"
    precomputed_counts = {}
    if REQUIRE_PRECOMPUTED:
        for directory in ("latents", "conditions"):
            paths = sorted((precomputed / directory).glob("*.pt"))
            precomputed_counts[directory] = len(paths)
            if len(paths) != EXPECTED_SAMPLES["train"]:
                errors.append(f"precomputed {directory} count is {len(paths)}, expected {EXPECTED_SAMPLES['train']}")
        condition_paths = sorted((precomputed / "conditions").glob("*.pt"))
        if condition_paths and len({path.stat().st_ino for path in condition_paths}) != 1:
            errors.append("identical COCO prompts were not hardlink-deduplicated")
        for tier, sample_ids in curriculum["tiers"].items():
            path = precomputed / "latents" / f"{sample_ids[0]}.pt"
            if path.is_file():
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if tuple(payload["latents"].shape) != (128, 7, 16, 16):
                    errors.append(f"invalid {tier} latent shape {tuple(payload['latents'].shape)}")
                if payload.get("num_frames") != 7:
                    errors.append(f"invalid {tier} latent frame metadata")
    payload = {
        "dataset": DATASET_NAME,
        "pixel_frames": FRAME_COUNT,
        "latent_frames": 7,
        "identities": {split: len(values) for split, values in source_sets.items()},
        "samples": {split: len(rows) for split, rows in split_rows.items()},
        "tier_counts": tier_counts,
        "precomputed": precomputed_counts,
        "errors": errors,
        "passed": not errors,
    }
    atomic_json(PRECOMPUTED_ROOT / DATASET_NAME / "validation.json", payload)
    print(json.dumps(payload, indent=2))
    if errors:
        raise RuntimeError(f"COCO aggressive 49-frame validation failed with {len(errors)} errors")


if __name__ == "__main__":
    main()
