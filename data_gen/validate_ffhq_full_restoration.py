import json
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from data_gen.build_ffhq_full_restoration_dataset import (
    DATASET_NAME,
    DISTANCE_UNITS,
    FAMILIES,
    FRAME_COUNTS,
    STRENGTHS,
    TEST_IDENTITIES,
    TIERS,
    TRAIN_IDENTITIES,
    VALIDATION_IDENTITIES,
)
from data_gen.precompute_ltx import record_spec
from data_gen.restoration_trajectory import build_restoration_frames
from data_gen.trajectory import center_crop_512, cosine_severity
from project_config import FFHQ_ROOT, MANIFEST_ROOT
from restoration.io import atomic_json, read_jsonl

OUTPUT_PATH = MANIFEST_ROOT / DATASET_NAME / "validation.json"


def mse(image: Image.Image, clean: Image.Image) -> float:
    left = np.asarray(image, dtype=np.float32) / 255.0
    right = np.asarray(clean, dtype=np.float32) / 255.0
    return float(np.mean((left - right) ** 2))


def load_splits() -> dict[str, list[dict]]:
    return {
        split: list(read_jsonl(MANIFEST_ROOT / DATASET_NAME / f"{split}.jsonl"))
        for split in ("train", "validation", "test")
    }


def trajectory_checks(records: list[dict]) -> tuple[dict[str, bool], dict[str, dict]]:
    representatives = {}
    for record in records:
        key = (record["render"]["degradation_family"], record["render"]["curriculum_tier"])
        representatives.setdefault(key, record)
    diagnostics = {}
    endpoints = True
    monotonic = True
    changed = True
    dimensions = True
    for family in FAMILIES:
        for tier in TIERS:
            record = representatives[(family, tier)]
            clean = center_crop_512(Image.open(FFHQ_ROOT / record["source_relpath"]))
            frames = build_restoration_frames(clean, record_spec(record), record["render"]["frame_count"])
            errors = [mse(frame, clean) for frame in frames]
            key = f"{family}:{tier}"
            diagnostics[key] = {
                "frames": len(frames),
                "initial_mse": errors[0],
                "final_mse": errors[-1],
                "largest_mse_increase": max(
                    [errors[index + 1] - errors[index] for index in range(len(errors) - 1)],
                    default=0.0,
                ),
                "distinct_mse_levels": len({round(value, 9) for value in errors}),
            }
            endpoints = endpoints and np.array_equal(np.asarray(frames[-1]), np.asarray(clean))
            monotonic = monotonic and all(
                errors[index + 1] <= errors[index] + 1e-7 for index in range(len(errors) - 1)
            )
            changed = changed and errors[0] > 0 and diagnostics[key]["distinct_mse_levels"] >= 3
            dimensions = dimensions and all(frame.size == (512, 512) for frame in frames)
    return {
        "trajectory_endpoints": endpoints,
        "trajectory_mse_monotonic": monotonic,
        "trajectory_changes": changed,
        "trajectory_dimensions": dimensions,
    }, diagnostics


def main() -> None:
    splits = load_splits()
    expected_counts = {
        "train": TRAIN_IDENTITIES,
        "validation": VALIDATION_IDENTITIES * len(FAMILIES) * len(TIERS),
        "test": TEST_IDENTITIES * len(FAMILIES) * len(TIERS),
    }
    identities = {
        split: {record["source_relpath"] for record in records}
        for split, records in splits.items()
    }
    train_cells = Counter(
        (record["render"]["degradation_family"], record["render"]["curriculum_tier"])
        for record in splits["train"]
    )
    validation_cells = defaultdict(set)
    for record in splits["validation"]:
        validation_cells[record["source_relpath"]].add(
            (record["render"]["degradation_family"], record["render"]["curriculum_tier"])
        )
    expected_cells = {(family, tier) for family in FAMILIES for tier in TIERS}
    all_records = [record for records in splits.values() for record in records]
    frame_metadata = all(
        record["render"]["frame_count"] == FRAME_COUNTS[record["render"]["curriculum_tier"]]
        and record["render"]["latent_frame_count"] == 1 + (record["render"]["frame_count"] - 1) // 8
        and record["render"]["restoration_distance_units"] == DISTANCE_UNITS[record["render"]["curriculum_tier"]]
        and record["render"]["latent_frame_count"] - 1 == record["render"]["restoration_distance_units"]
        for record in all_records
    )
    normalized_api = all(
        cosine_severity(0, count) == 1.0 and cosine_severity(count - 1, count) == 0.0
        for count in FRAME_COUNTS.values()
    )
    trajectory_gates, diagnostics = trajectory_checks(splits["validation"])
    gates = {
        "exact_counts": {split: len(records) for split, records in splits.items()} == expected_counts,
        "unique_sample_ids": len({record["sample_id"] for record in all_records}) == len(all_records),
        "identity_split": identities["train"].isdisjoint(identities["validation"])
        and identities["train"].isdisjoint(identities["test"])
        and identities["validation"].isdisjoint(identities["test"]),
        "train_family_tier_balance": set(train_cells) == expected_cells
        and max(train_cells.values()) - min(train_cells.values()) <= 1,
        "validation_coverage": len(validation_cells) == VALIDATION_IDENTITIES
        and all(cells == expected_cells for cells in validation_cells.values()),
        "strengths_complete": set(STRENGTHS) == set(FAMILIES)
        and all(set(STRENGTHS[family]) == set(TIERS) for family in FAMILIES),
        "frame_metadata": frame_metadata,
        "fixed_endpoint_homotopy": all(
            record["render"].get("trajectory_builder") == "fixed_endpoint_homotopy_v1"
            for record in all_records
        ),
        "normalized_severity_api": normalized_api,
        **trajectory_gates,
    }
    payload = {
        "passed": all(gates.values()),
        "gates": gates,
        "counts": {split: len(records) for split, records in splits.items()},
        "identities": {split: len(values) for split, values in identities.items()},
        "train_cells": {f"{family}:{tier}": count for (family, tier), count in sorted(train_cells.items())},
        "trajectory_diagnostics": diagnostics,
    }
    atomic_json(OUTPUT_PATH, payload)
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise RuntimeError("FFHQ full restoration manifest validation failed")


if __name__ == "__main__":
    main()
