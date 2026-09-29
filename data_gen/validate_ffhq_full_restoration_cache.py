import json
from collections import defaultdict
from pathlib import Path

import torch

from data_gen.build_ffhq_full_restoration_dataset import DATASET_NAME, FAMILIES, TIERS, TRAIN_IDENTITIES
from project_config import MANIFEST_ROOT, PRECOMPUTED_ROOT
from restoration.io import atomic_json, read_jsonl

CACHE_ROOT = PRECOMPUTED_ROOT / DATASET_NAME / "train/.precomputed"
OUTPUT_PATH = PRECOMPUTED_ROOT / DATASET_NAME / "validation.json"


def main() -> None:
    records = list(read_jsonl(MANIFEST_ROOT / DATASET_NAME / "train.jsonl"))
    expected_ids = {record["sample_id"] for record in records}
    latent_files = {path.stem: path for path in (CACHE_ROOT / "latents").glob("*.pt")}
    condition_files = {path.stem: path for path in (CACHE_ROOT / "conditions").glob("*.pt")}
    representatives = {}
    for record in records:
        key = (record["render"]["degradation_family"], record["render"]["curriculum_tier"])
        representatives.setdefault(key, record)
    shapes = defaultdict(set)
    payload_integrity = True
    for (family, tier), record in representatives.items():
        latent = torch.load(latent_files[record["sample_id"]], map_location="cpu", weights_only=True)
        condition = torch.load(condition_files[record["sample_id"]], map_location="cpu", weights_only=True)
        shape = tuple(latent["latents"].shape)
        shapes[tier].add(shape)
        expected_time = record["render"]["latent_frame_count"]
        payload_integrity = payload_integrity and shape == (128, expected_time, 16, 16)
        payload_integrity = payload_integrity and latent["num_frames"] == expected_time
        payload_integrity = payload_integrity and condition["video_prompt_embeds"].numel() > 0
        payload_integrity = payload_integrity and condition["prompt_attention_mask"].numel() > 0
    curriculum = json.loads((CACHE_ROOT.parent / "curriculum.json").read_text(encoding="utf-8"))
    curriculum_ids = {
        sample_id
        for members in curriculum["tiers"].values()
        for sample_id in members
    }
    gates = {
        "manifest_count": len(records) == TRAIN_IDENTITIES,
        "latent_ids": set(latent_files) == expected_ids,
        "condition_ids": set(condition_files) == expected_ids,
        "nonempty_files": all(path.stat().st_size > 0 for path in (*latent_files.values(), *condition_files.values())),
        "curriculum_ids": curriculum_ids == expected_ids,
        "family_tier_representatives": set(representatives) == {(family, tier) for family in FAMILIES for tier in TIERS},
        "payload_integrity": payload_integrity,
        "adaptive_shapes": {tier: values for tier, values in shapes.items()} == {
            "easy": {(128, 3, 16, 16)},
            "medium": {(128, 4, 16, 16)},
            "hard": {(128, 5, 16, 16)},
            "extreme": {(128, 7, 16, 16)},
        },
    }
    payload = {
        "passed": all(gates.values()),
        "gates": gates,
        "precomputed": {"latents": len(latent_files), "conditions": len(condition_files)},
        "representative_shapes": {tier: [list(shape) for shape in sorted(values)] for tier, values in shapes.items()},
    }
    atomic_json(OUTPUT_PATH, payload)
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise RuntimeError("FFHQ full restoration cache validation failed")


if __name__ == "__main__":
    main()
