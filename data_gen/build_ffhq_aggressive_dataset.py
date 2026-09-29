import data_gen.build_ffhq_dataset as builder
from project_config import PROJECT_ROOT

DATASET_NAME = "ffhq_pixel_aggressive"
TOTAL_STEPS = 5000
TIER_FACTORS = {
    "easy": (8, 12, 16),
    "medium": (12, 16, 24),
    "hard": (16, 24, 32),
}
CURRICULUM_ANCHORS = (
    {"step": 1, "weights": {"easy": 0.70, "medium": 0.25, "hard": 0.05}},
    {"step": 1000, "weights": {"easy": 0.45, "medium": 0.40, "hard": 0.15}},
    {"step": 2500, "weights": {"easy": 0.30, "medium": 0.40, "hard": 0.30}},
    {"step": 4000, "weights": {"easy": 0.20, "medium": 0.35, "hard": 0.45}},
    {"step": 5000, "weights": {"easy": 0.15, "medium": 0.30, "hard": 0.55}},
)
VALIDATION_ROOT = PROJECT_ROOT / "data/ffhq_aggressive_validation"


def main() -> None:
    builder.DATASET_NAME = DATASET_NAME
    builder.TOTAL_STEPS = TOTAL_STEPS
    builder.TIER_FACTORS = TIER_FACTORS
    builder.CURRICULUM_ANCHORS = CURRICULUM_ANCHORS
    builder.VALIDATION_ROOT = VALIDATION_ROOT
    builder.main()


if __name__ == "__main__":
    main()
