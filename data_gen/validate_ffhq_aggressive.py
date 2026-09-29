import data_gen.validate_ffhq as validator

DATASET_NAME = "ffhq_pixel_aggressive"
EXPECTED_TIERS = {
    "easy": {8, 12, 16},
    "medium": {12, 16, 24},
    "hard": {16, 24, 32},
}
REQUIRE_PRECOMPUTED = True


def main() -> None:
    validator.DATASET_NAME = DATASET_NAME
    validator.EXPECTED_TIERS = EXPECTED_TIERS
    validator.REQUIRE_PRECOMPUTED = REQUIRE_PRECOMPUTED
    validator.main()


if __name__ == "__main__":
    main()
