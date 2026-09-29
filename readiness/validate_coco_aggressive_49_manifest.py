import data_gen.validate_coco_aggressive_49 as validator


def main() -> None:
    validator.REQUIRE_PRECOMPUTED = False
    validator.main()


if __name__ == "__main__":
    main()
