import data_gen.precompute_coco_aggressive_49 as job


LIMIT = 3


def main() -> None:
    job.LIMIT = LIMIT
    job.TELEGRAM_ENABLED = False
    job.main()


if __name__ == "__main__":
    main()
