import data_gen.precompute_ffhq_aggressive as job

LIMIT = 3


def main() -> None:
    job.LIMIT = LIMIT
    job.TELEGRAM_ENABLED = False
    job.main()


if __name__ == "__main__":
    main()
