import data_gen.precompute_ffhq_full_restoration as job

LIMIT = 20


def main() -> None:
    job.LIMIT = LIMIT
    job.TELEGRAM_ENABLED = False
    job.main()


if __name__ == "__main__":
    main()
