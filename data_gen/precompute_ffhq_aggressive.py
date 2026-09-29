import json

import data_gen.precompute_ltx as precompute
from project_config import MANIFEST_ROOT, PRECOMPUTED_ROOT
from restoration.io import atomic_json
from telegram_training_helper import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TelegramBot

DATASET_NAME = "ffhq_pixel_aggressive"
SPLIT = "train"
PHASE = "both"
LIMIT = None
OVERWRITE = False
TELEGRAM_ENABLED = True
TELEGRAM_INTERVAL = 500


def main() -> None:
    if TELEGRAM_ENABLED and (not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID):
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
    bot = (
        TelegramBot(auto_start=True, poll_updates=False, enable_terminal_commands=False)
        if TELEGRAM_ENABLED
        else None
    )

    def progress(phase: str, complete: int, total: int) -> None:
        if bot is not None and (complete % TELEGRAM_INTERVAL == 0 or complete == total):
            bot.send_message(
                f"Aggressive FFHQ precompute {phase}: {complete}/{total} ({100.0 * complete / total:.1f}%)",
                wait=False,
            )

    curriculum_path = MANIFEST_ROOT / DATASET_NAME / "curriculum.json"
    curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
    destination = PRECOMPUTED_ROOT / DATASET_NAME / SPLIT
    atomic_json(destination / "curriculum.json", curriculum)
    precompute.DATASET_NAME = DATASET_NAME
    precompute.SPLIT = SPLIT
    precompute.PHASE = PHASE
    precompute.LIMIT = LIMIT
    precompute.OVERWRITE = OVERWRITE
    precompute.RETAIN_RGB = False
    precompute.PROGRESS_CALLBACK = progress
    if bot is not None:
        bot.send_message("Aggressive FFHQ LTX precomputation started", wait=True)
    try:
        precompute.main()
        if bot is not None:
            bot.send_message("Aggressive FFHQ LTX precomputation completed", wait=True)
    finally:
        if bot is not None:
            bot.stop(wait=True)


if __name__ == "__main__":
    main()
