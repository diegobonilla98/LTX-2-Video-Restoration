import os

from telegram_training_helper import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TelegramBot

MESSAGE = os.environ.get("VTIR_TELEGRAM_MESSAGE", "Video restoration status update")


def main() -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
    bot = TelegramBot(
        auto_start=True,
        poll_updates=False,
        enable_terminal_commands=False,
    )
    bot.send_message(MESSAGE, wait=True)
    bot.stop(wait=True)


if __name__ == "__main__":
    main()
