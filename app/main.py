import argparse
import asyncio

from aiogram import Bot, Dispatcher

from .config import TELEGRAM_BOT_TOKEN, logger, validate_settings
from .handlers import router
from .hints import test_fusionbrain_api


async def run_bot(skip_api_check: bool = False) -> None:
    validate_settings()
    if not skip_api_check:
        await test_fusionbrain_api()
    bot = Bot(TELEGRAM_BOT_TOKEN)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    logger.info("Bot is starting…")
    await dispatcher.start_polling(bot)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quiz bot entrypoint")
    parser.add_argument(
        "--skip-fusionbrain-check",
        action="store_true",
        help="Skip probing FusionBrain API on startup",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_bot(skip_api_check=args.skip_fusionbrain_check))
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped")


if __name__ == "__main__":
    main()
