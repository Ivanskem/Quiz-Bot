import logging
import os
from pathlib import Path
from typing import Iterable


def _parse_env_line(line: str) -> tuple[str, str] | None:
    if not line or line.startswith("#"):
        return None
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip().strip('"').strip("'")
    return key, value


def load_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(raw_line.strip())
            if not parsed:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)
    except OSError as exc:  # noqa: BLE001
        logging.getLogger("quizbot").warning("Не удалось прочитать .env: %s", exc)


load_env()

TELEGRAM_BOT_TOKEN = os.getenv("BOT_TOKEN", "")
FUSIONBRAIN_API_KEY = os.getenv("FBRAIN_API_KEY", "")
FUSIONBRAIN_SECRET = os.getenv("FBRAIN_SECRET", "")
FB_CONCURRENCY = int(os.getenv("FB_CONCURRENCY", "1") or 1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quizbot")


def validate_settings() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Не указан TELEGRAM_BOT_TOKEN")
    if not (FUSIONBRAIN_API_KEY and FUSIONBRAIN_SECRET):
        raise RuntimeError("Не заданы ключ и секрет FusionBrain API — проверьте переменные окружения.")
