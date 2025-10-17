import re
from difflib import SequenceMatcher


def normalize_answer(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\s\-а-яa-z0-9]", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value)
    return value


def is_correct(user_answer: str, gold_answer: str, threshold: float = 0.82) -> bool:
    user = normalize_answer(user_answer)
    gold = normalize_answer(gold_answer)
    if not user:
        return False
    if user in gold or gold in user:
        return True
    ratio = SequenceMatcher(None, user, gold).ratio()
    return ratio >= threshold
