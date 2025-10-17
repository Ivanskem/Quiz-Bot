import asyncio
import re
from typing import List, Optional, Tuple

from .models import QAPair

RU_STOP = {
    "и", "в", "во", "не", "что", "он", "она", "оно", "как", "а", "но", "это", "из",
    "у", "я", "мы", "вы", "они", "бы", "к", "по", "же", "же", "же", "так", "же",
    "ли", "или", "если", "то", "на", "с", "со", "же", "же", "от", "до", "за", "при",
    "для", "о", "об", "обо", "еще", "уже", "также", "где", "когда", "чтобы", "что",
}


def tokenize_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def pick_keyword(sentence: str) -> Optional[str]:
    words = re.findall(r"[A-Za-zА-Яа-я0-9\-]+", sentence)
    words_norm = [w.lower() for w in words if w and w.lower() not in RU_STOP]
    if not words_norm:
        return None
    words_norm.sort(key=lambda w: (-len(w), w))
    return words_norm[0]


def make_gap_question(sentence: str, keyword: str) -> Tuple[str, str]:
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    masked = pattern.sub("_____ ", sentence, count=1)
    question = f"Заполните пропуск: {masked}"
    answer = keyword
    return question, answer


def generate_qa_pairs_simple(text: str, n: int) -> List[QAPair]:
    sentences = tokenize_sentences(text)
    if not sentences:
        raise ValueError("Не удалось выделить предложения из текста.")
    step = max(1, len(sentences) // max(1, n))
    qa: List[QAPair] = []
    i = 0
    while len(qa) < n and i < len(sentences):
        sent = sentences[i]
        keyword = pick_keyword(sent)
        if keyword and len(keyword) >= 3:
            question, answer = make_gap_question(sent, keyword)
            qa.append(QAPair(question=question, answer=answer))
        i += step if step > 0 else 1
    if len(qa) < n:
        for sent in sentences:
            if len(qa) >= n:
                break
            keyword = pick_keyword(sent)
            if keyword and len(keyword) >= 3:
                question, answer = make_gap_question(sent, keyword)
                qa.append(QAPair(question=question, answer=answer))
    return qa[:n]


async def generate_qa_pairs(text: str, n: int) -> List[QAPair]:
    return await asyncio.to_thread(generate_qa_pairs_simple, text, n)
