import asyncio
import random
import re
from typing import List, Optional, Tuple

from .models import QAPair

RU_STOP = {
    "и", "в", "во", "не", "что", "он", "она", "как", "а", "но", "это", "из",
    "у", "я", "мы", "вы", "они", "бы", "к", "по", "же", "ли", "или", "если",
    "то", "на", "с", "со", "от", "до", "за", "при", "для", "о", "об", "обо",
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


def make_choice_question(key: str, distractors: List[str], sentence: str) -> Tuple[str, str]:
    pool = [d for d in distractors if d.lower() != key.lower() and len(d) >= 3]
    random.shuffle(pool)
    options = [key] + pool[:3]
    random.shuffle(options)
    q = (
        f"Выберите правильное слово (1 вариант из 4):\n"
        f"{re.sub(re.escape(key), '_____ ', sentence, flags=re.IGNORECASE, count=1)}\n"
        f"Варианты: {', '.join(options)}"
    )
    return q, key


def _unique_push(qa: List[QAPair], q: str, a: str, seen: set) -> None:
    norm_q = re.sub(r"\s+", " ", q.strip().lower())
    if norm_q in seen:
        return
    seen.add(norm_q)
    qa.append(QAPair(question=q, answer=a))


def generate_qa_pairs_simple(text: str, n: int) -> List[QAPair]:
    sentences = tokenize_sentences(text)
    if not sentences:
        raise ValueError("Не удалось выделить предложения из текста.")

    # Собираем слова для отвлекающих вариантов
    all_keywords: List[str] = []
    for s in sentences:
        k = pick_keyword(s)
        if k and len(k) >= 3:
            all_keywords.append(k)

    step = max(1, len(sentences) // max(1, n))
    qa: List[QAPair] = []
    seen_q: set = set()
    i = 0
    toggle = True  # чередуем типы вопросов
    while len(qa) < n and i < len(sentences):
        sent = sentences[i]
        key = pick_keyword(sent)
        if key and len(key) >= 3:
            if toggle and len(all_keywords) >= 4:
                q, a = make_choice_question(key, all_keywords, sent)
                _unique_push(qa, q, a, seen_q)
            else:
                q, a = make_gap_question(sent, key)
                _unique_push(qa, q, a, seen_q)
            toggle = not toggle
        i += step if step > 0 else 1

    # добиваем до n, проходя по всем предложениям
    if len(qa) < n:
        for sent in sentences:
            if len(qa) >= n:
                break
            key = pick_keyword(sent)
            if not key or len(key) < 3:
                continue
            # попробуем оба варианта, чтобы повысить разнообразие
            for maker in (make_gap_question, lambda s, k: make_choice_question(k, all_keywords, s)):
                if len(qa) >= n:
                    break
                q, a = maker(sent, key)
                _unique_push(qa, q, a, seen_q)

    return qa[:n]


async def generate_qa_pairs(text: str, n: int) -> List[QAPair]:
    return await asyncio.to_thread(generate_qa_pairs_simple, text, n)
