from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecommendationContext:
    total: int
    correct: int
    wrong: int
    text_chars: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def build_recommendation(context: RecommendationContext) -> str:
    accuracy = context.accuracy
    volume_hint = "Объём текста небольшой" if context.text_chars < 1000 else "Текст довольно объёмный"

    if accuracy >= 0.85:
        return (
            "Отличный результат! Попробуйте усложнить задание: задайте больше вопросов или попросите "
            "бота построить вопросы по другим темам. "
            f"{volume_hint}, можно поэкспериментировать с другими источниками."
        )
    if accuracy >= 0.5:
        return (
            "Хороший прогресс. Рекомендую пересмотреть места, где возникли ошибки, и попробовать ещё раз "
            "с дополнительными 2-3 вопросами. "
            f"{volume_hint}, возможно, стоит выделить ключевые термины заранее."
        )
    return (
        "Кажется, материал дался непросто. Сформируйте конспект из исходного текста или попросите бота "
        "разбить материал на абзацы, затем повторите викторину с меньшим числом вопросов."
    )
