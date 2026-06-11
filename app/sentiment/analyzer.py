"""Определение тональности новости и ранжирование выборки.

Подход — лексиконно-правиловый (lexicon + rules):
  * по словарю тональности начисляются веса сработавших термов;
  * частица отрицания («не», «без», …) инвертирует знак следующих N термов;
  * усилители/ослабители («резко», «значительно», «немного») масштабируют вес;
  * итоговая сумма сжимается через tanh в диапазон [-1, 1] — так одиночный сильный
    терм даёт выраженную оценку, а накопление однополярных термов плавно насыщается.

Ранжирование: новости сортируются по возрастанию score, поэтому ранг 1 получает
самая негативная новость — то, что в задаче мониторинга рисков нужно видеть первым.

Подход прозрачен и не требует обучающих данных/GPU — это осознанный выбор для
MVP. Путь развития (в production) описан в docs/architecture.md: дообученный
RuBERT-классификатор тональности, лексикон как fallback и источник объяснений.
"""

from __future__ import annotations

import math

from app.config import settings
from app.models import NewsItem, SentimentResult
from app.sentiment.lexicon import (
    INTENSIFIERS_STEMMED,
    MULTIWORD,
    NEGATION_WINDOW,
    NEGATIONS_STEMMED,
    NEGATIVE,
    POSITIVE,
)
from app.text import normalize

# Масштаб сжатия суммы весов в tanh. Меньше → быстрее насыщение к ±1.
_TANH_SCALE = 2.0


class SentimentAnalyzer:
    def __init__(self, neutral_band: float | None = None):
        self.neutral_band = (
            neutral_band if neutral_band is not None else settings.sentiment_neutral_band
        )

    def _label(self, score: float) -> str:
        if score <= -0.5:
            return "резко негативная"
        if score < -self.neutral_band:
            return "негативная"
        if score <= self.neutral_band:
            return "нейтральная"
        if score < 0.5:
            return "позитивная"
        return "резко позитивная"

    def score_text(self, text: str) -> tuple[float, list[str]]:
        """Возвращает (score в [-1,1], список сработавших термов)."""
        stems = normalize(text, drop_stopwords=False)
        total = 0.0
        matched: list[str] = []

        negation_countdown = 0
        pending_intensifier = 1.0

        for token in stems:
            if token in NEGATIONS_STEMMED:
                negation_countdown = NEGATION_WINDOW
                continue
            if token in INTENSIFIERS_STEMMED:
                pending_intensifier = INTENSIFIERS_STEMMED[token]
                # `continue` пропускает декремент окна отрицания, поэтому стоящий перед
                # усилителем «не» сохраняет действие на последующий тональный терм.
                continue

            weight = NEGATIVE.get(token) or POSITIVE.get(token)
            if weight is not None:
                weight *= pending_intensifier
                if negation_countdown > 0:
                    weight = -weight * 0.8  # инверсия с лёгким затуханием
                total += weight
                matched.append(token)
                pending_intensifier = 1.0

            if negation_countdown > 0:
                negation_countdown -= 1

        # Многословные термы — по подстроке исходного текста.
        low = text.lower()
        for phrase, weight in MULTIWORD.items():
            if phrase in low:
                total += weight
                matched.append(phrase)

        score = math.tanh(total / _TANH_SCALE)
        return round(score, 4), matched

    def analyze(self, items: list[NewsItem]) -> list[SentimentResult]:
        """Оценивает выборку и проставляет ранги (1 — самая негативная)."""
        scored: list[tuple[NewsItem, float, list[str]]] = []
        for it in items:
            score, matched = self.score_text(it.full_text)
            scored.append((it, score, matched))

        # Ранг по возрастанию score; при равенстве — стабильно по id.
        order = sorted(range(len(scored)), key=lambda i: (scored[i][1], scored[i][0].id))
        rank_of = {idx: rank for rank, idx in enumerate(order, start=1)}

        results: list[SentimentResult] = []
        for idx, (it, score, matched) in enumerate(scored):
            results.append(
                SentimentResult(
                    id=it.id,
                    score=score,
                    label=self._label(score),
                    rank=rank_of[idx],
                    matched_terms=matched,
                )
            )
        return results


def analyze_sentiment(items: list[NewsItem]) -> list[SentimentResult]:
    return SentimentAnalyzer().analyze(items)
