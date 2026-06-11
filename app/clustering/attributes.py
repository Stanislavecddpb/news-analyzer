"""Извлечение ключевых атрибутов новости: компания, регион, отрасль.

Стратегия (гибридная, словарь + правила):
  1. Компании  — поиск известных названий из справочника + эвристика по юр. формам
                 (ООО «…», ПАО "…") для компаний вне справочника.
  2. Регионы   — сопоставление с справочником субъектов/городов РФ.
  3. Отрасли   — ключевые слова отрасли по стеммированным основам текста.

Возвращается структура `Attributes`. Все списки упорядочены по убыванию частоты
упоминания в тексте, что позволяет легко взять «доминирующий» атрибут (первый).
"""

from __future__ import annotations

import re
from collections import Counter

from app.clustering.dictionaries import (
    INDUSTRIES,
    KNOWN_COMPANIES,
    LEGAL_FORMS,
    LOCATIONS,
)
from app.models import Attributes
from app.text import stem, tokenize

# «ООО Ромашка», ПАО «Газпром нефть», АО "ТД Перекрёсток" — захватываем имя после юр. формы.
_LEGAL_FORM_RE = re.compile(
    r"\b(?:" + "|".join(LEGAL_FORMS) + r")\b[\s«\"']*([А-ЯЁA-Z][\w\-\s«»\"']{1,40}?)(?=[»\"'.,;]|\s—|$)",
    re.IGNORECASE,
)


def _rank_by_frequency(counter: Counter[str]) -> list[str]:
    return [item for item, _ in counter.most_common()]


def _extract_companies(text: str, stems: set[str], joined_stem: str) -> list[str]:
    found: Counter[str] = Counter()

    # 1. Известные компании из справочника (по подстроке в стеммированном тексте).
    for key, canonical in KNOWN_COMPANIES.items():
        if " " in key:
            if key in joined_stem or key in text.lower():
                found[canonical] += 1
        elif stem(key) in stems or key in stems:
            found[canonical] += 1

    # 2. Компании вне справочника — по юридической форме.
    for match in _LEGAL_FORM_RE.finditer(text):
        name = re.sub(r"[«»\"']", "", match.group(1)).strip(" -—")
        # Берём первые 1-3 слова имени, отбрасывая хвост предложения.
        name = " ".join(name.split()[:3]).strip()
        if len(name) >= 2 and name.lower() not in KNOWN_COMPANIES:
            found[name.title()] += 1

    return _rank_by_frequency(found)


def _extract_locations(text: str, joined_stem: str) -> list[str]:
    found: Counter[str] = Counter()
    low = text.lower()
    for key, canonical in LOCATIONS.items():
        if " " in key:
            if key in joined_stem or key in low:
                found[canonical] += 1
        elif key in joined_stem or key in low:
            found[canonical] += 1
    return _rank_by_frequency(found)


def _extract_industries(stems: set[str], joined_stem: str) -> list[str]:
    found: Counter[str] = Counter()
    for industry, keywords in INDUSTRIES.items():
        hits = 0
        for kw in keywords:
            if " " in kw or "-" in kw:
                if kw in joined_stem:
                    hits += 1
            elif any(s.startswith(kw) for s in stems):
                hits += 1
        if hits:
            found[industry] += hits
    return _rank_by_frequency(found)


def extract_attributes(title: str, text: str) -> Attributes:
    """Извлекает (компании, регионы, отрасли) из заголовка и тела новости."""
    full = f"{title} {text}".strip()
    tokens = tokenize(full)
    stems = {stem(t) for t in tokens}
    joined_stem = " ".join(stem(t) for t in tokens)

    return Attributes(
        companies=_extract_companies(full, stems, joined_stem),
        locations=_extract_locations(full, joined_stem),
        industries=_extract_industries(stems, joined_stem),
    )
