"""Конфигурация сервиса. Значения читаются из окружения (12-factor)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # Порог косинусного расстояния для агломеративной кластеризации.
    # Меньше — кластеры «строже» (только очень похожие новости попадают вместе).
    cluster_distance_threshold: float = _get_float("CLUSTER_DISTANCE_THRESHOLD", 0.6)

    # Во сколько раз «утяжелять» атрибуты (компания/регион/отрасль) в векторе текста.
    attribute_boost: int = int(os.environ.get("ATTRIBUTE_BOOST", "3"))

    # Граница, ниже которой |score| считается нейтральной тональностью.
    sentiment_neutral_band: float = _get_float("SENTIMENT_NEUTRAL_BAND", 0.05)

    app_title: str = "News Analyzer — анализ новостного фона компаний (.ru)"
    app_version: str = "0.1.0"


settings = Settings()
