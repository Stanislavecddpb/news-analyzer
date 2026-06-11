"""Кластеризация новостей по ключевым атрибутам и тематической близости.

Алгоритм (гибридный «атрибуты + текст»):

  1. Для каждой новости извлекаются атрибуты (компания, регион, отрасль).
  2. Текст обогащается атрибутами: канонические имена компании/региона/отрасли
     добавляются к тексту с весом `attribute_boost`. Так совпадение по компании
     сильнее тянет новости в один кластер, чем совпадение случайных слов.
  3. Обогащённые тексты векторизуются TF-IDF (стеммированные униграммы и биграммы).
  4. Агломеративная кластеризация (average linkage, косинусная метрика) с порогом
     расстояния `distance_threshold`. Число кластеров заранее не задаётся —
     алгоритм определяет его сам, что важно при неизвестном потоке тем.
  5. Каждый кластер размечается доминирующими атрибутами и человекочитаемой меткой.

Почему так, а не «просто GROUP BY компания»:
  — одна новость может упоминать несколько компаний, а часть новостей не содержит
    распознаваемой компании вовсе; текстовая близость склеивает такие случаи по
    событию (например, отраслевая новость про несколько игроков рынка).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from app.clustering.attributes import extract_attributes
from app.config import settings
from app.models import Attributes, Cluster, ClusterMember, NewsItem
from app.text import normalize


def _stemmed_analyzer(text: str) -> list[str]:
    """Анализатор для TfidfVectorizer: стеммированные униграммы + биграммы."""
    tokens = normalize(text)
    bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return tokens + bigrams


class Clusterer:
    def __init__(self, distance_threshold: float | None = None, attribute_boost: int | None = None):
        self.distance_threshold = (
            distance_threshold if distance_threshold is not None else settings.cluster_distance_threshold
        )
        self.attribute_boost = (
            attribute_boost if attribute_boost is not None else settings.attribute_boost
        )

    def _enrich(self, item: NewsItem, attrs: Attributes) -> str:
        """Добавляет к тексту атрибуты с повтором (boost), чтобы усилить их вес в TF-IDF."""
        boost_tokens: list[str] = []
        for value in attrs.companies[:1] + attrs.locations[:1] + attrs.industries[:1]:
            # пробелы заменяем на «_», чтобы атрибут стал единым токеном
            token = value.lower().replace(" ", "_")
            boost_tokens.extend([token] * self.attribute_boost)
        return item.full_text + " " + " ".join(boost_tokens)

    def cluster(self, items: list[NewsItem]) -> list[Cluster]:
        if not items:
            return []

        attrs_list = [extract_attributes(it.title, it.text) for it in items]

        if len(items) == 1:
            labels = np.array([0])
        else:
            corpus = [self._enrich(it, a) for it, a in zip(items, attrs_list)]
            vectorizer = TfidfVectorizer(
                analyzer=_stemmed_analyzer,
                min_df=1,
                sublinear_tf=True,
            )
            matrix = vectorizer.fit_transform(corpus)

            # Если словарь пуст (вырожденный вход) — каждый объект в свой кластер.
            if matrix.shape[1] == 0:
                labels = np.arange(len(items))
            else:
                model = AgglomerativeClustering(
                    n_clusters=None,
                    metric="cosine",
                    linkage="average",
                    distance_threshold=self.distance_threshold,
                )
                labels = model.fit_predict(matrix.toarray())

        return self._build_clusters(items, attrs_list, labels)

    @staticmethod
    def _dominant(values: list[str]) -> str | None:
        counter: Counter[str] = Counter(values)
        return counter.most_common(1)[0][0] if counter else None

    def _build_clusters(
        self, items: list[NewsItem], attrs_list: list[Attributes], labels: np.ndarray
    ) -> list[Cluster]:
        groups: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            groups.setdefault(int(lab), []).append(idx)

        clusters: list[Cluster] = []
        for new_id, (_, indices) in enumerate(
            sorted(groups.items(), key=lambda kv: -len(kv[1]))
        ):
            companies, locations, industries = [], [], []
            members: list[ClusterMember] = []
            for i in indices:
                a = attrs_list[i]
                companies += a.companies[:1]
                locations += a.locations[:1]
                industries += a.industries[:1]
                members.append(
                    ClusterMember(id=items[i].id, title=items[i].title, attributes=a)
                )

            dom_company = self._dominant(companies)
            dom_location = self._dominant(locations)
            dom_industry = self._dominant(industries)

            label_parts = [p for p in (dom_company, dom_industry, dom_location) if p]
            label = " · ".join(label_parts) if label_parts else "Без распознанных атрибутов"

            clusters.append(
                Cluster(
                    cluster_id=new_id,
                    size=len(indices),
                    label=label,
                    dominant_company=dom_company,
                    dominant_location=dom_location,
                    dominant_industry=dom_industry,
                    member_ids=[items[i].id for i in indices],
                    members=members,
                )
            )
        return clusters


# Удобный модульный shortcut.
def cluster_news(items: list[NewsItem], distance_threshold: float | None = None) -> list[Cluster]:
    return Clusterer(distance_threshold=distance_threshold).cluster(items)
