"""Pydantic-схемы запросов и ответов API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    """Входная новость. Минимально необходимый набор полей."""

    id: str = Field(..., description="Уникальный идентификатор новости")
    title: str = Field("", description="Заголовок")
    text: str = Field(..., description="Тело новости")
    url: str | None = Field(None, description="Источник (.ru)")
    published_at: str | None = Field(None, description="Дата публикации (ISO 8601)")

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}".strip()


# ---------- Тональность ----------

class SentimentResult(BaseModel):
    id: str
    score: float = Field(..., description="Тональность в диапазоне [-1, 1]")
    label: str = Field(..., description="резко негативная | негативная | нейтральная | позитивная | резко позитивная")
    rank: int = Field(..., description="Ранг по возрастанию score (1 — самая негативная)")
    matched_terms: list[str] = Field(default_factory=list, description="Сработавшие термы словаря")


class SentimentRequest(BaseModel):
    items: list[NewsItem]


class SentimentResponse(BaseModel):
    results: list[SentimentResult]


# ---------- Атрибуты / кластеризация ----------

class Attributes(BaseModel):
    companies: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)


class ClusterMember(BaseModel):
    id: str
    title: str
    attributes: Attributes


class Cluster(BaseModel):
    cluster_id: int
    size: int
    label: str = Field(..., description="Человекочитаемая метка кластера")
    dominant_company: str | None = None
    dominant_location: str | None = None
    dominant_industry: str | None = None
    member_ids: list[str]
    members: list[ClusterMember]


class ClusterRequest(BaseModel):
    items: list[NewsItem]
    distance_threshold: float | None = Field(
        None, description="Переопределить порог расстояния кластеризации"
    )


class ClusterResponse(BaseModel):
    clusters: list[Cluster]
    n_clusters: int


# ---------- Сводный анализ ----------

class AnalyzedItem(BaseModel):
    id: str
    title: str
    cluster_id: int
    attributes: Attributes
    sentiment: SentimentResult


class AnalyzeResponse(BaseModel):
    items: list[AnalyzedItem]
    clusters: list[Cluster]
    n_clusters: int
