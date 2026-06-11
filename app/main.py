"""FastAPI-приложение: точки входа для кластеризации, тональности и сводного анализа."""

from __future__ import annotations

from fastapi import FastAPI

from app.clustering import Clusterer
from app.config import settings
from app.models import (
    AnalyzedItem,
    AnalyzeResponse,
    ClusterRequest,
    ClusterResponse,
    SentimentRequest,
    SentimentResponse,
)
from app.sentiment import SentimentAnalyzer

app = FastAPI(title=settings.app_title, version=settings.app_version)


@app.get("/health", tags=["service"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": settings.app_version}


@app.post("/sentiment", response_model=SentimentResponse, tags=["analysis"])
def sentiment(req: SentimentRequest) -> SentimentResponse:
    """Тональность и ранжирование выборки новостей."""
    results = SentimentAnalyzer().analyze(req.items)
    return SentimentResponse(results=results)


@app.post("/cluster", response_model=ClusterResponse, tags=["analysis"])
def cluster(req: ClusterRequest) -> ClusterResponse:
    """Кластеризация новостей по компании/региону/отрасли и тематической близости."""
    clusters = Clusterer(distance_threshold=req.distance_threshold).cluster(req.items)
    return ClusterResponse(clusters=clusters, n_clusters=len(clusters))


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
def analyze(req: ClusterRequest) -> AnalyzeResponse:
    """Сводный анализ: кластеризация + тональность + ранжирование в одном ответе."""
    clusters = Clusterer(distance_threshold=req.distance_threshold).cluster(req.items)
    sentiments = {s.id: s for s in SentimentAnalyzer().analyze(req.items)}

    # id новости -> id кластера
    cluster_of: dict[str, int] = {}
    attrs_of = {}
    for c in clusters:
        for m in c.members:
            cluster_of[m.id] = c.cluster_id
            attrs_of[m.id] = m.attributes

    items = [
        AnalyzedItem(
            id=it.id,
            title=it.title,
            cluster_id=cluster_of.get(it.id, -1),
            attributes=attrs_of.get(it.id),
            sentiment=sentiments[it.id],
        )
        for it in req.items
    ]
    return AnalyzeResponse(items=items, clusters=clusters, n_clusters=len(clusters))
