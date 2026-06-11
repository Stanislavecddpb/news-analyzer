from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_PAYLOAD = {
    "items": [
        {"id": "1", "title": "Газпром рост", "text": "ПАО Газпром рекордная прибыль газ"},
        {"id": "2", "title": "Газпром авария", "text": "Авария на скважине Газпрома, уголовное дело"},
        {"id": "3", "title": "Сбербанк", "text": "Сбербанк повысил ставки по ипотеке"},
    ]
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_sentiment_endpoint():
    r = client.post("/sentiment", json=_PAYLOAD)
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 3
    assert all("rank" in x for x in results)


def test_cluster_endpoint():
    r = client.post("/cluster", json=_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["n_clusters"] >= 1


def test_analyze_endpoint():
    r = client.post("/analyze", json=_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 3
    assert all(item["cluster_id"] >= 0 for item in body["items"])
    assert all(item["sentiment"] for item in body["items"])
