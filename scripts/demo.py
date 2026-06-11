"""Демонстрация работы модулей без поднятия API.

Запуск:  python -m scripts.demo  [путь_к_json]
По умолчанию берётся data/sample_news.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.clustering import Clusterer
from app.models import NewsItem
from app.sentiment import SentimentAnalyzer


def main() -> None:
    # Windows-консоль по умолчанию не UTF-8 — без этого русский вывод превращается в кракозябры.
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample_news.json")
    items = [NewsItem(**row) for row in json.loads(path.read_text(encoding="utf-8"))]

    print(f"Загружено новостей: {len(items)}\n")

    # --- Кластеризация ---
    clusters = Clusterer().cluster(items)
    print(f"=== КЛАСТЕРЫ ({len(clusters)}) ===")
    for c in clusters:
        print(f"[#{c.cluster_id}] {c.label}  (новостей: {c.size})")
        for m in c.members:
            print(f"      - {m.id}: {m.title}")
    print()

    # --- Тональность и ранжирование ---
    results = sorted(SentimentAnalyzer().analyze(items), key=lambda r: r.rank)
    print("=== РАНЖИРОВАНИЕ ПО ТОНАЛЬНОСТИ (1 — самая негативная) ===")
    titles = {it.id: it.title for it in items}
    for r in results:
        terms = ", ".join(r.matched_terms[:5])
        print(f"  {r.rank:>2}. score={r.score:+.3f}  [{r.label}]  {titles[r.id]}")
        if terms:
            print(f"        термы: {terms}")


if __name__ == "__main__":
    main()
