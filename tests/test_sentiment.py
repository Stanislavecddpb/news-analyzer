from app.models import NewsItem
from app.sentiment import SentimentAnalyzer


def _score(text: str) -> float:
    return SentimentAnalyzer().score_text(text)[0]


def test_negative_text_scores_negative():
    assert _score("Компании грозит банкротство и арест активов по иску") < -0.2


def test_positive_text_scores_positive():
    assert _score("Компания показала рекордный рост прибыли и выручки") > 0.2


def test_neutral_text_near_zero():
    s = _score("Компания провела ежегодное собрание акционеров в офисе")
    assert abs(s) <= 0.1


def test_negation_flips_polarity():
    # Отрицание смотрит вперёд, поэтому тональный терм должен идти после частицы «не».
    positive = _score("компания получила прибыль")
    negated = _score("компания не получила прибыль")
    assert negated < positive


def test_intensifier_amplifies():
    base = _score("обвал акций")
    strong = _score("резкий обвал акций")
    assert base < 0
    assert strong < base  # усилитель делает негатив более выраженным


def test_score_bounded():
    text = "банкротство арест мошенничество хищение дефолт катастрофа крах обвал"
    s = _score(text)
    assert -1.0 <= s < 0


def test_ranking_orders_most_negative_first():
    items = [
        NewsItem(id="pos", text="рекордный рост прибыли и успех компании"),
        NewsItem(id="neg", text="банкротство и уголовное дело против компании"),
        NewsItem(id="neu", text="компания провела собрание акционеров"),
    ]
    results = {r.id: r for r in SentimentAnalyzer().analyze(items)}
    assert results["neg"].rank == 1
    assert results["pos"].rank == 3
    assert results["neg"].score < results["neu"].score < results["pos"].score
