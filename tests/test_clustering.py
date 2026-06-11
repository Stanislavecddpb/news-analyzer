from app.clustering import Clusterer, extract_attributes
from app.models import NewsItem


def test_extract_company_from_dictionary():
    attrs = extract_attributes("Газпром отчитался", "ПАО Газпром увеличил добычу газа")
    assert "Газпром" in attrs.companies
    assert "Нефтегаз" in attrs.industries


def test_extract_location():
    attrs = extract_attributes("Новость", "Завод в Екатеринбурге увеличил выпуск")
    assert "Екатеринбург" in attrs.locations


def test_extract_company_by_legal_form():
    attrs = extract_attributes("", 'ООО «Уральские моторы» нарастило экспорт')
    assert any("Уральск" in c for c in attrs.companies)


def test_same_company_news_cluster_together():
    items = [
        NewsItem(id="a", title="Газпром нарастил прибыль", text="ПАО Газпром рост выручки газ"),
        NewsItem(id="b", title="Газпром и дивиденды", text="Газпром выплатит дивиденды акционерам газ"),
        NewsItem(id="c", title="Сбербанк ставки", text="Сбербанк повысил ставки по ипотеке кредит банк"),
    ]
    clusters = Clusterer(distance_threshold=0.7).cluster(items)
    cluster_of = {mid: c.cluster_id for c in clusters for mid in c.member_ids}
    assert cluster_of["a"] == cluster_of["b"]
    assert cluster_of["c"] != cluster_of["a"]


def test_cluster_labels_have_dominant_company():
    items = [
        NewsItem(id="a", title="Газпром", text="Газпром добыча газа в Москве"),
        NewsItem(id="b", title="Газпром", text="Газпром экспорт газа"),
    ]
    clusters = Clusterer().cluster(items)
    assert clusters[0].dominant_company == "Газпром"


def test_empty_input():
    assert Clusterer().cluster([]) == []


def test_single_item():
    clusters = Clusterer().cluster([NewsItem(id="x", text="Сбербанк запускает сервис")])
    assert len(clusters) == 1
    assert clusters[0].size == 1
