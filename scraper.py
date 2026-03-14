"""
Türkçe Haber Scraper — Ücretsiz RSS + HTML parse
"""

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "tr-TR,tr;q=0.9",
}

# Ücretsiz RSS kaynaklarını tanımla
RSS_SOURCES = [
    {
        "name": "NTV",
        "feeds": [
            "https://www.ntv.com.tr/gundem.rss",
            "https://www.ntv.com.tr/ekonomi.rss",
            "https://www.ntv.com.tr/spor.rss",
            "https://www.ntv.com.tr/teknoloji.rss",
            "https://www.ntv.com.tr/saglik.rss",
            "https://www.ntv.com.tr/egitim.rss",
            "https://www.ntv.com.tr/dunya.rss",
            "https://www.ntv.com.tr/sanat.rss",
            "https://www.ntv.com.tr/otomobil.rss",
            "https://www.ntv.com.tr/yasam.rss",
        ],
    },
    {
        "name": "Sözcü",
        "feeds": [
            "https://www.sozcu.com.tr/rss/gundem.xml",
            "https://www.sozcu.com.tr/rss/ekonomi.xml",
            "https://www.sozcu.com.tr/rss/spor.xml",
            "https://www.sozcu.com.tr/rss/dunya.xml",
            "https://www.sozcu.com.tr/rss/teknoloji.xml",
            "https://www.sozcu.com.tr/rss/yasam.xml",
        ],
    },
    {
        "name": "Cumhuriyet",
        "feeds": [
            "https://www.cumhuriyet.com.tr/rss/son_dakika.xml",
            "https://www.cumhuriyet.com.tr/rss/turkiye.xml",
            "https://www.cumhuriyet.com.tr/rss/dunya.xml",
            "https://www.cumhuriyet.com.tr/rss/ekonomi.xml",
            "https://www.cumhuriyet.com.tr/rss/spor.xml",
            "https://www.cumhuriyet.com.tr/rss/teknoloji.xml",
            "https://www.cumhuriyet.com.tr/rss/saglik.xml",
            "https://www.cumhuriyet.com.tr/rss/kultur-sanat.xml",
        ],
    },
]


@dataclass
class NewsItem:
    title: str
    summary: str
    url: str
    source: str
    date: str
    category: str = ""
    confidence: float = 0.0


def clean_html(text: str) -> str:
    """HTML etiketlerini ve fazla boşlukları temizle."""
    if not text:
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_rss_feed(url: str, source_name: str) -> list[NewsItem]:
    """Bir RSS feed'i parse et."""
    items = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        # RSS 2.0 veya Atom
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        channel_items = root.findall(".//item")
        if not channel_items:
            channel_items = root.findall(".//atom:entry", ns)

        for item in channel_items:
            title = item.findtext("title") or item.findtext("atom:title", namespaces=ns) or ""
            desc = item.findtext("description") or item.findtext("atom:summary", namespaces=ns) or ""
            link = item.findtext("link") or ""
            if not link:
                link_el = item.find("atom:link", ns)
                link = link_el.get("href", "") if link_el is not None else ""
            pub_date = item.findtext("pubDate") or item.findtext("atom:updated", namespaces=ns) or ""

            title = clean_html(title)
            summary = clean_html(desc)

            if not title or len(title) < 5:
                continue

            items.append(NewsItem(
                title=title,
                summary=summary[:300],
                url=link.strip(),
                source=source_name,
                date=pub_date,
            ))
    except Exception as e:
        print(f"  RSS hata ({source_name} - {url}): {e}")
    return items


def fetch_all_news() -> list[NewsItem]:
    """Tüm RSS kaynaklarından haber çek."""
    all_news = []
    seen_titles = set()

    for source in RSS_SOURCES:
        for feed_url in source["feeds"]:
            items = parse_rss_feed(feed_url, source["name"])
            for item in items:
                key = item.title.lower().strip()
                if key not in seen_titles:
                    seen_titles.add(key)
                    all_news.append(item)
            time.sleep(0.3)  # Saygılı scraping

    print(f"Toplam {len(all_news)} benzersiz haber çekildi.")
    return all_news


def classify_news(news_list: list[NewsItem], classifier) -> list[NewsItem]:
    """Haberleri sınıflandır."""
    for item in news_list:
        text = f"{item.title}. {item.summary}" if item.summary else item.title
        result = classifier.predict(text)
        item.category = result["category"]
        item.confidence = result["confidence"]
    return news_list


def news_to_dict(news_list: list[NewsItem]) -> list[dict]:
    return [asdict(item) for item in news_list]


if __name__ == "__main__":
    news = fetch_all_news()
    for n in news[:5]:
        print(f"[{n.source}] {n.title}")
