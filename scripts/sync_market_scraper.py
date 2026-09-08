import json
import re
from pathlib import Path
from typing import Dict, Any, List
import requests
from bs4 import BeautifulSoup

BENCHMARK_FILE = Path.cwd() / "data" / "market_benchmarks.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-IN,en;q=0.9"
}

SOURCES = {
    "terracotta pottery": "https://shop.gaatha.com/search?q=terracotta+pot",
    "bamboo basket": "https://shop.gaatha.com/search?q=bamboo+basket",
    "wooden craft": "https://shop.gaatha.com/search?q=wooden+handicraft",
    "handloom textile": "https://shop.gaatha.com/search?q=handloom+stole"
}


def extract_prices_from_html(html_text: str) -> List[float]:
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text(" ", strip=True)
    matches = re.findall(r"(?:₹|Rs\.?|INR)\s*([0-9][0-9,]*(?:\.[0-9]+)?)", text)
    prices = []
    for m in matches:
        try:
            val = float(m.replace(",", ""))
            if 50.0 <= val <= 25000.0:
                prices.append(val)
        except ValueError:
            continue
    return prices


def compute_distribution(prices: List[float], fallback_median: float) -> Dict[str, Any]:
    if not prices or len(prices) < 3:
        return {
            "small": {"min": round(fallback_median * 0.6, 2), "median": fallback_median, "max": round(fallback_median * 1.4, 2), "count": len(prices)},
            "medium": {"min": round(fallback_median * 0.8, 2), "median": round(fallback_median * 1.3, 2), "max": round(fallback_median * 1.8, 2), "count": len(prices)},
            "large": {"min": fallback_median, "median": round(fallback_median * 1.8, 2), "max": round(fallback_median * 2.8, 2), "count": len(prices)}
        }

    prices.sort()
    n = len(prices)
    med = prices[n // 2]
    return {
        "small": {"min": round(prices[0], 2), "median": round(med * 0.75, 2), "max": round(med, 2), "count": n},
        "medium": {"min": round(med * 0.75, 2), "median": round(med, 2), "max": round(med * 1.4, 2), "count": n},
        "large": {"min": round(med, 2), "median": round(med * 1.5, 2), "max": round(prices[-1], 2), "count": n}
    }


def sync_market_prices():
    print("[-] Starting scheduled market benchmark synchronization...")
    benchmarks = {}
    if BENCHMARK_FILE.exists():
        try:
            with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
                benchmarks = json.load(f)
        except Exception:
            benchmarks = {}

    default_medians = {
        "terracotta pottery": 450.0,
        "bamboo basket": 599.0,
        "wooden craft": 1299.0,
        "handloom textile": 1199.0
    }

    for category, url in SOURCES.items():
        print(f"    ├─ Querying {category}...")
        extracted_prices = []
        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            if resp.status_code == 200:
                extracted_prices = extract_prices_from_html(resp.text)
        except Exception as e:
            print(f"    │  [!] Scrape skipped for {category}: {e}")

        benchmarks[category] = compute_distribution(extracted_prices, default_medians[category])

    BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_FILE, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"[+] Updated benchmarks written to {BENCHMARK_FILE}")


if __name__ == "__main__":
    sync_market_prices()