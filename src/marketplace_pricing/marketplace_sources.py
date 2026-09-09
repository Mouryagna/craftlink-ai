"""
SIH26090 Marketplace Source Collector — High-Precision Target Engine
Performs strict, targeted catalog harvesting and precision filtering
across Indian artisanal portals.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import requests
from bs4 import BeautifulSoup

REQUEST_TIMEOUT = 10
WORKER_THREADS = 10

PRIMARY_SOURCES = {
    "Okhai": "https://okhai.org/",
    "Gaatha": "https://shop.gaatha.com/",
    "GoSwadeshi": "https://goswadeshi.in/",
    "Santarms": "https://www.santarms.com/",
    "Kaarigar Online": "https://kaarigaronline.com/",
    "Karigar": "https://karigar.in/",
    "Kaari": "https://kaari.in/",
    "Kaarigo": "https://www.kaarigo.com/",
    "Jaypore": "https://www.jaypore.com/",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}


@dataclass
class MarketplaceCandidate:
    listing_id: str
    marketplace: str
    title: str
    price: float
    url: str
    product_type: Optional[str] = None
    material: Optional[str] = None
    features: Optional[List[str]] = None
    source_tier: str = "PRIMARY"

    def to_listing(self) -> Dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "marketplace": self.marketplace,
            "title": self.title,
            "price": self.price,
            "url": self.url,
            "product_type": self.product_type or "",
            "material": self.material or "",
            "features": self.features or [],
            "source_tier": self.source_tier,
        }


class HandmadeMarketplaceCollector:
    def __init__(self, primary_sources=None, secondary_sources=None):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.primary_sources = primary_sources or PRIMARY_SOURCES

    @staticmethod
    def _clean_price(val: Any) -> Optional[float]:
        if val is None:
            return None
        cleaned = re.sub(r"[^\d.]", "", str(val).replace(",", ""))
        try:
            p = float(cleaned)
            return p if 30.0 <= p <= 40000.0 else None
        except ValueError:
            return None

    # -----------------------------------------------------------------------
    # STRICT RELEVANCE GATE
    # -----------------------------------------------------------------------
    @staticmethod
    def is_strictly_relevant(title: str, target_type: str, target_mat: str) -> bool:
        """
        Hard negative filter:
        1. Must contain the specific product type noun (or its root stem).
        2. Must NOT match disjoint craft categories (e.g. coaster vs suit/dress/saree).
        """
        t_clean = title.lower()
        target_t = target_type.lower().strip()
        target_m = target_mat.lower().strip()

        # Disjoint negative words (if searching for hard crafts, reject apparel/jewelry)
        apparel_disjoint = {"suit", "kurta", "saree", "dupatta", "dress", "necklace", "earring", "tikka", "mathapatti"}
        tableware_disjoint = {"coaster", "runner", "bottle", "planter", "pot", "tray", "bowl"}

        if any(w in target_t for w in tableware_disjoint):
            if any(re.search(rf"\b{re.escape(bad)}\b", t_clean) for bad in apparel_disjoint):
                return False

        if any(w in target_t for w in apparel_disjoint):
            if any(re.search(rf"\b{re.escape(bad)}\b", t_clean) for bad in tableware_disjoint):
                return False

        # Must explicitly mention the core target product noun
        # e.g. "coaster" in "sabai grass coaster set" -> True
        keywords = [k for k in target_t.split() if len(k) > 3]
        if not keywords:
            keywords = [target_t]

        has_type_match = any(re.search(rf"\b{re.escape(k)}\b", t_clean) for k in keywords)

        # Or must contain the exact specific material
        has_mat_match = bool(target_m and len(target_m) > 3 and re.search(rf"\b{re.escape(target_m)}\b", t_clean))

        return has_type_match or has_mat_match

    # -----------------------------------------------------------------------
    # SHOPIFY SEARCH API (TARGETED QUERIES)
    # -----------------------------------------------------------------------
    def _fetch_shopify_targeted(self, base_url: str, source_name: str, query: str, target_type: str, target_mat: str) -> List[Dict[str, Any]]:
        base = base_url.rstrip("/")
        endpoint = f"{base}/search/suggest.json"
        params = {"q": query, "resources[type]": "product"}
        results = []

        try:
            resp = self.session.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return []

            data = resp.json()
            products = data.get("resources", {}).get("results", {}).get("products", [])

            for p in products:
                title = p.get("title", "").strip()
                if not self.is_strictly_relevant(title, target_type, target_mat):
                    continue

                raw_price = p.get("price")
                try:
                    price = float(raw_price) if raw_price else 0.0
                    if price > 15000:
                        price = price / 100.0  # Handle paisa conversion
                except (TypeError, ValueError):
                    price = 0.0

                if price <= 0:
                    continue

                url = urllib.parse.urljoin(base_url, p.get("url", ""))
                results.append({
                    "listing_id": f"{source_name[:4].upper()}-{hashlib.md5(url.encode()).hexdigest()[:8]}",
                    "marketplace": source_name,
                    "title": title[:90],
                    "price": price,
                    "url": url,
                    "product_type": target_type,
                    "material": target_mat,
                    "features": ["Handcrafted", "Artisan Direct"],
                    "source_tier": "PRIMARY",
                })
        except Exception:
            return []

        return results

    # -----------------------------------------------------------------------
    # WOOCOMMERCE / HTML DOM TARGETED SEARCH
    # -----------------------------------------------------------------------
    def _fetch_html_targeted(self, base_url: str, source_name: str, query: str, target_type: str, target_mat: str) -> List[Dict[str, Any]]:
        base = base_url.rstrip("/")
        search_url = f"{base}/?s={urllib.parse.quote(query)}&post_type=product"
        results = []

        try:
            resp = self.session.get(search_url, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, "html.parser")
            cards = soup.select("li.product, div.product-item, div.product, div.grid-item")

            for card in cards:
                t_elem = card.select_one("h2, h3, a.product-title, a.woocommerce-loop-product__title")
                p_elem = card.select_one("span.price, span.woocommerce-Price-amount, div.price")
                link = card.select_one("a[href]")

                if not t_elem or not p_elem:
                    continue

                title = t_elem.get_text(" ", strip=True)
                if not self.is_strictly_relevant(title, target_type, target_mat):
                    continue

                price = self._clean_price(p_elem.get_text())
                if not price:
                    continue

                href = link.get("href") if link else base
                results.append({
                    "listing_id": f"{source_name[:4].upper()}-{hashlib.md5(href.encode()).hexdigest()[:8]}",
                    "marketplace": source_name,
                    "title": title[:90],
                    "price": price,
                    "url": href,
                    "product_type": target_type,
                    "material": target_mat,
                    "features": ["Handmade", "Authentic Craft"],
                    "source_tier": "PRIMARY",
                })
        except Exception:
            return []

        return results

    # -----------------------------------------------------------------------
    # SITE SEARCH ORCHESTRATOR
    # -----------------------------------------------------------------------
    def _search_single_site(self, name_src: str, base_url: str, queries: List[str], target_type: str, target_mat: str) -> List[Dict[str, Any]]:
        site_candidates = []
        for q in queries:
            # Try Shopify suggest first
            items = self._fetch_shopify_targeted(base_url, name_src, q, target_type, target_mat)
            if not items:
                # Fallback to HTML crawl
                items = self._fetch_html_targeted(base_url, name_src, q, target_type, target_mat)
            site_candidates.extend(items)

        # Deduplicate within site
        unique = []
        seen = set()
        for item in site_candidates:
            key = (item["title"].lower().strip(), float(item["price"]))
            if key not in seen:
                seen.add(key)
                unique.append(item)

        return unique

    # -----------------------------------------------------------------------
    # MASTER PIPELINE
    # -----------------------------------------------------------------------
    def collect(
        self,
        product: Dict[str, Any],
        primary_limit_per_source: int = 15,
        secondary_limit_per_source: int = 0,
        min_primary_candidates: int = 5,
    ) -> List[Dict[str, Any]]:
        target_type = str(product.get("product_type", "")).strip()
        target_mat = str(product.get("material", "")).strip()

        # Build highly specific search queries
        queries = [
            f"{target_mat} {target_type}".strip(),
            f"{target_type}".strip(),
        ]

        print("\n" + "=" * 80)
        print(f"TARGETED DISCOVERY: Searching for '{target_mat} {target_type}'")
        print(f"Strict Query Variations : {queries}")
        print("=" * 80)

        all_results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=WORKER_THREADS) as executor:
            future_to_site = {
                executor.submit(self._search_single_site, name, url, queries, target_type, target_mat): name
                for name, url in self.primary_sources.items()
            }
            for future in as_completed(future_to_site):
                site_name = future_to_site[future]
                try:
                    site_items = future.result()
                    if site_items:
                        print(f"[PRIMARY] {site_name:<16} : Found {len(site_items)} strictly matched products")
                        all_results.extend(site_items)
                    else:
                        print(f"[PRIMARY] {site_name:<16} : 0 matches (irrelevant items excluded)")
                except Exception:
                    pass

        # Global deduplication
        master_pool = []
        seen_global = set()
        for item in all_results:
            key = (item["title"].lower().strip()[:40], float(item["price"]))
            if key not in seen_global:
                seen_global.add(key)
                master_pool.append(item)

        print("-" * 80)
        print(f"TOTAL STRICT MARKETPLACE COMPARABLES DISCOVERED: {len(master_pool)}")
        print("=" * 80)
        return master_pool


marketplace_collector = HandmadeMarketplaceCollector()

def collect_marketplace_candidates(product: Dict[str, Any], **kwargs):
    return marketplace_collector.collect(product, **kwargs)