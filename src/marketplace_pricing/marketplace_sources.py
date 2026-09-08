

"""
SIH26090 Marketplace Source Collector

Primary sources:
    Kaarigar Online
    Gaatha
    Jaypore
    GoSwadeshi
    Karigar
    Santarms
    Kaari
    Kaarigo
    Okhai

Secondary sources:
    Amazon India
    Flipkart

Design:
    1. Search primary Indian handmade/handloom sources first.
    2. Fetch product pages.
    3. Extract the same marketplace fields used by the pricing pipeline.
    4. If the product is handloom/textile and primary evidence is insufficient,
       search Amazon and Flipkart as secondary sources.
    5. Return NORMALIZED LISTING DICTS.
    6. An LLM product analyzer decides which fetched listings are relevant
       enough to enter the candidate pool.
    7. SimilarityEngine then performs the detailed multi-factor comparison.
       This module never calculates a price.

No CAPTCHA/anti-bot bypass is used.
If a site blocks normal requests, that source is skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = 8
SEARCH_TIMEOUT = 6
REQUEST_DELAY = 0.0
SEARCH_WORKERS = 24
SOURCE_WORKERS = 9
PAGE_WORKERS = 24
SITEMAP_TIMEOUT = 8
MAX_SITEMAP_URLS = 5000

PRIMARY_SOURCES = {
    "Kaarigar Online": "https://kaarigaronline.com/",
    "Gaatha": "https://shop.gaatha.com/",
    "Jaypore": "https://www.jaypore.com/",
    "GoSwadeshi": "https://goswadeshi.in/",
    "Karigar": "https://karigar.in/products/",
    "Santarms": "https://www.santarms.com/",
    "Kaari": "https://kaari.in/",
    "Kaarigo": "https://www.kaarigo.com/",
    "Okhai": "https://okhai.org/",
}

SECONDARY_SOURCES = {
    "Amazon": "https://www.amazon.in/",
    "Flipkart": "https://www.flipkart.com/",
}

PRIMARY_RESULT_LIMIT = 8
SECONDARY_RESULT_LIMIT = 5
MAX_PRIMARY_CANDIDATES = 60
MAX_TOTAL_CANDIDATES = 80

# LLM retrieval + candidate-analysis configuration. The LLM never calculates prices.
LLM_QUERY_MODEL = os.getenv("GEMINI_QUERY_MODEL", "gemini-2.5-flash-lite")
LLM_CLASSIFIER_MODEL = os.getenv("GEMINI_CLASSIFIER_MODEL", LLM_QUERY_MODEL)
LLM_QUERIES_PER_SOURCE = 6
LLM_MAX_QUERY_LENGTH = 140
LLM_CLASSIFIER_BATCH_SIZE = 12
LLM_CLASSIFIER_MIN_CONFIDENCE = 0.70

# Secondary search is intentionally limited to handloom/handicraft/textile
# queries. It is not the primary source pool.
HANDLOOM_TERMS = {
    "handloom",
    "handwoven",
    "hand woven",
    "weaving",
    "woven",
    "loom",
    "textile",
    "saree",
    "sari",
    "dupatta",
    "stole",
    "shawl",
    "khadi",
    "ikat",
    "jamdani",
    "chanderi",
    "banarasi",
    "kantha",
    "ajrakh",
    "block print",
    "embroidery",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}


# ---------------------------------------------------------------------------
# DATA MODEL
# ---------------------------------------------------------------------------

@dataclass
class MarketplaceCandidate:
    listing_id: str
    marketplace: str
    title: str
    price: float
    url: str

    rating: Optional[float] = None
    review_count: Optional[int] = None
    seller_name: Optional[str] = None
    handmade_evidence: Optional[str] = None

    product_type: Optional[str] = None
    material: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    dimensions: Optional[str] = None
    dimensions_length: Optional[float] = None
    dimensions_width: Optional[float] = None
    dimensions_height: Optional[float] = None
    features: Optional[List[str]] = None
    description: Optional[str] = None

    source_tier: str = "PRIMARY"

    def to_listing(self) -> Dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "marketplace": self.marketplace,
            "title": self.title,
            "price": self.price,
            "url": self.url,
            "rating": self.rating,
            "review_count": self.review_count,
            "seller_name": self.seller_name,
            "handmade_evidence": self.handmade_evidence,
            "product_type": self.product_type or "",
            "material": self.material or "",
            "color": self.color or "",
            "size": self.size or "",
            "dimensions": self.dimensions or "",
            "dimensions_length": self.dimensions_length,
            "dimensions_width": self.dimensions_width,
            "dimensions_height": self.dimensions_height,
            "features": self.features or [],
            "description": self.description or "",
            "source_tier": self.source_tier,
        }


# ---------------------------------------------------------------------------
# LLM PRODUCT CATEGORY CLASSIFIER
# ---------------------------------------------------------------------------

class LLMProductCategoryClassifier:
    """Resolve the concrete marketplace product family before retrieval.

    The classifier is used only for retrieval vocabulary.  It never scores
    listings and never calculates prices.  If Gemini is unavailable, the
    original product_type is retained and deterministic aliases are used.
    """

    def __init__(self, model: str = LLM_QUERY_MODEL):
        self.model = model
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            return None
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            return self._client
        except Exception:
            return None

    @staticmethod
    def _fallback(product: Dict[str, Any]) -> Dict[str, Any]:
        name = str(product.get("product_name", "") or "").lower()
        ptype = str(product.get("product_type", "") or "").strip().lower()
        rules = {
            "water bottle": ["water bottle", "terracotta bottle", "earthen bottle", "clay bottle", "flask"],
            "basket": ["basket", "bamboo basket", "woven basket", "storage basket", "fruit basket", "hamper"],
            "stool": ["stool", "handcrafted stool", "wooden stool", "cane stool"],
            "table": ["table", "side table", "coffee table", "lawn table"],
            "swing chair": ["swing chair", "hanging swing chair", "jhoola", "hanging chair"],
            "swing": ["swing", "jhoola", "swing chair", "hanging swing"],
            "chair": ["chair", "armchair", "cane chair", "wooden chair"],
        }
        if "water bottle" in name or "bottle" in name:
            ptype = "water bottle"
        elif "basket" in name or "hamper" in name:
            ptype = "basket"
        elif "stool" in name:
            ptype = "stool"
        elif "swing chair" in name:
            ptype = "swing chair"
        elif "swing" in name or "jhoola" in name:
            ptype = "swing"
        elif "table" in name:
            ptype = "table"
        aliases = rules.get(ptype, [ptype] if ptype else [])
        return {"product_category": ptype, "aliases": aliases}

    def classify(self, product: Dict[str, Any]) -> Dict[str, Any]:
        fallback = self._fallback(product)
        client = self._get_client()
        if client is None:
            return fallback

        payload = {
            "product_name": product.get("product_name", ""),
            "product_type": product.get("product_type", ""),
            "material": product.get("material", ""),
            "features": product.get("features", []),
            "description": product.get("description", ""),
        }
        prompt = f"""
You are the PRODUCT CATEGORY CLASSIFIER for an artisan marketplace pricing
system.

Determine the concrete PRODUCT FAMILY of this product for marketplace search.
Do not broaden it into a generic craft category. For example:
- terracotta water bottle -> water bottle
- bamboo basket -> basket
- cane stool -> stool
- rattan swing chair -> swing chair

Return useful commercial synonyms that still refer to the same buyer-purpose
product. Do NOT return unrelated adjacent products such as cups for bottles,
plates for baskets, tables for stools, or sarees for furniture.

PRODUCT:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return ONLY JSON:
{{
  "product_category": "concrete category",
  "aliases": ["synonym 1", "synonym 2", "synonym 3"],
  "confidence": 0.0
}}
"""
        schema = {
            "type": "OBJECT",
            "properties": {
                "product_category": {"type": "STRING"},
                "aliases": {"type": "ARRAY", "items": {"type": "STRING"}},
                "confidence": {"type": "NUMBER"},
            },
            "required": ["product_category", "aliases", "confidence"],
        }
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"response_mime_type": "application/json", "response_schema": schema, "temperature": 0.0},
            )
            data = json.loads(getattr(response, "text", "") or "{}")
            category = str(data.get("product_category", "")).strip().lower()
            aliases = [str(x).strip().lower() for x in data.get("aliases", []) if str(x).strip()]
            try:
                confidence = float(data.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if not category or confidence < 0.60:
                return fallback
            return {"product_category": category, "aliases": list(dict.fromkeys([category, *aliases]))}
        except Exception as exc:
            print(f"  LLM product category classification failed: {exc}", flush=True)
            return fallback


# ---------------------------------------------------------------------------
# SMART QUERY GENERATOR
# ---------------------------------------------------------------------------

class SmartQueryGenerator:
    """Generate marketplace-search queries from the product attributes.

    The LLM is deliberately restricted to retrieval assistance. It does not
    retrieve products, invent listings, score similarity, or calculate prices.
    Real URLs still come from the existing web-discovery layer, and the
    existing SimilarityEngine remains responsible for product matching.
    """

    def __init__(
        self,
        model: str = LLM_QUERY_MODEL,
        queries_per_source: int = LLM_QUERIES_PER_SOURCE,
    ) -> None:
        self.model = model
        self.queries_per_source = max(1, int(queries_per_source))
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self._client = None

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        try:
            return [str(x).strip() for x in value if str(x).strip()]
        except TypeError:
            text = str(value).strip()
            return [text] if text else []

    @staticmethod
    def _normalise_query(query: Any) -> str:
        query = re.sub(r"\s+", " ", str(query or "")).strip()
        query = query.strip('\"\'')
        return query

    @staticmethod
    def _product_payload(product: Dict[str, Any]) -> Dict[str, Any]:
        features = product.get("features", [])
        if isinstance(features, str):
            features = [x.strip() for x in features.split(",") if x.strip()]
        else:
            features = [str(x).strip() for x in features if str(x).strip()]

        return {
            "product_name": str(product.get("product_name", "")),
            "product_type": str(product.get("product_type", "")),
            "material": str(product.get("material", "")),
            "additional_materials": SmartQueryGenerator._as_list(
                product.get("additional_materials", [])
            ),
            "color": str(product.get("color", "")),
            "size": str(product.get("size", "")),
            "dimensions": str(product.get("dimensions", "")),
            "dimensions_length": product.get("dimensions_length"),
            "dimensions_width": product.get("dimensions_width"),
            "dimensions_height": product.get("dimensions_height"),
            "features": features,
            "description": str(product.get("description", "")),
            "search_terms": product.get("search_terms", []),
        }

    def _fallback_queries(
        self,
        product: Dict[str, Any],
    ) -> List[str]:
        """Deterministic fallback used only when Gemini is unavailable/fails."""
        data = self._product_payload(product)
        product_type = data["product_type"]
        material = data["material"]
        size = data["size"]
        color = data["color"]
        features = data["features"]

        feature_text = " ".join(features[:3])
        aliases = [str(x).strip() for x in data.get("search_terms", []) if str(x).strip()]
        core_terms = aliases[:4] if aliases else [product_type]
        queries = [
            f"handmade {material} {core_terms[0]}",
            f"handcrafted {material} {core_terms[0]}",
            f"handwoven {material} {core_terms[1] if len(core_terms) > 1 else core_terms[0]}" if any(
                "woven" in x.lower() or "weav" in x.lower()
                for x in features
            ) else f"traditional {material} {core_terms[0]}",
            f"{size} {material} {core_terms[2] if len(core_terms) > 2 else core_terms[0]}" if size else
            f"natural {material} {core_terms[0]}",
            f"{color} {material} {core_terms[3] if len(core_terms) > 3 else core_terms[0]}" if color else
            f"artisan {material} {core_terms[0]}",
            f"{feature_text} {material} {core_terms[0]}" if feature_text else
            f"traditional handmade {material} {core_terms[0]}",
        ]
        return queries

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not self.api_key:
            return None

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            return self._client
        except Exception:
            return None

    def generate(
        self,
        product: Dict[str, Any],
        source_name: str,
        source_domain: str,
    ) -> List[str]:
        """Return several independent search-bar-style queries for one source."""
        fallback = self._fallback_queries(product)
        client = self._get_client()

        if client is None:
            return fallback[:self.queries_per_source]

        payload = json.dumps(
            self._product_payload(product),
            ensure_ascii=False,
            indent=2,
        )

        prompt = f"""
You are the marketplace retrieval query generator for SIH26090.

Generate {self.queries_per_source} SHORT search-bar queries for finding
real products comparable to the artisan product below on this marketplace:
{source_name} ({source_domain})

PRODUCT ATTRIBUTES:
{payload}

RULES:
1. Preserve the exact product type and primary material.
2. Treat product type and primary material as mandatory anchors.
3. Use size, dimensions, color and craft features when useful.
4. You may use close wording variants such as handmade/handcrafted,
   handwoven/woven, traditional craft, storage basket, etc.
5. Do NOT change bamboo into rattan/cane/wood unless it is explicitly an
   additional material in the product.
6. Do NOT change the product type.
7. Do NOT include prices, seller names, fake product names, URLs, or
   marketplace names in the query.
8. Do NOT return queries containing site: because the application adds the
   source-domain restriction itself.
9. Queries should look like normal phrases a user would type into a
   marketplace search box.
10. Make the queries meaningfully different from each other so retrieval
    covers exact, feature-focused and semantic variants.

Return ONLY valid JSON matching this schema:
{{
  "queries": ["query 1", "query 2", "..."]
}}
"""

        schema = {
            "type": "OBJECT",
            "properties": {
                "queries": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                }
            },
            "required": ["queries"],
        }

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                    "temperature": 0.2,
                },
            )

            raw = getattr(response, "text", "") or ""
            data = json.loads(raw)
            queries = data.get("queries", []) if isinstance(data, dict) else []

            cleaned = []
            seen = set()
            for query in self._as_list(queries):
                query = self._normalise_query(query)
                if not query or len(query) > LLM_MAX_QUERY_LENGTH:
                    continue
                if "site:" in query.lower():
                    continue
                key = query.lower()
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(query)
                if len(cleaned) >= self.queries_per_source:
                    break

            if cleaned:
                return cleaned
        except Exception as exc:
            print(f"  LLM query generation failed for {source_name}: {exc}")
            print("  Falling back to deterministic attribute queries.")

        return fallback[:self.queries_per_source]


# ---------------------------------------------------------------------------
# LLM CANDIDATE ANALYZER
# ---------------------------------------------------------------------------

class LLMCandidateAnalyzer:
    """
    Semantic marketplace-product gate.

    Search engines are allowed to return noisy URLs.  A listing becomes a
    marketplace candidate ONLY after this analyzer decides that it is relevant
    to the target artisan product.

    This is deliberately different from similarity.py:
        - this class answers: "Should this marketplace product enter our pool?"
        - similarity.py answers: "How similar is the accepted product?"

    No numeric price is produced here.
    """

    def __init__(
        self,
        model: str = LLM_CLASSIFIER_MODEL,
        batch_size: int = LLM_CLASSIFIER_BATCH_SIZE,
        min_confidence: float = LLM_CLASSIFIER_MIN_CONFIDENCE,
    ):
        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.min_confidence = float(min_confidence)
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            return None
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            return self._client
        except Exception:
            return None

    @staticmethod
    def _target_payload(product: Dict[str, Any]) -> Dict[str, Any]:
        features = product.get("features", [])
        if isinstance(features, str):
            features = [x.strip() for x in features.split(",") if x.strip()]
        return {
            "product_name": str(product.get("product_name", "")),
            "product_type": str(product.get("product_type", "")),
            "material": str(product.get("material", "")),
            "additional_materials": product.get("additional_materials", []),
            "color": str(product.get("color", "")),
            "size": str(product.get("size", "")),
            "dimensions": str(product.get("dimensions", "")),
            "features": features,
            "description": str(product.get("description", "")),
        }

    @staticmethod
    def _candidate_payload(index: int, listing: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": index,
            "title": str(listing.get("title", "")),
            "product_type": str(listing.get("product_type", "")),
            "material": str(listing.get("material", "")),
            "color": str(listing.get("color", "")),
            "size": str(listing.get("size", "")),
            "dimensions": str(listing.get("dimensions", "")),
            "features": listing.get("features", []),
            "description": str(listing.get("description", ""))[:1200],
            "price": listing.get("price"),
        }

    def _classify_batch(
        self,
        product: Dict[str, Any],
        listings: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        client = self._get_client()
        if client is None:
            raise RuntimeError(
                "GEMINI_API_KEY is required for marketplace candidate analysis. "
                "No listing is accepted without the analyzer."
            )

        target = json.dumps(
            self._target_payload(product),
            ensure_ascii=False,
            indent=2,
        )
        candidates = [
            self._candidate_payload(i, listing)
            for i, listing in enumerate(listings)
        ]

        prompt = f"""
You are the PRODUCT RELEVANCE ANALYZER for SIH26090 marketplace pricing.

Your job is NOT to calculate price and NOT to perform final similarity scoring.
Your only job is to decide whether each marketplace listing is a valid product
candidate for the target artisan product.

TARGET ARTISAN PRODUCT:
{target}

MARKETPLACE LISTINGS:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

CLASSIFICATION LABELS:
- PRIMARY: same core product/category and same buyer purpose. Material may differ.
- RELATED: closely related product family or use, but not the same exact product.
- REJECT: different product category, different buyer purpose, accessory/part,
  generic page, bundle with a different core product, or clearly unrelated item.

CRITICAL RULES:
1. PRODUCT IDENTITY and BUYER PURPOSE are the strongest signals.
2. Do NOT accept a product merely because one word overlaps.
3. Do NOT accept cups/mugs/plates as water bottles just because they are pottery.
4. Do NOT accept sarees, paintings, vases, jewellery, etc. for a bottle/basket/stool/table.
5. Material mismatch alone is NOT a rejection when the product identity is the same.
6. Synonyms and commercial naming variants are allowed:
   bottle/flask/earthen bottle can be related to a water bottle;
   jhoola/swing/swing chair can be related when the target is a swing product;
   hamper/basket can be related when the target is a basket.
7. Judge the actual product described by the listing, not the marketplace domain.
8. Price must NEVER determine relevance.
9. If the listing is ambiguous, prefer REJECT rather than inventing relevance.
10. Return one decision for EVERY input id.

Return ONLY JSON matching this schema:
{{
  "results": [
    {{
      "id": 0,
      "decision": "PRIMARY|RELATED|REJECT",
      "confidence": 0.0,
      "reason": "short reason"
    }}
  ]
}}
"""

        schema = {
            "type": "OBJECT",
            "properties": {
                "results": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id": {"type": "INTEGER"},
                            "decision": {"type": "STRING"},
                            "confidence": {"type": "NUMBER"},
                            "reason": {"type": "STRING"},
                        },
                        "required": ["id", "decision", "confidence", "reason"],
                    },
                }
            },
            "required": ["results"],
        }

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": schema,
                "temperature": 0.0,
            },
        )
        raw = getattr(response, "text", "") or ""
        data = json.loads(raw)
        return data.get("results", []) if isinstance(data, dict) else []

    def filter_candidates(
        self,
        product: Dict[str, Any],
        listings: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Classify every extracted listing and keep only LLM-approved ones."""
        if not listings:
            return []

        accepted: List[Dict[str, Any]] = []
        print(
            f"  LLM product analyzer: evaluating {len(listings)} fetched listings...",
            flush=True,
        )

        for start in range(0, len(listings), self.batch_size):
            batch = listings[start:start + self.batch_size]
            try:
                decisions = self._classify_batch(product, batch)
            except Exception as exc:
                # Fail closed: no analyzer decision means no candidate.
                print(
                    f"  LLM product analyzer failed for batch {start // self.batch_size + 1}: {exc}",
                    flush=True,
                )
                continue

            by_id = {}
            for decision in decisions:
                try:
                    by_id[int(decision.get("id"))] = decision
                except (TypeError, ValueError):
                    continue

            for local_id, listing in enumerate(batch):
                decision = by_id.get(local_id)
                if not decision:
                    continue

                label = str(decision.get("decision", "REJECT")).upper().strip()
                try:
                    confidence = float(decision.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                confidence = max(0.0, min(1.0, confidence))

                listing["llm_relevance"] = label
                listing["llm_relevance_confidence"] = confidence
                listing["llm_relevance_reason"] = str(decision.get("reason", ""))

                if label in {"PRIMARY", "RELATED"} and confidence >= self.min_confidence:
                    accepted.append(listing)
                    print(
                        f"  [LLM {label}] {listing.get('title', '')[:90]} "
                        f"| ₹{listing.get('price')} | confidence={confidence:.2f}",
                        flush=True,
                    )

        return accepted


# ---------------------------------------------------------------------------
# COLLECTOR
# ---------------------------------------------------------------------------

class HandmadeMarketplaceCollector:
    """
    Generic product collector.

    It intentionally does not contain marketplace-specific CAPTCHA bypasses.
    Each source is searched normally, then the product page is parsed using
    JSON-LD, meta tags and visible page text.
    """

    def __init__(
        self,
        primary_sources: Optional[Dict[str, str]] = None,
        secondary_sources: Optional[Dict[str, str]] = None,
    ):
        self.primary_sources = primary_sources or PRIMARY_SOURCES
        self.secondary_sources = secondary_sources or SECONDARY_SOURCES

        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.query_generator = SmartQueryGenerator()
        self.candidate_analyzer = LLMCandidateAnalyzer()
        self._sitemap_cache = {}
        self._query_log = {}

    # -----------------------------------------------------------------------
    # BASIC HELPERS
    # -----------------------------------------------------------------------

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, (list, tuple)):
            value = " ".join(str(x) for x in value)

        return re.sub(r"\s+", " ", str(value)).strip()

    @staticmethod
    def _parse_price(value: Any) -> Optional[float]:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            value = float(value)
            return value if value >= 0 else None

        text = str(value).replace(",", "")
        matches = re.findall(r"\d+(?:\.\d+)?", text)

        if not matches:
            return None

        try:
            price = float(matches[-1])
            return price if price >= 0 else None
        except ValueError:
            return None

    @staticmethod
    def _parse_int(value: Any) -> Optional[int]:
        if value is None:
            return None

        match = re.search(r"\d[\d,]*", str(value))
        if not match:
            return None

        try:
            return int(match.group(0).replace(",", ""))
        except ValueError:
            return None

    @staticmethod
    def _parse_float(value: Any) -> Optional[float]:
        if value is None:
            return None

        match = re.search(r"\d+(?:\.\d+)?", str(value))
        if not match:
            return None

        try:
            return float(match.group(0))
        except ValueError:
            return None

    @staticmethod
    def _make_listing_id(marketplace: str, url: str) -> str:
        digest = hashlib.sha1(
            f"{marketplace}|{url}".encode("utf-8")
        ).hexdigest()[:16]

        prefix = re.sub(
            r"[^A-Z0-9]",
            "",
            marketplace.upper(),
        )[:8]

        return f"{prefix}-{digest}"

    @staticmethod
    def _domain_allowed(url: str, domain: str) -> bool:
        try:
            host = urlparse(url).netloc.lower()
            domain = domain.lower().replace("www.", "")
            return host == domain or host.endswith("." + domain)
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # SEARCH ENGINE
    # -----------------------------------------------------------------------

    def _resolve_search_href(self, href: str) -> str:
        """Resolve common search-engine redirect wrappers to the real URL."""
        href = (href or "").strip()
        if not href:
            return ""
        parsed = urlparse(href)
        if parsed.path.startswith("/l/"):
            target = parse_qs(parsed.query).get("uddg", [""])[0]
            if target:
                return unquote(target)
        for key in ("url", "u", "target"):
            target = parse_qs(parsed.query).get(key, [""])[0]
            if target.startswith(("http://", "https://")):
                return unquote(target)
        return href

    def _search_duckduckgo(self, query: str, allowed_domain: str, limit: int) -> List[str]:
        url = "https://html.duckduckgo.com/html/?q=" + quote(query)
        try:
            response = requests.get(url, headers=HEADERS, timeout=SEARCH_TIMEOUT)
            if response.status_code != 200:
                return []
            soup = BeautifulSoup(response.text, "html.parser")
            links = []
            for anchor in soup.select("a.result__a"):
                href = self._resolve_search_href(anchor.get("href", ""))
                if self._domain_allowed(href, allowed_domain) and href not in links:
                    links.append(href)
                    if len(links) >= limit:
                        break
            return links
        except requests.RequestException:
            return []

    def _search_bing(self, query: str, allowed_domain: str, limit: int) -> List[str]:
        url = "https://www.bing.com/search?q=" + quote(query)
        try:
            response = requests.get(url, headers=HEADERS, timeout=SEARCH_TIMEOUT)
            if response.status_code != 200:
                return []
            soup = BeautifulSoup(response.text, "html.parser")
            links = []
            for anchor in soup.select("li.b_algo h2 a"):
                href = self._resolve_search_href(anchor.get("href", ""))
                if self._domain_allowed(href, allowed_domain) and href not in links:
                    links.append(href)
                    if len(links) >= limit:
                        break
            return links
        except requests.RequestException:
            return []

    def _direct_search_urls(self, base_url: str, query: str) -> List[str]:
        q = quote(query)
        base = base_url.rstrip("/")
        domain = urlparse(base_url).netloc.lower().replace("www.", "")

        # Marketplace-specific public search pages first.
        if domain == "amazon.in":
            return [
                f"https://www.amazon.in/s?k={q}",
                f"{base}/search?q={q}",
            ]

        if domain == "flipkart.com":
            return [
                f"https://www.flipkart.com/search?q={q}",
                f"{base}/search?q={q}",
            ]

        return [
            f"{base}/search?q={q}",
            f"{base}/?s={q}&post_type=product",
            f"{base}/products/?q={q}",
            f"{base}/products/search?q={q}",
        ]

    def _search_direct_marketplace(self, base_url: str, allowed_domain: str, query: str, limit: int) -> List[str]:
        candidates = []
        for search_url in self._direct_search_urls(base_url, query):
            try:
                response = requests.get(search_url, headers=HEADERS, timeout=SEARCH_TIMEOUT)
                if response.status_code != 200:
                    continue
                soup = BeautifulSoup(response.text, "html.parser")
                for obj in self._jsonld_objects(soup):
                    obj_type = obj.get("@type", "")
                    types = {str(x).lower() for x in obj_type} if isinstance(obj_type, list) else {str(obj_type).lower()}
                    if "product" in types:
                        obj_url = obj.get("url")
                        if obj_url:
                            absolute = urljoin(search_url, str(obj_url))
                            if self._domain_allowed(absolute, allowed_domain) and absolute not in candidates:
                                candidates.append(absolute)
                for anchor in soup.select("a[href]"):
                    href = urljoin(search_url, anchor.get("href", "").strip())
                    text = self._clean_text(anchor.get_text(" ", strip=True))
                    href_lower = href.lower()
                    if not self._domain_allowed(href, allowed_domain):
                        continue

                    # Amazon product detail links.
                    is_amazon_product = (
                        allowed_domain == "amazon.in"
                        and ("/dp/" in href_lower or "/gp/product/" in href_lower)
                    )

                    # Flipkart product detail links normally contain /p/.
                    is_flipkart_product = (
                        allowed_domain == "flipkart.com"
                        and "/p/" in href_lower
                    )

                    generic_product = (
                        "/product/" in href_lower
                        or "/products/" in href_lower
                        or "product" in text.lower()
                    )

                    if (is_amazon_product or is_flipkart_product or generic_product) and href not in candidates:
                        candidates.append(href)
                    if len(candidates) >= limit:
                        return candidates[:limit]
            except requests.RequestException:
                continue
        return candidates[:limit]

    def _sitemap_urls(self, base_url: str, allowed_domain: str) -> List[str]:
        roots = [
            urljoin(base_url.rstrip("/") + "/", "sitemap.xml"),
            urljoin(base_url.rstrip("/") + "/", "sitemap_index.xml"),
        ]
        visited = set()
        product_urls = []
        queue = list(roots)
        while queue and len(visited) < 20 and len(product_urls) < MAX_SITEMAP_URLS:
            sitemap_url = queue.pop(0)
            if sitemap_url in visited:
                continue
            visited.add(sitemap_url)
            try:
                response = requests.get(sitemap_url, headers=HEADERS, timeout=SITEMAP_TIMEOUT)
                if response.status_code != 200:
                    continue
                root = ET.fromstring(response.content)
            except (requests.RequestException, ET.ParseError):
                continue
            for loc in root.iter():
                if loc.tag.lower().endswith("loc") and loc.text:
                    target = loc.text.strip()
                    if target.endswith(".xml") or "sitemap" in target.lower():
                        if target not in visited:
                            queue.append(target)
                    elif self._domain_allowed(target, allowed_domain):
                        product_urls.append(target)
                        if len(product_urls) >= MAX_SITEMAP_URLS:
                            break
        return list(dict.fromkeys(product_urls))

    def _search_sitemap(self, base_url: str, allowed_domain: str, query: str, limit: int) -> List[str]:
        cache_key = allowed_domain
        if cache_key not in self._sitemap_cache:
            self._sitemap_cache[cache_key] = self._sitemap_urls(base_url, allowed_domain)
        urls = self._sitemap_cache[cache_key]
        if not urls:
            return []
        tokens = [
            token.lower()
            for token in re.findall(r"[a-zA-Z0-9]+", query)
            if len(token) >= 3 and token.lower() not in {"the", "and", "for", "with"}
        ]
        scored = []
        for url in urls:
            value = url.lower()
            score = sum(1 for token in tokens if token in value)
            if score > 0:
                scored.append((score, url))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [url for _, url in scored[:limit]]

    def _search_web(
        self,
        query: str,
        allowed_domain: str,
        limit: int = 8,
        base_url: Optional[str] = None,
    ) -> List[str]:
        """
        Discover marketplace product URLs.

        Retrieval only:
          DDG + Bing + direct marketplace search + sitemap fallback.

        IMPORTANT:
        This function does NOT perform similarity matching.
        It deliberately collects a broad candidate URL pool. The
        SimilarityEngine is responsible for deciding comparability.
        """
        limit = max(1, int(limit))
        results: List[str] = []
        seen = set()

        def add_urls(urls):
            for url in urls or []:
                if not url:
                    continue
                normalized = str(url).strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    results.append(normalized)

        # Search engines in parallel.
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    self._search_duckduckgo,
                    query,
                    allowed_domain,
                    limit,
                ),
                executor.submit(
                    self._search_bing,
                    query,
                    allowed_domain,
                    limit,
                ),
            ]

            for future in as_completed(futures):
                try:
                    add_urls(future.result())
                except Exception:
                    pass

        # If search engines do not understand the site-qualified query, retry
        # with the plain marketplace phrase and still enforce the target domain.
        if not results and base_url and allowed_domain in {"amazon.in", "flipkart.com"}:
            plain_query = re.sub(r"^site:[^ ]+\s*", "", query, flags=re.IGNORECASE).strip()
            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(self._search_duckduckgo, plain_query, allowed_domain, limit),
                        executor.submit(self._search_bing, plain_query, allowed_domain, limit),
                    ]
                    for future in as_completed(futures):
                        try:
                            add_urls(future.result())
                        except Exception:
                            pass
            except Exception:
                pass

        # Direct marketplace search ALWAYS gets a chance.
        if base_url and len(results) < limit:
            try:
                direct = self._search_direct_marketplace(
                    base_url,
                    allowed_domain,
                    query,
                    limit,
                )
                add_urls(direct)
            except Exception:
                pass

        # Sitemap is a controlled fallback, not the first choice.
        if base_url and len(results) < limit:
            try:
                sitemap = self._search_sitemap(
                    base_url,
                    allowed_domain,
                    query,
                    limit,
                )
                add_urls(sitemap)
            except Exception:
                pass

        return results[:limit]

    # -----------------------------------------------------------------------
    # PAGE FETCHING
    # -----------------------------------------------------------------------

    def _fetch_page(
        self,
        url: str,
    ) -> Optional[BeautifulSoup]:
        try:
            response = self.session.get(
                url,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                return None

            content_type = response.headers.get(
                "Content-Type",
                "",
            ).lower()

            if "html" not in content_type:
                return None

            return BeautifulSoup(
                response.text,
                "html.parser",
            )

        except requests.RequestException:
            return None

    # -----------------------------------------------------------------------
    # JSON-LD
    # -----------------------------------------------------------------------

    @staticmethod
    def _jsonld_objects(
        soup: BeautifulSoup,
    ) -> List[Dict[str, Any]]:
        objects = []

        for script in soup.select(
            'script[type="application/ld+json"]'
        ):
            raw = script.string or script.get_text()

            if not raw:
                continue

            try:
                data = json.loads(raw)
            except Exception:
                continue

            if isinstance(data, dict):
                objects.append(data)

                graph = data.get("@graph")

                if isinstance(graph, list):
                    objects.extend(
                        x for x in graph
                        if isinstance(x, dict)
                    )

            elif isinstance(data, list):
                objects.extend(
                    x for x in data
                    if isinstance(x, dict)
                )

        return objects

    @staticmethod
    def _find_product_jsonld(
        objects: Iterable[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        for obj in objects:
            obj_type = obj.get("@type", "")

            if isinstance(obj_type, list):
                types = {
                    str(x).lower()
                    for x in obj_type
                }
            else:
                types = {
                    str(obj_type).lower()
                }

            if (
                "product" in types
                or "productgroup" in types
            ):
                return obj

        return None

    # -----------------------------------------------------------------------
    # FIELD EXTRACTION
    # -----------------------------------------------------------------------

    def _extract_meta(
        self,
        soup: BeautifulSoup,
        *names: str,
    ) -> str:
        for name in names:
            tag = soup.find(
                "meta",
                attrs={
                    "property": name,
                },
            )

            if not tag:
                tag = soup.find(
                    "meta",
                    attrs={
                        "name": name,
                    },
                )

            if tag and tag.get("content"):
                return self._clean_text(
                    tag["content"]
                )

        return ""

    def _extract_features(
        self,
        soup: BeautifulSoup,
        text: str,
    ) -> List[str]:
        features = []

        feature_terms = [
            "handmade",
            "handcrafted",
            "handwoven",
            "hand woven",
            "traditional",
            "artisan",
            "weaving",
            "loom",
            "natural",
            "organic",
            "block print",
            "embroidered",
            "embroidery",
            "carved",
            "woven",
            "khadi",
            "ikat",
            "ajrakh",
            "kantha",
            "dabu",
        ]

        normalized = text.lower()

        for term in feature_terms:
            if term in normalized:
                features.append(term)

        # Add useful list items containing craft signals.
        for item in soup.select("li"):
            value = self._clean_text(
                item.get_text(" ", strip=True)
            )

            if (
                5 <= len(value) <= 120
                and any(
                    term in value.lower()
                    for term in feature_terms
                )
            ):
                if value not in features:
                    features.append(value)

        return features[:15]

    def _extract_dimensions(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """
        Parse common forms such as:
            30 x 20 cm
            30x20x10 cm
            L 30 W 20 H 10 cm
        """
        patterns = [
            r"(\d+(?:\.\d+)?)\s*[x×]\s*"
            r"(\d+(?:\.\d+)?)\s*[x×]\s*"
            r"(\d+(?:\.\d+)?)\s*(?:cm|inch|in)?",

            r"(\d+(?:\.\d+)?)\s*[x×]\s*"
            r"(\d+(?:\.\d+)?)\s*(?:cm|inch|in)?",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                values = [
                    float(x)
                    for x in match.groups()
                    if x is not None
                ]

                dimensions = " x ".join(
                    f"{x:g}"
                    for x in values
                )

                return {
                    "dimensions": dimensions,
                    "dimensions_length": values[0]
                    if len(values) >= 1 else None,
                    "dimensions_width": values[1]
                    if len(values) >= 2 else None,
                    "dimensions_height": values[2]
                    if len(values) >= 3 else None,
                }

        return {
            "dimensions": "",
            "dimensions_length": None,
            "dimensions_width": None,
            "dimensions_height": None,
        }

    def _extract_product(
        self,
        marketplace: str,
        url: str,
        soup: BeautifulSoup,
        source_tier: str,
    ) -> Optional[MarketplaceCandidate]:

        objects = self._jsonld_objects(soup)
        product_json = self._find_product_jsonld(
            objects
        )

        title = ""

        if product_json:
            title = self._clean_text(
                product_json.get("name")
            )

        if not title:
            title = self._extract_meta(
                soup,
                "og:title",
                "twitter:title",
            )

        if not title and soup.title:
            title = self._clean_text(
                soup.title.get_text()
            )

        description = ""

        if product_json:
            description = self._clean_text(
                product_json.get("description")
            )

        if not description:
            description = self._extract_meta(
                soup,
                "description",
                "og:description",
            )

        visible_text = self._clean_text(
            soup.get_text(" ", strip=True)
        )

        combined_text = (
            f"{title} {description} {visible_text}"
        )

        # ---------------------------------------------------------------
        # PRICE
        # ---------------------------------------------------------------

        price = None

        if product_json:
            offers = product_json.get(
                "offers"
            )

            if isinstance(offers, dict):
                price = self._parse_price(
                    offers.get("price")
                )

            elif isinstance(offers, list):
                for offer in offers:
                    if isinstance(offer, dict):
                        price = self._parse_price(
                            offer.get("price")
                        )

                        if price is not None:
                            break

        if price is None:
            meta_price = self._extract_meta(
                soup,
                "product:price:amount",
                "og:price:amount",
            )

            price = self._parse_price(
                meta_price
            )

        if price is None:
            # Common visible price patterns.
            matches = re.findall(
                r"(?:₹|Rs\.?|INR)\s*"
                r"([0-9][0-9,]*(?:\.[0-9]+)?)",
                visible_text,
                flags=re.IGNORECASE,
            )

            if matches:
                price = self._parse_price(
                    matches[0]
                )

        if price is None or price <= 0:
            return None

        # ---------------------------------------------------------------
        # RATING / REVIEWS
        # ---------------------------------------------------------------

        rating = None
        review_count = None

        if product_json:
            aggregate = product_json.get(
                "aggregateRating"
            )

            if isinstance(aggregate, dict):
                rating = self._parse_float(
                    aggregate.get(
                        "ratingValue"
                    )
                )

                review_count = self._parse_int(
                    aggregate.get(
                        "reviewCount"
                    )
                )

        # ---------------------------------------------------------------
        # SELLER
        # ---------------------------------------------------------------

        seller_name = None

        if product_json:
            brand = product_json.get("brand")

            if isinstance(brand, dict):
                seller_name = self._clean_text(
                    brand.get("name")
                )
            elif brand:
                seller_name = self._clean_text(
                    brand
                )

        # ---------------------------------------------------------------
        # PRODUCT ATTRIBUTES
        # ---------------------------------------------------------------

        material = ""
        color = ""
        size = ""

        if product_json:
            material = self._clean_text(
                product_json.get(
                    "material"
                )
            )

            color = self._clean_text(
                product_json.get(
                    "color"
                )
            )

            size = self._clean_text(
                product_json.get(
                    "size"
                )
            )

        # Search common attribute labels.
        if not material:
            match = re.search(
                r"(?:material|fabric)\s*[:\-]\s*"
                r"([^|;,]{2,80})",
                combined_text,
                flags=re.IGNORECASE,
            )
            if match:
                material = self._clean_text(
                    match.group(1)
                )

        if not color:
            match = re.search(
                r"color\s*[:\-]\s*"
                r"([^|;,]{2,50})",
                combined_text,
                flags=re.IGNORECASE,
            )
            if match:
                color = self._clean_text(
                    match.group(1)
                )

        if not size:
            match = re.search(
                r"size\s*[:\-]\s*"
                r"([^|;,]{1,30})",
                combined_text,
                flags=re.IGNORECASE,
            )
            if match:
                size = self._clean_text(
                    match.group(1)
                )

        dimension_data = self._extract_dimensions(
            combined_text
        )

        features = self._extract_features(
            soup,
            combined_text,
        )

        # ---------------------------------------------------------------
        # HANDMADE EVIDENCE
        # ---------------------------------------------------------------

        evidence_terms = [
            "handmade",
            "handcrafted",
            "handwoven",
            "hand woven",
            "artisan",
            "karigar",
            "traditional craft",
            "handloom",
        ]

        evidence = [
            term
            for term in evidence_terms
            if term in combined_text.lower()
        ]

        handmade_evidence = (
            ", ".join(evidence)
            if evidence
            else None
        )

        # ---------------------------------------------------------------
        # PRODUCT TYPE
        # ---------------------------------------------------------------

        # Product identity is title-first. Marketplace JSON-LD categories
        # can be broad or incorrect, so they are only used as a fallback.
        product_type = self._infer_product_type(
            title,
            "",
        )

        if not product_type and product_json:
            category = product_json.get("category")
            if category:
                product_type = self._clean_text(category)

        if not product_type:
            product_type = self._infer_product_type(
                title,
                combined_text,
            )

        listing_id = self._make_listing_id(
            marketplace,
            url,
        )

        return MarketplaceCandidate(
            listing_id=listing_id,
            marketplace=marketplace,
            title=title,
            price=float(price),
            url=url,
            rating=rating,
            review_count=review_count,
            seller_name=seller_name,
            handmade_evidence=handmade_evidence,
            product_type=product_type,
            material=material,
            color=color,
            size=size,
            dimensions=dimension_data[
                "dimensions"
            ],
            dimensions_length=dimension_data[
                "dimensions_length"
            ],
            dimensions_width=dimension_data[
                "dimensions_width"
            ],
            dimensions_height=dimension_data[
                "dimensions_height"
            ],
            features=features,
            description=description,
            source_tier=source_tier,
        )

    # -----------------------------------------------------------------------
    # PRODUCT TYPE INFERENCE
    # -----------------------------------------------------------------------

    @staticmethod
    def _infer_product_type(
        title: str,
        text: str,
    ) -> str:
        """
        Infer a concrete product family.

        Title has priority. Broad words such as "home", "decor",
        "storage", "furniture", etc. are intentionally not product types.
        """
        title_value = re.sub(r"\s+", " ", str(title or "")).lower().strip()
        full_value = f"{title_value} {str(text or '').lower()}"

        type_map = {
            "swing": [
                "swing chair",
                "hanging swing",
                "swing",
            ],
            "water bottle": [
                "water bottle",
                "bottle",
                "flask",
            ],
            "basket": [
                "basket",
                "hamper",
            ],
            "stool": [
                "stool",
                "ottoman",
            ],
            "chair": [
                "chair",
                "armchair",
            ],
            "table": [
                "table",
                "coffee table",
            ],
            "lamp": [
                "lamp",
                "lantern",
                "light",
            ],
            "bag": [
                "bag",
                "tote",
                "clutch",
            ],
            "saree": [
                "saree",
                "sari",
            ],
            "dupatta": [
                "dupatta",
            ],
            "stole": [
                "stole",
            ],
            "shawl": [
                "shawl",
            ],
            "bedsheet": [
                "bedsheet",
                "bed sheet",
            ],
            "rug": [
                "rug",
                "dhurrie",
                "durrie",
            ],
            "cushion": [
                "cushion",
                "pillow cover",
            ],
            "pottery": [
                "pottery",
                "terracotta pot",
                "earthen pot",
            ],
            "jug": [
                "jug",
                "pitcher",
            ],
        }

        # Long/more-specific phrases first.
        for product_type, terms in type_map.items():
            for term in sorted(terms, key=len, reverse=True):
                if re.search(r"\b" + re.escape(term) + r"\b", title_value):
                    return product_type

        # Only use broader page text as a fallback.
        for product_type, terms in type_map.items():
            for term in sorted(terms, key=len, reverse=True):
                if re.search(r"\b" + re.escape(term) + r"\b", full_value):
                    return product_type

        return ""

    # -----------------------------------------------------------------------
    # SOURCE SEARCH
    # -----------------------------------------------------------------------

    def _source_domain(
        self,
        base_url: str,
    ) -> str:
        host = urlparse(base_url).netloc.lower()

        if host.startswith("www."):
            host = host[4:]

        return host

    def search_source(self, source_name: str, base_url: str, product: Dict[str, Any], limit: int, source_tier: str) -> List[Dict[str, Any]]:
        """Run all six queries for one marketplace concurrently, then fetch pages concurrently."""
        domain = self._source_domain(base_url)
        category_info = self.category_classifier.classify(product)
        retrieval_product = dict(product)
        retrieval_product["product_type"] = category_info.get("product_category") or product.get("product_type", "")
        retrieval_product["search_terms"] = category_info.get("aliases", [])
        print(f"  LLM category: {retrieval_product.get('product_type', '')} | aliases: {retrieval_product.get('search_terms', [])[:6]}", flush=True)
        queries = self.query_generator.generate(product=retrieval_product, source_name=source_name, source_domain=domain)
        self._query_log[source_name] = list(queries)
        raw_urls = []
        print(f"  Searching {source_name} across web + direct marketplace search...", flush=True)
        with ThreadPoolExecutor(max_workers=min(len(queries), SEARCH_WORKERS)) as executor:
            futures = [
                executor.submit(self._search_web, f"site:{domain} {query}", domain, limit, base_url)
                for query in queries
            ]
            for future in as_completed(futures):
                try:
                    raw_urls.extend(future.result())
                except Exception:
                    pass
        raw_urls = list(dict.fromkeys(raw_urls))[:MAX_PRIMARY_CANDIDATES]

        # DEBUG/VERIFICATION MODE: show exactly what the search engine found
        # BEFORE the LLM analyzer or any relevance decision. This lets us
        # verify whether the marketplace actually contains useful products.
        print("", flush=True)
        print("  " + "=" * 78, flush=True)
        print("  DISCOVERED URL AUDIT - BEFORE LLM ANALYSIS", flush=True)
        print(f"  Target product: {product.get('product_name', '')}", flush=True)
        print(f"  Target type   : {product.get('product_type', '')}", flush=True)
        print(f"  Target material: {product.get('material', '')}", flush=True)
        print(f"  URLs discovered: {len(raw_urls)}", flush=True)
        print("  " + "=" * 78, flush=True)

        results = []
        discovered_audit = []
        with ThreadPoolExecutor(max_workers=min(PAGE_WORKERS, max(1, len(raw_urls)))) as executor:
            future_map = {executor.submit(self._fetch_page, url): url for url in raw_urls}
            for future in as_completed(future_map):
                url = future_map[future]
                try:
                    soup = future.result()
                except Exception:
                    soup = None

                if soup is None:
                    discovered_audit.append((url, "[PAGE FETCH FAILED]"))
                    continue

                try:
                    candidate = self._extract_product(
                        marketplace=source_name,
                        url=url,
                        soup=soup,
                        source_tier=source_tier,
                    )
                except Exception:
                    candidate = None

                if candidate is None:
                    # Try to expose the HTML title even if product extraction
                    # failed, so the user can still inspect the discovered URL.
                    page_title = soup.title.get_text(" ", strip=True) if soup.title else "[PRODUCT EXTRACTION FAILED]"
                    discovered_audit.append((url, page_title))
                    continue

                listing = candidate.to_listing()
                discovered_audit.append((url, listing.get("title", "[NO PRODUCT TITLE]")))

                if self._candidate_is_usable(listing, product):
                    results.append(listing)

        # Print in stable URL-discovery order rather than completion order.
        audit_map = {url: title for url, title in discovered_audit}
        for index, url in enumerate(raw_urls, start=1):
            print(f"  [{index:02d}] PRODUCT: {audit_map.get(url, '[NOT FETCHED]')}", flush=True)
            print(f"       URL    : {url}", flush=True)
        print("  " + "=" * 78, flush=True)

        # The search engine only finds pages. The LLM decides which extracted
        # products are actually relevant to the target artisan product.
        fetched = self.deduplicate(results)
        results = self.candidate_analyzer.filter_candidates(product, fetched)
        results = self.deduplicate(results)
        print(
            f"  [{source_name}] URLs discovered: {len(raw_urls)}",
            flush=True,
        )
        print(
            f"  [{source_name}] final usable: {len(results)}",
            flush=True,
        )
        return results

    @staticmethod
    def _generic_page_title(title: str, url: str = "") -> bool:
        value = re.sub(r"\s+", " ", str(title or "")).strip().lower()

        blocked = (
            "gift card",
            "shop all",
            "shop best",
            "homepage",
            "home |",
            "search results",
            "collection",
            "category",
            "all products",
            "contact us",
            "about us",
            "login",
            "sign up",
        )

        return any(term in value for term in blocked)

    @staticmethod
    def _effective_target_type(product: Dict[str, Any]) -> str:
        """Resolve the concrete product family used for retrieval hygiene.

        The upstream product may carry a broad craft cluster or an incorrect
        product_type.  The product title is therefore used as the authoritative
        source when it contains a concrete product family.
        """
        name = str(product.get("product_name", "") or product.get("title", ""))
        inferred = HandmadeMarketplaceCollector._infer_product_type(name, "")
        if inferred:
            return inferred
        return str(product.get("product_type", "") or "").strip().lower()

    @classmethod
    def _candidate_is_usable(
        cls,
        listing: Dict[str, Any],
        product: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Only perform non-semantic hygiene before LLM analysis.

        IMPORTANT: product relevance is NOT decided here.  The LLM candidate
        analyzer is the gate.  This method only prevents broken pages and
        invalid prices from being sent to the analyzer.
        """
        title = re.sub(r"\s+", " ", str(listing.get("title", ""))).strip()
        url = str(listing.get("url", "") or "")

        if not title or cls._generic_page_title(title, url):
            return False

        try:
            price = float(listing.get("price"))
        except (TypeError, ValueError):
            return False

        return price > 0


    # -----------------------------------------------------------------------
    # HANDLOOM DETECTION
    # -----------------------------------------------------------------------

    @staticmethod
    def is_handloom_product(
        product: Dict[str, Any],
    ) -> bool:
        values = [
            product.get(
                "product_name",
                ""
            ),
            product.get(
                "product_type",
                ""
            ),
            product.get(
                "material",
                ""
            ),
            product.get(
                "description",
                ""
            ),
        ]

        features = product.get(
            "features",
            []
        )

        if isinstance(features, str):
            values.append(features)
        else:
            values.extend(
                str(x)
                for x in features
            )

        text = " ".join(
            str(x).lower()
            for x in values
        )

        return any(
            term in text
            for term in HANDLOOM_TERMS
        )

    # -----------------------------------------------------------------------
    # DEDUPLICATION
    # -----------------------------------------------------------------------

    @staticmethod
    def deduplicate(
        listings: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results = []
        seen_urls = set()
        seen_keys = set()

        for listing in listings:
            url = str(
                listing.get(
                    "url",
                    ""
                )
            ).strip()

            title = re.sub(
                r"\s+",
                " ",
                str(
                    listing.get(
                        "title",
                        ""
                    )
                ).lower(),
            ).strip()

            price = listing.get(
                "price"
            )

            if url and url in seen_urls:
                continue

            key = (
                title,
                str(price),
            )

            if key in seen_keys:
                continue

            if url:
                seen_urls.add(url)

            seen_keys.add(key)
            results.append(listing)

        return results

    # -----------------------------------------------------------------------
    # PRIMARY + SECONDARY COLLECTION
    # -----------------------------------------------------------------------

    def collect(self, product: Dict[str, Any], primary_limit_per_source: int = 5, secondary_limit_per_source: int = 5, min_primary_candidates: int = 15) -> List[Dict[str, Any]]:
        print("\n" + "=" * 80)
        print("PRIMARY INDIAN HANDMADE / HANDLOOM MARKETPLACE SEARCH")
        print("=" * 80)

        def run_source(item: Tuple[str, str, str, int]):
            name, url, tier, limit = item
            return name, self.search_source(name, url, product, limit, tier)

        primary_items = [(name, url, "PRIMARY", primary_limit_per_source) for name, url in self.primary_sources.items()]
        primary_results = {}
        with ThreadPoolExecutor(max_workers=min(len(primary_items), SEARCH_WORKERS)) as executor:
            futures = [executor.submit(run_source, item) for item in primary_items]
            for future in as_completed(futures):
                try:
                    name, results = future.result()
                    primary_results[name] = results
                except Exception:
                    pass

        all_primary = []
        for name in self.primary_sources:
            results = primary_results.get(name, [])
            display_queries = self._query_log.get(name, self.query_generator._fallback_queries(product))[:LLM_QUERIES_PER_SOURCE]
            print(f"\n[PRIMARY] {name}")
            print(f"  Smart queries generated: {len(display_queries)}")
            for i, query in enumerate(display_queries, 1):
                print(f"    Q{i}: {query}")
            print(f"  Results accepted: {len(results)}")
            all_primary.extend(results)

        all_primary = self.deduplicate(all_primary)[:MAX_PRIMARY_CANDIDATES]
        print(f"\nPrimary candidate pool: {len(all_primary)}")

        # Always use Amazon/Flipkart as the secondary fallback when the
        # primary handmade marketplaces do not provide enough candidates.
        # This is intentionally not restricted to handloom/textile products:
        # baskets, pottery, furniture, bamboo/cane crafts, etc. are also valid
        # artisan products and need market coverage.
        use_secondary = len(all_primary) < min_primary_candidates
        all_secondary = []
        if use_secondary:
            print("\n" + "=" * 80)
            print("SECONDARY AMAZON / FLIPKART SEARCH")
            print("=" * 80)
            secondary_items = [(name, url, "SECONDARY", secondary_limit_per_source) for name, url in self.secondary_sources.items()]
            secondary_results = {}
            with ThreadPoolExecutor(max_workers=min(len(secondary_items), SEARCH_WORKERS)) as executor:
                futures = [executor.submit(run_source, item) for item in secondary_items]
                for future in as_completed(futures):
                    try:
                        name, results = future.result()
                        secondary_results[name] = results
                    except Exception:
                        pass
            for name in self.secondary_sources:
                results = secondary_results.get(name, [])
                display_queries = self._query_log.get(name, self.query_generator._fallback_queries(product))[:LLM_QUERIES_PER_SOURCE]
                print(f"\n[SECONDARY] {name}")
                print(f"  Smart queries generated: {len(display_queries)}")
                for i, query in enumerate(display_queries, 1):
                    print(f"    Q{i}: {query}")
                print(f"  Results accepted: {len(results)}")
                all_secondary.extend(results)
        else:
            print("\n[SECONDARY] Primary sources produced sufficient candidates; Amazon/Flipkart skipped.")

        combined = self.deduplicate([*all_primary, *all_secondary])[:MAX_TOTAL_CANDIDATES]
        print("\n" + "=" * 80)
        print("FINAL RAW CANDIDATE POOL")
        print("=" * 80)
        print(f"Primary candidates   : {len(all_primary)}")
        print(f"Secondary candidates : {len(all_secondary)}")
        print(f"Total candidates     : {len(combined)}")
        return combined

# ---------------------------------------------------------------------------
# APPLICATION HELPER
# ---------------------------------------------------------------------------

marketplace_collector = (
    HandmadeMarketplaceCollector()
)


def collect_marketplace_candidates(
    product: Dict[str, Any],
    primary_limit_per_source: int = 5,
    secondary_limit_per_source: int = 5,
) -> List[Dict[str, Any]]:
    return marketplace_collector.collect(
        product=product,
        primary_limit_per_source=primary_limit_per_source,
        secondary_limit_per_source=secondary_limit_per_source,
    )


# ---------------------------------------------------------------------------
# SELF TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    test_product = {
        "product_id": "TEST-BASKET-001",
        "product_name": "Handmade Bamboo Basket",
        "product_type": "Basket",
        "material": "Bamboo",
        "color": "Natural Brown",
        "size": "Medium",
        "dimensions_length": 30,
        "dimensions_width": 20,
        "dimensions_height": 10,
        "features": [
            "Handwoven",
            "Traditional weaving",
            "Natural bamboo",
            "Round shape",
            "Handcrafted",
        ],
        "description": (
            "Handmade traditional bamboo basket "
            "made using natural bamboo."
        ),
    }

    collector = HandmadeMarketplaceCollector()

    listings = collector.collect(
        product=test_product,
        primary_limit_per_source=2,
        secondary_limit_per_source=2,
    )

    print("\n" + "=" * 80)
    print("COLLECTOR SELF TEST")
    print("=" * 80)
    print(f"Collected listings: {len(listings)}")
    for index, listing in enumerate(listings, start=1):
        print(f"{index:02d}. [{listing.get('source_tier')}] {listing.get('marketplace')} | ₹{listing.get('price')} | {listing.get('title')}")

    print("\n" + "=" * 80)
    print("TOP SIMILAR MARKETPLACE PRODUCTS")
    print("=" * 80)
    try:
        from similarity import SimilarityEngine, ProductAttributes
        product_attributes = ProductAttributes(
            product_id=test_product["product_id"],
            product_name=test_product["product_name"],
            product_type=test_product["product_type"],
            material=test_product["material"],
            color=test_product["color"],
            size=test_product["size"],
            dimensions_length=test_product["dimensions_length"],
            dimensions_width=test_product["dimensions_width"],
            dimensions_height=test_product["dimensions_height"],
            features=test_product["features"],
            description=test_product["description"],
        )
        engine = SimilarityEngine()
        matches = engine.match_marketplace_listings(product_attributes, listings, top_k=10)
        if not matches:
            print("No candidates passed the existing similarity threshold.")
        else:
            for index, match in enumerate(matches, start=1):
                breakdown = getattr(match, "breakdown", None)
                score = getattr(breakdown, "total_score", None) if breakdown else None
                level = getattr(breakdown, "match_level", None) if breakdown else None
                listing_obj = getattr(match, "product", match)
                if isinstance(listing_obj, dict):
                    title = listing_obj.get("title", "")
                    marketplace = listing_obj.get("marketplace", "")
                    price = listing_obj.get("price", "")
                else:
                    title = getattr(listing_obj, "title", "")
                    marketplace = getattr(listing_obj, "marketplace", "")
                    price = getattr(listing_obj, "price", "")
                print(f"\nMATCH #{index}")
                print(f"  Marketplace : {marketplace}")
                print(f"  Product     : {title}")
                print(f"  Price       : ₹{price}")
                print(f"  Similarity  : {score:.4f}" if isinstance(score, (int, float)) else "  Similarity  : N/A")
                print(f"  Match level : {level}")
    except Exception as exc:
        print(f"Similarity self-test could not run: {exc}")

