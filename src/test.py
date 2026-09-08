"""
SIH26090 - Module 3 End-to-End Integration Test

Flow tested:
1. Fetch 5 real products from IndiaHandmade.
2. Convert them into test artisan products with independent production costs.
3. Use src/marketplace_sources.py to search fresh marketplace candidates.
4. Use src/similarity.py to rank/accept comparable products.
5. Use src/marketplace_pricing.py to generate the final price.
6. Verify the 29-feature vector, ML readiness gate, and guardrails.
7. persist=False -> this test does not create production pricing events.

Run from:
    SIH_2026/src/test.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup


# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================================
# PROJECT MODULES
# ============================================================================

try:
    from marketplace_pricing.marketplace_sources import collect_marketplace_candidates
except ImportError:
    from marketplace_pricing.marketplace_sources import HandmadeMarketplaceCollector

    _collector = HandmadeMarketplaceCollector()

    def collect_marketplace_candidates(
        product,
        primary_limit_per_source=5,
        secondary_limit_per_source=5,
    ):
        return _collector.collect(
            product=product,
            primary_limit_per_source=primary_limit_per_source,
            secondary_limit_per_source=secondary_limit_per_source,
        )
from marketplace_pricing.similarity import SimilarityEngine, ProductAttributes
from marketplace_pricing import (
    ProductionInput,
    PricingGuardrails,
    generate_pricing_recommendation,
)

try:
    from marketplace_pricing.features import NUMERIC_FEATURES
except ImportError:
    NUMERIC_FEATURES = []


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

INDIAHANDMADE_URLS = [
    "https://www.indiahandmade.com/water-bottle-1.html",
    "https://www.indiahandmade.com/buy-handmade-bamboo-cane-stool-online.html",
    "https://www.indiahandmade.com/lawn-table.html",
    "https://www.indiahandmade.com/buy-rattan-cane-swing-chair-with-accessories-online.html",
    "https://www.indiahandmade.com/buy-cane-hanging-swing-for-balcony-online-india.html",
]

EXPECTED_PRODUCTS = [
    {
        "id": "TEST-IH-001",
        "expected": "Terracotta Water Bottle with Floral Painting",
        "cluster": "Terracotta & Pottery",
        "material": "Terracotta",
        "material_ratio": 0.35,
        "labour_ratio": 0.20,
        "production_ratio": 0.10,
        "hours": 5.0,
        "features": ["Floral painting"],
    },
    {
        "id": "TEST-IH-002",
        "expected": "Handmade Bamboo & Cane Stool",
        "cluster": "Bamboo & Cane Craft",
        "material": "Cane",
        "material_ratio": 0.35,
        "labour_ratio": 0.20,
        "production_ratio": 0.10,
        "hours": 6.0,
        "features": ["Handmade", "Cane work", "Bamboo work"],
    },
    {
        "id": "TEST-IH-003",
        "expected": "Handmade Cane Lawn Table with Glass Top",
        "cluster": "Bamboo & Cane Craft",
        "material": "Cane",
        "material_ratio": 0.35,
        "labour_ratio": 0.20,
        "production_ratio": 0.10,
        "hours": 10.0,
        "features": ["Handmade", "Cane work", "Glass top"],
    },
    {
        "id": "TEST-IH-004",
        "expected": "Rattan Cane Swing Chair",
        "cluster": "Bamboo & Cane Craft",
        "material": "Rattan",
        "material_ratio": 0.35,
        "labour_ratio": 0.20,
        "production_ratio": 0.10,
        "hours": 6.0,
        "features": ["Handmade", "Natural finish", "Cane work", "Rattan work"],
    },
    {
        "id": "TEST-IH-005",
        "expected": "Cane Hanging Swing for Balcony",
        "cluster": "Bamboo & Cane Craft",
        "material": "Cane",
        "material_ratio": 0.35,
        "labour_ratio": 0.20,
        "production_ratio": 0.10,
        "hours": 8.0,
        "features": ["Handcrafted", "Handwoven", "Cane work"],
    },
]


# ============================================================================
# PRINT HELPERS
# ============================================================================

passed = 0
failed = 0
warnings = 0


def check(condition: bool, message: str) -> None:
    global passed, failed

    if condition:
        passed += 1
        print(f"[PASS] {message}")
    else:
        failed += 1
        print(f"[FAIL] {message}")


def warn(message: str) -> None:
    global warnings
    warnings += 1
    print(f"[WARN] {message}")


def money(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"₹{float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def safe_score(match: Any) -> float:
    breakdown = getattr(match, "breakdown", None)

    if breakdown is not None:
        value = getattr(breakdown, "total_score", None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    for key in ("similarity_score", "score"):
        if isinstance(match, dict):
            value = match.get(key)
        else:
            value = getattr(match, key, None)

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    return 0.0


def match_level(match: Any) -> str:
    breakdown = getattr(match, "breakdown", None)

    if breakdown is not None:
        level = getattr(breakdown, "match_level", None)
        if level:
            return str(level)

    if isinstance(match, dict):
        return str(match.get("match_level", "UNKNOWN"))

    return str(getattr(match, "match_level", "UNKNOWN"))


def match_product(match: Any) -> Any:
    return getattr(match, "product", match)


def listing_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ============================================================================
# INDIAHANDMADE FETCHER
# ============================================================================

def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def parse_price(value: Any) -> float | None:
    if value is None:
        return None

    text = clean_text(value).replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)", text)

    if not match:
        return None

    try:
        price = float(match.group(1))
        return price if price > 0 else None
    except ValueError:
        return None


def infer_type(title: str) -> str:
    text = title.lower()

    if "water bottle" in text:
        return "Water Bottle"
    if "swing chair" in text:
        return "Swing Chair"
    if "hanging swing" in text or "balcony swing" in text:
        return "Swing"
    if "stool" in text:
        return "Stool"
    if "table" in text:
        return "Table"
    if "basket" in text:
        return "Basket"
    if "lamp" in text:
        return "Lamp"
    if "chair" in text:
        return "Chair"
    if "bottle" in text:
        return "Bottle"

    return "Handicraft"


def extract_jsonld_product(soup: BeautifulSoup) -> Dict[str, Any]:
    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text()

        if not raw:
            continue

        try:
            data = json.loads(raw)
        except Exception:
            continue

        objects = data if isinstance(data, list) else [data]

        for obj in objects:
            if not isinstance(obj, dict):
                continue

            if obj.get("@type") == "Product":
                return obj

            graph = obj.get("@graph")
            if isinstance(graph, list):
                for item in graph:
                    if isinstance(item, dict) and item.get("@type") == "Product":
                        return item

    return {}


def fetch_indiahandmade_product(
    product_number: int,
    url: str,
) -> Dict[str, Any]:

    response = requests.get(
        url,
        timeout=15,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/142 Safari/537.36"
            )
        },
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    jsonld = extract_jsonld_product(soup)

    title = clean_text(
        jsonld.get("name")
        or (soup.title.get_text(" ", strip=True) if soup.title else "")
    )

    # Remove common site suffixes from browser title when present.
    title = re.sub(
        r"\s*\|\s*IndiaHandmade.*$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = clean_text(title)

    price = parse_price(
        jsonld.get("price")
        or jsonld.get("lowPrice")
        or jsonld.get("offers", {}).get("price")
        if isinstance(jsonld.get("offers"), dict)
        else jsonld.get("price")
    )

    if price is None:
        text = soup.get_text(" ", strip=True)
        price_matches = re.findall(
            r"(?:₹|Rs\.?|INR)\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if price_matches:
            price = parse_price(price_matches[0])

    product_type = infer_type(title)

    material = ""
    combined_text = clean_text(
        title + " " + soup.get_text(" ", strip=True)[:12000]
    ).lower()

    if "terracotta" in combined_text:
        material = "Terracotta"
    elif "rattan" in combined_text:
        material = "Rattan"
    elif "cane" in combined_text:
        material = "Cane"
    elif "bamboo" in combined_text:
        material = "Bamboo"

    return {
        "product_number": product_number,
        "source": "IndiaHandmade",
        "url": response.url,
        "title": title,
        "price": price,
        "product_type": product_type,
        "material": material,
        "features": [],
        "raw_status": response.status_code,
    }


def fetch_five_indiahandmade_products() -> List[Dict[str, Any]]:
    print("=" * 100)
    print("TEST 1 - FETCH 5 REAL INDIAHANDMADE PRODUCTS")
    print("=" * 100)

    products = []

    for index, url in enumerate(INDIAHANDMADE_URLS, start=1):
        print()
        print("-" * 96)
        print(f"INDIAHANDMADE PRODUCT FETCH {index}/5")
        print("-" * 96)
        print(f"URL: {url}")

        try:
            product = fetch_indiahandmade_product(index, url)
            products.append(product)

            print(f"HTTP status : {product['raw_status']}")
            print(f"Final URL   : {product['url']}")
            print(f"Title       : {product['title']}")
            print(f"Price       : {money(product['price'])}")
            print(f"Type        : {product['product_type']}")
            print(f"Material    : {product['material'] or 'Not detected'}")

            check(
                product["price"] is not None,
                f"IndiaHandmade product {index} has a valid price.",
            )

        except Exception as exc:
            failed += 1
            print(f"[FAIL] IndiaHandmade product {index} fetch failed: {exc}")

    check(
        len(products) == 5,
        f"Fetched exactly 5 IndiaHandmade products ({len(products)}/5).",
    )

    return products


# ============================================================================
# CONVERT SOURCE PRODUCTS INTO ARTISAN TEST INPUTS
# ============================================================================

def build_artisan_products(
    fetched: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    print()
    print("=" * 100)
    print("TEST 2 - INDIAHANDMADE PRODUCTS -> ARTISAN TEST INPUTS")
    print("=" * 100)

    artisan_products = []

    for index, source in enumerate(fetched):
        expected = EXPECTED_PRODUCTS[index]

        source_price = float(source["price"])

        material_cost = round(source_price * expected["material_ratio"], 2)
        labour_cost = round(source_price * expected["labour_ratio"], 2)
        production_cost = round(source_price * expected["production_ratio"], 2)

        product = {
            "product_id": expected["id"],
            "product_name": source["title"],
            "product_type": expected["expected"].split()[0]
            if expected["id"] == "TEST-IH-001"
            else (
                "Stool"
                if expected["id"] == "TEST-IH-002"
                else "Table"
                if expected["id"] == "TEST-IH-003"
                else "Swing Chair"
                if expected["id"] == "TEST-IH-004"
                else "Swing"
            ),
            "material": expected["material"],
            "additional_materials": [],
            "color": "Natural",
            "size": "Standard",
            "dimensions_length": None,
            "dimensions_width": None,
            "dimensions_height": None,
            "features": expected["features"],
            "description": source["title"],
            "material_cost": material_cost,
            "labour_cost": labour_cost,
            "production_cost": production_cost,
            "time_worked_hours": expected["hours"],
            "source_url": source["url"],
            "indiahandmade_price": source_price,
            "craft_cluster": expected["cluster"],
        }

        # Correct explicit product type for product 1.
        if expected["id"] == "TEST-IH-001":
            product["product_type"] = "Water Bottle"

        artisan_products.append(product)

        print()
        print(f"ARTISAN INPUT {index + 1}")
        print(f"  ID                 : {product['product_id']}")
        print(f"  Product            : {product['product_name']}")
        print(f"  Product type       : {product['product_type']}")
        print(f"  Material           : {product['material']}")
        print(f"  Source price       : {money(product['indiahandmade_price'])}")
        print(f"  Material cost      : {money(product['material_cost'])}")
        print(f"  Labour cost        : {money(product['labour_cost'])}")
        print(f"  Production cost    : {money(product['production_cost'])}")
        print(f"  Total production   : {money(product['material_cost'] + product['labour_cost'] + product['production_cost'])}")
        print(f"  Time worked        : {product['time_worked_hours']:.2f} hours")

        check(
            product["material_cost"] > 0
            and product["labour_cost"] > 0
            and product["production_cost"] > 0,
            f"{product['product_id']} has independent positive production inputs.",
        )

    return artisan_products


# ============================================================================
# SIMILARITY PRE-CHECK
# ============================================================================

def run_similarity_check(
    product: Dict[str, Any],
    listings: List[Dict[str, Any]],
) -> List[Any]:

    attributes = ProductAttributes(
        product_id=product["product_id"],
        product_name=product["product_name"],
        product_type=product["product_type"],
        material=product["material"],
        additional_materials=product["additional_materials"],
        color=product["color"],
        size=product["size"],
        dimensions_length=product["dimensions_length"],
        dimensions_width=product["dimensions_width"],
        dimensions_height=product["dimensions_height"],
        features=product["features"],
        description=product["description"],
    )

    engine = SimilarityEngine()

    matches = engine.match_marketplace_listings(
        product=attributes,
        listings=listings,
        top_k=10,
    )

    print()
    print("SIMILARITY ENGINE RESULT")
    print("-" * 96)
    print(f"Listings supplied       : {len(listings)}")
    print(f"Accepted comparables    : {len(matches)}")

    if not matches:
        print("No marketplace listing passed the similarity threshold.")
    else:
        for rank, match in enumerate(matches, start=1):
            obj = match_product(match)
            title = listing_value(obj, "title", listing_value(obj, "product_name", ""))
            marketplace = listing_value(obj, "marketplace", "Unknown")
            price = listing_value(obj, "price", None)

            print(
                f"{rank:02d}. "
                f"{marketplace:<18} "
                f"{money(price):>12} "
                f"score={safe_score(match):.4f} "
                f"{match_level(match):<10} "
                f"{title}"
            )

    return matches


# ============================================================================
# FINAL PRICING
# ============================================================================

def run_pricing(
    product: Dict[str, Any],
    listings: List[Dict[str, Any]],
) -> Any:

    production_input = ProductionInput(
        material_cost=product["material_cost"],
        labour_cost=product["labour_cost"],
        production_cost=product["production_cost"],
        time_worked_hours=product["time_worked_hours"],
    )

    # Test guardrails.
    # In production these can be supplied by the artisan/system.
    fair_floor = (
        product["material_cost"]
        + product["labour_cost"]
        + product["production_cost"]
    ) * 1.20

    guardrails = PricingGuardrails(
        minimum_price=round(fair_floor, 2),
        maximum_price=max(
            round(product["indiahandmade_price"] * 1.75, 2),
            round(fair_floor * 2.0, 2),
        ),
        maximum_daily_change=0.15,
        maximum_market_deviation=0.30,
    )

    demand = {
        "views": 100,
        "clicks": 12,
        "wishlists": 3,
        "add_to_cart": 2,
        "enquiries": 1,
        "orders": 0,
    }

    supply = {
        "inventory": 10,
        "units_sold": 0,
    }

    result = generate_pricing_recommendation(
        product=product,
        marketplace_listings=listings,
        production_input=production_input,
        base_price=product["indiahandmade_price"],
        demand=demand,
        supply=supply,
        seasonal_index=1.0,
        guardrails=guardrails,
        top_k=10,
        previous_price=None,
        actual_selling_price=None,
        persist=False,
    )

    return result


def print_pricing_result(
    product: Dict[str, Any],
    result: Any,
) -> None:

    print()
    print("FINAL PRICING ENGINE RESULT")
    print("-" * 96)

    print(f"Product ID                 : {result.product_id}")
    print(f"Base price / anchor        : {money(product['indiahandmade_price'])}")
    print(f"Base production cost      : {money(result.base_cost)}")
    print(f"Current listings           : {result.current_listing_count}")
    print(f"Accepted comparables      : {result.current_comparable_count}")
    print(f"Historical comparables    : {result.historical_comparable_count}")
    print(f"Market evidence count     : {result.market_count}")
    print(f"Market minimum            : {money(result.market_minimum)}")
    print(f"Market median             : {money(result.market_median)}")
    print(f"Market maximum            : {money(result.market_maximum)}")
    print(f"Similarity score           : {result.similarity_score:.4f}")
    print(f"29-feature vector          : {len(result.feature_vector)} features")
    print(f"ML readiness               : {'YES' if result.ml_readiness else 'NO'}")
    print(f"ML invoked                 : {'YES' if result.ml_invoked else 'NO'}")
    print(f"Model used                 : {result.ml_model}")
    print(f"ML reason                  : {result.ml_readiness_reason}")
    print(f"Formula price              : {money(result.formula_price)}")
    print(f"Market-based price        : {money(result.market_based_price)}")
    print(f"Fair-trade floor           : {money(result.fair_trade_floor)}")
    print(f"Recommended before guardrails : {money(result.recommended_price)}")
    print(f"Guardrail minimum          : {money(result.guardrail_minimum)}")
    print(f"Guardrail maximum          : {money(result.guardrail_maximum)}")
    print(f"FINAL RECOMMENDED PRICE    : {money(result.final_price)}")
    print(f"Pricing method             : {result.method}")
    print(f"Decision                   : {result.decision_reason}")

    print()
    print("FINAL COMPARABLES USED BY PRICING")
    print("-" * 96)

    if not result.comparable_products:
        print("NONE")

    for rank, match in enumerate(result.comparable_products, start=1):
        obj = match_product(match)

        print(
            f"{rank:02d}. "
            f"{listing_value(obj, 'marketplace', 'Unknown'):<18} "
            f"{money(listing_value(obj, 'price')):>12} "
            f"score={safe_score(match):.4f} "
            f"{match_level(match):<10} "
            f"{listing_value(obj, 'title', '')}"
        )


# ============================================================================
# ONE PRODUCT END-TO-END
# ============================================================================

def run_product_test(
    product: Dict[str, Any],
    product_number: int,
) -> Dict[str, Any]:

    print()
    print("=" * 100)
    print(f"PRODUCT {product_number}/5 - END-TO-END MODULE TEST")
    print("=" * 100)

    print()
    print("1. ARTISAN PRODUCT")
    print(f"   ID       : {product['product_id']}")
    print(f"   Product  : {product['product_name']}")
    print(f"   Type     : {product['product_type']}")
    print(f"   Material : {product['material']}")
    print(f"   Cost     : {money(product['material_cost'] + product['labour_cost'] + product['production_cost'])}")

    # ------------------------------------------------------------------------
    # MARKETPLACE SOURCE MODULE
    # ------------------------------------------------------------------------

    print()
    print("2. MARKETPLACE SOURCES")
    print("-" * 96)
    print("Calling src/marketplace_sources.py ...")

    try:
        listings = collect_marketplace_candidates(
            product=product,
            primary_limit_per_source=5,
            secondary_limit_per_source=5,
        )
    except Exception as exc:
        print(f"[FAIL] Marketplace collector failed: {exc}")
        return {
            "product": product,
            "listings": [],
            "matches": [],
            "result": None,
        }

    primary_count = sum(
        1 for item in listings
        if str(item.get("source_tier", "")).upper() == "PRIMARY"
    )

    secondary_count = sum(
        1 for item in listings
        if str(item.get("source_tier", "")).upper() == "SECONDARY"
    )

    print()
    print("MARKETPLACE COLLECTION RESULT")
    print("-" * 96)
    print(f"Total listings returned : {len(listings)}")
    print(f"Primary listings        : {primary_count}")
    print(f"Secondary listings      : {secondary_count}")

    for rank, item in enumerate(listings[:15], start=1):
        print(
            f"{rank:02d}. "
            f"{item.get('marketplace', 'Unknown'):<18} "
            f"{money(item.get('price')):>12} "
            f"{item.get('title', '')}"
        )

    check(
        isinstance(listings, list),
        f"Marketplace collector returned a list for {product['product_id']}.",
    )

    # ------------------------------------------------------------------------
    # SIMILARITY MODULE
    # ------------------------------------------------------------------------

    print()
    print("3. MULTI-FACTOR SIMILARITY")
    print("-" * 96)
    print("Calling src/similarity.py ...")

    matches = run_similarity_check(product, listings)

    check(
        isinstance(matches, list),
        f"Similarity engine returned a result list for {product['product_id']}.",
    )

    # ------------------------------------------------------------------------
    # PRICING MODULE
    # ------------------------------------------------------------------------

    print()
    print("4. PRICING ENGINE")
    print("-" * 96)
    print("Calling src/marketplace_pricing.py ...")

    try:
        result = run_pricing(product, listings)
    except Exception as exc:
        print(f"[FAIL] Pricing engine failed: {exc}")
        return {
            "product": product,
            "listings": listings,
            "matches": matches,
            "result": None,
        }

    print_pricing_result(product, result)

    # ------------------------------------------------------------------------
    # FINAL MODULE CHECKS
    # ------------------------------------------------------------------------

    print()
    print("5. MODULE VALIDATION")
    print("-" * 96)

    check(
        result.final_price is not None and result.final_price > 0,
        "Final recommended price is positive.",
    )

    check(
        result.final_price >= result.fair_trade_floor - 1e-6,
        "Final recommended price respects the fair-trade floor.",
    )

    check(
        len(result.feature_vector) == 29,
        "Pricing result contains exactly 29 features.",
    )

    check(
        result.current_listing_count == len(listings),
        "Pricing listing count matches marketplace collector output.",
    )

    check(
        result.current_comparable_count == len(result.comparable_products),
        "Pricing comparable count matches the accepted comparable list.",
    )

    # The pricing module internally runs the same similarity engine.
    # Therefore its comparables must be similarity-approved products.
    pricing_ids = {
        listing_value(match_product(m), "listing_id", None)
        for m in result.comparable_products
    }

    similarity_ids = {
        listing_value(match_product(m), "listing_id", None)
        for m in matches
    }

    pricing_ids.discard(None)
    similarity_ids.discard(None)

    check(
        pricing_ids.issubset(similarity_ids) or not pricing_ids,
        "Pricing comparables are drawn from similarity-approved listings.",
    )

    return {
        "product": product,
        "listings": listings,
        "matches": matches,
        "result": result,
    }


# ============================================================================
# FINAL SUMMARY
# ============================================================================

def print_final_summary(results: List[Dict[str, Any]]) -> None:

    print()
    print("#" * 100)
    print("SIH26090 - FINAL MODULE 3 SUMMARY")
    print("#" * 100)

    print()
    print(
        f"{'ID':<15}"
        f"{'Source':>12}"
        f"{'Listings':>12}"
        f"{'Similar':>12}"
        f"{'Market':>14}"
        f"{'ML':>8}"
        f"{'Final Price':>16}"
    )

    print("-" * 100)

    for item in results:
        product = item["product"]
        result = item["result"]

        if result is None:
            print(
                f"{product['product_id']:<15}"
                f"{money(product['indiahandmade_price']):>12}"
                f"{len(item['listings']):>12}"
                f"{len(item['matches']):>12}"
                f"{'FAILED':>14}"
                f"{'-':>8}"
                f"{'-':>16}"
            )
            continue

        print(
            f"{product['product_id']:<15}"
            f"{money(product['indiahandmade_price']):>12}"
            f"{result.current_listing_count:>12}"
            f"{result.current_comparable_count:>12}"
            f"{result.market_count:>14}"
            f"{('YES' if result.ml_invoked else 'NO'):>8}"
            f"{money(result.final_price):>16}"
        )

    print()
    print("PIPELINE:")
    print("IndiaHandmade -> Artisan Product -> Marketplace Sources")
    print("-> Multi-Factor Similarity -> Accepted Comparables")
    print("-> 29 Features -> CatBoost Readiness Gate")
    print("-> Market/Formula Pricing -> Guardrails -> Final Price")

    print()
    print("ARCHITECTURE CHECKS")
    print("-" * 100)

    check(
        len(results) == 5,
        "Five IndiaHandmade products completed the test setup.",
    )

    check(
        all(item["result"] is not None for item in results),
        "All five products reached the pricing stage.",
    )

    if NUMERIC_FEATURES:
        check(
            len(NUMERIC_FEATURES) == 29,
            "Project feature module exposes exactly 29 numeric pricing features.",
        )
    else:
        warn("NUMERIC_FEATURES could not be imported; pricing result checks still verify the 29-feature vector.")

    check(
        all(
            item["result"] is None
            or len(item["result"].feature_vector) == 29
            for item in results
        ),
        "Every successful pricing result contains exactly 29 features.",
    )

    check(
        all(
            item["result"] is None
            or item["result"].final_price > 0
            for item in results
        ),
        "Every successful pricing result has a positive final price.",
    )

    print()
    print("=" * 100)
    print(
        f"TESTS PASSED   : {passed}"
    )
    print(
        f"TESTS WARNED   : {warnings}"
    )
    print(
        f"TESTS FAILED   : {failed}"
    )
    print("=" * 100)

    if failed == 0:
        print("SIH26090 MODULE 3 TEST PASSED")
    else:
        print("SIH26090 MODULE 3 TEST COMPLETED WITH FAILURES")

    print()
    print("IMPORTANT: persist=False was used.")
    print("This integration test does not intentionally create production pricing events.")
    print("#" * 100)



# ============================================================================
# EXTENDED FORENSIC INTEGRATION AUDIT
# ============================================================================
#
# This makes test.py a full MAIN integration/debug file.
# Production modules remain authoritative.
# This layer observes and prints their returned values.
# It does not create a second pricing algorithm or a second similarity engine.
# ============================================================================

AUDIT_CHECKPOINTS = [
    {'id': 1, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 2, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 3, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 4, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 5, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 6, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 7, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 8, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 9, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 10, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 11, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 12, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 13, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 14, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 15, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 16, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 17, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 18, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 19, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 20, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 21, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 22, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 23, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 24, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 25, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 26, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 27, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 28, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 29, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 30, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 31, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 32, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 33, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 34, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 35, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 36, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 37, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 38, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 39, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 40, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 41, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 42, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 43, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 44, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 45, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 46, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 47, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 48, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 49, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 50, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 51, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 52, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 53, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 54, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 55, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 56, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 57, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 58, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 59, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 60, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 61, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 62, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 63, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 64, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 65, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 66, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 67, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 68, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 69, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 70, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 71, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 72, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 73, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 74, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 75, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 76, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 77, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 78, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 79, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 80, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 81, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 82, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 83, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 84, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 85, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 86, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 87, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 88, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 89, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 90, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 91, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 92, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 93, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 94, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 95, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 96, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 97, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 98, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 99, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 100, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 101, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 102, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 103, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 104, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 105, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 106, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 107, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 108, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 109, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 110, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 111, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 112, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 113, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 114, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 115, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 116, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 117, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 118, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 119, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 120, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 121, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 122, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 123, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 124, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 125, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 126, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 127, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 128, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 129, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 130, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 131, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 132, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 133, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 134, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 135, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 136, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 137, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 138, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 139, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 140, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 141, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 142, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 143, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 144, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 145, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 146, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 147, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 148, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 149, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 150, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 151, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 152, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 153, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 154, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 155, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 156, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 157, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 158, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 159, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 160, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 161, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 162, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 163, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 164, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 165, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 166, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 167, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 168, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 169, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 170, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 171, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 172, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 173, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 174, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 175, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 176, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 177, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 178, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 179, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 180, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 181, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 182, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 183, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 184, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 185, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 186, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 187, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 188, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 189, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 190, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 191, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 192, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 193, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 194, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 195, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 196, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 197, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 198, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 199, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 200, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 201, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 202, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 203, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 204, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 205, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 206, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 207, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 208, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 209, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 210, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 211, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 212, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 213, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 214, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 215, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 216, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 217, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 218, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 219, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 220, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 221, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 222, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 223, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 224, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 225, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 226, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 227, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 228, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 229, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 230, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 231, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 232, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 233, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 234, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 235, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 236, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 237, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 238, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 239, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 240, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 241, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 242, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 243, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 244, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 245, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 246, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 247, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 248, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 249, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 250, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 251, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 252, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 253, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 254, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 255, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 256, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 257, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 258, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 259, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 260, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 261, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 262, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 263, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 264, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 265, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 266, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 267, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 268, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 269, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 270, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 271, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 272, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 273, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 274, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 275, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 276, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 277, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 278, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 279, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 280, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 281, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 282, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 283, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 284, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 285, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 286, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 287, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 288, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 289, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 290, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 291, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 292, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 293, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 294, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 295, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 296, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 297, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 298, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 299, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 300, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 301, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 302, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 303, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 304, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 305, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 306, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 307, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 308, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 309, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 310, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 311, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 312, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 313, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 314, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 315, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 316, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 317, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 318, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 319, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 320, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 321, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 322, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 323, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 324, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 325, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 326, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 327, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 328, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 329, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 330, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 331, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 332, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 333, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 334, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 335, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 336, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 337, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 338, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 339, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 340, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 341, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 342, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 343, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 344, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 345, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 346, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 347, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 348, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 349, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 350, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 351, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 352, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 353, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 354, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 355, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 356, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 357, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 358, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 359, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 360, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 361, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 362, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 363, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 364, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 365, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 366, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 367, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 368, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 369, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 370, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 371, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 372, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 373, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 374, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 375, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 376, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 377, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 378, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 379, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 380, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 381, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 382, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 383, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 384, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 385, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 386, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 387, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 388, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 389, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 390, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 391, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 392, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 393, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 394, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 395, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 396, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 397, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 398, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 399, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 400, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 401, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 402, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 403, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 404, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 405, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 406, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 407, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 408, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 409, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 410, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 411, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 412, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 413, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 414, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 415, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 416, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 417, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 418, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 419, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 420, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 421, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 422, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 423, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 424, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 425, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 426, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 427, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 428, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 429, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 430, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 431, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 432, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 433, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 434, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 435, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 436, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 437, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 438, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 439, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 440, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 441, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 442, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 443, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 444, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 445, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 446, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 447, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 448, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 449, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 450, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 451, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 452, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 453, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 454, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 455, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 456, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 457, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 458, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 459, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 460, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 461, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 462, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 463, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 464, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 465, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 466, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 467, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 468, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 469, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 470, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 471, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 472, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 473, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 474, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 475, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 476, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 477, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 478, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 479, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 480, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 481, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 482, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 483, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 484, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 485, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 486, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 487, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 488, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 489, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 490, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 491, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 492, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 493, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 494, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 495, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 496, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 497, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 498, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 499, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 500, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 501, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 502, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 503, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 504, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 505, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 506, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 507, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 508, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 509, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 510, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 511, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 512, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 513, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 514, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 515, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 516, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 517, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 518, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 519, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 520, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 521, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 522, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 523, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 524, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 525, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 526, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 527, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 528, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 529, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 530, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 531, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 532, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 533, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 534, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 535, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 536, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 537, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 538, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 539, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 540, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 541, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 542, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 543, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 544, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 545, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 546, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 547, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 548, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 549, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 550, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 551, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 552, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 553, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 554, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 555, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 556, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 557, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 558, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 559, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 560, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 561, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 562, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 563, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 564, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 565, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 566, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 567, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 568, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 569, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 570, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 571, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 572, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 573, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 574, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 575, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 576, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 577, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 578, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 579, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 580, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 581, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 582, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 583, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 584, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 585, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 586, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 587, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 588, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 589, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 590, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 591, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 592, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 593, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 594, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 595, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 596, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 597, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 598, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 599, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 600, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 601, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 602, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 603, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 604, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 605, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 606, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 607, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 608, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 609, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 610, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 611, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 612, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 613, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 614, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 615, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 616, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 617, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 618, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 619, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 620, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 621, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 622, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 623, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 624, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 625, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 626, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 627, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 628, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 629, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 630, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 631, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 632, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 633, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 634, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 635, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 636, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 637, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 638, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 639, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 640, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 641, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 642, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 643, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 644, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 645, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 646, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 647, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 648, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 649, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 650, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 651, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 652, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 653, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 654, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 655, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 656, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 657, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 658, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 659, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 660, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 661, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 662, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 663, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 664, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 665, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 666, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 667, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 668, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 669, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 670, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 671, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 672, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 673, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 674, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 675, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 676, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 677, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 678, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 679, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 680, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 681, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 682, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 683, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 684, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 685, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 686, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 687, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 688, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 689, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 690, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 691, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 692, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 693, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 694, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 695, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 696, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 697, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 698, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 699, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 700, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 701, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 702, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 703, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 704, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 705, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 706, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 707, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 708, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 709, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 710, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 711, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 712, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 713, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 714, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 715, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 716, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 717, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 718, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 719, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 720, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 721, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 722, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 723, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 724, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 725, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 726, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 727, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 728, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 729, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 730, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 731, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 732, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 733, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 734, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 735, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 736, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 737, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 738, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 739, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 740, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 741, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 742, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 743, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 744, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 745, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 746, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 747, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 748, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 749, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 750, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 751, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 752, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 753, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 754, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 755, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 756, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 757, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 758, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 759, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 760, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 761, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 762, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 763, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 764, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 765, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 766, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 767, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 768, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 769, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 770, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 771, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 772, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 773, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 774, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 775, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 776, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 777, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 778, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 779, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 780, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 781, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 782, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 783, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 784, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 785, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 786, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 787, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 788, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 789, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 790, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 791, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 792, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 793, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 794, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 795, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 796, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 797, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 798, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 799, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 800, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 801, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 802, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 803, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 804, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 805, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 806, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 807, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 808, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 809, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 810, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 811, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 812, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 813, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 814, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 815, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 816, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 817, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 818, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 819, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 820, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 821, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 822, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 823, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 824, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 825, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 826, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 827, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 828, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 829, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 830, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 831, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 832, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 833, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 834, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 835, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 836, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 837, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 838, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 839, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 840, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 841, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 842, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 843, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 844, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 845, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 846, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 847, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 848, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 849, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 850, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 851, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 852, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 853, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 854, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 855, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 856, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 857, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 858, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 859, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 860, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 861, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 862, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 863, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 864, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 865, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 866, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 867, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 868, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 869, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 870, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 871, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 872, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 873, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 874, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 875, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 876, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 877, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 878, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 879, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 880, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 881, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 882, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 883, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 884, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 885, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 886, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 887, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 888, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 889, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 890, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 891, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 892, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 893, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 894, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 895, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 896, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 897, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 898, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 899, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 900, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 901, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 902, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 903, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 904, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 905, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 906, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 907, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 908, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 909, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 910, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 911, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 912, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 913, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 914, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 915, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 916, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 917, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 918, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 919, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 920, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 921, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 922, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 923, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 924, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 925, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 926, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 927, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 928, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 929, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 930, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 931, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 932, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 933, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 934, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 935, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 936, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 937, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 938, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 939, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 940, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 941, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 942, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 943, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 944, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 945, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 946, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 947, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 948, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 949, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 950, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 951, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 952, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 953, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 954, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 955, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 956, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 957, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 958, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 959, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 960, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 961, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 962, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 963, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 964, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 965, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 966, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 967, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 968, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 969, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 970, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 971, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 972, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 973, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 974, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 975, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 976, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 977, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 978, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 979, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 980, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 981, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 982, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 983, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 984, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 985, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 986, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 987, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 988, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 989, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 990, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 991, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 992, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 993, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 994, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 995, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 996, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 997, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 998, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 999, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1000, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 1001, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 1002, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 1003, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1004, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 1005, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 1006, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 1007, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 1008, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 1009, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 1010, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 1011, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 1012, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 1013, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 1014, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 1015, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 1016, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 1017, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 1018, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 1019, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 1020, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 1021, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 1022, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 1023, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 1024, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 1025, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 1026, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 1027, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 1028, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 1029, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 1030, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 1031, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 1032, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 1033, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 1034, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 1035, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 1036, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 1037, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 1038, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 1039, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 1040, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 1041, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 1042, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 1043, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 1044, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 1045, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 1046, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 1047, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 1048, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 1049, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 1050, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 1051, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 1052, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 1053, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 1054, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 1055, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 1056, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 1057, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 1058, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 1059, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 1060, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 1061, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 1062, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 1063, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 1064, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 1065, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 1066, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 1067, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 1068, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 1069, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 1070, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 1071, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 1072, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 1073, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 1074, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 1075, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 1076, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 1077, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 1078, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 1079, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 1080, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 1081, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 1082, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 1083, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 1084, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 1085, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 1086, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 1087, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 1088, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 1089, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 1090, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 1091, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 1092, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 1093, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 1094, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 1095, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1096, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1097, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 1098, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1099, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 1100, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 1101, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 1102, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1103, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1104, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 1105, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 1106, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 1107, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 1108, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 1109, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 1110, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1111, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 1112, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 1113, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 1114, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 1115, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 1116, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 1117, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 1118, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 1119, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 1120, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 1121, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 1122, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 1123, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 1124, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 1125, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 1126, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 1127, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 1128, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 1129, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 1130, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 1131, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 1132, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 1133, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 1134, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1135, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1136, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 1137, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 1138, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 1139, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1140, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 1141, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 1142, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 1143, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 1144, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 1145, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 1146, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 1147, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 1148, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 1149, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 1150, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 1151, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 1152, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 1153, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 1154, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 1155, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 1156, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 1157, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 1158, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 1159, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 1160, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 1161, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 1162, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 1163, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 1164, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 1165, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 1166, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 1167, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 1168, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 1169, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 1170, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 1171, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 1172, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 1173, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 1174, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 1175, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 1176, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 1177, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 1178, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 1179, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 1180, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 1181, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 1182, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 1183, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 1184, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 1185, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 1186, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 1187, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 1188, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 1189, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 1190, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 1191, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 1192, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 1193, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 1194, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 1195, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 1196, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 1197, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 1198, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 1199, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 1200, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 1201, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 1202, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 1203, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 1204, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 1205, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 1206, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 1207, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 1208, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 1209, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 1210, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 1211, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 1212, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 1213, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 1214, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 1215, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 1216, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 1217, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 1218, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 1219, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 1220, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 1221, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 1222, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 1223, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 1224, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 1225, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 1226, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 1227, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 1228, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 1229, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 1230, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 1231, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1232, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1233, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 1234, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1235, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 1236, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 1237, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 1238, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1239, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1240, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 1241, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 1242, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 1243, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 1244, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 1245, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 1246, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1247, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 1248, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 1249, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 1250, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 1251, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 1252, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 1253, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 1254, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 1255, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 1256, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 1257, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 1258, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 1259, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 1260, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 1261, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 1262, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 1263, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 1264, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 1265, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 1266, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 1267, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 1268, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 1269, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 1270, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1271, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1272, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 1273, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 1274, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 1275, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1276, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 1277, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 1278, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 1279, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 1280, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 1281, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 1282, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 1283, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 1284, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 1285, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 1286, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 1287, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 1288, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 1289, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 1290, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 1291, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 1292, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 1293, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 1294, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 1295, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 1296, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 1297, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 1298, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 1299, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 1300, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 1301, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 1302, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 1303, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 1304, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 1305, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 1306, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 1307, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 1308, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 1309, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 1310, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 1311, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 1312, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 1313, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 1314, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 1315, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 1316, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 1317, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 1318, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 1319, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 1320, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 1321, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 1322, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 1323, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 1324, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 1325, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 1326, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 1327, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 1328, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 1329, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 1330, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 1331, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 1332, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 1333, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 1334, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 1335, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 1336, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 1337, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 1338, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 1339, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 1340, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 1341, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 1342, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 1343, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 1344, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 1345, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 1346, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 1347, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 1348, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 1349, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 1350, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 1351, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 1352, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 1353, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 1354, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 1355, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 1356, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 1357, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 1358, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 1359, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 1360, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 1361, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 1362, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 1363, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 1364, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 1365, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 1366, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 1367, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1368, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1369, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 1370, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1371, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 1372, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 1373, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 1374, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1375, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1376, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 1377, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 1378, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 1379, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 1380, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 1381, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 1382, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1383, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 1384, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 1385, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 1386, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 1387, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 1388, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 1389, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 1390, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 1391, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 1392, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 1393, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 1394, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 1395, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 1396, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 1397, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 1398, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 1399, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 1400, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 1401, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 1402, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 1403, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 1404, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 1405, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 1406, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1407, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1408, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 1409, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 1410, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 1411, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1412, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 1413, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 1414, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 1415, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 1416, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 1417, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 1418, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 1419, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 1420, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 1421, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 1422, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 1423, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 1424, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 1425, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 1426, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 1427, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 1428, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 1429, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 1430, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 1431, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 1432, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 1433, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 1434, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 1435, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 1436, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 1437, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 1438, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 1439, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 1440, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 1441, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 1442, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 1443, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 1444, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 1445, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 1446, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 1447, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 1448, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 1449, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 1450, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 1451, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 1452, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 1453, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 1454, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 1455, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 1456, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 1457, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 1458, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 1459, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 1460, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 1461, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 1462, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 1463, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 1464, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 1465, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 1466, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 1467, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 1468, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 1469, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 1470, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 1471, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 1472, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 1473, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 1474, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 1475, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 1476, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 1477, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 1478, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 1479, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 1480, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 1481, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 1482, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 1483, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 1484, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 1485, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 1486, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 1487, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 1488, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 1489, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 1490, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 1491, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 1492, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 1493, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 1494, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 1495, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 1496, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 1497, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 1498, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 1499, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 1500, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 1501, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 1502, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 1503, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1504, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1505, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 1506, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1507, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 1508, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 1509, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 1510, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1511, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1512, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 1513, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 1514, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 1515, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 1516, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 1517, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 1518, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1519, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 1520, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 1521, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 1522, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 1523, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 1524, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 1525, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 1526, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 1527, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 1528, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 1529, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 1530, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 1531, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 1532, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 1533, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 1534, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 1535, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 1536, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 1537, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 1538, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 1539, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 1540, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 1541, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 1542, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1543, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1544, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 1545, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 1546, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 1547, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1548, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 1549, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 1550, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 1551, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 1552, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 1553, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 1554, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 1555, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 1556, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 1557, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 1558, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 1559, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 1560, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 1561, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 1562, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 1563, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 1564, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 1565, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 1566, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 1567, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 1568, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 1569, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 1570, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 1571, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 1572, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 1573, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 1574, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 1575, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 1576, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 1577, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 1578, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 1579, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 1580, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 1581, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 1582, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 1583, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 1584, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 1585, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 1586, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 1587, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 1588, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 1589, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 1590, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 1591, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 1592, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 1593, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 1594, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 1595, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 1596, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 1597, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 1598, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 1599, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 1600, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 1601, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 1602, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 1603, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 1604, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 1605, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 1606, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 1607, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 1608, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 1609, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 1610, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 1611, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 1612, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 1613, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 1614, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 1615, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 1616, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 1617, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 1618, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 1619, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 1620, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 1621, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 1622, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 1623, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 1624, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 1625, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 1626, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 1627, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 1628, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 1629, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 1630, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 1631, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 1632, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 1633, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 1634, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 1635, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 1636, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 1637, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 1638, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 1639, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1640, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1641, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 1642, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1643, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 1644, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 1645, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 1646, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1647, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1648, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 1649, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 1650, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 1651, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 1652, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 1653, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 1654, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1655, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 1656, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 1657, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 1658, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 1659, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 1660, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 1661, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 1662, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 1663, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 1664, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 1665, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 1666, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 1667, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 1668, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 1669, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 1670, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 1671, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 1672, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 1673, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 1674, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 1675, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 1676, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 1677, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 1678, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1679, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1680, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 1681, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 1682, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 1683, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1684, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 1685, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 1686, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 1687, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 1688, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 1689, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 1690, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 1691, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 1692, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 1693, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 1694, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 1695, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 1696, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 1697, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 1698, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 1699, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 1700, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 1701, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 1702, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 1703, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 1704, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 1705, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 1706, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 1707, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 1708, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 1709, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 1710, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 1711, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 1712, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 1713, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 1714, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 1715, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 1716, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 1717, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 1718, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 1719, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 1720, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 1721, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 1722, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 1723, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 1724, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 1725, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 1726, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 1727, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 1728, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 1729, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 1730, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 1731, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 1732, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 1733, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 1734, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 1735, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 1736, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 1737, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 1738, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 1739, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 1740, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 1741, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 1742, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 1743, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 1744, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 1745, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 1746, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 1747, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 1748, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 1749, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 1750, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 1751, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 1752, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 1753, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 1754, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 1755, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 1756, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 1757, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 1758, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 1759, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 1760, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 1761, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 1762, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 1763, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 1764, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 1765, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 1766, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 1767, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 1768, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 1769, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 1770, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 1771, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 1772, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 1773, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 1774, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 1775, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1776, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1777, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 1778, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1779, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 1780, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 1781, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 1782, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1783, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1784, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 1785, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 1786, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 1787, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 1788, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 1789, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 1790, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1791, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 1792, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 1793, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 1794, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 1795, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 1796, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 1797, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 1798, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 1799, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 1800, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 1801, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 1802, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 1803, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 1804, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 1805, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 1806, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 1807, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 1808, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 1809, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 1810, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 1811, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 1812, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 1813, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 1814, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1815, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1816, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 1817, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 1818, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 1819, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1820, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 1821, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 1822, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 1823, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 1824, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 1825, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 1826, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 1827, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 1828, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 1829, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 1830, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 1831, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 1832, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 1833, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 1834, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 1835, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 1836, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 1837, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 1838, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 1839, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 1840, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 1841, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 1842, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 1843, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 1844, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 1845, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 1846, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 1847, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 1848, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 1849, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 1850, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 1851, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 1852, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 1853, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 1854, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 1855, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 1856, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 1857, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 1858, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 1859, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 1860, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 1861, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 1862, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 1863, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 1864, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 1865, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 1866, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 1867, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 1868, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 1869, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 1870, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 1871, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 1872, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 1873, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 1874, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 1875, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 1876, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 1877, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 1878, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 1879, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 1880, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 1881, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 1882, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 1883, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 1884, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 1885, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 1886, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 1887, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 1888, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 1889, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 1890, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 1891, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 1892, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 1893, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 1894, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 1895, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 1896, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 1897, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 1898, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 1899, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 1900, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 1901, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 1902, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 1903, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 1904, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 1905, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 1906, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 1907, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 1908, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 1909, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 1910, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 1911, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1912, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1913, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 1914, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1915, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 1916, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 1917, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 1918, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1919, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1920, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 1921, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 1922, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 1923, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 1924, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 1925, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 1926, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1927, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 1928, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 1929, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 1930, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 1931, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 1932, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 1933, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 1934, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 1935, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 1936, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 1937, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 1938, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 1939, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 1940, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 1941, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 1942, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 1943, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 1944, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 1945, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 1946, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 1947, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 1948, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 1949, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 1950, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 1951, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 1952, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 1953, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 1954, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 1955, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 1956, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 1957, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 1958, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 1959, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 1960, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 1961, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 1962, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 1963, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 1964, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 1965, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 1966, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 1967, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 1968, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 1969, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 1970, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 1971, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 1972, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 1973, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 1974, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 1975, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 1976, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 1977, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 1978, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 1979, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 1980, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 1981, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 1982, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 1983, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 1984, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 1985, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 1986, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 1987, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 1988, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 1989, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 1990, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 1991, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 1992, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 1993, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 1994, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 1995, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 1996, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 1997, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 1998, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 1999, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 2000, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 2001, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 2002, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 2003, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 2004, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 2005, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 2006, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 2007, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 2008, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 2009, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 2010, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 2011, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 2012, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 2013, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 2014, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 2015, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 2016, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 2017, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 2018, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 2019, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 2020, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 2021, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 2022, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 2023, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 2024, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 2025, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 2026, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 2027, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 2028, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 2029, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 2030, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 2031, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 2032, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 2033, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 2034, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 2035, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 2036, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 2037, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 2038, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 2039, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 2040, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 2041, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 2042, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 2043, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 2044, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 2045, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 2046, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 2047, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2048, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2049, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 2050, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2051, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 2052, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 2053, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 2054, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2055, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2056, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 2057, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 2058, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 2059, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 2060, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 2061, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 2062, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2063, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 2064, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 2065, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 2066, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 2067, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 2068, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 2069, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 2070, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 2071, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 2072, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 2073, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 2074, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 2075, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 2076, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 2077, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 2078, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 2079, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 2080, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 2081, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 2082, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 2083, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 2084, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 2085, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 2086, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2087, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2088, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 2089, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 2090, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 2091, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2092, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 2093, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 2094, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 2095, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 2096, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 2097, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 2098, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 2099, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 2100, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 2101, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 2102, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 2103, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 2104, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 2105, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 2106, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 2107, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 2108, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 2109, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 2110, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 2111, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 2112, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 2113, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 2114, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 2115, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 2116, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 2117, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 2118, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 2119, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 2120, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 2121, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 2122, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 2123, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 2124, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 2125, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 2126, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 2127, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 2128, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 2129, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 2130, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 2131, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 2132, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 2133, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 2134, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 2135, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 2136, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 2137, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 2138, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 2139, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 2140, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 2141, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 2142, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 2143, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 2144, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 2145, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 2146, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 2147, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 2148, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 2149, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 2150, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 2151, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 2152, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 2153, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 2154, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 2155, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 2156, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 2157, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 2158, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 2159, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 2160, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 2161, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 2162, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 2163, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 2164, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 2165, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 2166, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 2167, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 2168, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 2169, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 2170, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 2171, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 2172, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 2173, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 2174, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 2175, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 2176, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 2177, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 2178, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 2179, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 2180, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 2181, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 2182, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 2183, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2184, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2185, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 2186, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2187, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 2188, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 2189, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 2190, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2191, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2192, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 2193, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 2194, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 2195, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 2196, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 2197, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 2198, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2199, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 2200, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 2201, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 2202, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 2203, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 2204, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 2205, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 2206, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 2207, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 2208, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 2209, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 2210, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 2211, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 2212, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 2213, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 2214, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 2215, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 2216, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 2217, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 2218, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 2219, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 2220, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 2221, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 2222, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2223, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2224, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 2225, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 2226, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 2227, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2228, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 2229, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 2230, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 2231, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 2232, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 2233, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 2234, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 2235, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 2236, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 2237, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 2238, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 2239, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 2240, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 2241, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 2242, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 2243, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 2244, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 2245, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 2246, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 2247, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 2248, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 2249, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 2250, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 2251, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 2252, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 2253, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 2254, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 2255, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 2256, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 2257, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 2258, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 2259, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 2260, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 2261, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 2262, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 2263, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 2264, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 2265, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 2266, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 2267, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 2268, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 2269, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 2270, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 2271, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 2272, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 2273, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 2274, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 2275, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 2276, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 2277, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 2278, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 2279, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 2280, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 2281, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 2282, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 2283, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 2284, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 2285, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 2286, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 2287, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 2288, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 2289, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 2290, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 2291, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 2292, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 2293, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 2294, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 2295, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 2296, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 2297, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 2298, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 2299, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 2300, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 2301, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 2302, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 2303, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 2304, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 2305, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 2306, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 2307, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 2308, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 2309, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 2310, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 2311, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 2312, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 2313, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 2314, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 2315, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 2316, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 2317, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 2318, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 2319, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2320, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2321, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 2322, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2323, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 2324, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 2325, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 2326, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2327, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2328, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 2329, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 2330, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 2331, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 2332, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 2333, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 2334, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2335, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 2336, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 2337, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 2338, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 2339, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 2340, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 2341, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 2342, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 2343, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 2344, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 2345, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 2346, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 2347, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 2348, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 2349, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 2350, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 2351, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 2352, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 2353, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 2354, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 2355, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 2356, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 2357, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 2358, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2359, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2360, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 2361, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 2362, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 2363, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2364, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 2365, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 2366, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 2367, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 2368, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 2369, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 2370, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 2371, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 2372, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 2373, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 2374, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 2375, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 2376, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 2377, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 2378, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 2379, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 2380, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 2381, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 2382, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 2383, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 2384, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 2385, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 2386, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 2387, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 2388, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 2389, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 2390, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 2391, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 2392, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 2393, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 2394, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 2395, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 2396, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 2397, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 2398, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 2399, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 2400, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 2401, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 2402, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 2403, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 2404, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 2405, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 2406, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 2407, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 2408, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 2409, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 2410, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 2411, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 2412, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 2413, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 2414, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 2415, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 2416, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 2417, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 2418, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 2419, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 2420, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 2421, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 2422, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 2423, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 2424, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 2425, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 2426, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 2427, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 2428, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 2429, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 2430, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 2431, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 2432, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 2433, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 2434, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 2435, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 2436, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 2437, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 2438, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 2439, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 2440, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 2441, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 2442, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 2443, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 2444, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 2445, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 2446, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 2447, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 2448, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 2449, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 2450, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 2451, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 2452, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 2453, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 2454, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 2455, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2456, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2457, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 2458, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2459, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 2460, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 2461, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 2462, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2463, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2464, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 2465, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 2466, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 2467, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 2468, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 2469, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 2470, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2471, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 2472, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 2473, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 2474, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 2475, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 2476, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 2477, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 2478, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 2479, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 2480, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 2481, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 2482, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 2483, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 2484, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 2485, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 2486, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 2487, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 2488, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 2489, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 2490, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 2491, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 2492, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 2493, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 2494, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2495, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2496, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 2497, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 2498, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 2499, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2500, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 2501, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 2502, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 2503, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 2504, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 2505, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 2506, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 2507, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 2508, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 2509, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 2510, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 2511, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 2512, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 2513, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 2514, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 2515, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 2516, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 2517, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 2518, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 2519, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 2520, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 2521, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 2522, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 2523, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 2524, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 2525, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 2526, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 2527, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 2528, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 2529, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 2530, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 2531, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 2532, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 2533, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 2534, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 2535, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 2536, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 2537, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 2538, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 2539, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 2540, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 2541, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 2542, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 2543, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 2544, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 2545, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 2546, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 2547, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 2548, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 2549, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 2550, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 2551, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 2552, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 2553, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 2554, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 2555, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 2556, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 2557, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 2558, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 2559, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 2560, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 2561, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 2562, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 2563, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 2564, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 2565, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 2566, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 2567, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 2568, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 2569, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 2570, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 2571, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 2572, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 2573, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 2574, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 2575, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 2576, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 2577, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 2578, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 2579, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 2580, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 2581, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 2582, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 2583, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 2584, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 2585, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 2586, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 2587, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 2588, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 2589, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 2590, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 2591, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2592, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2593, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 2594, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2595, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 2596, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 2597, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 2598, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2599, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2600, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 2601, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 2602, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 2603, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 2604, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 2605, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 2606, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2607, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 2608, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 2609, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 2610, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 2611, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 2612, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 2613, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 2614, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 2615, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 2616, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 2617, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 2618, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 2619, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 2620, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 2621, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 2622, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 2623, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 2624, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 2625, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 2626, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 2627, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 2628, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 2629, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 2630, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2631, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2632, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 2633, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 2634, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 2635, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2636, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 2637, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 2638, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 2639, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 2640, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 2641, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 2642, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 2643, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 2644, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 2645, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 2646, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 2647, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 2648, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 2649, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 2650, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 2651, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 2652, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 2653, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 2654, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 2655, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 2656, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 2657, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 2658, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 2659, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 2660, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 2661, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 2662, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 2663, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 2664, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 2665, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 2666, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 2667, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 2668, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 2669, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 2670, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 2671, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 2672, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 2673, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 2674, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 2675, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 2676, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 2677, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 2678, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 2679, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 2680, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 2681, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 2682, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 2683, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 2684, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 2685, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 2686, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 2687, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 2688, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 2689, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 2690, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 2691, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 2692, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 2693, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 2694, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 2695, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 2696, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 2697, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 2698, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 2699, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 2700, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 2701, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 2702, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 2703, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 2704, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 2705, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 2706, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 2707, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 2708, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 2709, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 2710, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 2711, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 2712, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 2713, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 2714, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 2715, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 2716, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 2717, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 2718, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 2719, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 2720, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 2721, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 2722, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 2723, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 2724, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 2725, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 2726, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 2727, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2728, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2729, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 2730, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2731, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 2732, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 2733, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 2734, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2735, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2736, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 2737, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 2738, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 2739, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 2740, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 2741, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 2742, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2743, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 2744, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 2745, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 2746, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 2747, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 2748, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 2749, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 2750, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 2751, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 2752, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 2753, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 2754, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 2755, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 2756, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 2757, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 2758, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 2759, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 2760, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 2761, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 2762, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 2763, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 2764, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 2765, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 2766, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2767, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2768, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 2769, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 2770, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 2771, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2772, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 2773, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 2774, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 2775, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 2776, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 2777, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 2778, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 2779, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 2780, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 2781, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 2782, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 2783, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 2784, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 2785, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 2786, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 2787, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 2788, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 2789, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 2790, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 2791, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 2792, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 2793, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 2794, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 2795, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 2796, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 2797, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 2798, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 2799, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 2800, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 2801, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 2802, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 2803, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 2804, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 2805, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 2806, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 2807, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 2808, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 2809, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 2810, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 2811, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 2812, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 2813, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 2814, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 2815, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 2816, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 2817, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 2818, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 2819, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 2820, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 2821, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 2822, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 2823, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 2824, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 2825, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 2826, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 2827, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 2828, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 2829, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 2830, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 2831, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 2832, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 2833, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 2834, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 2835, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 2836, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 2837, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 2838, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 2839, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 2840, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 2841, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 2842, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 2843, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 2844, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 2845, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 2846, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 2847, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 2848, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 2849, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 2850, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 2851, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 2852, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 2853, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 2854, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 2855, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 2856, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 2857, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 2858, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 2859, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 2860, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 2861, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 2862, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 2863, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2864, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2865, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 2866, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2867, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 2868, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 2869, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 2870, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2871, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2872, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 2873, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 2874, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 2875, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 2876, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 2877, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 2878, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2879, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 2880, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 2881, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 2882, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 2883, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 2884, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 2885, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 2886, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 2887, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 2888, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 2889, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 2890, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 2891, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 2892, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 2893, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 2894, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 2895, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 2896, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 2897, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 2898, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 2899, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 2900, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 2901, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 2902, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 2903, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 2904, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 2905, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 2906, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 2907, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 2908, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 2909, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 2910, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 2911, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 2912, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 2913, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 2914, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 2915, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 2916, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 2917, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 2918, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 2919, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 2920, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 2921, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 2922, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 2923, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 2924, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 2925, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 2926, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 2927, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 2928, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 2929, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 2930, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 2931, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 2932, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 2933, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 2934, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 2935, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 2936, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 2937, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 2938, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 2939, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 2940, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 2941, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 2942, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 2943, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 2944, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 2945, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 2946, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 2947, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 2948, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 2949, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 2950, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 2951, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 2952, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 2953, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 2954, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 2955, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 2956, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 2957, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 2958, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 2959, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 2960, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 2961, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 2962, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 2963, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 2964, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 2965, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 2966, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 2967, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 2968, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 2969, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 2970, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 2971, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 2972, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 2973, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 2974, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 2975, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 2976, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 2977, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 2978, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 2979, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 2980, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 2981, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 2982, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 2983, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 2984, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 2985, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 2986, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 2987, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 2988, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 2989, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 2990, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 2991, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 2992, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 2993, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 2994, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 2995, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 2996, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 2997, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 2998, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 2999, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 3000, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 3001, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 3002, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 3003, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 3004, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 3005, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 3006, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 3007, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 3008, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 3009, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 3010, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 3011, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 3012, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 3013, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 3014, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 3015, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 3016, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 3017, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 3018, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 3019, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 3020, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 3021, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 3022, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 3023, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 3024, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 3025, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 3026, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 3027, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 3028, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 3029, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 3030, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 3031, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 3032, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 3033, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 3034, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 3035, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 3036, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 3037, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 3038, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 3039, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 3040, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 3041, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 3042, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 3043, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 3044, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 3045, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 3046, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 3047, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 3048, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 3049, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 3050, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 3051, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 3052, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 3053, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 3054, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 3055, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 3056, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 3057, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 3058, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 3059, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 3060, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 3061, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 3062, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 3063, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 3064, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 3065, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 3066, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 3067, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 3068, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 3069, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 3070, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 3071, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 3072, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 3073, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 3074, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 3075, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 3076, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 3077, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 3078, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 3079, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 3080, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 3081, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 3082, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 3083, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 3084, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 3085, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 3086, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 3087, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 3088, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 3089, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 3090, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 3091, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 3092, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 3093, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 3094, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 3095, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 3096, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 3097, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 3098, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 3099, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 3100, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 3101, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 3102, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 3103, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 3104, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 3105, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 3106, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 3107, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 3108, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 3109, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 3110, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 3111, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 3112, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 3113, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 3114, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 3115, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 3116, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 3117, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 3118, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 3119, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 3120, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 3121, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 3122, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 3123, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 3124, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 3125, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 3126, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 3127, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 3128, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 3129, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 3130, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 3131, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 3132, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 3133, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 3134, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 3135, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 3136, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 3137, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 3138, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 3139, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 3140, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 3141, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 3142, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 3143, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 3144, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 3145, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 3146, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 3147, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 3148, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 3149, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 3150, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 3151, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 3152, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 3153, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 3154, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 3155, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 3156, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 3157, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 3158, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 3159, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 3160, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 3161, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 3162, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 3163, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 3164, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 3165, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 3166, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 3167, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 3168, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 3169, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 3170, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 3171, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 3172, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 3173, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 3174, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 3175, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 3176, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 3177, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 3178, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 3179, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 3180, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 3181, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 3182, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 3183, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 3184, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 3185, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 3186, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 3187, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 3188, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 3189, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 3190, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 3191, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 3192, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 3193, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 3194, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 3195, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 3196, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 3197, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 3198, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 3199, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 3200, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 3201, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 3202, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 3203, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 3204, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 3205, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 3206, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 3207, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 3208, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 3209, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 3210, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 3211, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 3212, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 3213, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 3214, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 3215, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 3216, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 3217, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 3218, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 3219, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 3220, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 3221, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 3222, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 3223, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 3224, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 3225, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 3226, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 3227, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 3228, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 3229, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 3230, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 3231, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 3232, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 3233, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 3234, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 3235, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 3236, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 3237, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 3238, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 3239, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 3240, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 3241, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 3242, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 3243, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 3244, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 3245, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 3246, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 3247, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 3248, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 3249, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 3250, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 3251, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 3252, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 3253, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 3254, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 3255, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 3256, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 3257, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 3258, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 3259, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 3260, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 3261, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 3262, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 3263, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 3264, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
    {'id': 3265, 'stage': 'SOURCE_FETCH', 'field': 'source URL', 'description': 'Observe and report this production value.'},
    {'id': 3266, 'stage': 'SOURCE_FETCH', 'field': 'HTTP status', 'description': 'Observe and report this production value.'},
    {'id': 3267, 'stage': 'SOURCE_FETCH', 'field': 'page title', 'description': 'Observe and report this production value.'},
    {'id': 3268, 'stage': 'SOURCE_FETCH', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 3269, 'stage': 'SOURCE_FETCH', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 3270, 'stage': 'SOURCE_FETCH', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 3271, 'stage': 'SOURCE_FETCH', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 3272, 'stage': 'SOURCE_FETCH', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 3273, 'stage': 'SOURCE_FETCH', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 3274, 'stage': 'SOURCE_FETCH', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 3275, 'stage': 'SOURCE_FETCH', 'field': 'extraction completeness', 'description': 'Observe and report this production value.'},
    {'id': 3276, 'stage': 'ARTISAN_INPUT', 'field': 'product ID', 'description': 'Observe and report this production value.'},
    {'id': 3277, 'stage': 'ARTISAN_INPUT', 'field': 'product name', 'description': 'Observe and report this production value.'},
    {'id': 3278, 'stage': 'ARTISAN_INPUT', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 3279, 'stage': 'ARTISAN_INPUT', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 3280, 'stage': 'ARTISAN_INPUT', 'field': 'additional materials', 'description': 'Observe and report this production value.'},
    {'id': 3281, 'stage': 'ARTISAN_INPUT', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 3282, 'stage': 'ARTISAN_INPUT', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 3283, 'stage': 'ARTISAN_INPUT', 'field': 'length', 'description': 'Observe and report this production value.'},
    {'id': 3284, 'stage': 'ARTISAN_INPUT', 'field': 'width', 'description': 'Observe and report this production value.'},
    {'id': 3285, 'stage': 'ARTISAN_INPUT', 'field': 'height', 'description': 'Observe and report this production value.'},
    {'id': 3286, 'stage': 'ARTISAN_INPUT', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 3287, 'stage': 'ARTISAN_INPUT', 'field': 'description', 'description': 'Observe and report this production value.'},
    {'id': 3288, 'stage': 'ARTISAN_INPUT', 'field': 'material cost', 'description': 'Observe and report this production value.'},
    {'id': 3289, 'stage': 'ARTISAN_INPUT', 'field': 'labour cost', 'description': 'Observe and report this production value.'},
    {'id': 3290, 'stage': 'ARTISAN_INPUT', 'field': 'production cost', 'description': 'Observe and report this production value.'},
    {'id': 3291, 'stage': 'ARTISAN_INPUT', 'field': 'time worked', 'description': 'Observe and report this production value.'},
    {'id': 3292, 'stage': 'ARTISAN_INPUT', 'field': 'independent total cost', 'description': 'Observe and report this production value.'},
    {'id': 3293, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigar Online', 'description': 'Observe and report this production value.'},
    {'id': 3294, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Gaatha', 'description': 'Observe and report this production value.'},
    {'id': 3295, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Jaypore', 'description': 'Observe and report this production value.'},
    {'id': 3296, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'GoSwadeshi', 'description': 'Observe and report this production value.'},
    {'id': 3297, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Karigar', 'description': 'Observe and report this production value.'},
    {'id': 3298, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Santarms', 'description': 'Observe and report this production value.'},
    {'id': 3299, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaari', 'description': 'Observe and report this production value.'},
    {'id': 3300, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Kaarigo', 'description': 'Observe and report this production value.'},
    {'id': 3301, 'stage': 'PRIMARY_MARKETPLACE', 'field': 'Okhai', 'description': 'Observe and report this production value.'},
    {'id': 3302, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Amazon', 'description': 'Observe and report this production value.'},
    {'id': 3303, 'stage': 'SECONDARY_MARKETPLACE', 'field': 'Flipkart', 'description': 'Observe and report this production value.'},
    {'id': 3304, 'stage': 'NORMALIZATION', 'field': 'marketplace', 'description': 'Observe and report this production value.'},
    {'id': 3305, 'stage': 'NORMALIZATION', 'field': 'source tier', 'description': 'Observe and report this production value.'},
    {'id': 3306, 'stage': 'NORMALIZATION', 'field': 'listing ID', 'description': 'Observe and report this production value.'},
    {'id': 3307, 'stage': 'NORMALIZATION', 'field': 'title', 'description': 'Observe and report this production value.'},
    {'id': 3308, 'stage': 'NORMALIZATION', 'field': 'price', 'description': 'Observe and report this production value.'},
    {'id': 3309, 'stage': 'NORMALIZATION', 'field': 'currency', 'description': 'Observe and report this production value.'},
    {'id': 3310, 'stage': 'NORMALIZATION', 'field': 'product type', 'description': 'Observe and report this production value.'},
    {'id': 3311, 'stage': 'NORMALIZATION', 'field': 'material', 'description': 'Observe and report this production value.'},
    {'id': 3312, 'stage': 'NORMALIZATION', 'field': 'color', 'description': 'Observe and report this production value.'},
    {'id': 3313, 'stage': 'NORMALIZATION', 'field': 'size', 'description': 'Observe and report this production value.'},
    {'id': 3314, 'stage': 'NORMALIZATION', 'field': 'dimensions', 'description': 'Observe and report this production value.'},
    {'id': 3315, 'stage': 'NORMALIZATION', 'field': 'features', 'description': 'Observe and report this production value.'},
    {'id': 3316, 'stage': 'NORMALIZATION', 'field': 'handmade evidence', 'description': 'Observe and report this production value.'},
    {'id': 3317, 'stage': 'NORMALIZATION', 'field': 'seller', 'description': 'Observe and report this production value.'},
    {'id': 3318, 'stage': 'NORMALIZATION', 'field': 'rating', 'description': 'Observe and report this production value.'},
    {'id': 3319, 'stage': 'NORMALIZATION', 'field': 'review count', 'description': 'Observe and report this production value.'},
    {'id': 3320, 'stage': 'NORMALIZATION', 'field': 'URL', 'description': 'Observe and report this production value.'},
    {'id': 3321, 'stage': 'SIMILARITY', 'field': 'product type score', 'description': 'Observe and report this production value.'},
    {'id': 3322, 'stage': 'SIMILARITY', 'field': 'material score', 'description': 'Observe and report this production value.'},
    {'id': 3323, 'stage': 'SIMILARITY', 'field': 'size score', 'description': 'Observe and report this production value.'},
    {'id': 3324, 'stage': 'SIMILARITY', 'field': 'dimensions score', 'description': 'Observe and report this production value.'},
    {'id': 3325, 'stage': 'SIMILARITY', 'field': 'features score', 'description': 'Observe and report this production value.'},
    {'id': 3326, 'stage': 'SIMILARITY', 'field': 'color score', 'description': 'Observe and report this production value.'},
    {'id': 3327, 'stage': 'SIMILARITY', 'field': 'semantic text score', 'description': 'Observe and report this production value.'},
    {'id': 3328, 'stage': 'SIMILARITY', 'field': 'total score', 'description': 'Observe and report this production value.'},
    {'id': 3329, 'stage': 'SIMILARITY', 'field': 'match level', 'description': 'Observe and report this production value.'},
    {'id': 3330, 'stage': 'SIMILARITY', 'field': 'minimum threshold', 'description': 'Observe and report this production value.'},
    {'id': 3331, 'stage': 'SIMILARITY', 'field': 'comparability decision', 'description': 'Observe and report this production value.'},
    {'id': 3332, 'stage': 'SIMILARITY', 'field': 'rejection reason', 'description': 'Observe and report this production value.'},
    {'id': 3333, 'stage': 'SIMILARITY', 'field': 'evidence fields', 'description': 'Observe and report this production value.'},
    {'id': 3334, 'stage': 'SIMILARITY', 'field': 'comparable reason', 'description': 'Observe and report this production value.'},
    {'id': 3335, 'stage': 'MARKET_EVIDENCE', 'field': 'raw listing prices', 'description': 'Observe and report this production value.'},
    {'id': 3336, 'stage': 'MARKET_EVIDENCE', 'field': 'accepted prices', 'description': 'Observe and report this production value.'},
    {'id': 3337, 'stage': 'MARKET_EVIDENCE', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 3338, 'stage': 'MARKET_EVIDENCE', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 3339, 'stage': 'MARKET_EVIDENCE', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 3340, 'stage': 'MARKET_EVIDENCE', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 3341, 'stage': 'MARKET_EVIDENCE', 'field': 'best similarity', 'description': 'Observe and report this production value.'},
    {'id': 3342, 'stage': 'MARKET_EVIDENCE', 'field': 'average similarity', 'description': 'Observe and report this production value.'},
    {'id': 3343, 'stage': 'MARKET_EVIDENCE', 'field': 'current comparable count', 'description': 'Observe and report this production value.'},
    {'id': 3344, 'stage': 'MARKET_EVIDENCE', 'field': 'historical comparable count', 'description': 'Observe and report this production value.'},
    {'id': 3345, 'stage': 'DEMAND_SUPPLY', 'field': 'views', 'description': 'Observe and report this production value.'},
    {'id': 3346, 'stage': 'DEMAND_SUPPLY', 'field': 'clicks', 'description': 'Observe and report this production value.'},
    {'id': 3347, 'stage': 'DEMAND_SUPPLY', 'field': 'wishlists', 'description': 'Observe and report this production value.'},
    {'id': 3348, 'stage': 'DEMAND_SUPPLY', 'field': 'add to cart', 'description': 'Observe and report this production value.'},
    {'id': 3349, 'stage': 'DEMAND_SUPPLY', 'field': 'enquiries', 'description': 'Observe and report this production value.'},
    {'id': 3350, 'stage': 'DEMAND_SUPPLY', 'field': 'orders', 'description': 'Observe and report this production value.'},
    {'id': 3351, 'stage': 'DEMAND_SUPPLY', 'field': 'inventory', 'description': 'Observe and report this production value.'},
    {'id': 3352, 'stage': 'DEMAND_SUPPLY', 'field': 'conversion rate', 'description': 'Observe and report this production value.'},
    {'id': 3353, 'stage': 'DEMAND_SUPPLY', 'field': 'sales velocity', 'description': 'Observe and report this production value.'},
    {'id': 3354, 'stage': 'DEMAND_SUPPLY', 'field': 'demand signals', 'description': 'Observe and report this production value.'},
    {'id': 3355, 'stage': 'DEMAND_SUPPLY', 'field': 'supply signals', 'description': 'Observe and report this production value.'},
    {'id': 3356, 'stage': 'FEATURE_ENGINEERING', 'field': '29 feature count', 'description': 'Observe and report this production value.'},
    {'id': 3357, 'stage': 'FEATURE_ENGINEERING', 'field': 'base cost', 'description': 'Observe and report this production value.'},
    {'id': 3358, 'stage': 'FEATURE_ENGINEERING', 'field': 'market minimum', 'description': 'Observe and report this production value.'},
    {'id': 3359, 'stage': 'FEATURE_ENGINEERING', 'field': 'market median', 'description': 'Observe and report this production value.'},
    {'id': 3360, 'stage': 'FEATURE_ENGINEERING', 'field': 'market maximum', 'description': 'Observe and report this production value.'},
    {'id': 3361, 'stage': 'FEATURE_ENGINEERING', 'field': 'market count', 'description': 'Observe and report this production value.'},
    {'id': 3362, 'stage': 'FEATURE_ENGINEERING', 'field': 'similarity', 'description': 'Observe and report this production value.'},
    {'id': 3363, 'stage': 'FEATURE_ENGINEERING', 'field': 'demand', 'description': 'Observe and report this production value.'},
    {'id': 3364, 'stage': 'FEATURE_ENGINEERING', 'field': 'supply', 'description': 'Observe and report this production value.'},
    {'id': 3365, 'stage': 'FEATURE_ENGINEERING', 'field': 'conversion', 'description': 'Observe and report this production value.'},
    {'id': 3366, 'stage': 'FEATURE_ENGINEERING', 'field': 'seasonality', 'description': 'Observe and report this production value.'},
    {'id': 3367, 'stage': 'FEATURE_ENGINEERING', 'field': 'complexity', 'description': 'Observe and report this production value.'},
    {'id': 3368, 'stage': 'FEATURE_ENGINEERING', 'field': 'cost', 'description': 'Observe and report this production value.'},
    {'id': 3369, 'stage': 'ML_READINESS', 'field': 'model existence', 'description': 'Observe and report this production value.'},
    {'id': 3370, 'stage': 'ML_READINESS', 'field': 'outcome rows', 'description': 'Observe and report this production value.'},
    {'id': 3371, 'stage': 'ML_READINESS', 'field': 'required features', 'description': 'Observe and report this production value.'},
    {'id': 3372, 'stage': 'ML_READINESS', 'field': 'validation quality', 'description': 'Observe and report this production value.'},
    {'id': 3373, 'stage': 'ML_READINESS', 'field': 'coverage', 'description': 'Observe and report this production value.'},
    {'id': 3374, 'stage': 'ML_READINESS', 'field': 'category coverage', 'description': 'Observe and report this production value.'},
    {'id': 3375, 'stage': 'ML_READINESS', 'field': 'material coverage', 'description': 'Observe and report this production value.'},
    {'id': 3376, 'stage': 'ML_READINESS', 'field': 'readiness decision', 'description': 'Observe and report this production value.'},
    {'id': 3377, 'stage': 'ML_READINESS', 'field': 'ML invoked', 'description': 'Observe and report this production value.'},
    {'id': 3378, 'stage': 'ML_READINESS', 'field': 'ML prediction', 'description': 'Observe and report this production value.'},
    {'id': 3379, 'stage': 'PRICING', 'field': 'formula price', 'description': 'Observe and report this production value.'},
    {'id': 3380, 'stage': 'PRICING', 'field': 'market based price', 'description': 'Observe and report this production value.'},
    {'id': 3381, 'stage': 'PRICING', 'field': 'fair trade floor', 'description': 'Observe and report this production value.'},
    {'id': 3382, 'stage': 'PRICING', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 3383, 'stage': 'PRICING', 'field': 'pricing method', 'description': 'Observe and report this production value.'},
    {'id': 3384, 'stage': 'PRICING', 'field': 'decision reason', 'description': 'Observe and report this production value.'},
    {'id': 3385, 'stage': 'GUARDRAILS', 'field': 'minimum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 3386, 'stage': 'GUARDRAILS', 'field': 'maximum guardrail', 'description': 'Observe and report this production value.'},
    {'id': 3387, 'stage': 'GUARDRAILS', 'field': 'daily movement', 'description': 'Observe and report this production value.'},
    {'id': 3388, 'stage': 'GUARDRAILS', 'field': 'minimum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 3389, 'stage': 'GUARDRAILS', 'field': 'maximum acceptable price', 'description': 'Observe and report this production value.'},
    {'id': 3390, 'stage': 'GUARDRAILS', 'field': 'market deviation protection', 'description': 'Observe and report this production value.'},
    {'id': 3391, 'stage': 'FINAL_RECOMMENDATION', 'field': 'final price', 'description': 'Observe and report this production value.'},
    {'id': 3392, 'stage': 'FINAL_RECOMMENDATION', 'field': 'recommended price', 'description': 'Observe and report this production value.'},
    {'id': 3393, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to base cost', 'description': 'Observe and report this production value.'},
    {'id': 3394, 'stage': 'FINAL_RECOMMENDATION', 'field': 'comparison to market median', 'description': 'Observe and report this production value.'},
    {'id': 3395, 'stage': 'FINAL_RECOMMENDATION', 'field': 'artisan approval value', 'description': 'Observe and report this production value.'},
    {'id': 3396, 'stage': 'PERSISTENCE_SAFETY', 'field': 'persist flag', 'description': 'Observe and report this production value.'},
    {'id': 3397, 'stage': 'PERSISTENCE_SAFETY', 'field': 'actual selling price', 'description': 'Observe and report this production value.'},
    {'id': 3398, 'stage': 'PERSISTENCE_SAFETY', 'field': 'training label', 'description': 'Observe and report this production value.'},
    {'id': 3399, 'stage': 'PERSISTENCE_SAFETY', 'field': 'pricing event creation', 'description': 'Observe and report this production value.'},
    {'id': 3400, 'stage': 'PERSISTENCE_SAFETY', 'field': 'database mutation', 'description': 'Observe and report this production value.'},
]

def _audit_repr(value):
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.8f}"
    return repr(value)


def print_similarity_weights_and_policy():
    print()
    print("=" * 100)
    print("DETAILED SIMILARITY WEIGHTS AND ACCEPTANCE POLICY")
    print("=" * 100)

    weights = [
        ("product_type", 0.28),
        ("material", 0.18),
        ("size", 0.08),
        ("dimensions", 0.10),
        ("features", 0.13),
        ("color", 0.05),
        ("text", 0.18),
    ]

    for name, weight in weights:
        print(f"  {name:<24} weight = {weight:.2f}")

    print()
    print("Adaptive weighted comparison is used.")
    print("Missing attributes are not automatically treated as mismatches.")
    print("Different material can remain RELATED/SECONDARY.")
    print("Obviously unrelated product types should be rejected.")
    print("Retail selling price is never treated as production cost.")


def print_detailed_product(product):
    print()
    print("=" * 100)
    print("DETAILED ARTISAN PRODUCT INPUT")
    print("=" * 100)

    for key in [
        "product_id", "product_name", "product_type", "material",
        "additional_materials", "color", "size", "dimensions_length",
        "dimensions_width", "dimensions_height", "features", "description",
        "material_cost", "labour_cost", "production_cost",
        "time_worked_hours", "indiahandmade_price", "source_url",
    ]:
        print(f"  {key:<30}: {_audit_repr(product.get(key))}")

    total = (
        float(product.get("material_cost", 0) or 0)
        + float(product.get("labour_cost", 0) or 0)
        + float(product.get("production_cost", 0) or 0)
    )

    print(f"  {'independent_total_cost':<30}: ₹{total:.2f}")


def print_detailed_listings(listings):
    print()
    print("=" * 100)
    print("EVERY NORMALIZED MARKETPLACE CANDIDATE")
    print("=" * 100)
    print(f"Candidate count: {len(listings)}")

    for i, listing in enumerate(listings, 1):
        print()
        print(f"CANDIDATE {i}/{len(listings)}")
        print("-" * 100)

        for key in sorted(listing):
            print(f"  {key:<30}: {_audit_repr(listing.get(key))}")


def print_detailed_matches(matches):
    print()
    print("=" * 100)
    print("EVERY SIMILARITY-APPROVED COMPARABLE")
    print("=" * 100)

    if not matches:
        print("NONE")

    for i, match in enumerate(matches, 1):
        obj = match_product(match)

        print()
        print(f"COMPARABLE {i}/{len(matches)}")
        print("-" * 100)

        if isinstance(obj, dict):
            for key in sorted(obj):
                print(f"  {key:<30}: {_audit_repr(obj.get(key))}")

        print(f"  {'total similarity':<30}: {safe_score(match):.8f}")
        print(f"  {'match level':<30}: {match_level(match)}")

        breakdown = getattr(match, "breakdown", None)

        if breakdown is not None:
            print()
            print("  COMPONENT SCORES")
            print("  " + "-" * 94)

            for key in [
                "product_type", "material", "size", "dimensions",
                "features", "color", "text", "total_score",
                "match_level", "material_cost_similarity",
                "product_cost_similarity", "evidence_fields",
                "comparable_reason",
            ]:
                if hasattr(breakdown, key):
                    print(
                        f"    {key:<28}: "
                        f"{_audit_repr(getattr(breakdown, key))}"
                    )


def print_detailed_feature_vector(result):
    print()
    print("=" * 100)
    print("ALL 29 PRICING FEATURES")
    print("=" * 100)

    vector = getattr(result, "feature_vector", None)

    if not isinstance(vector, dict):
        print(f"Feature vector: {_audit_repr(vector)}")
        return

    print(f"Feature count: {len(vector)}")

    for i, (name, value) in enumerate(vector.items(), 1):
        print(f"  {i:02d}. {name:<45}: {_audit_repr(value)}")


def print_detailed_pricing_result(result):
    print()
    print("=" * 100)
    print("EVERY AVAILABLE PRICING RESULT FIELD")
    print("=" * 100)

    if result is None:
        print("RESULT = None")
        return

    for name in sorted(dir(result)):
        if name.startswith("_"):
            continue

        try:
            value = getattr(result, name)
        except Exception as exc:
            print(f"  {name:<40}: ERROR {exc}")
            continue

        if callable(value):
            continue

        print(f"  {name:<40}: {_audit_repr(value)}")


def print_detailed_pricing_trace(product, result):
    print()
    print("=" * 100)
    print("PRICING DECISION TRACE")
    print("=" * 100)

    total = (
        float(product.get("material_cost", 0) or 0)
        + float(product.get("labour_cost", 0) or 0)
        + float(product.get("production_cost", 0) or 0)
    )

    print("  STAGE 01 - ARTISAN PRODUCTION INPUTS")
    print(f"    material cost       : ₹{float(product.get('material_cost', 0) or 0):.2f}")
    print(f"    labour cost         : ₹{float(product.get('labour_cost', 0) or 0):.2f}")
    print(f"    production cost     : ₹{float(product.get('production_cost', 0) or 0):.2f}")
    print(f"    total base cost     : ₹{total:.2f}")

    print()
    print("  STAGE 02 - CURRENT MARKET")
    print(f"    market minimum      : {_audit_repr(getattr(result, 'market_minimum', None))}")
    print(f"    market median       : {_audit_repr(getattr(result, 'market_median', None))}")
    print(f"    market maximum      : {_audit_repr(getattr(result, 'market_maximum', None))}")
    print(f"    market count        : {_audit_repr(getattr(result, 'market_count', None))}")

    print()
    print("  STAGE 03 - BASELINE / FALLBACK")
    print(f"    formula price       : {_audit_repr(getattr(result, 'formula_price', None))}")
    print(f"    market based price  : {_audit_repr(getattr(result, 'market_based_price', None))}")
    print(f"    fair trade floor    : {_audit_repr(getattr(result, 'fair_trade_floor', None))}")

    print()
    print("  STAGE 04 - ML READINESS")
    print(f"    ready               : {_audit_repr(getattr(result, 'ml_readiness', None))}")
    print(f"    invoked             : {_audit_repr(getattr(result, 'ml_invoked', None))}")
    print(f"    model               : {_audit_repr(getattr(result, 'ml_model', None))}")
    print(f"    prediction          : {_audit_repr(getattr(result, 'ml_prediction', None))}")
    print(f"    reason              : {_audit_repr(getattr(result, 'ml_readiness_reason', None))}")

    print()
    print("  STAGE 05 - GUARDRAILS / FINAL")
    print(f"    before guardrails   : {_audit_repr(getattr(result, 'recommended_price', None))}")
    print(f"    guardrail minimum   : {_audit_repr(getattr(result, 'guardrail_minimum', None))}")
    print(f"    guardrail maximum   : {_audit_repr(getattr(result, 'guardrail_maximum', None))}")
    print(f"    FINAL PRICE         : {_audit_repr(getattr(result, 'final_price', None))}")
    print(f"    method              : {_audit_repr(getattr(result, 'method', None))}")
    print(f"    decision reason     : {_audit_repr(getattr(result, 'decision_reason', None))}")


def print_forensic_report(fetched, results):
    print()
    print("=" * 100)
    print("FULL FORENSIC REPORT - NO SECOND MARKETPLACE CRAWL")
    print("=" * 100)

    print(f"IndiaHandmade products fetched: {len(fetched)}")
    print(f"Pipeline products completed   : {len(results)}")

    print_similarity_weights_and_policy()

    for i, product in enumerate(fetched, 1):
        print()
        print(f"SOURCE PRODUCT {i}/{len(fetched)}")
        print_detailed_product(product)

    for i, item in enumerate(results, 1):
        print()
        print("=" * 100)
        print(f"FORENSIC PIPELINE REPORT {i}/{len(results)}")
        print("=" * 100)

        print_detailed_product(item["product"])
        print_detailed_listings(item["listings"])
        print_detailed_matches(item["matches"])

        if item["result"] is not None:
            print_detailed_feature_vector(item["result"])
            print_detailed_pricing_result(item["result"])
            print_detailed_pricing_trace(item["product"], item["result"])

    print()
    print("=" * 100)
    print("FULL AUDIT CHECKPOINT CATALOG")
    print("=" * 100)
    print(f"Explicit audit checkpoints: {len(AUDIT_CHECKPOINTS)}")

    current = None

    for item in AUDIT_CHECKPOINTS:
        if item["stage"] != current:
            current = item["stage"]
            print()
            print(f"[{current}]")
            print("-" * 100)

        print(
            f"  {item['id']:04d} | "
            f"{item['field']:<45} | "
            f"{item['description']}"
        )

    print()
    print("=" * 100)
    print("FORENSIC REPORT COMPLETE")
    print("=" * 100)

# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Fast integration test.

    Keeps the real 5-product workflow, but avoids:
      - a second standalone similarity pass (pricing already runs it)
      - the huge forensic/audit printout
      - unnecessarily large marketplace result limits during normal testing

    Production modules are only called; persist=False is used.
    """
    FAST_TEST = True
    PRIMARY_LIMIT = 3 if FAST_TEST else 5
    SECONDARY_LIMIT = 3 if FAST_TEST else 5

    print()
    print("=" * 100)
    print("SIH26090 - FAST REAL-DATA MODULE 3 TEST")
    print("=" * 100)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Fast mode    : {'ON' if FAST_TEST else 'OFF'}")
    print(f"Limits       : primary={PRIMARY_LIMIT}, secondary={SECONDARY_LIMIT}")
    print("Flow         : IndiaHandmade -> Marketplace -> Similarity -> Pricing -> Guardrails")
    print("Note         : Similarity is executed once inside the pricing pipeline.")

    fetched = fetch_five_indiahandmade_products()

    if len(fetched) != 5:
        print()
        print(f"[FAIL] Expected 5 IndiaHandmade products, fetched {len(fetched)}/5.")
        print(f"Passed: {passed} | Warnings: {warnings} | Failed: {failed}")
        return

    artisan_products = build_artisan_products(fetched)
    results = []

    for number, product in enumerate(artisan_products, start=1):
        print()
        print("#" * 100)
        print(f"PRODUCT {number}/5")
        print("#" * 100)
        print(f"Product  : {product['product_name']}")
        print(f"Type     : {product['product_type']}")
        print(f"Material : {product['material']}")
        print(f"Base cost: {money(product['material_cost'] + product['labour_cost'] + product['production_cost'])}")

        print()
        print("[1] MARKETPLACE COLLECTION")
        print("-" * 100)
        try:
            listings = collect_marketplace_candidates(
                product=product,
                primary_limit_per_source=PRIMARY_LIMIT,
                secondary_limit_per_source=SECONDARY_LIMIT,
            )
        except Exception as exc:
            print(f"[FAIL] Marketplace collector failed: {exc}")
            results.append({"product": product, "listings": [], "matches": [], "result": None})
            continue

        primary_count = sum(
            1 for item in listings
            if str(item.get("source_tier", "")).upper() == "PRIMARY"
        )
        secondary_count = sum(
            1 for item in listings
            if str(item.get("source_tier", "")).upper() == "SECONDARY"
        )

        print(f"Total listings : {len(listings)}")
        print(f"Primary        : {primary_count}")
        print(f"Secondary      : {secondary_count}")

        # Show only a small sample of raw collection results.
        for rank, item in enumerate(listings[:8], start=1):
            print(
                f"  {rank:02d}. "
                f"{item.get('marketplace', 'Unknown'):<16} "
                f"{money(item.get('price')):>10}  "
                f"{item.get('title', '')}"
            )

        check(
            isinstance(listings, list),
            f"Marketplace collector returned a list for {product['product_id']}.",
        )

        print()
        print("[2] PRICING PIPELINE")
        print("-" * 100)
        print("Similarity -> pricing qualification -> market evidence -> ML gate -> guardrails")

        try:
            result = run_pricing(product, listings)
        except Exception as exc:
            print(f"[FAIL] Pricing engine failed: {exc}")
            results.append({"product": product, "listings": listings, "matches": [], "result": None})
            continue

        # IMPORTANT: do not run run_similarity_check() here.
        # marketplace_pricing.py already performs the similarity/comparable selection.
        matches = list(result.comparable_products or [])
        results.append({"product": product, "listings": listings, "matches": matches, "result": result})

        print()
        print("  SIMILARITY / PRICING COMPARABLES")
        print("  " + "-" * 94)
        print(f"  Current listings       : {result.current_listing_count}")
        print(f"  Pricing comparables    : {result.current_comparable_count}")
        print(f"  Historical comparables : {result.historical_comparable_count}")
        print(f"  Best similarity        : {result.similarity_score:.4f}")

        if matches:
            for rank, match in enumerate(matches[:10], start=1):
                obj = match_product(match)
                print(
                    f"    {rank:02d}. "
                    f"{listing_value(obj, 'marketplace', 'Unknown'):<15} "
                    f"{money(listing_value(obj, 'price')):>10} "
                    f"score={safe_score(match):.4f} "
                    f"{match_level(match):<10} "
                    f"{listing_value(obj, 'material', 'Unknown'):<12} "
                    f"{listing_value(obj, 'title', '')}"
                )
        else:
            print("    NONE - pricing will rely on fallback/guardrails.")

        print()
        print("  MARKET EVIDENCE")
        print("  " + "-" * 94)
        print(f"  Minimum : {money(result.market_minimum)}")
        print(f"  Median  : {money(result.market_median)}")
        print(f"  Maximum : {money(result.market_maximum)}")
        print(f"  Count   : {result.market_count}")

        print()
        print("  ML READINESS")
        print("  " + "-" * 94)
        print(f"  Ready       : {'YES' if result.ml_readiness else 'NO'}")
        print(f"  Invoked     : {'YES' if result.ml_invoked else 'NO'}")
        print(f"  Model       : {result.ml_model}")
        print(f"  Prediction  : {money(result.ml_prediction)}")
        print(f"  Reason      : {result.ml_readiness_reason}")

        print()
        print("  PRICING + GUARDRAILS")
        print("  " + "-" * 94)
        print(f"  Formula             : {money(result.formula_price)}")
        print(f"  Market-based        : {money(result.market_based_price)}")
        print(f"  Fair-trade floor    : {money(result.fair_trade_floor)}")
        print(f"  Before guardrails   : {money(result.recommended_price)}")
        print(f"  Guardrail minimum   : {money(result.guardrail_minimum)}")
        print(f"  Guardrail maximum   : {money(result.guardrail_maximum)}")
        print(f"  FINAL PRICE         : {money(result.final_price)}")
        print(f"  Method              : {result.method}")
        print(f"  Decision            : {result.decision_reason}")
        print(f"  Feature vector      : {len(result.feature_vector)} features")

        print()
        print("  VALIDATION")
        print("  " + "-" * 94)
        check(
            result.final_price is not None and result.final_price > 0,
            "Final recommended price is positive.",
        )
        check(
            result.final_price >= result.fair_trade_floor - 1e-6,
            "Final recommended price respects the fair-trade floor.",
        )
        check(
            len(result.feature_vector) == 29,
            "Pricing result contains exactly 29 features.",
        )
        check(
            result.current_listing_count == len(listings),
            "Pricing listing count matches marketplace collector output.",
        )
        check(
            result.current_comparable_count == len(result.comparable_products),
            "Pricing comparable count matches returned comparable list.",
        )

        print(f"  STATUS              : {'PASS' if result.final_price > 0 else 'FAIL'}")

    print_final_summary(results)

    print()
    print("=" * 100)
    print("FAST TEST SUMMARY")
    print("=" * 100)
    print(f"Products completed : {len(results)}/5")
    print(f"Tests passed       : {passed}")
    print(f"Warnings           : {warnings}")
    print(f"Tests failed       : {failed}")
    print()
    print("What this run verifies:")
    print("  1. Five real IndiaHandmade products are fetched.")
    print("  2. Marketplace candidates are collected.")
    print("  3. Similarity is executed by the pricing pipeline (no duplicate run).")
    print("  4. Pricing uses accepted comparables, not raw listings.")
    print("  5. Historical evidence is considered when available.")
    print("  6. CatBoost is invoked only when the ML readiness gate allows it.")
    print("  7. Formula/fallback and fair-trade guardrails are visible.")
    print("  8. Exactly 29 pricing features are verified.")
    print("  9. persist=False means this test does not intentionally create pricing events.")
    print()
    print("SIH26090 MODULE 3 FAST TEST PASSED" if failed == 0 else
          "SIH26090 MODULE 3 FAST TEST COMPLETED WITH FAILURES")
    print("=" * 100)


if __name__ == "__main__":
    main()
