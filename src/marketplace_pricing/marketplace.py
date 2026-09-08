"""
src/marketplace.py

Marketplace Analysis Layer
==========================

Responsibilities:
    - Store marketplace listings/snapshots
    - Retrieve historical marketplace data
    - Remove duplicate listings
    - Calculate current market statistics
    - Select comparable marketplace products
    - Combine historical and current market evidence

This module does NOT:
    - calculate the final price
    - train CatBoost
    - decide artisan approval
    - update product prices
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from database import (
    save_marketplace_snapshot,
    get_marketplace_snapshots,
)

from similarity import (
    SimilarityEngine,
    ProductAttributes,
    SimilarProduct,
)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class MarketplaceListing:
    """
    Normalized marketplace listing.
    """

    marketplace: str
    listing_id: Optional[str]
    title: str
    price: float

    url: Optional[str] = None

    currency: str = "INR"

    rating: Optional[float] = None
    review_count: Optional[int] = None
    seller_name: Optional[str] = None

    handmade_evidence: Optional[str] = None

    product_type: Optional[str] = None
    material: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    dimensions: Optional[str] = None
    features: Optional[List[str]] = None

    similarity_score: Optional[float] = None

    snapshot_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MarketStatistics:
    """
    Statistics calculated from comparable marketplace prices.
    """

    minimum: Optional[float]
    median: Optional[float]
    maximum: Optional[float]
    count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "minimum": self.minimum,
            "median": self.median,
            "maximum": self.maximum,
            "count": self.count,
        }


@dataclass
class MarketplaceAnalysis:
    """
    Complete marketplace analysis result.
    """

    current_listings: List[MarketplaceListing]

    historical_listings: List[MarketplaceListing]

    comparables: List[SimilarProduct]

    current_statistics: MarketStatistics

    historical_statistics: MarketStatistics

    combined_statistics: MarketStatistics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_listings": [
                item.to_dict()
                for item in self.current_listings
            ],
            "historical_listings": [
                item.to_dict()
                for item in self.historical_listings
            ],
            "comparables": [
                item.to_dict()
                for item in self.comparables
            ],
            "current_statistics": (
                self.current_statistics.to_dict()
            ),
            "historical_statistics": (
                self.historical_statistics.to_dict()
            ),
            "combined_statistics": (
                self.combined_statistics.to_dict()
            ),
        }


# ============================================================
# MARKETPLACE SERVICE
# ============================================================

class MarketplaceService:
    """
    Application-side marketplace service.

    Current marketplace analysis is ALWAYS performed.

    Historical data is supplementary evidence and does not
    replace fresh marketplace analysis.
    """

    def __init__(
        self,
        similarity_engine: Optional[SimilarityEngine] = None,
    ) -> None:

        self.similarity_engine = (
            similarity_engine
            if similarity_engine is not None
            else SimilarityEngine()
        )

    # ========================================================
    # GENERAL HELPERS
    # ========================================================

    @staticmethod
    def _get(
        obj: Any,
        key: str,
        default: Any = None,
    ) -> Any:

        if obj is None:
            return default

        if isinstance(obj, Mapping):
            return obj.get(key, default)

        return getattr(obj, key, default)

    @staticmethod
    def _safe_float(
        value: Any,
        default: Optional[float] = None,
    ) -> Optional[float]:

        try:
            if value is None:
                return default

            return float(value)

        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(
        value: Any,
        default: Optional[int] = None,
    ) -> Optional[int]:

        try:
            if value is None:
                return default

            return int(value)

        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_features(
        features: Any,
    ) -> List[str]:

        if features is None:
            return []

        if isinstance(features, str):
            return [
                item.strip()
                for item in features.split(",")
                if item.strip()
            ]

        if isinstance(features, Iterable):
            return [
                str(item).strip()
                for item in features
                if str(item).strip()
            ]

        return []

    # ========================================================
    # LISTING NORMALIZATION
    # ========================================================

    def normalize_listing(
        self,
        listing: Any,
    ) -> MarketplaceListing:

        return MarketplaceListing(
            marketplace=str(
                self._get(
                    listing,
                    "marketplace",
                    "",
                )
            ),

            listing_id=self._get(
                listing,
                "listing_id",
                None,
            ),

            title=str(
                self._get(
                    listing,
                    "title",
                    self._get(
                        listing,
                        "product_name",
                        "",
                    ),
                )
            ),

            price=self._safe_float(
                self._get(
                    listing,
                    "price",
                    self._get(
                        listing,
                        "retail_price_inr",
                        0,
                    ),
                ),
                default=0.0,
            ) or 0.0,

            url=self._get(
                listing,
                "url",
                None,
            ),

            currency=str(
                self._get(
                    listing,
                    "currency",
                    "INR",
                )
            ),

            rating=self._safe_float(
                self._get(
                    listing,
                    "rating",
                    None,
                )
            ),

            review_count=self._safe_int(
                self._get(
                    listing,
                    "review_count",
                    None,
                )
            ),

            seller_name=self._get(
                listing,
                "seller_name",
                None,
            ),

            handmade_evidence=self._get(
                listing,
                "handmade_evidence",
                None,
            ),

            product_type=self._get(
                listing,
                "product_type",
                None,
            ),

            material=self._get(
                listing,
                "material",
                None,
            ),

            color=self._get(
                listing,
                "color",
                None,
            ),

            size=self._get(
                listing,
                "size",
                None,
            ),

            dimensions=self._get(
                listing,
                "dimensions",
                None,
            ),

            features=self._normalize_features(
                self._get(
                    listing,
                    "features",
                    [],
                )
            ),

            similarity_score=self._safe_float(
                self._get(
                    listing,
                    "similarity_score",
                    None,
                )
            ),

            snapshot_date=self._get(
                listing,
                "snapshot_date",
                None,
            ),
        )

    # ========================================================
    # SAVE MARKETPLACE DATA
    # ========================================================

    def save_listings(
        self,
        product_id: str,
        listings: Iterable[Any],
        snapshot_date: Optional[str] = None,
    ) -> int:
        """
        Save marketplace listings as database snapshots.

        Returns:
            Number of listings successfully saved.
        """

        if snapshot_date is None:
            snapshot_date = date.today().isoformat()

        saved_count = 0

        for raw_listing in listings:

            listing = self.normalize_listing(
                raw_listing
            )

            if listing.price <= 0:
                continue

            data = {
                "product_id": product_id,

                "marketplace": listing.marketplace,

                "listing_id": listing.listing_id,

                "title": listing.title,

                "url": listing.url,

                "price": listing.price,

                "currency": listing.currency,

                "rating": listing.rating,

                "review_count": listing.review_count,

                "seller_name": listing.seller_name,

                "handmade_evidence": (
                    listing.handmade_evidence
                ),

                "product_type": listing.product_type,

                "material": listing.material,

                "color": listing.color,

                "size": listing.size,

                "dimensions": listing.dimensions,

                "features": listing.features,

                "similarity_score": (
                    listing.similarity_score
                ),

                "snapshot_date": (
                    listing.snapshot_date
                    or snapshot_date
                ),
            }

            try:
                save_marketplace_snapshot(
                    data
                )

                saved_count += 1

            except Exception as exc:

                print(
                    "[MarketplaceService] "
                    f"Could not save listing: {exc}"
                )

        return saved_count

    # ========================================================
    # HISTORICAL SNAPSHOTS
    # ========================================================

    def get_historical_snapshots(
        self,
        product_id: str,
        limit: int = 100,
    ) -> List[MarketplaceListing]:
        """
        Retrieve previously stored marketplace snapshots.
        """

        rows = get_marketplace_snapshots(
            product_id=product_id,
            limit=limit,
        )

        results = []

        for row in rows:

            try:
                results.append(
                    self.normalize_listing(row)
                )

            except Exception:
                continue

        return results

    # ========================================================
    # ROW CONVERSION
    # ========================================================

    def row_to_listing(
        self,
        row: Any,
    ) -> MarketplaceListing:

        return self.normalize_listing(row)

    # ========================================================
    # PRICE EXTRACTION
    # ========================================================

    def extract_prices(
        self,
        listings: Iterable[Any],
    ) -> List[float]:

        prices = []

        for listing in listings:

            price = self._safe_float(
                self._get(
                    listing,
                    "price",
                    None,
                )
            )

            if price is None:
                continue

            if price <= 0:
                continue

            prices.append(price)

        return prices

    # ========================================================
    # MARKET STATISTICS
    # ========================================================

    def calculate_statistics(
        self,
        listings: Iterable[Any],
    ) -> MarketStatistics:

        prices = self.extract_prices(
            listings
        )

        if not prices:

            return MarketStatistics(
                minimum=None,
                median=None,
                maximum=None,
                count=0,
            )

        return MarketStatistics(
            minimum=round(
                min(prices),
                2,
            ),

            median=round(
                median(prices),
                2,
            ),

            maximum=round(
                max(prices),
                2,
            ),

            count=len(prices),
        )

    # ========================================================
    # DUPLICATE REMOVAL
    # ========================================================

    def remove_duplicates(
        self,
        listings: Iterable[Any],
    ) -> List[MarketplaceListing]:
        """
        Remove duplicate marketplace listings.

        Priority:
            1. marketplace + listing_id
            2. marketplace + title + price
        """

        unique = []

        seen_ids = set()
        seen_content = set()

        for raw_listing in listings:

            listing = self.normalize_listing(
                raw_listing
            )

            if listing.price <= 0:
                continue

            if listing.listing_id:

                identifier = (
                    listing.marketplace.lower(),
                    str(listing.listing_id).lower(),
                )

                if identifier in seen_ids:
                    continue

                seen_ids.add(identifier)

            else:

                content_key = (
                    listing.marketplace.lower(),
                    listing.title.lower().strip(),
                    round(listing.price, 2),
                )

                if content_key in seen_content:
                    continue

                seen_content.add(content_key)

            unique.append(listing)

        return unique

    # ========================================================
    # TOP COMPARABLES
    # ========================================================

    def select_top_comparables(
        self,
        product: ProductAttributes,
        listings: Iterable[Any],
        top_k: int = 10,
    ) -> List[SimilarProduct]:
        """
        Find the strongest marketplace comparables.

        SimilarityEngine performs the actual multi-factor
        matching.
        """

        normalized = [
            self.normalize_listing(item)
            for item in listings
        ]

        raw_listings = [
            item.to_dict()
            for item in normalized
        ]

        matches = (
            self.similarity_engine
            .match_marketplace_listings(
                product=product,
                listings=raw_listings,
                top_k=top_k,
            )
        )

        return matches

    # ========================================================
    # CURRENT MARKET ANALYSIS
    # ========================================================

    def analyze_current_market(
        self,
        product: ProductAttributes,
        listings: Iterable[Any],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Analyze the CURRENT marketplace.

        This is always performed even if historical data exists.
        """

        unique_listings = (
            self.remove_duplicates(
                listings
            )
        )

        comparables = (
            self.select_top_comparables(
                product=product,
                listings=unique_listings,
                top_k=top_k,
            )
        )

        comparable_listings = []

        for match in comparables:

            comparable_listings.append(
                MarketplaceListing(
                    marketplace=match.marketplace,
                    listing_id=match.listing_id,
                    title=match.title,
                    price=match.price,
                    url=match.url,
                    product_type=match.product_type,
                    material=match.material,
                    color=match.color,
                    size=match.size,
                    dimensions=match.dimensions,
                    features=match.features or [],
                    similarity_score=(
                        match.similarity_score
                    ),
                )
            )

        statistics = self.calculate_statistics(
            comparable_listings
        )

        return {
            "listings": unique_listings,

            "comparables": comparables,

            "comparable_listings": (
                comparable_listings
            ),

            "statistics": statistics,

            "market_minimum": statistics.minimum,

            "market_median": statistics.median,

            "market_maximum": statistics.maximum,

            "market_count": statistics.count,
        }

    # ========================================================
    # COMBINE MARKET EVIDENCE
    # ========================================================

    def combine_market_evidence(
        self,
        current_listings: Iterable[Any],
        historical_listings: Iterable[Any],
    ) -> MarketStatistics:
        """
        Combine current and historical marketplace prices.

        Historical data supplements current evidence.
        """

        current = self.remove_duplicates(
            current_listings
        )

        historical = self.remove_duplicates(
            historical_listings
        )

        prices = (
            self.extract_prices(current)
            +
            self.extract_prices(historical)
        )

        if not prices:

            return MarketStatistics(
                minimum=None,
                median=None,
                maximum=None,
                count=0,
            )

        return MarketStatistics(
            minimum=round(
                min(prices),
                2,
            ),

            median=round(
                median(prices),
                2,
            ),

            maximum=round(
                max(prices),
                2,
            ),

            count=len(prices),
        )

    # ========================================================
    # COMPLETE MARKET ANALYSIS
    # ========================================================

    def analyze_market(
        self,
        product: ProductAttributes,
        current_listings: Iterable[Any],
        historical_listings: Optional[
            Iterable[Any]
        ] = None,
        top_k: int = 10,
    ) -> MarketplaceAnalysis:
        """
        Complete marketplace analysis.

        Flow:

            Current marketplace
                    ↓
              deduplication
                    ↓
              similarity matching
                    ↓
             current statistics
                    ↓
            historical evidence
                    ↓
          combined market evidence
        """

        current = self.remove_duplicates(
            current_listings
        )

        historical = (
            self.remove_duplicates(
                historical_listings
            )
            if historical_listings is not None
            else []
        )

        comparables = (
            self.select_top_comparables(
                product=product,
                listings=current,
                top_k=top_k,
            )
        )

        current_statistics = (
            self.calculate_statistics(
                comparables
            )
        )

        historical_statistics = (
            self.calculate_statistics(
                historical
            )
        )

        combined_statistics = (
            self.combine_market_evidence(
                current_listings=current,
                historical_listings=historical,
            )
        )

        return MarketplaceAnalysis(
            current_listings=current,

            historical_listings=historical,

            comparables=comparables,

            current_statistics=(
                current_statistics
            ),

            historical_statistics=(
                historical_statistics
            ),

            combined_statistics=(
                combined_statistics
            ),
        )

    # ========================================================
    # CURRENT MARKET STATISTICS HELPER
    # ========================================================

    def get_current_market_statistics(
        self,
        product: ProductAttributes,
        listings: Iterable[Any],
        top_k: int = 10,
    ) -> MarketStatistics:

        analysis = (
            self.analyze_current_market(
                product=product,
                listings=list(listings),
                top_k=top_k,
            )
        )

        return analysis["statistics"]


# ============================================================
# DEFAULT SERVICE
# ============================================================

marketplace_service = MarketplaceService()


# ============================================================
# APPLICATION HELPER
# ============================================================

def analyze_market(
    product: ProductAttributes,
    current_listings: Iterable[Any],
    historical_listings: Optional[
        Iterable[Any]
    ] = None,
    top_k: int = 10,
) -> MarketplaceAnalysis:

    return marketplace_service.analyze_market(
        product=product,
        current_listings=current_listings,
        historical_listings=historical_listings,
        top_k=top_k,
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SIH26090 MARKETPLACE SERVICE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # ARTISAN PRODUCT
    # --------------------------------------------------------

    product = ProductAttributes(
        product_id="ART-000001",
        product_name="Handmade Bamboo Basket",
        product_type="Basket",
        material="Bamboo",
        color="Natural Brown",
        size="Medium",
        dimensions_length=30,
        dimensions_width=20,
        dimensions_height=10,
        features=[
            "Handwoven",
            "Traditional weaving",
            "Natural bamboo",
            "Round shape",
            "Handcrafted",
        ],
        description=(
            "Handmade traditional bamboo basket."
        ),
    )

    # --------------------------------------------------------
    # CURRENT MARKETPLACE DATA
    # --------------------------------------------------------

    listings = [
        {
            "listing_id": "AMZ-001",
            "marketplace": "Amazon",
            "title": (
                "SAI BALAJI Natural Bamboo "
                "Medium Round Basket"
            ),
            "price": 699,
            "url": "",
            "product_type": "Basket",
            "material": "Bamboo",
            "color": "Natural Brown",
            "size": "Medium",
            "dimensions": "30x20 cm",
            "features": [
                "Handwoven",
                "Natural bamboo",
                "Round shape",
            ],
        },
        {
            "listing_id": "AMZ-002",
            "marketplace": "Amazon",
            "title": (
                "Handwoven Bamboo Basket"
            ),
            "price": 159,
            "url": "",
            "product_type": "Basket",
            "material": "Bamboo",
            "color": "Natural Brown",
            "size": "Medium",
            "dimensions": "30x20 cm",
            "features": [
                "Handwoven",
                "Natural bamboo",
            ],
        },
        {
            "listing_id": "AMZ-003",
            "marketplace": "Amazon",
            "title": (
                "NISKANET Natural Bamboo "
                "Medium Round Baskets"
            ),
            "price": 599,
            "url": "",
            "product_type": "Basket",
            "material": "Bamboo",
            "color": "Natural Brown",
            "size": "Medium",
            "features": [
                "Handwoven",
                "Round shape",
            ],
        },
        {
            "listing_id": "AMZ-004",
            "marketplace": "Amazon",
            "title": (
                "SAI PRASEEDA Natural Bamboo "
                "Medium Round Baskets"
            ),
            "price": 699,
            "url": "",
            "product_type": "Basket",
            "material": "Bamboo",
            "color": "Natural Brown",
            "size": "Medium",
            "features": [
                "Handwoven",
                "Natural bamboo",
                "Round shape",
            ],
        },
        {
            "listing_id": "AMZ-005",
            "marketplace": "Amazon",
            "title": (
                "TAVASYA Natural Bamboo "
                "Medium Round Baskets"
            ),
            "price": 599,
            "url": "",
            "product_type": "Basket",
            "material": "Bamboo",
            "color": "Natural Brown",
            "size": "Medium",
            "features": [
                "Handwoven",
                "Round shape",
            ],
        },
        {
            "listing_id": "AMZ-006",
            "marketplace": "Amazon",
            "title": (
                "Wicker Storage Basket"
            ),
            "price": 2999,
            "url": "",
            "product_type": "Basket",
            "material": "Wicker",
            "size": "Large",
            "features": [
                "Handcrafted",
            ],
        },
    ]

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    service = MarketplaceService()

    analysis = service.analyze_market(
        product=product,
        current_listings=listings,
        historical_listings=[],
        top_k=10,
    )

    current_stats = (
        analysis.current_statistics
    )

    combined_stats = (
        analysis.combined_statistics
    )

    print()
    print("Current listings:",
          len(analysis.current_listings))

    print(
        "Comparable products:",
        len(analysis.comparables),
    )

    print()
    print(
        "Current minimum:",
        current_stats.minimum,
    )

    print(
        "Current median:",
        current_stats.median,
    )

    print(
        "Current maximum:",
        current_stats.maximum,
    )

    print(
        "Current count:",
        current_stats.count,
    )

    print()
    print(
        "Combined minimum:",
        combined_stats.minimum,
    )

    print(
        "Combined median:",
        combined_stats.median,
    )

    print(
        "Combined maximum:",
        combined_stats.maximum,
    )

    print(
        "Combined count:",
        combined_stats.count,
    )

    # --------------------------------------------------------
    # ASSERTIONS
    # --------------------------------------------------------

    assert len(
        analysis.current_listings
    ) == 6

    assert len(
        analysis.comparables
    ) > 0

    assert (
        current_stats.count
        > 0
    )

    assert (
        current_stats.minimum
        is not None
    )

    assert (
        current_stats.median
        is not None
    )

    assert (
        current_stats.maximum
        is not None
    )

    assert (
        current_stats.minimum
        <= current_stats.median
        <= current_stats.maximum
    )

    assert (
        combined_stats.count
        >= current_stats.count
    )

    print()
    print("=" * 70)
    print("MARKETPLACE SERVICE TEST PASSED")
    print("=" * 70)