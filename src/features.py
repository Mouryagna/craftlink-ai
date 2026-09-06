# src/features.py

"""
Feature Engineering Module
==========================

Purpose:
    Convert product, marketplace, similarity, demand, supply, and seasonal
    information into a stable numerical feature set for the pricing model.

Architecture:

    Product
        |
    Similarity Engine
        |
    Marketplace Analysis
        |
    Feature Engineering
        |
    CatBoost Regressor

Important:
    - This module does NOT calculate the final price.
    - This module does NOT decide ML readiness.
    - This module does NOT generate target labels.
    - actual_selling_price must come from real selling outcomes.
    - Base price / market statistics are model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import math
import re


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_SEASONAL_INDEX = 1.0

DEFAULT_DEMAND_VALUES = {
    "views": 0.0,
    "clicks": 0.0,
    "wishlists": 0.0,
    "add_to_cart": 0.0,
    "enquiries": 0.0,
    "orders": 0.0,
}

DEFAULT_SUPPLY_VALUES = {
    "inventory": 0.0,
}

NUMERIC_FEATURES = [
    # Product / cost
    "material_cost",
    "labour_cost",
    "production_cost",
    "base_cost",
    "time_worked_hours",

    # Pricing anchors
    "base_price",
    "market_minimum",
    "market_median",
    "market_maximum",
    "market_count",

    # Similarity
    "similarity_score",
    "primary_match_count",
    "secondary_match_count",

    # Similarity component scores
    "similarity_product_type",
    "similarity_material",
    "similarity_size",
    "similarity_dimensions",
    "similarity_features",
    "similarity_color",
    "similarity_text",

    # Demand
    "views",
    "clicks",
    "wishlists",
    "add_to_cart",
    "enquiries",
    "orders",
    "conversion_rate",

    # Supply
    "inventory",

    # Other
    "seasonal_index",
]


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class DemandSignals:
    """Current product demand signals."""

    views: float = 0.0
    clicks: float = 0.0
    wishlists: float = 0.0
    add_to_cart: float = 0.0
    enquiries: float = 0.0
    orders: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class SupplySignals:
    """Current product supply signals."""

    inventory: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class SimilarityFeatures:
    """Aggregated similarity information."""

    similarity_score: float = 0.0

    primary_match_count: int = 0
    secondary_match_count: int = 0

    product_type_score: float = 0.0
    material_score: float = 0.0
    size_score: float = 0.0
    dimensions_score: float = 0.0
    features_score: float = 0.0
    color_score: float = 0.0
    text_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "similarity_score": self.similarity_score,
            "primary_match_count": self.primary_match_count,
            "secondary_match_count": self.secondary_match_count,
            "similarity_product_type": self.product_type_score,
            "similarity_material": self.material_score,
            "similarity_size": self.size_score,
            "similarity_dimensions": self.dimensions_score,
            "similarity_features": self.features_score,
            "similarity_color": self.color_score,
            "similarity_text": self.text_score,
        }


@dataclass
class MarketFeatures:
    """Current marketplace statistics."""

    minimum: float = 0.0
    median: float = 0.0
    maximum: float = 0.0
    count: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "market_minimum": self.minimum,
            "market_median": self.median,
            "market_maximum": self.maximum,
            "market_count": self.count,
        }


@dataclass
class ProductFeatures:
    """Production-related product information."""

    material_cost: float = 0.0
    labour_cost: float = 0.0
    production_cost: float = 0.0
    time_worked_hours: float = 0.0
    base_cost: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# ============================================================
# FEATURE ENGINEERING CLASS
# ============================================================

class PricingFeatureEngineer:
    """
    Builds the feature vector used by the pricing model.

    This class is deliberately independent from CatBoost so that:
        1. Training and prediction use identical transformations.
        2. Feature engineering can be tested separately.
        3. The application can call this module without importing ML code.
    """

    def __init__(
        self,
        seasonal_index: float = DEFAULT_SEASONAL_INDEX,
    ) -> None:
        self.seasonal_index = self._safe_non_negative(
            seasonal_index,
            default=DEFAULT_SEASONAL_INDEX,
        )

    # ========================================================
    # BASIC HELPERS
    # ========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:
        """
        Convert a value to float safely.
        """

        if value is None:
            return default

        if isinstance(value, bool):
            return float(value)

        try:
            number = float(value)

            if not math.isfinite(number):
                return default

            return number

        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_non_negative(
        value: Any,
        default: float = 0.0,
    ) -> float:
        number = PricingFeatureEngineer._safe_float(
            value,
            default=default,
        )

        return max(0.0, number)

    @staticmethod
    def _get(
        data: Any,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Read a value from either:
            - dictionary
            - dataclass
            - normal object
        """

        if data is None:
            return default

        if isinstance(data, Mapping):
            return data.get(key, default)

        return getattr(data, key, default)

    # ========================================================
    # PRODUCT FEATURES
    # ========================================================

    def extract_product_features(
        self,
        product: Any,
    ) -> ProductFeatures:
        """
        Extract production/cost information.

        base_cost:
            material + labour + production cost

        This does NOT include the selling price.
        """

        material_cost = self._safe_non_negative(
            self._get(product, "material_cost", 0.0)
        )

        labour_cost = self._safe_non_negative(
            self._get(product, "labour_cost", 0.0)
        )

        production_cost = self._safe_non_negative(
            self._get(product, "production_cost", 0.0)
        )

        time_worked_hours = self._safe_non_negative(
            self._get(product, "time_worked_hours", 0.0)
        )

        base_cost = (
            material_cost
            + labour_cost
            + production_cost
        )

        return ProductFeatures(
            material_cost=material_cost,
            labour_cost=labour_cost,
            production_cost=production_cost,
            time_worked_hours=time_worked_hours,
            base_cost=base_cost,
        )

    # ========================================================
    # MARKET FEATURES
    # ========================================================

    def extract_market_features(
        self,
        market: Any = None,
        prices: Optional[Sequence[float]] = None,
    ) -> MarketFeatures:
        """
        Extract current marketplace statistics.

        Can accept:
            MarketStats object
            dictionary
            list of prices
        """

        if prices is None and market is not None:
            prices = self._get(market, "prices", None)

        if prices is not None:
            clean_prices = []

            for price in prices:
                value = self._safe_float(price, default=-1.0)

                if value >= 0:
                    clean_prices.append(value)

            if clean_prices:
                clean_prices.sort()

                count = len(clean_prices)

                minimum = clean_prices[0]
                maximum = clean_prices[-1]

                middle = count // 2

                if count % 2 == 0:
                    median = (
                        clean_prices[middle - 1]
                        + clean_prices[middle]
                    ) / 2.0
                else:
                    median = clean_prices[middle]

                return MarketFeatures(
                    minimum=minimum,
                    median=median,
                    maximum=maximum,
                    count=count,
                )

        minimum = self._safe_non_negative(
            self._get(market, "minimum", 0.0)
        )

        median = self._safe_non_negative(
            self._get(market, "median", 0.0)
        )

        maximum = self._safe_non_negative(
            self._get(market, "maximum", 0.0)
        )

        count = int(
            self._safe_non_negative(
                self._get(market, "count", 0)
            )
        )

        return MarketFeatures(
            minimum=minimum,
            median=median,
            maximum=maximum,
            count=count,
        )

    # ========================================================
    # SIMILARITY FEATURES
    # ========================================================

    def extract_similarity_features(
        self,
        similarities: Optional[Iterable[Any]],
    ) -> SimilarityFeatures:
        """
        Aggregate the strongest comparable-product matches.

        Similarity engine already determines the individual match.
        This module only converts those results into model features.
        """

        if similarities is None:
            return SimilarityFeatures()

        items = list(similarities)

        if not items:
            return SimilarityFeatures()

        scores = []
        primary_count = 0
        secondary_count = 0

        component_values = {
            "product_type_score": [],
            "material_score": [],
            "size_score": [],
            "dimensions_score": [],
            "features_score": [],
            "color_score": [],
            "text_score": [],
        }

        for item in items:

            score = self._safe_float(
                self._get(item, "score", None),
                default=0.0,
            )

            if score > 0:
                scores.append(score)

            match_level = str(
                self._get(item, "match_level", "")
            ).upper()

            if match_level == "PRIMARY":
                primary_count += 1

            elif match_level == "SECONDARY":
                secondary_count += 1

            breakdown = self._get(
                item,
                "breakdown",
                None,
            )

            if breakdown is not None:

                field_mapping = {
                    "product_type_score": "product_type",
                    "material_score": "material",
                    "size_score": "size",
                    "dimensions_score": "dimensions",
                    "features_score": "features",
                    "color_score": "color",
                    "text_score": "text",
                }

                for output_key, input_key in field_mapping.items():

                    value = self._get(
                        breakdown,
                        input_key,
                        None,
                    )

                    if value is not None:
                        component_values[
                            output_key
                        ].append(
                            self._safe_float(value)
                        )

        def average(values: List[float]) -> float:
            if not values:
                return 0.0

            return sum(values) / len(values)

        return SimilarityFeatures(
            similarity_score=max(scores) if scores else 0.0,
            primary_match_count=primary_count,
            secondary_match_count=secondary_count,
            product_type_score=average(
                component_values["product_type_score"]
            ),
            material_score=average(
                component_values["material_score"]
            ),
            size_score=average(
                component_values["size_score"]
            ),
            dimensions_score=average(
                component_values["dimensions_score"]
            ),
            features_score=average(
                component_values["features_score"]
            ),
            color_score=average(
                component_values["color_score"]
            ),
            text_score=average(
                component_values["text_score"]
            ),
        )

    # ========================================================
    # DEMAND FEATURES
    # ========================================================

    def extract_demand_features(
        self,
        demand: Any = None,
    ) -> DemandSignals:
        """
        Extract current demand signals.

        Conversion rate:

            orders / views

        if views > 0.

        Otherwise conversion_rate = 0.
        """

        values = {}

        for field in DEFAULT_DEMAND_VALUES:
            values[field] = self._safe_non_negative(
                self._get(
                    demand,
                    field,
                    DEFAULT_DEMAND_VALUES[field],
                )
            )

        views = values["views"]
        orders = values["orders"]

        if views > 0:
            conversion_rate = orders / views
        else:
            conversion_rate = 0.0

        return DemandSignals(
            views=values["views"],
            clicks=values["clicks"],
            wishlists=values["wishlists"],
            add_to_cart=values["add_to_cart"],
            enquiries=values["enquiries"],
            orders=values["orders"],
        ), conversion_rate

    # ========================================================
    # SUPPLY FEATURES
    # ========================================================

    def extract_supply_features(
        self,
        supply: Any = None,
    ) -> SupplySignals:
        """
        Extract current supply/inventory information.
        """

        inventory = self._safe_non_negative(
            self._get(
                supply,
                "inventory",
                0.0,
            )
        )

        return SupplySignals(
            inventory=inventory
        )

    # ========================================================
    # BASE PRICE
    # ========================================================

    def extract_base_price(
        self,
        product: Any = None,
        base_price: Optional[float] = None,
    ) -> float:
        """
        Extract the current/base price anchor.

        The base price is an input to the model.

        It is NOT generated by this feature engineer.
        """

        if base_price is None:
            base_price = self._get(
                product,
                "base_price",
                0.0,
            )

        return self._safe_non_negative(
            base_price
        )

    # ========================================================
    # MAIN FEATURE VECTOR
    # ========================================================

    def build_features(
        self,
        product: Any,
        market: Any = None,
        similarities: Optional[Iterable[Any]] = None,
        demand: Any = None,
        supply: Any = None,
        base_price: Optional[float] = None,
        seasonal_index: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Build the complete numerical feature vector.

        This is the main function used by the application.

        Returns:
            Dict[str, float]
        """

        product_features = self.extract_product_features(
            product
        )

        market_features = self.extract_market_features(
            market
        )

        similarity_features = self.extract_similarity_features(
            similarities
        )

        demand_features, conversion_rate = (
            self.extract_demand_features(
                demand
            )
        )

        supply_features = self.extract_supply_features(
            supply
        )

        price = self.extract_base_price(
            product=product,
            base_price=base_price,
        )

        if seasonal_index is None:
            seasonal_index = self.seasonal_index

        seasonal_index = self._safe_non_negative(
            seasonal_index,
            default=DEFAULT_SEASONAL_INDEX,
        )

        features: Dict[str, float] = {}

        # ----------------------------------------------------
        # Product
        # ----------------------------------------------------

        features.update(
            product_features.to_dict()
        )

        # ----------------------------------------------------
        # Base price
        # ----------------------------------------------------

        features["base_price"] = price

        # ----------------------------------------------------
        # Market
        # ----------------------------------------------------

        features.update(
            market_features.to_dict()
        )

        # ----------------------------------------------------
        # Similarity
        # ----------------------------------------------------

        features.update(
            similarity_features.to_dict()
        )

        # ----------------------------------------------------
        # Demand
        # ----------------------------------------------------

        features.update(
            demand_features.to_dict()
        )

        features["conversion_rate"] = conversion_rate

        # ----------------------------------------------------
        # Supply
        # ----------------------------------------------------

        features.update(
            supply_features.to_dict()
        )

        # ----------------------------------------------------
        # Seasonal
        # ----------------------------------------------------

        features["seasonal_index"] = seasonal_index

        return self.sanitize_features(
            features
        )

    # ========================================================
    # SANITIZATION
    # ========================================================

    @staticmethod
    def sanitize_features(
        features: Mapping[str, Any],
    ) -> Dict[str, float]:
        """
        Ensure every feature is numeric and finite.

        Missing features are filled with 0.

        This keeps training and inference stable.
        """

        result = {}

        for feature_name in NUMERIC_FEATURES:

            value = features.get(
                feature_name,
                0.0,
            )

            try:
                value = float(value)

                if not math.isfinite(value):
                    value = 0.0

            except (TypeError, ValueError):
                value = 0.0

            result[feature_name] = value

        return result

    # ========================================================
    # MODEL COLUMN ORDER
    # ========================================================

    @staticmethod
    def feature_names() -> List[str]:
        """
        Return the exact feature order expected by the model.
        """

        return NUMERIC_FEATURES.copy()

    # ========================================================
    # VECTOR
    # ========================================================

    @classmethod
    def to_vector(
        cls,
        features: Mapping[str, Any],
    ) -> List[float]:
        """
        Convert feature dictionary into ordered numerical vector.
        """

        clean = cls.sanitize_features(
            features
        )

        return [
            clean[name]
            for name in NUMERIC_FEATURES
        ]

    # ========================================================
    # TRAINING DATA
    # ========================================================

    @classmethod
    def prepare_training_row(
        cls,
        features: Mapping[str, Any],
        actual_selling_price: Optional[float],
    ) -> Dict[str, float]:
        """
        Prepare one training row.

        IMPORTANT:
            actual_selling_price must be the REAL observed selling
            outcome.

        Never pass formula-generated recommended_price as the target.
        """

        if actual_selling_price is None:
            raise ValueError(
                "actual_selling_price is required for a training row."
            )

        target = float(
            actual_selling_price
        )

        if not math.isfinite(target):
            raise ValueError(
                "actual_selling_price must be finite."
            )

        if target < 0:
            raise ValueError(
                "actual_selling_price cannot be negative."
            )

        row = cls.sanitize_features(
            features
        )

        row["actual_selling_price"] = target

        return row

    # ========================================================
    # BATCH TRAINING DATA
    # ========================================================

    @classmethod
    def prepare_training_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
    ) -> List[Dict[str, float]]:
        """
        Prepare multiple historical rows.

        Rows without actual selling outcomes are skipped.
        """

        result = []

        for row in rows:

            actual_price = row.get(
                "actual_selling_price"
            )

            if actual_price is None:
                continue

            feature_values = {
                key: row.get(key, 0.0)
                for key in NUMERIC_FEATURES
            }

            result.append(
                cls.prepare_training_row(
                    feature_values,
                    actual_price,
                )
            )

        return result


# ============================================================
# DEFAULT INSTANCE
# ============================================================

feature_engineer = PricingFeatureEngineer()


# ============================================================
# SIMPLE APPLICATION HELPER
# ============================================================

def build_pricing_features(
    product: Any,
    market: Any = None,
    similarities: Optional[Iterable[Any]] = None,
    demand: Any = None,
    supply: Any = None,
    base_price: Optional[float] = None,
    seasonal_index: Optional[float] = None,
) -> Dict[str, float]:
    """
    Application-level helper.

    Example:

        features = build_pricing_features(
            product=product,
            market=market,
            similarities=similarities,
            demand=demand,
            supply=supply,
            base_price=1000,
            seasonal_index=1.1,
        )
    """

    return feature_engineer.build_features(
        product=product,
        market=market,
        similarities=similarities,
        demand=demand,
        supply=supply,
        base_price=base_price,
        seasonal_index=seasonal_index,
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PRICING FEATURE ENGINEERING TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Example product
    # --------------------------------------------------------

    product = {
        "product_id": "ART-000001",
        "product_name": "Handmade Bamboo Basket",
        "product_type": "Basket",
        "material": "Bamboo",
        "color": "Natural Brown",
        "size": "Medium",
        "dimensions_length": 30,
        "dimensions_width": 20,
        "time_worked_hours": 6,

        "material_cost": 350,
        "labour_cost": 250,
        "production_cost": 100,
    }

    # --------------------------------------------------------
    # Current marketplace prices
    # --------------------------------------------------------

    market_prices = [
        699,
        599,
        159,
        699,
        599,
        1199,
    ]

    # --------------------------------------------------------
    # Similarity results
    # --------------------------------------------------------

    similarities = [
        {
            "score": 0.7521,
            "match_level": "PRIMARY",
            "breakdown": {
                "product_type": 1.0,
                "material": 1.0,
                "size": 1.0,
                "dimensions": 0.0,
                "features": 0.8,
                "color": 1.0,
                "text": 0.21,
            },
        },
        {
            "score": 0.6975,
            "match_level": "SECONDARY",
            "breakdown": {
                "product_type": 1.0,
                "material": 1.0,
                "size": 1.0,
                "dimensions": 0.0,
                "features": 0.4,
                "color": 1.0,
                "text": 0.25,
            },
        },
        {
            "score": 0.6600,
            "match_level": "SECONDARY",
            "breakdown": {
                "product_type": 1.0,
                "material": 1.0,
                "size": 1.0,
                "dimensions": 0.0,
                "features": 0.2,
                "color": 1.0,
                "text": 0.20,
            },
        },
    ]

    # --------------------------------------------------------
    # Demand
    # --------------------------------------------------------

    demand = {
        "views": 1000,
        "clicks": 120,
        "wishlists": 25,
        "add_to_cart": 15,
        "enquiries": 8,
        "orders": 5,
    }

    # --------------------------------------------------------
    # Supply
    # --------------------------------------------------------

    supply = {
        "inventory": 12,
    }

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    engineer = PricingFeatureEngineer(
        seasonal_index=1.1
    )

    market = engineer.extract_market_features(
        prices=market_prices
    )

    features = engineer.build_features(
        product=product,
        market=market,
        similarities=similarities,
        demand=demand,
        supply=supply,
        base_price=840,
        seasonal_index=1.1,
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print("\nGenerated features:\n")

    for name in engineer.feature_names():
        print(
            f"{name:<30} : "
            f"{features[name]:.6f}"
        )

    print("\n" + "-" * 70)

    vector = engineer.to_vector(
        features
    )

    print(
        f"Feature count : {len(vector)}"
    )

    print(
        f"Feature names : {len(engineer.feature_names())}"
    )

    assert len(vector) == len(
        engineer.feature_names()
    )

    assert features["base_cost"] == 700.0

    assert features["market_minimum"] == 159.0

    assert features["market_maximum"] == 1199.0

    assert features["primary_match_count"] == 1

    assert features["secondary_match_count"] == 2

    assert features["conversion_rate"] == 0.005

    # --------------------------------------------------------
    # Training-row test
    # --------------------------------------------------------

    training_row = engineer.prepare_training_row(
        features=features,
        actual_selling_price=900,
    )

    assert (
        training_row["actual_selling_price"]
        == 900.0
    )

    print(
        "\nTraining row target : "
        f"₹{training_row['actual_selling_price']:.2f}"
    )

    print("\n" + "=" * 70)
    print("PRICING FEATURE ENGINEERING TEST PASSED")
    print("=" * 70)