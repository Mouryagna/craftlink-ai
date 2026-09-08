"""
src/marketplace_pricing.py

SIH26090 - Module 3: Smart Pricing Pipeline

Final workflow
--------------

ARTISAN PRODUCT
      |
      v
Production cost + base price
      |
      +------------------------------+
      |                              |
      v                              v
Historical DB search          FRESH MARKETPLACE DATA
      |                              |
      +---------------+--------------+
                      v
              Multi-factor similarity
                      |
                      v
             Comparable products
                      |
                      v
        Current + historical evidence
                      |
                      v
             29 pricing features
                      |
                      v
              ML readiness check
                 /           \\
             READY          NOT READY
                |               |
             CatBoost      Market / formula
                |               |
                +-------+-------+
                        v
                 Pricing engine
                        |
                        v
                    Guardrails
                        |
                        v
                Final recommendation
                        |
                        v
                 Artisan approval
                        |
                        v
                       SELL
                        |
                        v
             Actual selling outcome
                        |
                        v
                  pricing_events
                        |
                        v
          End-of-day CatBoost retraining

Important design rules
----------------------
- The LLM is not used for numeric price calculation.
- Base price is an anchor/input, not an ML output.
- Current marketplace analysis is performed on every pricing run.
- Similarity is multi-factor: product type, material, size,
  dimensions, craft features, color and text.
- Different product types are excluded by SimilarityEngine.
- Historical prices are evidence, not a replacement for fresh market data.
- CatBoost trains only on actual_selling_price outcomes.
- Cold-start / insufficient-data cases use market/formula fallback.
- Guardrails remain the final safety layer.
- Existing generated data is only appended to; nothing is overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from similarity import SimilarityEngine, ProductAttributes
from features import PricingFeatureEngineer, NUMERIC_FEATURES
from model import PricingModel
from pricing import (
    ProductionInput,
    PricingGuardrails,
    generate_price_recommendation,
)

# Database is deliberately used only for persistence/history.
# Marketplace collection itself remains the responsibility of the
# marketplace layer; this module consumes its fresh listings.
try:
    from database import (
        save_marketplace_snapshot,
        get_marketplace_snapshots,
        save_pricing_event,
        get_training_data,
    )
except ImportError:
    # Allows isolated unit testing of the pipeline without DB access.
    save_marketplace_snapshot = None
    get_marketplace_snapshots = None
    save_pricing_event = None
    get_training_data = None


# ============================================================
# RESULT
# ============================================================

@dataclass
class PricingPipelineResult:
    product_id: str
    method: str
    base_cost: float

    market_minimum: Optional[float]
    market_median: Optional[float]
    market_maximum: Optional[float]
    market_count: int

    similarity_score: float
    ml_prediction: Optional[float]

    formula_price: float
    market_based_price: Optional[float]
    fair_trade_floor: float

    recommended_price: float
    final_price: float

    feature_vector: Dict[str, float]
    comparable_products: List[Any]

    # Decision transparency
    ml_invoked: bool = False
    ml_model: str = "FALLBACK"
    ml_readiness: bool = False
    ml_readiness_reason: str = ""

    current_listing_count: int = 0
    current_comparable_count: int = 0
    historical_comparable_count: int = 0
    historical_price_count: int = 0

    guardrail_minimum: Optional[float] = None
    guardrail_maximum: Optional[float] = None
    decision_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        def serialise(value: Any) -> Any:
            if hasattr(value, "to_dict"):
                return value.to_dict()
            return value

        return {
            "product_id": self.product_id,
            "method": self.method,
            "base_cost": self.base_cost,
            "market_minimum": self.market_minimum,
            "market_median": self.market_median,
            "market_maximum": self.market_maximum,
            "market_count": self.market_count,
            "similarity_score": self.similarity_score,
            "ml_prediction": self.ml_prediction,
            "formula_price": self.formula_price,
            "market_based_price": self.market_based_price,
            "fair_trade_floor": self.fair_trade_floor,
            "recommended_price": self.recommended_price,
            "final_price": self.final_price,
            "feature_vector": self.feature_vector,
            "comparable_products": [
                serialise(item) for item in self.comparable_products
            ],
            "ml_invoked": self.ml_invoked,
            "ml_model": self.ml_model,
            "ml_readiness": self.ml_readiness,
            "ml_readiness_reason": self.ml_readiness_reason,
            "current_listing_count": self.current_listing_count,
            "current_comparable_count": self.current_comparable_count,
            "historical_comparable_count": self.historical_comparable_count,
            "historical_price_count": self.historical_price_count,
            "guardrail_minimum": self.guardrail_minimum,
            "guardrail_maximum": self.guardrail_maximum,
            "decision_reason": self.decision_reason,
        }


# ============================================================
# PIPELINE
# ============================================================

class PricingPipeline:
    """Complete application-side pricing orchestrator."""

    def __init__(
        self,
        similarity_engine: Optional[SimilarityEngine] = None,
        feature_engineer: Optional[PricingFeatureEngineer] = None,
        pricing_model: Optional[PricingModel] = None,
    ) -> None:
        self.similarity_engine = similarity_engine or SimilarityEngine()
        self.feature_engineer = feature_engineer or PricingFeatureEngineer()
        self.pricing_model = pricing_model or PricingModel()

    # ========================================================
    # GENERIC HELPERS
    # ========================================================

    @staticmethod
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [x.strip() for x in value.split(",") if x.strip()]
        try:
            return [str(x).strip() for x in value if str(x).strip()]
        except TypeError:
            text = str(value).strip()
            return [text] if text else []

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    # ========================================================
    # PRODUCT NORMALISATION
    # ========================================================

    def _convert_product(self, product: Any) -> Any:
        if not isinstance(product, Mapping):
            return product

        features = self._as_list(
            product.get("features", product.get("craft_features", []))
        )

        dimensions = product.get("dimensions", "")
        if not dimensions:
            parts = []
            for key in (
                "dimensions_length",
                "dimensions_width",
                "dimensions_height",
            ):
                value = product.get(key)
                if value is not None:
                    parts.append(str(value))
            dimensions = "x".join(parts)

        return ProductAttributes(
            product_id=product.get("product_id"),
            product_name=str(
                product.get(
                    "product_name",
                    product.get("title", ""),
                )
            ),
            product_type=str(product.get("product_type", "")),
            material=str(product.get("material", "")),
            additional_materials=self._as_list(
                product.get("additional_materials", [])
            ),
            color=str(product.get("color", "")),
            size=str(product.get("size", "")),
            dimensions_length=product.get("dimensions_length"),
            dimensions_width=product.get("dimensions_width"),
            dimensions_height=product.get("dimensions_height"),
            dimensions=str(dimensions),
            features=features,
            description=str(product.get("description", "")),
        )

    # ========================================================
    # MARKETPLACE LISTINGS
    # ========================================================

    @staticmethod
    def _normalise_listing(listing: Mapping[str, Any]) -> Dict[str, Any]:
        """Normalise common collector output without changing its meaning."""
        result = dict(listing)

        if "title" not in result:
            result["title"] = result.get("product_name", "")

        if "price" not in result:
            result["price"] = result.get("retail_price_inr")

        if "marketplace" not in result:
            result["marketplace"] = result.get("source", "")

        result["features"] = PricingPipeline._as_list(
            result.get("features", result.get("craft_features", []))
        )

        return result

    def _prepare_listings(
        self,
        listings: Iterable[Any],
    ) -> List[Dict[str, Any]]:
        prepared = []
        for listing in listings:
            if isinstance(listing, Mapping):
                prepared.append(self._normalise_listing(listing))
        return prepared

    # ========================================================
    # SIMILARITY
    # ========================================================

    def find_comparables(
        self,
        product: Any,
        listings: Iterable[Mapping[str, Any]],
        top_k: int = 10,
    ) -> List[Any]:
        product_attributes = self._convert_product(product)
        listings = list(listings)
        if not listings:
            return []

        # IMPORTANT: SimilarityEngine performs marketplace-listing conversion
        # itself. Do not call a nonexistent module-level converter here.
        return self.similarity_engine.match_marketplace_listings(
            product=product_attributes,
            listings=listings,
            top_k=top_k,
        )

    @staticmethod
    def _similarity_score(match: Any) -> float:
        return PricingPipeline._as_float(
            PricingPipeline._get(
                match,
                "similarity_score",
                PricingPipeline._get(match, "score", 0.0),
            )
        )

    @staticmethod
    def _match_level(match: Any) -> str:
        return str(PricingPipeline._get(match, "match_level", ""))

    @staticmethod
    def _match_price(match: Any) -> Optional[float]:
        value = PricingPipeline._get(match, "price", None)
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        return price if price > 0 else None

    # ========================================================
    # HISTORICAL MARKET EVIDENCE
    # ========================================================

    def get_historical_snapshots(self, product_id: str) -> List[Dict[str, Any]]:
        if get_marketplace_snapshots is None:
            return []
        try:
            return list(get_marketplace_snapshots(product_id))
        except Exception:
            return []

    def find_historical_comparables(
        self,
        product: Any,
        product_id: str,
        top_k: int = 20,
    ) -> List[Any]:
        snapshots = self.get_historical_snapshots(product_id)
        if not snapshots:
            return []

        # DB returns newest snapshots first. Keep the newest observation for
        # each listing before similarity matching so repeated daily tracking
        # of the same listing cannot dominate historical evidence.
        latest_by_listing = {}
        without_id = []
        for snapshot in snapshots:
            listing_id = snapshot.get("listing_id")
            if listing_id:
                key = str(listing_id)
                if key not in latest_by_listing:
                    latest_by_listing[key] = snapshot
            else:
                without_id.append(snapshot)

        prepared = self._prepare_listings(
            list(latest_by_listing.values()) + without_id
        )
        return self.find_comparables(product, prepared, top_k=top_k)

    # ========================================================
    # MARKET EVIDENCE
    # ========================================================

    @staticmethod
    def _unique_historical_prices(matches: Sequence[Any]) -> List[float]:
        """
        Keep the latest observed price per listing when possible.
        This prevents one listing tracked on many days from dominating
        the cold-start market statistics.
        """
        prices: List[float] = []
        seen_listing_ids = set()

        for match in matches:
            listing_id = PricingPipeline._get(match, "listing_id", None)
            if listing_id:
                if listing_id in seen_listing_ids:
                    continue
                seen_listing_ids.add(listing_id)

            price = PricingPipeline._match_price(match)
            if price is not None:
                prices.append(price)

        return prices

    @staticmethod
    def _current_prices(matches: Sequence[Any]) -> List[float]:
        prices = []
        for match in matches:
            price = PricingPipeline._match_price(match)
            if price is not None:
                prices.append(price)
        return prices

    @staticmethod
    def calculate_market_statistics(prices: Sequence[float]) -> Dict[str, float]:
        clean = []
        for price in prices:
            try:
                value = float(price)
            except (TypeError, ValueError):
                continue
            if value > 0:
                clean.append(value)

        if not clean:
            return {
                "minimum": 0.0,
                "median": 0.0,
                "maximum": 0.0,
                "count": 0,
            }

        return {
            "minimum": min(clean),
            "median": float(median(clean)),
            "maximum": max(clean),
            "count": len(clean),
        }

    # ========================================================
    # ML READINESS
    # ========================================================

    def is_ml_ready(self, comparable_count: int = 0) -> bool:
        """Backward-compatible readiness check used by older test/app code.

        A product is not considered ML-ready when no accepted comparable
        exists. The full reason is available from check_ml_readiness().
        """
        ready, _ = self.check_ml_readiness(comparable_count)
        return ready

    def check_ml_readiness(
        self,
        comparable_count: int,
    ) -> Tuple[bool, str]:
        """
        Prototype readiness gate.

        Required:
        - trained/loadable CatBoost model
        - exact 29-feature interface
        - at least 10 actual-outcome rows
        - at least one comparable for the current product

        The model itself remains responsible for its validation metrics.
        """
        if len(NUMERIC_FEATURES) != 29:
            return False, "Feature schema is not the required 29-feature schema."

        if comparable_count <= 0:
            return False, "No accepted comparable product for this product."

        if get_training_data is None:
            training_rows = []
        else:
            try:
                training_rows = get_training_data()
            except Exception:
                training_rows = []

        if len(training_rows) < 10:
            return (
                False,
                f"Only {len(training_rows)} actual-outcome rows; at least 10 are recommended.",
            )

        try:
            if self.pricing_model.is_trained():
                return True, "CatBoost model is already loaded and usable."

            if not self.pricing_model.model_path.exists():
                return False, "Trained CatBoost model file is not available."

            self.pricing_model.load()
            if not self.pricing_model.is_trained():
                return False, "CatBoost model could not be loaded."

            return True, "CatBoost model loaded successfully."
        except Exception as exc:
            return False, f"CatBoost readiness check failed: {exc}"

    def predict_if_ready(
        self,
        features: Mapping[str, Any],
        comparable_count: int,
    ) -> Tuple[Optional[float], bool, str, str]:
        ready, reason = self.check_ml_readiness(comparable_count)
        if not ready:
            return None, False, "FALLBACK", reason

        try:
            result = self.pricing_model.predict(features)
            prediction = float(result.predicted_price)
            if prediction <= 0:
                return None, False, "FALLBACK", "CatBoost returned a non-positive prediction."
            return prediction, True, str(result.model_used), reason
        except Exception as exc:
            return None, False, "FALLBACK", f"CatBoost prediction failed: {exc}"

    # ========================================================
    # PERSIST CURRENT SNAPSHOTS
    # ========================================================

    def save_current_snapshots(
        self,
        product_id: str,
        listings: Sequence[Mapping[str, Any]],
        comparables: Sequence[Any],
    ) -> None:
        if save_marketplace_snapshot is None:
            return

        score_by_listing_id = {}
        for match in comparables:
            listing_id = self._get(match, "listing_id", None)
            if listing_id:
                score_by_listing_id[str(listing_id)] = self._similarity_score(match)

        snapshot_date = self._now()

        for listing in listings:
            try:
                listing_id = listing.get("listing_id")
                score = score_by_listing_id.get(str(listing_id)) if listing_id else None

                features = self._as_list(
                    listing.get("features", listing.get("craft_features", []))
                )

                save_marketplace_snapshot({
                    "product_id": product_id,
                    "marketplace": listing.get("marketplace"),
                    "listing_id": listing_id,
                    "title": listing.get("title", listing.get("product_name", "")),
                    "url": listing.get("url"),
                    "price": listing.get("price", listing.get("retail_price_inr")),
                    "currency": listing.get("currency", "INR"),
                    "rating": listing.get("rating"),
                    "review_count": listing.get("review_count"),
                    "seller_name": listing.get("seller_name"),
                    "handmade_evidence": listing.get("handmade_evidence"),
                    "product_type": listing.get("product_type"),
                    "material": listing.get("material"),
                    "color": listing.get("color"),
                    "size": listing.get("size"),
                    "dimensions": listing.get("dimensions"),
                    "features": str(features),
                    "similarity_score": score,
                    "snapshot_date": snapshot_date,
                })
            except Exception:
                # A persistence failure must not change the pricing decision.
                continue

    # ========================================================
    # PRICING EVENT PERSISTENCE
    # ========================================================

    def save_pricing_result(
        self,
        result: PricingPipelineResult,
        production_input: ProductionInput,
        demand: Any,
        supply: Any,
        seasonal_index: float,
        previous_price: Optional[float],
        actual_selling_price: Optional[float] = None,
    ) -> None:
        if save_pricing_event is None:
            return

        demand = demand or {}
        supply = supply or {}

        event = {
            "product_id": result.product_id,
            "event_date": self._now(),
            "previous_price": previous_price,
            "recommended_price": result.final_price,
            "approved_price": None,
            "actual_selling_price": actual_selling_price,
            "base_cost": result.base_cost,
            "market_minimum": result.market_minimum,
            "market_median": result.market_median,
            "market_maximum": result.market_maximum,
            "market_count": result.market_count,
            "formula_price": result.formula_price,
            "market_based_price": result.market_based_price,
            "fair_trade_floor": result.fair_trade_floor,
            "time_factor": None,
            "complexity_factor": None,
            "seasonal_index": seasonal_index,
            "similarity_score": result.similarity_score,
            "pricing_method": result.method,
            "views": self._as_float(self._get(demand, "views", 0)),
            "clicks": self._as_float(self._get(demand, "clicks", 0)),
            "wishlists": self._as_float(self._get(demand, "wishlists", 0)),
            "add_to_cart": self._as_float(self._get(demand, "add_to_cart", 0)),
            "enquiries": self._as_float(self._get(demand, "enquiries", 0)),
            "orders": self._as_float(self._get(demand, "orders", 0)),
            "inventory": self._as_float(self._get(supply, "inventory", 0)),
            "conversion_rate": result.feature_vector.get("conversion_rate", 0.0),
            "created_at": self._now(),
        }

        try:
            save_pricing_event(event)
        except Exception:
            pass

    # ========================================================
    # COMPLETE PIPELINE
    # ========================================================

    def generate_recommendation(
        self,
        product: Any,
        marketplace_listings: Iterable[Any],
        production_input: ProductionInput,
        base_price: float,
        demand: Any = None,
        supply: Any = None,
        seasonal_index: float = 1.0,
        guardrails: Optional[PricingGuardrails] = None,
        top_k: int = 10,
        previous_price: Optional[float] = None,
        actual_selling_price: Optional[float] = None,
        persist: bool = True,
    ) -> PricingPipelineResult:
        if guardrails is None:
            guardrails = PricingGuardrails()

        product_id = str(self._get(product, "product_id", "UNKNOWN"))
        current_listings = self._prepare_listings(marketplace_listings)
        product_attributes = self._convert_product(product)

        # ----------------------------------------------------
        # 1. FRESH MARKETPLACE DATA -> SIMILARITY
        # ----------------------------------------------------
        # Every marketplace result is only a CANDIDATE. Raw marketplace
        # prices are NEVER allowed to enter the pricing calculation.
        # Only products accepted by SimilarityEngine become comparables.
        current_comparables = self.find_comparables(
            product=product_attributes,
            listings=current_listings,
            top_k=top_k,
        )

        # ----------------------------------------------------
        # 2. HISTORICAL DB -> SAME SIMILARITY ENGINE
        # ----------------------------------------------------
        # Historical listings are also passed through the same similarity
        # engine. Therefore historical prices can influence pricing only
        # when the historical listing is itself a valid comparable.
        historical_comparables = self.find_historical_comparables(
            product=product_attributes,
            product_id=product_id,
            top_k=max(top_k, 20),
        )

        # ----------------------------------------------------
        # 3. COMPARABLE-ONLY MARKET EVIDENCE
        # ----------------------------------------------------
        # CRITICAL RULE:
        #   current_listings       -> NEVER used directly for pricing
        #   current_comparables    -> USED
        #   historical_comparables -> USED
        #
        # Irrelevant products, even if very cheap or very expensive, have
        # ZERO influence on market minimum/median/maximum or pricing.
        current_prices = self._current_prices(current_comparables)
        historical_prices = self._unique_historical_prices(
            historical_comparables
        )

        # Current comparable evidence is primary; historical comparable
        # evidence is supporting evidence.
        market_prices = current_prices + historical_prices
        market_stats = self.calculate_market_statistics(market_prices)

        # ----------------------------------------------------
        # 4. BASE PRICE / ANCHOR
        # ----------------------------------------------------
        base_price = max(0.0, self._as_float(base_price))

        # ----------------------------------------------------
        # 5. 29 FEATURE VECTOR
        # ----------------------------------------------------
        market = {
            "minimum": market_stats["minimum"],
            "median": market_stats["median"],
            "maximum": market_stats["maximum"],
            "count": market_stats["count"],
        }

        features = self.feature_engineer.build_features(
            product=product,
            market=market,
            similarities=current_comparables,
            demand=demand,
            supply=supply,
            base_price=base_price,
            seasonal_index=seasonal_index,
        )

        if len(features) != 29:
            raise RuntimeError(
                f"Pricing feature vector must contain 29 features; got {len(features)}."
            )

        # ----------------------------------------------------
        # 6. ML READINESS -> CATBOOST OR FALLBACK
        # ----------------------------------------------------
        ml_prediction, ml_invoked, ml_model, ml_reason = self.predict_if_ready(
            features=features,
            comparable_count=len(current_comparables),
        )

        # ----------------------------------------------------
        # 7. PRICING ENGINE + GUARDRAILS
        # ----------------------------------------------------
        product_features = self._as_list(
            self._get(product, "features", self._get(product, "craft_features", []))
        )

        # Pass ONLY prices belonging to accepted comparable products.
        # Raw marketplace listings are intentionally NOT passed here.
        pricing_result = generate_price_recommendation(
            production=production_input,
            features=product_features,
            seasonal_index=seasonal_index,
            market_prices=market_prices,
            guardrails=guardrails,
            previous_price=previous_price,
            ml_prediction=ml_prediction,
        )

        # ----------------------------------------------------
        # 8. DECISION REASON
        # ----------------------------------------------------
        if ml_invoked:
            decision_reason = (
                "CatBoost invoked because the trained model is ready, the 29-feature "
                "schema is valid, actual outcome data is sufficient, and current "
                "market comparables were found."
            )
        elif current_comparables:
            decision_reason = (
                "CatBoost was not invoked; pricing used current comparable-market "
                "evidence with the formula/fair-trade baseline as fallback. "
                f"Reason: {ml_reason}"
            )
        elif historical_comparables:
            decision_reason = (
                "No current comparable passed similarity. Historical comparable "
                "evidence was used; ML/fallback readiness remained gated. "
                f"Reason: {ml_reason}"
            )
        else:
            decision_reason = (
                "No comparable marketplace product was accepted. Pricing falls back "
                "to cost/time/formula logic with guardrails. "
                f"Reason: {ml_reason}"
            )

        result = PricingPipelineResult(
            product_id=product_id,
            method=pricing_result.pricing_method,
            base_cost=pricing_result.base_cost,
            market_minimum=pricing_result.market_minimum,
            market_median=pricing_result.market_median,
            market_maximum=pricing_result.market_maximum,
            market_count=pricing_result.market_count,
            similarity_score=(
                self._similarity_score(current_comparables[0])
                if current_comparables
                else 0.0
            ),
            ml_prediction=ml_prediction,
            formula_price=pricing_result.formula_price,
            market_based_price=pricing_result.market_based_price,
            fair_trade_floor=pricing_result.fair_trade_floor,
            recommended_price=pricing_result.recommended_price_before_guardrails,
            final_price=pricing_result.final_recommended_price,
            feature_vector=dict(features),
            comparable_products=list(current_comparables),
            ml_invoked=ml_invoked,
            ml_model=ml_model,
            ml_readiness=(ml_model == "CATBOOST"),
            ml_readiness_reason=ml_reason,
            current_listing_count=len(current_listings),
            current_comparable_count=len(current_comparables),
            historical_comparable_count=len(historical_comparables),
            historical_price_count=len(historical_prices),
            guardrail_minimum=pricing_result.guardrail_minimum,
            guardrail_maximum=pricing_result.guardrail_maximum,
            decision_reason=decision_reason,
        )

        # ----------------------------------------------------
        # 9. APPEND CURRENT MARKET SNAPSHOTS + PRICING EVENT
        # ----------------------------------------------------
        if persist:
            self.save_current_snapshots(
                product_id=product_id,
                listings=current_listings,
                comparables=current_comparables,
            )
            self.save_pricing_result(
                result=result,
                production_input=production_input,
                demand=demand,
                supply=supply,
                seasonal_index=seasonal_index,
                previous_price=previous_price,
                actual_selling_price=actual_selling_price,
            )

        return result

    # ========================================================
    # END-OF-DAY RETRAINING
    # ========================================================

    def retrain_end_of_day(self) -> Optional[Any]:
        """
        Retrain CatBoost using every accumulated event that has a real
        actual_selling_price. Existing database rows are never modified.
        """
        if get_training_data is None:
            raise RuntimeError("Database training-data access is unavailable.")

        rows = get_training_data()
        if len(rows) < 10:
            print(
                f"END-OF-DAY RETRAINING SKIPPED: {len(rows)} actual outcome rows."
            )
            return None

        metrics = self.pricing_model.retrain(rows, validation_fraction=0.20)
        self.pricing_model.save()
        return metrics

    # ========================================================
    # DEBUG / DEMO OUTPUT
    # ========================================================

    @staticmethod
    def print_result(result: PricingPipelineResult) -> None:
        print("\n" + "=" * 78)
        print("SIH26090 MODULE 3 - SMART PRICING DECISION")
        print("=" * 78)
        print(f"Product ID                 : {result.product_id}")
        print(f"Current listings fetched   : {result.current_listing_count}")
        print(f"Current comparables USED   : {result.current_comparable_count}")
        print(f"Historical comparables USED: {result.historical_comparable_count}")
        print(f"Historical price evidence  : {result.historical_price_count}")
        print("RAW NON-COMPARABLE LISTINGS: NOT USED FOR PRICING")
        print(f"Best similarity score      : {result.similarity_score:.4f}")
        print("-" * 78)
        print(f"ML readiness               : {'YES' if result.ml_readiness else 'NO'}")
        print(f"ML invoked                 : {'YES' if result.ml_invoked else 'NO'}")
        print(f"Model used                 : {result.ml_model}")
        print(f"ML decision reason         : {result.ml_readiness_reason}")
        print(f"ML prediction              : {result.ml_prediction}")
        print("-" * 78)
        print(f"Base cost                  : ₹{result.base_cost:.2f}")
        print(f"Market minimum             : ₹{result.market_minimum or 0:.2f}")
        print(f"Market median              : ₹{result.market_median or 0:.2f}")
        print(f"Market maximum             : ₹{result.market_maximum or 0:.2f}")
        print(f"Market evidence count     : {result.market_count}")
        print(f"Formula price              : ₹{result.formula_price:.2f}")
        print(f"Market-based price         : ₹{result.market_based_price or 0:.2f}")
        print(f"Fair-trade floor           : ₹{result.fair_trade_floor:.2f}")
        print(f"Pre-guardrail recommendation: ₹{result.recommended_price:.2f}")
        print(f"Guardrail minimum          : ₹{result.guardrail_minimum or 0:.2f}")
        print(f"Guardrail maximum          : ₹{result.guardrail_maximum or 0:.2f}")
        print(f"FINAL RECOMMENDATION       : ₹{result.final_price:.2f}")
        print(f"Pricing method              : {result.method}")
        print("-" * 78)
        print(f"Decision                    : {result.decision_reason}")
        print(f"Feature count              : {len(result.feature_vector)}")
        print("=" * 78)

        if result.comparable_products:
            print("TOP COMPARABLES")
            print("-" * 78)
            for index, match in enumerate(result.comparable_products, start=1):
                title = PricingPipeline._get(match, "title", "")
                marketplace = PricingPipeline._get(match, "marketplace", "")
                price = PricingPipeline._match_price(match)
                score = PricingPipeline._similarity_score(match)
                level = PricingPipeline._match_level(match)
                print(
                    f"{index:02d}. {marketplace:<10} "
                    f"₹{price or 0:<8.2f} "
                    f"score={score:.4f} "
                    f"{level:<10} "
                    f"{title}"
                )
        else:
            print("TOP COMPARABLES: NONE ACCEPTED")


# ============================================================
# DEFAULT INSTANCE + APPLICATION HELPER
# ============================================================

pricing_pipeline = PricingPipeline()


def generate_pricing_recommendation(
    product: Any,
    marketplace_listings: Iterable[Any],
    production_input: ProductionInput,
    base_price: float,
    demand: Any = None,
    supply: Any = None,
    seasonal_index: float = 1.0,
    guardrails: Optional[PricingGuardrails] = None,
    top_k: int = 10,
    previous_price: Optional[float] = None,
    actual_selling_price: Optional[float] = None,
    persist: bool = True,
) -> PricingPipelineResult:
    return pricing_pipeline.generate_recommendation(
        product=product,
        marketplace_listings=marketplace_listings,
        production_input=production_input,
        base_price=base_price,
        demand=demand,
        supply=supply,
        seasonal_index=seasonal_index,
        guardrails=guardrails,
        top_k=top_k,
        previous_price=previous_price,
        actual_selling_price=actual_selling_price,
        persist=persist,
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    product = {
        "product_id": "ART-000001",
        "product_name": "Handmade Bamboo Basket",
        "product_type": "Basket",
        "material": "Bamboo",
        "additional_materials": [],
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
        "description": "Handmade traditional bamboo basket.",
        "material_cost": 350,
        "labour_cost": 250,
        "production_cost": 100,
        "time_worked_hours": 6,
    }

    production = ProductionInput(
        material_cost=350,
        production_cost=100,
        time_worked_hours=6,
        craft_labour_rate=100,
    )

    guardrails = PricingGuardrails(
        minimum_price=840,
        maximum_price=1500,
        maximum_daily_change=0.15,
        maximum_market_deviation=0.30,
    )

    demand = {
        "views": 1000,
        "clicks": 120,
        "wishlists": 25,
        "add_to_cart": 15,
        "enquiries": 8,
        "orders": 5,
    }

    supply = {"inventory": 12}

    # These are application-level collector outputs. In the real application,
    # marketplace.py supplies the fresh Amazon / Flipkart / IndiaHandmade data.
    marketplace_listings = [
        {
            "listing_id": "AMZ-001",
            "marketplace": "Amazon",
            "title": "SAI BALAJI Natural Bamboo Medium Round Basket",
            "price": 699,
            "url": "",
            "product_type": "Basket",
            "material": "Bamboo",
            "color": "Natural Brown",
            "size": "Medium",
            "dimensions": "30x20 cm",
            "features": ["Handwoven", "Natural bamboo", "Round shape"],
        },
        {
            "listing_id": "AMZ-002",
            "marketplace": "Amazon",
            "title": "NISKANET Natural Bamboo Medium Round Basket",
            "price": 599,
            "url": "",
            "product_type": "Basket",
            "material": "Bamboo",
            "color": "Natural Brown",
            "size": "Medium",
            "dimensions": "30x20 cm",
            "features": ["Handwoven", "Natural bamboo"],
        },
        {
            "listing_id": "AMZ-003",
            "marketplace": "Amazon",
            "title": "Handwoven Bamboo Basket",
            "price": 159,
            "url": "",
            "product_type": "Basket",
            "material": "Bamboo",
            "color": "Natural Brown",
            "size": "Medium",
            "dimensions": "30x20 cm",
            "features": ["Handwoven"],
        },
        {
            "listing_id": "FLP-001",
            "marketplace": "Flipkart",
            "title": "Handcrafted Natural Bamboo Wall Hanging",
            "price": 358,
            "url": "",
            "product_type": "Wall Hanging",
            "material": "Bamboo",
            "features": ["Handcrafted"],
        },
        {
            "listing_id": "FLP-002",
            "marketplace": "Flipkart",
            "title": "Bamboo Wall Hanging Flower Basket",
            "price": 162,
            "url": "",
            "product_type": "Wall Hanging",
            "material": "Bamboo",
            "features": ["Decorative"],
        },
    ]

    pipeline = PricingPipeline()

    print("=" * 78)
    print("SIH26090 MODULE 3 PIPELINE TEST")
    print("=" * 78)

    result = pipeline.generate_recommendation(
        product=product,
        marketplace_listings=marketplace_listings,
        production_input=production,
        base_price=840,
        demand=demand,
        supply=supply,
        seasonal_index=1.1,
        guardrails=guardrails,
        top_k=10,
        persist=False,
    )

    pipeline.print_result(result)

    assert len(result.feature_vector) == 29
    assert result.current_listing_count == 5
    assert result.current_comparable_count >= 1
    assert result.final_price >= result.fair_trade_floor
    assert result.final_price <= 1500

    print("\nSIH26090 MODULE 3 PIPELINE TEST PASSED")
