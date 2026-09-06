from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable, Optional


# ============================================================
# CONFIGURATION
# ============================================================

FAIR_TRADE_MULTIPLIER = 1.20

REFERENCE_WORKING_HOURS = 6.0

MIN_TIME_FACTOR = 0.80
MAX_TIME_FACTOR = 1.50

DEFAULT_MAX_DAILY_PRICE_CHANGE = 0.15
DEFAULT_MAX_MARKET_DEVIATION = 0.30


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ProductionInput:
    material_cost: float
    labour_cost: float
    production_cost: float
    time_worked_hours: float


@dataclass
class MarketStats:
    prices: list[float]

    @property
    def count(self) -> int:
        return len(self.prices)

    @property
    def minimum(self) -> Optional[float]:
        if not self.prices:
            return None

        return min(self.prices)

    @property
    def maximum(self) -> Optional[float]:
        if not self.prices:
            return None

        return max(self.prices)

    @property
    def median_price(self) -> Optional[float]:
        if not self.prices:
            return None

        return median(self.prices)


@dataclass
class PricingGuardrails:
    minimum_price: Optional[float] = None
    maximum_price: Optional[float] = None

    maximum_daily_change: float = (
        DEFAULT_MAX_DAILY_PRICE_CHANGE
    )

    maximum_market_deviation: float = (
        DEFAULT_MAX_MARKET_DEVIATION
    )


@dataclass
class PricingResult:
    base_cost: float

    time_factor: float
    complexity_factor: float
    seasonal_index: float

    market_minimum: Optional[float]
    market_median: Optional[float]
    market_maximum: Optional[float]
    market_count: int

    fair_trade_floor: float

    formula_price: float
    market_based_price: Optional[float]

    recommended_price_before_guardrails: float
    final_recommended_price: float

    guardrail_minimum: float
    guardrail_maximum: Optional[float]

    pricing_method: str


# ============================================================
# VALIDATION
# ============================================================

def _validate_non_negative(
    value: float,
    field_name: str,
) -> None:

    if value < 0:
        raise ValueError(
            f"{field_name} cannot be negative."
        )


def validate_production_input(
    production: ProductionInput,
) -> None:

    _validate_non_negative(
        production.material_cost,
        "Material cost",
    )

    _validate_non_negative(
        production.labour_cost,
        "Labour cost",
    )

    _validate_non_negative(
        production.production_cost,
        "Production cost",
    )

    _validate_non_negative(
        production.time_worked_hours,
        "Time worked",
    )


def validate_guardrails(
    guardrails: PricingGuardrails,
) -> None:

    if guardrails.minimum_price is not None:

        _validate_non_negative(
            guardrails.minimum_price,
            "Minimum price",
        )

    if guardrails.maximum_price is not None:

        _validate_non_negative(
            guardrails.maximum_price,
            "Maximum price",
        )

    if (
        guardrails.minimum_price is not None
        and guardrails.maximum_price is not None
        and guardrails.minimum_price
        > guardrails.maximum_price
    ):
        raise ValueError(
            "Minimum price cannot be greater than maximum price."
        )

    if not 0 <= guardrails.maximum_daily_change <= 1:

        raise ValueError(
            "Maximum daily price change must be "
            "between 0 and 1."
        )

    if not 0 <= guardrails.maximum_market_deviation <= 1:

        raise ValueError(
            "Maximum market deviation must be "
            "between 0 and 1."
        )


# ============================================================
# BASE PRODUCTION COST
# ============================================================

def calculate_base_cost(
    production: ProductionInput,
) -> float:
    """
    Base production cost supplied by the artisan.

    material cost
    + labour cost
    + other production cost
    """

    validate_production_input(production)

    return (
        production.material_cost
        + production.labour_cost
        + production.production_cost
    )


# ============================================================
# TIME FACTOR
# ============================================================

def calculate_time_factor(
    time_worked_hours: float,
    reference_hours: float = REFERENCE_WORKING_HOURS,
) -> float:
    """
    Converts production time into a bounded factor.

    This prevents extremely large or extremely small
    working-hour values from dominating the price.
    """

    _validate_non_negative(
        time_worked_hours,
        "Time worked",
    )

    if reference_hours <= 0:

        raise ValueError(
            "Reference working hours must be greater than zero."
        )

    factor = (
        time_worked_hours
        / reference_hours
    )

    return max(
        MIN_TIME_FACTOR,
        min(
            MAX_TIME_FACTOR,
            factor,
        ),
    )


# ============================================================
# CRAFT COMPLEXITY
# ============================================================

def calculate_complexity_factor(
    features: Iterable[str],
) -> float:
    """
    Derives craft complexity from product features.

    The artisan does not manually enter a complexity multiplier.
    """

    feature_list = [
        str(feature).strip().lower()
        for feature in features
        if str(feature).strip()
    ]

    if not feature_list:
        return 1.0

    complexity_keywords = {
        "handwoven",
        "handcrafted",
        "traditional",
        "traditional weaving",
        "intricate",
        "detailed",
        "embroidered",
        "carved",
        "engraved",
        "braided",
        "woven",
        "decorative",
        "artisan",
        "ornamental",
    }

    matched_features = 0

    for feature in feature_list:

        if any(
            keyword in feature
            for keyword in complexity_keywords
        ):
            matched_features += 1

    if matched_features == 0:
        return 1.00

    if matched_features == 1:
        return 1.05

    if matched_features == 2:
        return 1.10

    if matched_features == 3:
        return 1.15

    return 1.20


# ============================================================
# FAIR-TRADE FLOOR
# ============================================================

def calculate_fair_trade_floor(
    base_cost: float,
    multiplier: float = FAIR_TRADE_MULTIPLIER,
) -> float:

    _validate_non_negative(
        base_cost,
        "Base cost",
    )

    if multiplier <= 0:

        raise ValueError(
            "Fair-trade multiplier must be greater than zero."
        )

    return (
        base_cost
        * multiplier
    )


# ============================================================
# MARKET STATISTICS
# ============================================================

def calculate_market_stats(
    prices: Iterable[float],
) -> MarketStats:

    cleaned_prices = []

    for price in prices:

        price = float(price)

        if price <= 0:
            continue

        cleaned_prices.append(price)

    return MarketStats(
        prices=cleaned_prices,
    )


# ============================================================
# FORMULA BASELINE
# ============================================================

def calculate_formula_price(
    base_cost: float,
    time_factor: float,
    complexity_factor: float,
    seasonal_index: float,
) -> float:

    if base_cost < 0:
        raise ValueError(
            "Base cost cannot be negative."
        )

    if time_factor <= 0:
        raise ValueError(
            "Time factor must be greater than zero."
        )

    if complexity_factor <= 0:
        raise ValueError(
            "Complexity factor must be greater than zero."
        )

    if seasonal_index <= 0:
        raise ValueError(
            "Seasonal index must be greater than zero."
        )

    return (
        base_cost
        * time_factor
        * complexity_factor
        * seasonal_index
    )


# ============================================================
# MARKET-BASED PRICE
# ============================================================

def calculate_market_based_price(
    market_stats: MarketStats,
) -> Optional[float]:
    """
    Uses the current marketplace median as the market anchor.

    Historical data and similarity information are supplied
    by the other modules.

    This function only handles the market-price calculation.
    """

    if market_stats.median_price is None:
        return None

    return market_stats.median_price


# ============================================================
# MARKET DEVIATION GUARDRAIL
# ============================================================

def apply_market_deviation_guardrail(
    price: float,
    market_stats: MarketStats,
    maximum_deviation: float,
) -> float:
    """
    Keeps the recommendation within an acceptable deviation
    from the current market median.
    """

    if market_stats.median_price is None:
        return price

    median_price = market_stats.median_price

    lower_bound = (
        median_price
        * (1 - maximum_deviation)
    )

    upper_bound = (
        median_price
        * (1 + maximum_deviation)
    )

    return max(
        lower_bound,
        min(
            upper_bound,
            price,
        ),
    )


# ============================================================
# DAILY PRICE MOVEMENT GUARDRAIL
# ============================================================

def apply_daily_change_guardrail(
    price: float,
    previous_price: Optional[float],
    maximum_daily_change: float,
) -> float:
    """
    Prevents excessive day-to-day price movement.
    """

    if previous_price is None:
        return price

    if previous_price <= 0:
        return price

    lower_bound = (
        previous_price
        * (1 - maximum_daily_change)
    )

    upper_bound = (
        previous_price
        * (1 + maximum_daily_change)
    )

    return max(
        lower_bound,
        min(
            upper_bound,
            price,
        ),
    )


# ============================================================
# PRICE FLOOR / CEILING GUARDRAILS
# ============================================================

def apply_price_guardrails(
    price: float,
    fair_trade_floor: float,
    guardrails: PricingGuardrails,
) -> float:

    validate_guardrails(
        guardrails
    )

    # System floor always applies.
    effective_minimum = fair_trade_floor

    # Artisan-defined minimum can only increase
    # the effective minimum.
    if guardrails.minimum_price is not None:

        effective_minimum = max(
            effective_minimum,
            guardrails.minimum_price,
        )

    price = max(
        price,
        effective_minimum,
    )

    # Optional artisan maximum.
    if guardrails.maximum_price is not None:

        if (
            guardrails.maximum_price
            < effective_minimum
        ):
            raise ValueError(
                "Maximum price is below the "
                "effective minimum price."
            )

        price = min(
            price,
            guardrails.maximum_price,
        )

    return price


# ============================================================
# APPLY ALL GUARDRAILS
# ============================================================

def apply_all_guardrails(
    price: float,
    fair_trade_floor: float,
    market_stats: MarketStats,
    guardrails: PricingGuardrails,
    previous_price: Optional[float] = None,
) -> float:
    """
    Applies pricing guardrails in the fixed order:

    1. Market deviation
    2. Maximum daily movement
    3. System/artisan minimum
    4. Artisan maximum
    """

    price = apply_market_deviation_guardrail(
        price=price,
        market_stats=market_stats,
        maximum_deviation=(
            guardrails.maximum_market_deviation
        ),
    )

    price = apply_daily_change_guardrail(
        price=price,
        previous_price=previous_price,
        maximum_daily_change=(
            guardrails.maximum_daily_change
        ),
    )

    price = apply_price_guardrails(
        price=price,
        fair_trade_floor=fair_trade_floor,
        guardrails=guardrails,
    )

    return round(
        price,
        2,
    )


# ============================================================
# FINAL PRICING CALCULATION
# ============================================================

def calculate_pricing_components(
    production: ProductionInput,
    features: Iterable[str],
    seasonal_index: float,
    market_prices: Iterable[float],
):
    """
    Calculates all non-ML pricing components.

    This does NOT decide whether CatBoost should be used.
    """

    validate_production_input(
        production
    )

    base_cost = calculate_base_cost(
        production
    )

    time_factor = calculate_time_factor(
        production.time_worked_hours
    )

    complexity_factor = calculate_complexity_factor(
        features
    )

    fair_trade_floor = calculate_fair_trade_floor(
        base_cost
    )

    market_stats = calculate_market_stats(
        market_prices
    )

    formula_price = calculate_formula_price(
        base_cost=base_cost,
        time_factor=time_factor,
        complexity_factor=complexity_factor,
        seasonal_index=seasonal_index,
    )

    # Formula can never recommend below the
    # system-derived fair-trade floor.
    formula_price = max(
        formula_price,
        fair_trade_floor,
    )

    market_based_price = (
        calculate_market_based_price(
            market_stats
        )
    )

    return {
        "base_cost": base_cost,
        "time_factor": time_factor,
        "complexity_factor": complexity_factor,
        "seasonal_index": seasonal_index,
        "fair_trade_floor": fair_trade_floor,
        "market_stats": market_stats,
        "formula_price": formula_price,
        "market_based_price": market_based_price,
    }


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

def generate_price_recommendation(
    production: ProductionInput,
    features: Iterable[str],
    seasonal_index: float,
    market_prices: Iterable[float],
    guardrails: PricingGuardrails,
    previous_price: Optional[float] = None,
    ml_prediction: Optional[float] = None,
) -> PricingResult:
    """
    Final pricing engine.

    Important:
    This function does NOT determine ML readiness.

    The caller decides:

        ML ready
            -> pass CatBoost prediction

        ML not ready
            -> pass ml_prediction=None

    Therefore the workflow remains:

        ML
        OR
        formula/market fallback
    """

    validate_guardrails(
        guardrails
    )

    components = calculate_pricing_components(
        production=production,
        features=features,
        seasonal_index=seasonal_index,
        market_prices=market_prices,
    )

    base_cost = components["base_cost"]
    time_factor = components["time_factor"]
    complexity_factor = components["complexity_factor"]
    fair_trade_floor = components["fair_trade_floor"]
    market_stats = components["market_stats"]
    formula_price = components["formula_price"]
    market_based_price = components["market_based_price"]

    # ========================================================
    # SOURCE SELECTION
    # ========================================================

    if ml_prediction is not None:

        if ml_prediction <= 0:
            raise ValueError(
                "ML prediction must be greater than zero."
            )

        recommended_price = float(
            ml_prediction
        )

        pricing_method = "CATBOOST"

    elif market_based_price is not None:

        # Market evidence is available.
        # Formula remains the fallback/baseline.
        recommended_price = max(
            formula_price,
            market_based_price,
        )

        pricing_method = "MARKET_BASED"

    else:

        # No current market evidence.
        recommended_price = formula_price

        pricing_method = "COST_TIME_BASED"

    # Never allow the raw recommendation below
    # the system-derived floor.
    recommended_price = max(
        recommended_price,
        fair_trade_floor,
    )

    recommended_before_guardrails = (
        recommended_price
    )

    # ========================================================
    # GUARDRAILS
    # ========================================================

    final_price = apply_all_guardrails(
        price=recommended_price,
        fair_trade_floor=fair_trade_floor,
        market_stats=market_stats,
        guardrails=guardrails,
        previous_price=previous_price,
    )

    # ========================================================
    # EFFECTIVE MINIMUM
    # ========================================================

    effective_minimum = fair_trade_floor

    if guardrails.minimum_price is not None:

        effective_minimum = max(
            effective_minimum,
            guardrails.minimum_price,
        )

    # ========================================================
    # RESULT
    # ========================================================

    return PricingResult(
        base_cost=round(
            base_cost,
            2,
        ),

        time_factor=round(
            time_factor,
            4,
        ),

        complexity_factor=round(
            complexity_factor,
            4,
        ),

        seasonal_index=round(
            seasonal_index,
            4,
        ),

        market_minimum=(
            round(
                market_stats.minimum,
                2,
            )
            if market_stats.minimum is not None
            else None
        ),

        market_median=(
            round(
                market_stats.median_price,
                2,
            )
            if market_stats.median_price is not None
            else None
        ),

        market_maximum=(
            round(
                market_stats.maximum,
                2,
            )
            if market_stats.maximum is not None
            else None
        ),

        market_count=market_stats.count,

        fair_trade_floor=round(
            fair_trade_floor,
            2,
        ),

        formula_price=round(
            formula_price,
            2,
        ),

        market_based_price=(
            round(
                market_based_price,
                2,
            )
            if market_based_price is not None
            else None
        ),

        recommended_price_before_guardrails=round(
            recommended_before_guardrails,
            2,
        ),

        final_recommended_price=final_price,

        guardrail_minimum=round(
            effective_minimum,
            2,
        ),

        guardrail_maximum=(
            round(
                guardrails.maximum_price,
                2,
            )
            if guardrails.maximum_price is not None
            else None
        ),

        pricing_method=pricing_method,
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SIH26090 PRICING ENGINE TEST")
    print("=" * 60)

    production = ProductionInput(
        material_cost=350,
        labour_cost=250,
        production_cost=100,
        time_worked_hours=6,
    )

    features = [
        "Handwoven",
        "Traditional weaving",
        "Natural bamboo",
        "Round shape",
        "Handcrafted",
    ]

    market_prices = [
        699,
        599,
        159,
        699,
        599,
    ]

    guardrails = PricingGuardrails(
        minimum_price=None,
        maximum_price=None,
        maximum_daily_change=0.15,
        maximum_market_deviation=0.30,
    )

    result = generate_price_recommendation(
        production=production,
        features=features,
        seasonal_index=1.10,
        market_prices=market_prices,
        guardrails=guardrails,
        previous_price=None,
        ml_prediction=None,
    )

    print()
    print("Base cost:", result.base_cost)
    print("Time factor:", result.time_factor)
    print("Complexity factor:", result.complexity_factor)
    print("Seasonal index:", result.seasonal_index)

    print()
    print("Market minimum:", result.market_minimum)
    print("Market median:", result.market_median)
    print("Market maximum:", result.market_maximum)
    print("Market count:", result.market_count)

    print()
    print("Fair-trade floor:", result.fair_trade_floor)
    print("Formula price:", result.formula_price)
    print("Market-based price:", result.market_based_price)

    print()
    print(
        "Recommended before guardrails:",
        result.recommended_price_before_guardrails,
    )

    print(
        "Final recommended price:",
        result.final_recommended_price,
    )

    print(
        "Pricing method:",
        result.pricing_method,
    )

    print()
    print("PRICING ENGINE TEST PASSED")