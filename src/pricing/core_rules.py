import math
from typing import Optional

FAIR_TRADE_MULTIPLIER = 1.20
DEFAULT_LABOUR_RATE = 100.0


def calculate_dampened_labour(hours: float, rate: float = DEFAULT_LABOUR_RATE) -> float:
    """Uses square-root dampening to prevent high hours from inflating cost."""
    safe_hours = max(0.0, float(hours))
    return round(math.sqrt(safe_hours) * rate, 2)


def calculate_base_cost(material_cost: float, labour_cost: float, overhead_cost: float) -> float:
    return round(max(0.0, material_cost) + max(0.0, labour_cost) + max(0.0, overhead_cost), 2)


def calculate_fair_trade_floor(base_cost: float) -> float:
    return round(base_cost * FAIR_TRADE_MULTIPLIER, 2)


def calculate_complexity_factor(score: int) -> float:
    scale = {1: 1.00, 2: 1.05, 3: 1.10, 4: 1.15, 5: 1.20}
    return scale.get(int(score), 1.05)


def apply_guardrails(
    recommended_price: float,
    fair_trade_floor: float,
    artisan_minimum: Optional[float] = None,
    market_median: Optional[float] = None
) -> float:
    effective_floor = fair_trade_floor
    if artisan_minimum is not None and artisan_minimum > 0:
        effective_floor = max(effective_floor, artisan_minimum)

    price = max(recommended_price, effective_floor)

    if market_median and market_median > 0:
        price = min(price, market_median * 1.35)

    return round(max(price, effective_floor), 2)