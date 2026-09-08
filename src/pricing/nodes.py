from typing import Dict, Any

from src.pricing.state import (
    PricingState,
    CostBreakdown,
    RetailTiers,
    PricingModuleOutput
)
from src.pricing.core_rules import (
    calculate_dampened_labour,
    calculate_base_cost,
    calculate_fair_trade_floor,
    calculate_complexity_factor,
    apply_guardrails,
    DEFAULT_LABOUR_RATE
)
from src.pricing.benchmark_service import get_market_benchmark
from src.pricing.ml_service import predict_market_fair_price
from scripts.train_pricing_model import NUMERIC_FEATURES


def calculate_cost_node(state: PricingState) -> Dict[str, Any]:
    hours = float(state.get("time_worked_hours", 4.0))
    stated_mat_cost = state.get("stated_material_cost")

    # Safe fallback if artisan didn't specify raw material costs
    material_cost = float(stated_mat_cost) if stated_mat_cost is not None else 120.0
    labour_cost = calculate_dampened_labour(hours, rate=DEFAULT_LABOUR_RATE)
    overhead = round(0.15 * (material_cost + labour_cost), 2)

    base_cost = calculate_base_cost(material_cost, labour_cost, overhead)
    fair_trade_floor = calculate_fair_trade_floor(base_cost)

    breakdown = CostBreakdown(
        material_cost=material_cost,
        labour_cost=labour_cost,
        production_overhead=overhead,
        base_cost=base_cost,
        fair_trade_floor=fair_trade_floor
    )

    return {"cost_breakdown": breakdown.model_dump()}


def fetch_benchmark_node(state: PricingState) -> Dict[str, Any]:
    craft_type = state.get("craft_type", "terracotta pottery")
    size_category = state.get("size_category", "medium")

    benchmark = get_market_benchmark(craft_type=craft_type, size_category=size_category)
    return {"market_benchmark": benchmark}


def price_valuation_node(state: PricingState) -> Dict[str, Any]:
    cost_data = state["cost_breakdown"]
    benchmark = state["market_benchmark"]
    complexity_score = int(state.get("visual_complexity_score", 3))
    artisan_floor = state.get("artisan_floor_price")

    base_cost = float(cost_data["base_cost"])
    fair_trade_floor = float(cost_data["fair_trade_floor"])
    market_median = float(benchmark.get("median", 450.0))

    feature_map = {feat: 0.0 for feat in NUMERIC_FEATURES}
    feature_map.update({
        "material_cost": float(cost_data["material_cost"]),
        "labour_cost": float(cost_data["labour_cost"]),
        "production_cost": float(cost_data["production_overhead"]),
        "base_cost": base_cost,
        "time_worked_hours": float(state.get("time_worked_hours", 4.0)),
        "base_price": fair_trade_floor,
        "market_minimum": float(benchmark.get("min", market_median * 0.7)),
        "market_median": market_median,
        "market_maximum": float(benchmark.get("max", market_median * 1.5)),
        "market_count": float(benchmark.get("count", 25)),
        "similarity_score": 0.85,
        "primary_match_count": 2.0,
        "secondary_match_count": 4.0,
        "similarity_product_type": 1.0,
        "similarity_material": 0.9,
        "similarity_size": 0.9,
        "similarity_dimensions": 0.8,
        "similarity_features": 0.85,
        "similarity_color": 0.9,
        "similarity_text": 0.8,
        "views": 350.0,
        "clicks": 40.0,
        "wishlists": 10.0,
        "add_to_cart": 5.0,
        "enquiries": 2.0,
        "orders": 3.0,
        "conversion_rate": 3.0 / 350.0,
        "inventory": 10.0,
        "seasonal_index": 1.0
    })

    ml_prediction = predict_market_fair_price(feature_map)
    complexity_factor = calculate_complexity_factor(complexity_score)
    pricing_method = "CATBOOST_REGRESSOR"

    if ml_prediction is not None:
        raw_recommended = ml_prediction * complexity_factor
    else:
        raw_recommended = max(fair_trade_floor, market_median) * complexity_factor
        pricing_method = "FORMULA_MARKET_FALLBACK"

    recommended_price = apply_guardrails(
        recommended_price=raw_recommended,
        fair_trade_floor=fair_trade_floor,
        artisan_minimum=float(artisan_floor) if artisan_floor else None,
        market_median=market_median
    )

    floor_price = max(fair_trade_floor, float(artisan_floor or 0.0))
    premium_price = round(recommended_price * 1.25, 2)

    retail_tiers = RetailTiers(
        floor_price=round(floor_price, 2),
        recommended_price=round(recommended_price, 2),
        premium_price=round(premium_price, 2),
        pricing_method=pricing_method
    )

    final_output = PricingModuleOutput(
        product_id=state.get("product_id", "ART-000001"),
        cost_breakdown=CostBreakdown(**cost_data),
        market_benchmark=benchmark,
        retail_tiers=retail_tiers
    )

    return {"final_output": final_output.model_dump()}