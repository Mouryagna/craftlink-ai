from typing import Optional, Dict, Any, TypedDict
from pydantic import BaseModel


class CostBreakdown(BaseModel):
    material_cost: float
    labour_cost: float
    production_overhead: float
    base_cost: float
    fair_trade_floor: float


class RetailTiers(BaseModel):
    floor_price: float
    recommended_price: float
    premium_price: float
    pricing_method: str


class PricingModuleOutput(BaseModel):
    product_id: str
    cost_breakdown: CostBreakdown
    market_benchmark: Dict[str, Any]
    retail_tiers: RetailTiers


class PricingState(TypedDict, total=False):
    product_id: str
    craft_type: str
    size_category: str
    time_worked_hours: float
    stated_material_cost: Optional[float]
    artisan_floor_price: Optional[float]
    visual_complexity_score: int

    cost_breakdown: Dict[str, Any]
    market_benchmark: Dict[str, Any]
    final_output: Dict[str, Any]