import json
from langgraph.graph import StateGraph, START, END

from src.pricing.state import PricingState
from src.pricing.nodes import (
    calculate_cost_node,
    fetch_benchmark_node,
    price_valuation_node
)

# -------------------------------------------------------------------------
# Compile Pricing Subgraph Pipeline
# -------------------------------------------------------------------------
builder = StateGraph(PricingState)

builder.add_node("calculate_cost", calculate_cost_node)
builder.add_node("fetch_benchmark", fetch_benchmark_node)
builder.add_node("price_valuation", price_valuation_node)

builder.add_edge(START, "calculate_cost")
builder.add_edge("calculate_cost", "fetch_benchmark")
builder.add_edge("fetch_benchmark", "price_valuation")
builder.add_edge("price_valuation", END)

pricing_subgraph = builder.compile()


# -------------------------------------------------------------------------
# Standalone Self-Test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("[-] Running Pricing Module Subgraph Self-Test...")

    sample_input: PricingState = {
        "product_id": "ART-000001",
        "craft_type": "terracotta pottery",
        "size_category": "medium",
        "time_worked_hours": 6.0,
        "stated_material_cost": 150.0,
        "artisan_floor_price": 400.0,
        "visual_complexity_score": 4
    }

    result = pricing_subgraph.invoke(sample_input)
    final_output = result.get("final_output", {})

    print("\n==========================================")
    print("1. COST BREAKDOWN")
    print("==========================================")
    print(json.dumps(final_output.get("cost_breakdown", {}), indent=2, ensure_ascii=False))

    print("\n==========================================")
    print("2. MARKET BENCHMARK (LOCAL LOOKUP)")
    print("==========================================")
    print(json.dumps(final_output.get("market_benchmark", {}), indent=2, ensure_ascii=False))

    print("\n==========================================")
    print("3. RETAIL PRICING TIERS")
    print("==========================================")
    print(json.dumps(final_output.get("retail_tiers", {}), indent=2, ensure_ascii=False))

    print("\n[+] Pricing Subgraph executed successfully.")