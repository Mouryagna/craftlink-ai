import os
import joblib
import pandas as pd
from catboost import CatBoostRegressor

# Import from your existing Module 3 codebase
from marketplace_sources import fetch_marketplace_candidates
from pricing import (
    calculate_complexity_factor,
    calculate_market_stats,
    calculate_base_cost,
    calculate_time_factor,
    ProductionInput,
    generate_price_recommendation,
    PricingGuardrails
)

MODEL_FILE = os.path.join(os.path.dirname(__file__), "catboost_pricing_model.pkl")

# ============================================================
# 1. ARTISAN INPUT (ART-000001)
# ============================================================
product_query = "Handmade Bamboo Basket"

artisan_prod = ProductionInput(
    material_cost=350.0,
    production_cost=100.0,
    time_worked_hours=6.0,
    craft_labour_rate=100.0
)

features = [
    "Handwoven",
    "Traditional weaving",
    "Natural bamboo",
    "Round shape",
    "Handcrafted"
]

seasonal_index = 1.10

# ============================================================
# 2. FETCH REAL-TIME MARKET DATA
# ============================================================
print(f"Fetching real-time market data for: '{product_query}'...")
raw_comparables = fetch_marketplace_candidates(product_query)

if not raw_comparables:
    raise RuntimeError("No live listings found from marketplace sources.")

# Extract real prices from the live scraped listings
real_prices = [
    float(item["price"])
    for item in raw_comparables
    if item.get("price") is not None and float(item.get("price", 0)) > 0
]

print(f"Fetched {len(real_prices)} live market price points: {real_prices}")

# Compute live market statistics
market_stats = calculate_market_stats(real_prices)
live_median = market_stats.median_price

if live_median is None:
    raise RuntimeError("Could not compute median from live market prices.")

print(f"Live Market Median: ₹{live_median}")

# ============================================================
# 3. BUILD REAL-TIME TRAINING DATASET
# ============================================================
# Using the live fetched market distribution to train CatBoost
complexity = calculate_complexity_factor(features)

training_rows = []
for p in real_prices:
    training_rows.append({
        "material_cost": artisan_prod.material_cost,
        "production_cost": artisan_prod.production_cost,
        "time_worked_hours": artisan_prod.time_worked_hours,
        "craft_labour_rate": artisan_prod.craft_labour_rate,
        "complexity_factor": complexity,
        "seasonal_index": seasonal_index,
        "market_median": live_median,
        "target_price": p  # Real market sale price
    })

df_train = pd.DataFrame(training_rows)

feature_cols = [
    "material_cost",
    "production_cost",
    "time_worked_hours",
    "craft_labour_rate",
    "complexity_factor",
    "seasonal_index",
    "market_median"
]

X = df_train[feature_cols]
y = df_train["target_price"]

# ============================================================
# 4. TRAIN AND SAVE CATBOOST MODEL (.PKL)
# ============================================================
model = CatBoostRegressor(
    iterations=150,
    learning_rate=0.05,
    depth=3,
    verbose=0,
    random_seed=42
)

model.fit(X, y)
joblib.dump(model, MODEL_FILE)
print(f"Saved real-time CatBoost model to: {MODEL_FILE}")

# ============================================================
# 5. TEST PREDICTION AND RUN PRICING ENGINE
# ============================================================
loaded_model = joblib.load(MODEL_FILE)

inference_input = pd.DataFrame([{
    "material_cost": artisan_prod.material_cost,
    "production_cost": artisan_prod.production_cost,
    "time_worked_hours": artisan_prod.time_worked_hours,
    "craft_labour_rate": artisan_prod.craft_labour_rate,
    "complexity_factor": complexity,
    "seasonal_index": seasonal_index,
    "market_median": live_median
}])

ml_price = float(loaded_model.predict(inference_input)[0])
print(f"Real-Time ML Predicted Price: ₹{round(ml_price, 2)}")

# Run final pricing engine with live ML prediction
result = generate_price_recommendation(
    production=artisan_prod,
    features=features,
    seasonal_index=seasonal_index,
    market_prices=real_prices,
    guardrails=PricingGuardrails(),
    previous_price=None,
    ml_prediction=ml_price
)

print("\n--- FINAL ENGINE OUTPUT ---")
print(f"Base Cost: ₹{result.base_cost}")
print(f"Fair-Trade Floor: ₹{result.fair_trade_floor}")
print(f"Market Median: ₹{result.market_median}")
print(f"Recommended Before Guardrails: ₹{result.recommended_price_before_guardrails}")
print(f"Final Price: ₹{result.final_recommended_price}")
print(f"Pricing Method: {result.pricing_method}")