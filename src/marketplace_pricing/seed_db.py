import datetime
from database import save_pricing_event, get_training_data
from model import train_pricing_model, MODEL_PATH

PRODUCT_ID = "ART-000001"

print(f"Seeding historical sales outcomes for {PRODUCT_ID} into database...")

# Seed 15 realistic historical pricing & sales events
base_date = datetime.date(2026, 1, 1)

for i in range(15):
    event_date = (base_date + datetime.timedelta(days=i * 3)).isoformat()

    # Realistic feature values around ART-000001
    cost = 694.95
    med = 599.0 + (i * 10)
    actual_sold_price = 850.0 + (i * 12)  # Non-null actual selling price

    event = {
        "product_id": PRODUCT_ID,
        "event_date": event_date,
        "previous_price": 840.0,
        "recommended_price": 879.11,
        "approved_price": 850.0,
        "actual_selling_price": actual_sold_price,
        "base_cost": cost,
        "market_minimum": 159.0,
        "market_median": med,
        "market_maximum": 699.0,
        "market_count": 5,
        "formula_price": 879.11,
        "market_based_price": med,
        "fair_trade_floor": 833.94,
        "time_factor": 1.0,
        "complexity_factor": 1.15,
        "seasonal_index": 1.10,
        "similarity_score": 0.89,
        "pricing_method": "FAIR_TRADE_FLOOR",
        "views": 200 + (i * 15),
        "clicks": 40 + (i * 3),
        "wishlists": 10 + i,
        "add_to_cart": 8 + i,
        "enquiries": 3,
        "orders": 2 + (i // 3),
        "inventory": 20 - i,
        "conversion_rate": (2 + (i // 3)) / (200 + (i * 15)),
        "created_at": datetime.datetime.now().isoformat()
    }
    save_pricing_event(event)

# Verify rows loaded from database
rows = get_training_data()
print(f"Database now has {len(rows)} historical actual-outcome rows.")

# Train and persist the model
print("Training CatBoost model...")
metrics = train_pricing_model(rows)

print("\n--- Training Successful ---")
print(f"Model saved to: {MODEL_PATH}")
print(f"MAE: {metrics.mae:.2f}")
print(f"RMSE: {metrics.rmse:.2f}")
print(f"R2: {metrics.r2:.2f}")
print(f"Trained rows: {metrics.train_rows}")