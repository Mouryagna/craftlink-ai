import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

NUMERIC_FEATURES = [
    "material_cost", "labour_cost", "production_cost", "base_cost", "time_worked_hours",
    "base_price", "market_minimum", "market_median", "market_maximum", "market_count",
    "similarity_score", "primary_match_count", "secondary_match_count",
    "similarity_product_type", "similarity_material", "similarity_size",
    "similarity_dimensions", "similarity_features", "similarity_color", "similarity_text",
    "views", "clicks", "wishlists", "add_to_cart", "enquiries", "orders", "conversion_rate",
    "inventory", "seasonal_index"
]

PROJECT_ROOT = Path.cwd()
MODEL_DIR = PROJECT_ROOT / "models" / "pricing"
MODEL_PATH = MODEL_DIR / "catboost_pricing_model.cbm"
FEATURES_PATH = MODEL_DIR / "feature_names.json"


def generate_seed_data(num_samples: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    records = []

    craft_profiles = [
        {"mat": (100, 250), "hours": (3, 8), "med": (350, 650)},
        {"mat": (150, 350), "hours": (4, 12), "med": (450, 850)},
        {"mat": (400, 900), "hours": (8, 20), "med": (1100, 2200)},
        {"mat": (300, 700), "hours": (6, 16), "med": (800, 1800)},
    ]

    for _ in range(num_samples):
        profile = craft_profiles[rng.integers(0, len(craft_profiles))]
        mat_cost = float(rng.uniform(*profile["mat"]))
        hours = float(rng.uniform(*profile["hours"]))
        rate = 100.0
        labour_cost = round(math.sqrt(hours) * rate, 2)
        prod_overhead = float(rng.uniform(30, 100))
        base_cost = round(mat_cost + labour_cost + prod_overhead, 2)

        m_median = float(rng.uniform(*profile["med"]))
        m_min = round(m_median * float(rng.uniform(0.65, 0.85)), 2)
        m_max = round(m_median * float(rng.uniform(1.20, 1.60)), 2)

        views = float(rng.integers(100, 1500))
        orders = float(rng.integers(1, max(2, int(views * 0.05))))
        conversion_rate = orders / views

        noise = float(rng.normal(0, 15))
        actual_price = round(
            (0.50 * (base_cost * 1.20)) + (0.45 * m_median) + (50 * conversion_rate) + noise,
            2
        )
        actual_price = max(base_cost * 1.10, actual_price)

        row = {
            "material_cost": mat_cost,
            "labour_cost": labour_cost,
            "production_cost": prod_overhead,
            "base_cost": base_cost,
            "time_worked_hours": hours,
            "base_price": base_cost * 1.20,
            "market_minimum": m_min,
            "market_median": m_median,
            "market_maximum": m_max,
            "market_count": int(rng.integers(8, 35)),
            "similarity_score": float(rng.uniform(0.70, 0.95)),
            "primary_match_count": int(rng.integers(1, 5)),
            "secondary_match_count": int(rng.integers(2, 8)),
            "similarity_product_type": 1.0,
            "similarity_material": float(rng.uniform(0.80, 1.0)),
            "similarity_size": float(rng.uniform(0.70, 1.0)),
            "similarity_dimensions": float(rng.uniform(0.60, 0.95)),
            "similarity_features": float(rng.uniform(0.60, 0.90)),
            "similarity_color": float(rng.uniform(0.70, 1.0)),
            "similarity_text": float(rng.uniform(0.50, 0.85)),
            "views": views,
            "clicks": float(rng.integers(10, int(views * 0.2))),
            "wishlists": float(rng.integers(2, 25)),
            "add_to_cart": float(rng.integers(1, 15)),
            "enquiries": float(rng.integers(0, 8)),
            "orders": orders,
            "conversion_rate": conversion_rate,
            "inventory": float(rng.integers(3, 20)),
            "seasonal_index": 1.0,
            "actual_selling_price": actual_price
        }
        records.append(row)

    return pd.DataFrame(records)


def train_and_save():
    print("[-] Generating seed training dataset...")
    df = generate_seed_data(num_samples=250)

    X = df[NUMERIC_FEATURES]
    y = df["actual_selling_price"]

    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[-] Fitting 29-feature CatBoost Regressor ({len(X_train)} train, {len(X_val)} val)...")
    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.04,
        depth=5,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"[+] CatBoost weights saved: {MODEL_PATH}")

    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(NUMERIC_FEATURES, f, indent=2)
    print(f"[+] Feature definition verified: {FEATURES_PATH}")


if __name__ == "__main__":
    train_and_save()