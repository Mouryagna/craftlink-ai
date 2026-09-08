from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from catboost import CatBoostRegressor

from scripts.train_pricing_model import NUMERIC_FEATURES

MODEL_PATH = Path.cwd() / "models" / "pricing" / "catboost_pricing_model.cbm"
_MODEL_INSTANCE: Optional[CatBoostRegressor] = None


def get_loaded_model() -> Optional[CatBoostRegressor]:
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None and MODEL_PATH.exists():
        try:
            model = CatBoostRegressor()
            model.load_model(str(MODEL_PATH))
            _MODEL_INSTANCE = model
        except Exception:
            _MODEL_INSTANCE = None
    return _MODEL_INSTANCE


def predict_market_fair_price(feature_dict: Dict[str, float]) -> Optional[float]:
    model = get_loaded_model()
    if model is None:
        return None

    ordered_values = {feature: [feature_dict.get(feature, 0.0)] for feature in NUMERIC_FEATURES}
    df = pd.DataFrame(ordered_values)

    try:
        pred = model.predict(df)[0]
        return round(float(pred), 2) if pred > 0 else None
    except Exception:
        return None