# src/model.py

"""
CatBoost Pricing Model
======================

Application-side ML layer for SIH26090 pricing.

Responsibilities:
    - Train CatBoost Regressor
    - Chronological validation
    - Save/load trained model
    - Predict recommended market price
    - Retrain using accumulated historical outcomes

Important:
    - Target = actual_selling_price
    - Never train on formula-generated recommended prices
    - Base price remains an input feature
    - Feature engineering is handled by src.features
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import json
import math

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor
except ImportError as exc:
    raise ImportError(
        "CatBoost is required. Install it with:\n"
        "pip install catboost"
    ) from exc

try:
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required. Install it with:\n"
        "pip install scikit-learn"
    ) from exc

from features import (
    NUMERIC_FEATURES,
    PricingFeatureEngineer,
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models" / "pricing"

MODEL_PATH = MODEL_DIR / "catboost_pricing_model.cbm"

FEATURES_PATH = MODEL_DIR / "feature_names.json"

METRICS_PATH = MODEL_DIR / "training_metrics.json"


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": False,
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ModelMetrics:
    """Validation metrics."""

    mae: float
    rmse: float
    r2: float
    train_rows: int
    validation_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
        }


@dataclass
class PredictionResult:
    """Result returned by the pricing model."""

    predicted_price: float
    model_used: str
    feature_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_price": self.predicted_price,
            "model_used": self.model_used,
            "feature_count": self.feature_count,
        }


# ============================================================
# PRICING MODEL
# ============================================================

class PricingModel:
    """
    CatBoost-based pricing model.

    The class is responsible only for ML.

    It does NOT:
        - collect marketplace data
        - calculate similarity
        - calculate pricing guardrails
        - approve prices
        - update the artisan's price
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.model_path = Path(model_path)

        self.feature_names = NUMERIC_FEATURES.copy()

        self.model_params = (
            model_params.copy()
            if model_params is not None
            else DEFAULT_MODEL_PARAMS.copy()
        )

        self.model: Optional[CatBoostRegressor] = None

        self.metrics: Optional[ModelMetrics] = None

        self.feature_engineer = PricingFeatureEngineer()

    # ========================================================
    # BASIC HELPERS
    # ========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:
            value = float(value)

            if not math.isfinite(value):
                return default

            return value

        except (TypeError, ValueError):
            return default

    # ========================================================
    # DATA PREPARATION
    # ========================================================

    def prepare_dataframe(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> pd.DataFrame:
        """
        Convert historical pricing rows into a model dataframe.

        Only rows with actual_selling_price are retained.
        """

        rows = list(rows)

        if not rows:
            raise ValueError(
                "No training rows were supplied."
            )

        prepared_rows = []

        for row in rows:

            actual_price = row.get(
                "actual_selling_price"
            )

            if actual_price is None:
                continue

            actual_price = self._safe_float(
                actual_price,
                default=-1.0,
            )

            if actual_price < 0:
                continue

            prepared = {}

            for feature in self.feature_names:
                prepared[feature] = self._safe_float(
                    row.get(feature, 0.0)
                )

            prepared["actual_selling_price"] = actual_price

            prepared_rows.append(prepared)

        if not prepared_rows:
            raise ValueError(
                "No rows contain a valid actual_selling_price."
            )

        df = pd.DataFrame(
            prepared_rows
        )

        return df

    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    @staticmethod
    def chronological_split(
        df: pd.DataFrame,
        validation_fraction: float = 0.20,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if not 0 < validation_fraction < 1:
            raise ValueError(
                "validation_fraction must be between 0 and 1."
            )

        df = df.reset_index(
            drop=True
        )

        total_rows = len(df)

        if total_rows < 2:
            raise ValueError(
                "At least 2 rows are required."
            )

        validation_rows = max(
            1,
            int(
                total_rows
                * validation_fraction
            ),
        )

        train_rows = total_rows - validation_rows

        if train_rows < 1:
            raise ValueError(
                "Not enough rows for training."
            )

        train_df = df.iloc[
            :train_rows
        ].copy()

        validation_df = df.iloc[
            train_rows:
        ].copy()

        return train_df, validation_df

    # ========================================================
    # TRAIN
    # ========================================================

    def train(
        self,
        rows: Iterable[Mapping[str, Any]],
        validation_fraction: float = 0.20,
    ) -> ModelMetrics:
        """
        Train CatBoost using historical actual selling outcomes.

        Data order is preserved.

        Earlier rows:
            training

        Latest rows:
            validation
        """

        df = self.prepare_dataframe(
            rows
        )

        if len(df) < 10:
            raise ValueError(
                "At least 10 historical outcome rows are "
                "recommended before training CatBoost."
            )

        train_df, validation_df = (
            self.chronological_split(
                df,
                validation_fraction,
            )
        )

        X_train = train_df[
            self.feature_names
        ]

        y_train = train_df[
            "actual_selling_price"
        ]

        X_validation = validation_df[
            self.feature_names
        ]

        y_validation = validation_df[
            "actual_selling_price"
        ]

        self.model = CatBoostRegressor(
            **self.model_params
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=(
                X_validation,
                y_validation,
            ),
            use_best_model=True,
            verbose=False,
        )

        predictions = self.model.predict(
            X_validation
        )

        predictions = np.asarray(
            predictions,
            dtype=float,
        )

        mae = mean_absolute_error(
            y_validation,
            predictions,
        )

        rmse = math.sqrt(
            mean_squared_error(
                y_validation,
                predictions,
            )
        )

        r2 = r2_score(
            y_validation,
            predictions,
        )

        self.metrics = ModelMetrics(
            mae=float(mae),
            rmse=float(rmse),
            r2=float(r2),
            train_rows=len(train_df),
            validation_rows=len(validation_df),
        )

        return self.metrics

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
    ) -> None:
        """
        Save the trained CatBoost model and metadata.
        """

        if self.model is None:
            raise RuntimeError(
                "Cannot save because the model has not been trained."
            )

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.save_model(
            str(self.model_path)
        )

        with open(
            FEATURES_PATH,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.feature_names,
                file,
                indent=2,
            )

        if self.metrics is not None:

            with open(
                METRICS_PATH,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    self.metrics.to_dict(),
                    file,
                    indent=2,
                )

    # ========================================================
    # LOAD
    # ========================================================

    def load(
        self,
    ) -> None:
        """
        Load an existing CatBoost model.
        """

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Pricing model not found:\n"
                f"{self.model_path}"
            )

        self.model = CatBoostRegressor()

        self.model.load_model(
            str(self.model_path)
        )

        if FEATURES_PATH.exists():

            with open(
                FEATURES_PATH,
                "r",
                encoding="utf-8",
            ) as file:

                saved_features = json.load(
                    file
                )

            if saved_features != self.feature_names:
                raise ValueError(
                    "Saved model feature order does not "
                    "match current feature engineering."
                )

    # ========================================================
    # IS TRAINED
    # ========================================================

    def is_trained(self) -> bool:
        """
        Return True if a usable model is loaded.
        """

        return self.model is not None

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        features: Mapping[str, Any],
    ) -> PredictionResult:
        """
        Predict a price from the feature dictionary.
        """

        if self.model is None:

            if self.model_path.exists():
                self.load()

            else:
                raise RuntimeError(
                    "Pricing model is not trained or loaded."
                )

        clean_features = (
            self.feature_engineer.sanitize_features(
                features
            )
        )

        vector = pd.DataFrame(
            [
                [
                    clean_features[name]
                    for name in self.feature_names
                ]
            ],
            columns=self.feature_names,
        )

        prediction = self.model.predict(
            vector
        )

        predicted_price = self._safe_float(
            prediction[0],
            default=0.0,
        )

        predicted_price = max(
            0.0,
            predicted_price,
        )

        return PredictionResult(
            predicted_price=predicted_price,
            model_used="CATBOOST",
            feature_count=len(
                self.feature_names
            ),
        )

    # ========================================================
    # RETRAIN
    # ========================================================

    def retrain(
        self,
        rows: Iterable[Mapping[str, Any]],
        validation_fraction: float = 0.20,
    ) -> ModelMetrics:
        """
        Retrain the model using the accumulated historical data.

        Existing model is replaced only after successful training.
        """

        new_model = PricingModel(
            model_path=self.model_path,
            model_params=self.model_params,
        )

        metrics = new_model.train(
            rows=rows,
            validation_fraction=validation_fraction,
        )

        new_model.save()

        self.model = new_model.model
        self.metrics = new_model.metrics

        return metrics

    # ========================================================
    # MODEL INFO
    # ========================================================

    def get_model_info(
        self,
    ) -> Dict[str, Any]:

        return {
            "model_type": "CatBoostRegressor",
            "model_path": str(
                self.model_path
            ),
            "trained": self.is_trained(),
            "feature_count": len(
                self.feature_names
            ),
            "features": self.feature_names.copy(),
            "metrics": (
                self.metrics.to_dict()
                if self.metrics is not None
                else None
            ),
        }


# ============================================================
# DEFAULT INSTANCE
# ============================================================

pricing_model = PricingModel()


# ============================================================
# APPLICATION HELPERS
# ============================================================

def train_pricing_model(
    rows: Iterable[Mapping[str, Any]],
    validation_fraction: float = 0.20,
) -> ModelMetrics:
    """
    Train and save the application pricing model.
    """

    model = PricingModel()

    metrics = model.train(
        rows=rows,
        validation_fraction=validation_fraction,
    )

    model.save()

    return metrics


def predict_price(
    features: Mapping[str, Any],
) -> PredictionResult:
    """
    Predict using the saved CatBoost model.
    """

    return pricing_model.predict(
        features
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CATBOOST PRICING MODEL TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Generate synthetic HISTORICAL OUTCOME data
    #
    # This is ONLY a self-test.
    # It is NOT project training data.
    # --------------------------------------------------------

    rng = np.random.default_rng(
        seed=42
    )

    training_rows = []

    for day in range(30):

        base_cost = (
            500
            + day * 8
        )

        market_median = (
            650
            + day * 5
        )

        base_price = (
            800
            + day * 5
        )

        views = (
            500
            + day * 20
        )

        clicks = (
            50
            + day * 2
        )

        wishlists = (
            10
            + day
        )

        add_to_cart = (
            8
            + day * 0.5
        )

        enquiries = (
            3
            + day * 0.2
        )

        orders = (
            2
            + day * 0.15
        )

        inventory = max(
            1,
            20 - day * 0.3,
        )

        conversion_rate = (
            orders / views
        )

        # Simulated actual outcome.
        #
        # Again: ONLY for testing that CatBoost
        # can train and predict.
        actual_price = (
            0.45 * base_cost
            + 0.35 * market_median
            + 0.10 * base_price
            + 300 * conversion_rate
            + rng.normal(
                0,
                10,
            )
        )

        row = {
            "material_cost": base_cost * 0.5,
            "labour_cost": base_cost * 0.3,
            "production_cost": base_cost * 0.2,
            "base_cost": base_cost,
            "time_worked_hours": 6.0,

            "base_price": base_price,

            "market_minimum": market_median - 100,
            "market_median": market_median,
            "market_maximum": market_median + 200,
            "market_count": 6,

            "similarity_score": 0.75,
            "primary_match_count": 3,
            "secondary_match_count": 2,

            "similarity_product_type": 1.0,
            "similarity_material": 1.0,
            "similarity_size": 0.9,
            "similarity_dimensions": 0.7,
            "similarity_features": 0.8,
            "similarity_color": 0.9,
            "similarity_text": 0.75,

            "views": views,
            "clicks": clicks,
            "wishlists": wishlists,
            "add_to_cart": add_to_cart,
            "enquiries": enquiries,
            "orders": orders,
            "conversion_rate": conversion_rate,

            "inventory": inventory,

            "seasonal_index": 1.0,

            # REAL target in production:
            # actual_selling_price
            "actual_selling_price": max(
                1,
                actual_price,
            ),
        }

        training_rows.append(
            row
        )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = PricingModel()

    metrics = model.train(
        training_rows
    )

    print("\nModel metrics:\n")

    print(
        f"MAE               : "
        f"{metrics.mae:.4f}"
    )

    print(
        f"RMSE              : "
        f"{metrics.rmse:.4f}"
    )

    print(
        f"R²                : "
        f"{metrics.r2:.4f}"
    )

    print(
        f"Training rows     : "
        f"{metrics.train_rows}"
    )

    print(
        f"Validation rows   : "
        f"{metrics.validation_rows}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    model.save()

    print(
        f"\nModel saved to:\n"
        f"{MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Prediction test
    # --------------------------------------------------------

    sample_features = {
        key: training_rows[-1].get(
            key,
            0.0,
        )
        for key in NUMERIC_FEATURES
    }

    result = model.predict(
        sample_features
    )

    print(
        "\nPrediction:"
    )

    print(
        f"Predicted price : "
        f"₹{result.predicted_price:.2f}"
    )

    print(
        f"Model used      : "
        f"{result.model_used}"
    )

    print(
        f"Feature count   : "
        f"{result.feature_count}"
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert model.is_trained()

    assert len(
        model.feature_names
    ) == 29

    assert result.predicted_price >= 0

    assert (
        result.feature_count
        == len(NUMERIC_FEATURES)
    )

    print("\n" + "=" * 70)
    print("CATBOOST PRICING MODEL TEST PASSED")
    print("=" * 70)