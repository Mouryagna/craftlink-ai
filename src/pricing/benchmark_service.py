import json
from pathlib import Path
from typing import Dict, Any

BENCHMARK_PATH = Path.cwd() / "data" / "market_benchmarks.json"


def get_market_benchmark(craft_type: str, size_category: str) -> Dict[str, Any]:
    craft_type = craft_type.lower().strip()
    size_category = size_category.lower().strip()

    defaults = {"min": 249.0, "median": 450.0, "max": 799.0, "count": 25}

    if not BENCHMARK_PATH.exists():
        return defaults

    try:
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        matched_category = None
        for key in data:
            if key in craft_type or craft_type in key:
                matched_category = data[key]
                break

        if not matched_category:
            matched_category = list(data.values())[0]

        return matched_category.get(size_category, matched_category.get("medium", defaults))
    except Exception:
        return defaults