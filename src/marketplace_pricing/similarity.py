"""
SIH26090 - Production Similarity Engine
=======================================

Purpose
-------
Compare one structured artisan product with marketplace products and return
ranked comparables for pricing.

Design principles
-----------------
* No KNN is required. Every candidate can be compared directly.
* Similarity is multi-field, not a single name/material rule.
* Product type is important, but the engine does not require an exact string.
* Material is a strong factor, but a different material does NOT automatically
  reject an otherwise comparable product.
* Missing fields are ignored rather than converted into zero similarity.
* Marketplace scraped product_type is treated as weak/untrusted metadata;
  title/product text is used to validate it.
* A candidate is rejected only when the evidence says it is genuinely
  unrelated, not simply because one field is missing or different.
* Optional sentence-transformers embeddings improve semantic matching when
  installed. A dependency-free lexical scorer remains available.
* Optional Gemini classification is used only to improve ambiguous product
  type labels. It is never used for pairwise scoring and never calculates
  prices.

This module does NOT calculate prices, scrape marketplaces, or train CatBoost.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import logging
import math
import os
import re
from functools import lru_cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OPTIONAL SEMANTIC MODEL
# ---------------------------------------------------------------------------

_EMBEDDER = None
_EMBEDDER_ATTEMPTED = False


def _get_embedder():
    """Lazy-load a small sentence-transformer if available."""
    global _EMBEDDER, _EMBEDDER_ATTEMPTED
    if _EMBEDDER_ATTEMPTED:
        return _EMBEDDER
    _EMBEDDER_ATTEMPTED = True

    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv(
            "SIMILARITY_EMBEDDING_MODEL",
            "all-MiniLM-L6-v2",
        )
        _EMBEDDER = SentenceTransformer(model_name)
    except Exception as exc:
        logger.info("Semantic embedding model unavailable: %s", exc)
        _EMBEDDER = None

    return _EMBEDDER


def _cosine(a, b) -> float:
    try:
        denominator = float((a * a).sum() ** 0.5 * (b * b).sum() ** 0.5)
        if denominator <= 0:
            return 0.0
        return max(0.0, min(1.0, float((a * b).sum() / denominator)))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# NORMALIZATION HELPERS
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "handmade", "handcrafted", "handwoven", "artisan", "artisanal",
    "traditional", "authentic", "beautiful", "premium", "unique",
    "natural", "buy", "online", "best", "new", "set", "with", "for",
    "from", "the", "and", "of", "in", "to", "a", "an",
}


def _norm(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _tokens(value: Any, remove_stopwords: bool = False) -> set[str]:
    words = set(_norm(value).split())
    if remove_stopwords:
        words -= _STOPWORDS
    return words


def _token_jaccard(a: Any, b: Any, remove_stopwords: bool = False) -> float:
    aa = _tokens(a, remove_stopwords)
    bb = _tokens(b, remove_stopwords)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def _sequence_similarity(a: Any, b: Any) -> float:
    from difflib import SequenceMatcher
    aa, bb = _norm(a), _norm(b)
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb).ratio()


def _semantic_similarity(a: Any, b: Any) -> float:
    aa, bb = _norm(a), _norm(b)
    if not aa or not bb:
        return 0.0
    model = _get_embedder()
    if model is None:
        return 0.0
    try:
        vectors = model.encode([aa, bb], normalize_embeddings=True)
        return max(0.0, min(1.0, float(vectors[0] @ vectors[1])))
    except Exception:
        return 0.0


def _field_similarity(a: Any, b: Any, semantic: bool = True) -> float:
    """Hybrid lexical/semantic similarity for arbitrary text fields."""
    aa, bb = _norm(a), _norm(b)
    if not aa or not bb:
        return 0.0
    if aa == bb:
        return 1.0

    j = _token_jaccard(aa, bb, remove_stopwords=True)
    seq = _sequence_similarity(aa, bb)

    # Embeddings are particularly useful for phrases such as
    # "wicker"/"rattan basket" or "floor covering"/"rug".
    sem = _semantic_similarity(aa, bb) if semantic else 0.0

    if sem > 0:
        return max(0.0, min(1.0, 0.45 * sem + 0.30 * j + 0.25 * seq))
    return max(0.0, min(1.0, 0.55 * j + 0.45 * seq))


def _field_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return bool(str(value).strip())


# ---------------------------------------------------------------------------
# OPTIONAL LLM TYPE CLASSIFICATION
# ---------------------------------------------------------------------------

# This is intentionally only a label vocabulary for classification. It is NOT
# the similarity logic and does not decide acceptance by itself.
TYPE_LABELS = [
    "basket", "bag", "pouch", "textile", "saree", "scarf", "shawl",
    "dress", "garment", "jewellery", "necklace", "earrings", "bangle",
    "furniture", "chair", "stool", "table", "swing", "storage",
    "home decor", "wall decor", "lamp", "diya", "candle holder",
    "water bottle", "pottery", "vessel", "bottle", "bowl", "plate",
    "tray", "box", "kitchenware", "utensil", "toy", "doll", "painting",
    "art", "instrument", "accessory", "footwear", "rug", "mat", "carpet",
    "stationery", "pen holder", "idol", "other",
]


class ProductTypeClassifier:
    """Optional LLM helper for ambiguous marketplace type strings."""

    def __init__(self) -> None:
        self.model = os.getenv("GEMINI_QUERY_MODEL", "gemini-2.5-flash-lite")
        self._client = None
        self._initialized = False
        self._cache: Dict[str, str] = {}

    def _init(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            return
        try:
            from google import genai
            self._client = genai.Client(api_key=key)
        except Exception as exc:
            logger.info("Gemini classifier unavailable: %s", exc)

    def classify(self, name: str) -> str:
        name = _norm(name)
        if not name:
            return ""
        if name in self._cache:
            return self._cache[name]

        # Do not make an API request when the name already contains a useful
        # noun. This keeps collection/scoring practical for thousands of rows.
        direct = self._lexical_type(name)
        if direct:
            self._cache[name] = direct
            return direct

        self._init()
        if self._client:
            try:
                prompt = (
                    "Classify the product into one category. Return JSON only. "
                    f"Allowed labels: {', '.join(TYPE_LABELS)}. "
                    f"Product name: {name}"
                )
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": 0},
                )
                match = re.search(r"\{.*?\}", getattr(response, "text", "") or "", re.S)
                if match:
                    value = json.loads(match.group(0)).get("product_type", "")
                    value = _norm(value)
                    if value:
                        self._cache[name] = value
                        return value
            except Exception as exc:
                logger.debug("Type classification failed: %s", exc)

        self._cache[name] = ""
        return ""

    @staticmethod
    def _lexical_type(text: str) -> str:
        # Prefer concrete product nouns over broad grouping labels such as
        # storage/furniture/art/accessory. Choose the occurrence closest to
        # the end of the title when several concrete nouns are present.
        broad = {"storage", "furniture", "textile", "garment", "home decor", "art", "accessory", "other", "kitchenware"}
        hits = []
        for label in TYPE_LABELS:
            if label in broad:
                continue
            match = re.search(r"\b" + re.escape(label) + r"\b", text)
            if match:
                hits.append((match.start(), len(label), label))
        if not hits:
            return ""
        hits.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return hits[0][2]


product_type_classifier = ProductTypeClassifier()


# ---------------------------------------------------------------------------
# DATA MODELS - PUBLIC INTERFACES PRESERVED
# ---------------------------------------------------------------------------

@dataclass
class ProductAttributes:
    product_id: Optional[str] = None
    product_name: str = ""
    product_type: str = ""
    material: str = ""
    additional_materials: Optional[List[str]] = None
    color: str = ""
    size: str = ""
    dimensions_length: Optional[float] = None
    dimensions_width: Optional[float] = None
    dimensions_height: Optional[float] = None
    dimensions: Optional[str] = None
    features: Optional[List[str]] = None
    description: str = ""
    # Optional production economics. Marketplace records often do not have
    # these fields, so they are used only when genuinely available.
    material_cost: Optional[float] = None
    labour_cost: Optional[float] = None
    production_cost: Optional[float] = None
    base_cost: Optional[float] = None

    def __post_init__(self) -> None:
        self.product_name = str(self.product_name or "").strip()
        self.product_type = str(self.product_type or "").strip()
        self.material = str(self.material or "").strip()
        self.color = str(self.color or "").strip()
        self.size = str(self.size or "").strip()
        self.dimensions = str(self.dimensions or "").strip()
        self.description = str(self.description or "").strip()
        for field in ("material_cost", "labour_cost", "production_cost", "base_cost"):
            value = getattr(self, field)
            try:
                value = float(value) if value is not None else None
                if value is not None and (not math.isfinite(value) or value < 0):
                    value = None
            except (TypeError, ValueError):
                value = None
            setattr(self, field, value)
        self.additional_materials = [str(x).strip() for x in (self.additional_materials or []) if str(x).strip()]
        self.features = [str(x).strip() for x in (self.features or []) if str(x).strip()]
        for field in ("dimensions_length", "dimensions_width", "dimensions_height"):
            value = getattr(self, field)
            try:
                value = float(value) if value is not None else None
                if value is not None and (not math.isfinite(value) or value <= 0):
                    value = None
            except (TypeError, ValueError):
                value = None
            setattr(self, field, value)


@dataclass
class SimilarityBreakdown:
    product_type: float
    material: float
    size: float
    dimensions: float
    features: float
    color: float
    text: float
    total_score: float
    match_level: str
    # Extra diagnostics. Existing integrations can ignore these fields.
    material_cost_similarity: float = 0.0
    product_cost_similarity: float = 0.0
    evidence_fields: Optional[List[str]] = None
    comparable_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimilarProduct:
    marketplace: str
    title: str
    price: float
    similarity_score: float
    match_level: str
    listing_id: Optional[str] = None
    url: Optional[str] = None
    product_type: Optional[str] = None
    material: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    dimensions: Optional[str] = None
    features: Optional[List[str]] = None
    breakdown: Optional[SimilarityBreakdown] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# ENGINE
# ---------------------------------------------------------------------------

class SimilarityEngine:
    """Balanced, pricing-oriented multi-factor similarity engine."""

    # Product type and material matter most, but neither one is used as a
    # simplistic exact-match gate.
    WEIGHTS = {
        "product_type": 0.28,
        "material": 0.18,
        "size": 0.08,
        "dimensions": 0.10,
        "features": 0.13,
        "color": 0.05,
        "text": 0.18,
    }

    MINIMUM_MATCH_SCORE = 0.48
    STRONG_MATCH_SCORE = 0.74
    MEDIUM_MATCH_SCORE = 0.60

    def __init__(self, minimum_score: float = MINIMUM_MATCH_SCORE) -> None:
        if not 0 <= minimum_score <= 1:
            raise ValueError("minimum_score must be between 0 and 1")
        self.minimum_score = minimum_score

    # -------------------------- field scoring --------------------------

    @staticmethod
    def normalize_text(text: Optional[str]) -> str:
        return _norm(text)

    @classmethod
    def tokenize(cls, text: Optional[str]) -> set:
        return _tokens(text, remove_stopwords=False)

    def product_type_similarity(self, product: ProductAttributes, candidate: ProductAttributes) -> float:
        a = _norm(product.product_type)
        b = _norm(candidate.product_type)
        if not a or not b:
            # If the explicit type is missing, the title still gives evidence.
            return self._title_type_similarity(product.product_name, candidate.product_name)
        if a == b:
            return 1.0

        lexical = _field_similarity(a, b, semantic=True)
        title = self._title_type_similarity(product.product_name, candidate.product_name)

        # The explicit type and title evidence are blended. This prevents a
        # bad scraped category from dominating the comparison.
        return max(lexical, 0.65 * lexical + 0.35 * title)

    @staticmethod
    def _title_type_similarity(a: str, b: str) -> float:
        # Compare the complete names with stopwords removed. It is deliberately
        # soft: "lawn table" and "side table" can remain related without an
        # enormous hardcoded type ontology.
        return _field_similarity(a, b, semantic=True)

    def material_similarity(self, product: ProductAttributes, candidate: ProductAttributes) -> float:
        a = _norm(product.material)
        b = _norm(candidate.material)
        if not a or not b:
            return self._material_from_text_similarity(product, candidate)
        return _field_similarity(a, b, semantic=True)

    def _material_from_text_similarity(self, product: ProductAttributes, candidate: ProductAttributes) -> float:
        if not product.material:
            return 0.0
        # Candidate material may be missing. Compare the artisan material with
        # the candidate's title/description semantically instead of using a
        # fixed list of materials.
        candidate_text = " ".join(x for x in [candidate.product_name, candidate.description] if x)
        return _field_similarity(product.material, candidate_text, semantic=True) * 0.75

    def size_similarity(self, product: ProductAttributes, candidate: ProductAttributes) -> float:
        a, b = _norm(product.size), _norm(candidate.size)
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0

        order = {"xs": 1, "extra small": 1, "small": 2, "s": 2,
                 "medium": 3, "m": 3, "large": 4, "l": 4,
                 "xl": 5, "extra large": 5}
        if a in order and b in order:
            d = abs(order[a] - order[b])
            return {0: 1.0, 1: 0.78, 2: 0.50}.get(d, 0.25)
        return _field_similarity(a, b, semantic=False)

    @staticmethod
    def _dimension_vector(product: ProductAttributes) -> Optional[Tuple[float, ...]]:
        values = [product.dimensions_length, product.dimensions_width, product.dimensions_height]
        if not any(v is not None for v in values):
            return None
        return tuple(v for v in values if v is not None)

    def dimensions_similarity(self, product: ProductAttributes, candidate: ProductAttributes) -> float:
        a = self._dimension_vector(product)
        b = self._dimension_vector(candidate)
        if a is None or b is None:
            if product.dimensions and candidate.dimensions:
                return _field_similarity(product.dimensions, candidate.dimensions, semantic=False)
            return 0.0
        if len(a) != len(b):
            return 0.0

        vals = []
        for x, y in zip(a, b):
            rel = abs(x - y) / max(abs(x), abs(y), 1e-9)
            if rel <= 0.10:
                vals.append(1.0)
            elif rel <= 0.25:
                vals.append(0.82)
            elif rel <= 0.40:
                vals.append(0.55)
            elif rel <= 0.60:
                vals.append(0.25)
            else:
                vals.append(0.0)
        return sum(vals) / len(vals)

    def features_similarity(self, product: ProductAttributes, candidate: ProductAttributes) -> float:
        a = [_norm(x) for x in product.features if _norm(x)]
        b = [_norm(x) for x in candidate.features if _norm(x)]
        if not a or not b:
            return 0.0

        # Best-match aggregation handles wording differences and avoids the
        # old exact-set-intersection problem.
        pair_scores = []
        for x in a:
            pair_scores.append(max(_field_similarity(x, y, semantic=True) for y in b))
        for y in b:
            pair_scores.append(max(_field_similarity(y, x, semantic=True) for x in a))
        return sum(pair_scores) / len(pair_scores)

    def color_similarity(self, product: ProductAttributes, candidate: ProductAttributes) -> float:
        a, b = _norm(product.color), _norm(candidate.color)
        if not a or not b:
            return 0.0
        return _field_similarity(a, b, semantic=True)

    def product_text_similarity(self, product: ProductAttributes, candidate: ProductAttributes) -> float:
        # Title carries the most useful identity signal. Description is useful
        # as supporting evidence but marketplace descriptions are noisy.
        title = _field_similarity(product.product_name, candidate.product_name, semantic=True)
        if product.description and candidate.description:
            desc = _field_similarity(product.description, candidate.description, semantic=True)
            return 0.75 * title + 0.25 * desc
        return title

    # -------------------------- cost evidence ---------------------------

    @staticmethod
    def _get_number(obj: Any, *names: str) -> Optional[float]:
        for name in names:
            value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
            try:
                if value is not None:
                    value = float(value)
                    if math.isfinite(value) and value >= 0:
                        return value
            except (TypeError, ValueError):
                pass
        return None

    def cost_similarity(self, product: ProductAttributes, candidate: ProductAttributes) -> Tuple[float, float]:
        """Return (material-cost similarity, total/product-cost similarity).

        Marketplace rows normally contain retail price rather than production
        costs, so absent cost data is intentionally *not* treated as a zero.
        """
        p_material = self._get_number(product, "material_cost")
        c_material = self._get_number(candidate, "material_cost")
        p_total = self._get_number(product, "production_cost", "base_cost")
        c_total = self._get_number(candidate, "production_cost", "base_cost")

        def ratio(a, b):
            if a is None or b is None or max(a, b) <= 0:
                return 0.0
            return max(0.0, 1.0 - abs(a - b) / max(a, b))

        return ratio(p_material, c_material), ratio(p_total, c_total)

    # ------------------------- final scoring ----------------------------

    def _known(self, product: ProductAttributes, candidate: ProductAttributes) -> Dict[str, bool]:
        return {
            "product_type": bool(product.product_type and candidate.product_type),
            "material": bool(product.material and candidate.material),
            "size": bool(product.size and candidate.size),
            "dimensions": bool(self._dimension_vector(product) and self._dimension_vector(candidate)) or bool(product.dimensions and candidate.dimensions),
            "features": bool(product.features and candidate.features),
            "color": bool(product.color and candidate.color),
            "text": bool(product.product_name and candidate.product_name),
        }

    @staticmethod
    def _obviously_unrelated(type_score: float, text_score: float, product: ProductAttributes, candidate: ProductAttributes) -> bool:
        # Only reject when BOTH structural identity and textual evidence are
        # poor. A different material alone is never enough to reject.
        if type_score >= 0.58:
            return False
        if text_score >= 0.48:
            return False

        # If names have no meaningful token overlap and types disagree, the
        # candidate is very likely unrelated.
        name_overlap = _token_jaccard(product.product_name, candidate.product_name, remove_stopwords=True)
        return name_overlap < 0.08

    def _classify_level(self, score: float, type_score: float, material_score: float) -> str:
        if score >= 0.74 and type_score >= 0.78:
            if material_score >= 0.75:
                return "PRIMARY"
            return "RELATED"
        if score >= 0.60:
            return "SECONDARY"
        if score >= self.minimum_score:
            return "RELATED"
        return "WEAK"

    def _evidence_fields(self, product: ProductAttributes, candidate: ProductAttributes, scores: Dict[str, float], known: Dict[str, bool]) -> List[str]:
        labels = {
            "product_type": "product type",
            "material": "material",
            "size": "size",
            "dimensions": "dimensions",
            "features": "features/craft",
            "color": "color",
            "text": "product name/text",
        }
        return [labels[k] for k in scores if known[k] and scores[k] >= 0.55]

    def _reason(self, scores: Dict[str, float], level: str) -> str:
        parts = []
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        for field, value in ordered:
            if value >= 0.70:
                parts.append(f"strong {field.replace('_', ' ')}")
            elif value >= 0.50:
                parts.append(f"moderate {field.replace('_', ' ')}")
        if not parts:
            return "weak multi-field evidence"
        prefix = {
            "PRIMARY": "strong multi-field comparable",
            "SECONDARY": "good related comparable",
            "RELATED": "related product with useful market evidence",
            "WEAK": "weak evidence",
        }.get(level, "related product")
        return prefix + ": " + ", ".join(parts[:4])

    def calculate_similarity(self, product: ProductAttributes, candidate: ProductAttributes) -> SimilarityBreakdown:
        scores = {
            "product_type": self.product_type_similarity(product, candidate),
            "material": self.material_similarity(product, candidate),
            "size": self.size_similarity(product, candidate),
            "dimensions": self.dimensions_similarity(product, candidate),
            "features": self.features_similarity(product, candidate),
            "color": self.color_similarity(product, candidate),
            "text": self.product_text_similarity(product, candidate),
        }
        known = self._known(product, candidate)

        # Adaptive weighted average: unavailable fields disappear from the
        # denominator. This is the key fix for previous 0-score failures.
        numerator = sum(scores[k] * self.WEIGHTS[k] for k in scores if known[k])
        denominator = sum(self.WEIGHTS[k] for k in scores if known[k])
        total = numerator / denominator if denominator else 0.0

        # Economic evidence is deliberately a small *additional* signal.
        # Retail price is NOT treated as production cost. Only actual cost
        # fields supplied by the source are compared.
        material_cost_score, product_cost_score = self.cost_similarity(product, candidate)
        cost_values = []
        if product.material_cost is not None and candidate.material_cost is not None:
            cost_values.append((material_cost_score, 0.035))
        if (product.production_cost is not None or product.base_cost is not None) and (candidate.production_cost is not None or candidate.base_cost is not None):
            cost_values.append((product_cost_score, 0.035))
        if cost_values:
            cost_total = sum(v * w for v, w in cost_values) / sum(w for _, w in cost_values)
            total = 0.94 * total + 0.06 * cost_total

        # Small consistency bonuses/penalties. These use relationships already
        # visible in the fields and do not replace the weighted score.
        if scores["product_type"] >= 0.78 and scores["text"] >= 0.55:
            total += 0.025
        if scores["product_type"] >= 0.75 and scores["material"] >= 0.75:
            total += 0.025
        elif scores["product_type"] >= 0.75 and scores["material"] < 0.35 and known["material"]:
            # Same product but very different material: still usable, but not
            # as strong as a same-material comparable.
            total -= 0.035

        total = max(0.0, min(1.0, total))

        if self._obviously_unrelated(scores["product_type"], scores["text"], product, candidate):
            total = 0.0
            level = "EXCLUDED"
        else:
            level = self._classify_level(total, scores["product_type"], scores["material"])
            if level == "WEAK":
                # Keep WEAK out of pricing comparables, but do not confuse a
                # weak match with a zero mathematical similarity.
                total = 0.0
                level = "EXCLUDED"

        material_cost, product_cost = self.cost_similarity(product, candidate)
        evidence = self._evidence_fields(product, candidate, scores, known)
        reason = self._reason(scores, level)

        return SimilarityBreakdown(
            product_type=round(scores["product_type"], 4),
            material=round(scores["material"], 4),
            size=round(scores["size"], 4),
            dimensions=round(scores["dimensions"], 4),
            features=round(scores["features"], 4),
            color=round(scores["color"], 4),
            text=round(scores["text"], 4),
            total_score=round(total, 4),
            match_level=level,
            material_cost_similarity=round(material_cost, 4),
            product_cost_similarity=round(product_cost, 4),
            evidence_fields=evidence,
            comparable_reason=reason,
        )

    # -------------------------- conversion ------------------------------

    @staticmethod
    def candidate_to_attributes(candidate: Dict[str, Any]) -> ProductAttributes:
        if not isinstance(candidate, dict):
            raise TypeError("candidate must be a dictionary")

        features = candidate.get("features", [])
        if isinstance(features, str):
            features = [x.strip() for x in re.split(r"[,;|]", features) if x.strip()]

        name = candidate.get("title", candidate.get("product_name", "")) or ""
        supplied_type = str(candidate.get("product_type", "") or "").strip()
        classified_type = product_type_classifier.classify(name) if name else ""

        # If the title has an explicit type, prefer it. Otherwise use the
        # supplied structured type, then optional LLM classification.
        final_type = classified_type or supplied_type

        return ProductAttributes(
            product_id=candidate.get("product_id"),
            product_name=name,
            product_type=final_type,
            material=candidate.get("material", ""),
            additional_materials=candidate.get("additional_materials", []),
            color=candidate.get("color", ""),
            size=candidate.get("size", ""),
            dimensions_length=candidate.get("dimensions_length"),
            dimensions_width=candidate.get("dimensions_width"),
            dimensions_height=candidate.get("dimensions_height"),
            dimensions=candidate.get("dimensions", ""),
            features=features,
            description=candidate.get("description", ""),
            material_cost=candidate.get("material_cost"),
            labour_cost=candidate.get("labour_cost"),
            production_cost=candidate.get("production_cost"),
            base_cost=candidate.get("base_cost"),
        )

    def marketplace_listing_to_candidate(self, listing: Dict[str, Any]) -> ProductAttributes:
        candidate = self.candidate_to_attributes(listing)
        candidate._marketplace = listing.get("marketplace", "")
        candidate._price = listing.get("price")
        candidate._listing_id = listing.get("listing_id")
        candidate._url = listing.get("url")
        return candidate

    # ---------------------------- ranking --------------------------------

    def find_similar_products(self, product: ProductAttributes, candidates: Iterable[Dict[str, Any] | ProductAttributes], top_k: int = 10) -> List[SimilarProduct]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        results: List[SimilarProduct] = []
        for raw in candidates:
            try:
                candidate = raw if isinstance(raw, ProductAttributes) else self.candidate_to_attributes(raw)
                breakdown = self.calculate_similarity(product, candidate)
                if breakdown.match_level == "EXCLUDED" or breakdown.total_score < self.minimum_score:
                    continue

                price = getattr(candidate, "_price", None)
                if price is None:
                    # Direct database candidates may call the field price.
                    price = getattr(candidate, "price", None)
                if price is None:
                    continue
                price = float(price)
                if not math.isfinite(price) or price <= 0:
                    continue

                results.append(SimilarProduct(
                    marketplace=getattr(candidate, "_marketplace", ""),
                    title=candidate.product_name,
                    price=price,
                    similarity_score=breakdown.total_score,
                    match_level=breakdown.match_level,
                    listing_id=getattr(candidate, "_listing_id", None),
                    url=getattr(candidate, "_url", None),
                    product_type=candidate.product_type or None,
                    material=candidate.material or None,
                    color=candidate.color or None,
                    size=candidate.size or None,
                    dimensions=candidate.dimensions or None,
                    features=candidate.features or [],
                    breakdown=breakdown,
                ))
            except Exception as exc:
                logger.warning("Skipping similarity candidate: %s", exc)

        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]

    def match_marketplace_listings(self, product: ProductAttributes, listings: Sequence[Dict[str, Any]], top_k: int = 10) -> List[SimilarProduct]:
        candidates = []
        for listing in listings:
            try:
                candidates.append(self.marketplace_listing_to_candidate(listing))
            except Exception as exc:
                logger.warning("Invalid marketplace listing: %s", exc)
        return self.find_similar_products(product, candidates, top_k=top_k)


# ---------------------------------------------------------------------------
# MODULE-LEVEL COMPATIBILITY HELPERS
# ---------------------------------------------------------------------------

def candidate_to_attributes(candidate: Dict[str, Any]) -> ProductAttributes:
    return SimilarityEngine.candidate_to_attributes(candidate)


def find_similar_products(product: ProductAttributes | Dict[str, Any], candidates: Iterable[Dict[str, Any] | ProductAttributes], top_k: int = 10) -> List[SimilarProduct]:
    if isinstance(product, dict):
        product = ProductAttributes(
            product_id=product.get("product_id"),
            product_name=product.get("product_name", product.get("title", "")),
            product_type=product.get("product_type", ""),
            material=product.get("material", ""),
            additional_materials=product.get("additional_materials", []),
            color=product.get("color", ""),
            size=product.get("size", ""),
            dimensions_length=product.get("dimensions_length"),
            dimensions_width=product.get("dimensions_width"),
            dimensions_height=product.get("dimensions_height"),
            dimensions=product.get("dimensions", ""),
            features=product.get("features", []),
            description=product.get("description", ""),
            material_cost=product.get("material_cost"),
            labour_cost=product.get("labour_cost"),
            production_cost=product.get("production_cost"),
            base_cost=product.get("base_cost"),
        )
    return SimilarityEngine().find_similar_products(product, candidates, top_k=top_k)


def match_marketplace_listings(product: ProductAttributes | Dict[str, Any], listings: Sequence[Dict[str, Any]], top_k: int = 10) -> List[SimilarProduct]:
    if isinstance(product, dict):
        product = ProductAttributes(
            product_name=product.get("product_name", product.get("title", "")),
            product_type=product.get("product_type", ""),
            material=product.get("material", ""),
            color=product.get("color", ""),
            size=product.get("size", ""),
            dimensions_length=product.get("dimensions_length"),
            dimensions_width=product.get("dimensions_width"),
            dimensions_height=product.get("dimensions_height"),
            dimensions=product.get("dimensions", ""),
            features=product.get("features", []),
            description=product.get("description", ""),
            material_cost=product.get("material_cost"),
            labour_cost=product.get("labour_cost"),
            production_cost=product.get("production_cost"),
            base_cost=product.get("base_cost"),
        )
    return SimilarityEngine().match_marketplace_listings(product, listings, top_k=top_k)


# ---------------------------------------------------------------------------
# SELF TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    artisan = ProductAttributes(
        product_id="ART-000001",
        product_name="Handmade Bamboo Basket",
        product_type="Basket",
        material="Bamboo",
        color="Natural Brown",
        size="Medium",
        dimensions_length=30,
        dimensions_width=20,
        features=["Handwoven", "Traditional weaving", "Natural bamboo", "Round shape", "Handcrafted"],
        description="Handmade round bamboo basket for home storage and fruit.",
        material_cost=350,
        production_cost=700,
    )

    listings = [
        {"marketplace": "Amazon", "title": "SAI BALAJI Natural Bamboo Medium Round Basket", "price": 699,
         "product_type": "Basket", "material": "Bamboo", "size": "Medium",
         "features": ["Handwoven", "Round basket", "Natural bamboo"], "color": "Natural", "url": "https://example.com/1"},
        {"marketplace": "Amazon", "title": "NISKANET Natural Bamboo Medium Round Basket", "price": 599,
         "product_type": "Basket", "material": "Bamboo", "size": "Medium",
         "features": ["Bamboo basket", "Handcrafted"], "material_cost": 320, "production_cost": 680, "url": "https://example.com/2"},
        {"marketplace": "Gaatha", "title": "Handmade From Bamboo | Fruit Basket", "price": 625,
         "product_type": "Basket", "material": "Bamboo", "size": "Medium",
         "features": ["Handmade", "Fruit basket"], "dimensions": "23 x 20 x 8 cm", "url": "https://example.com/3"},
        {"marketplace": "IndiaHandmade", "title": "Handmade Cane Storage Basket", "price": 799,
         "product_type": "Basket", "material": "Cane", "size": "Medium",
         "features": ["Handwoven", "Storage"], "url": "https://example.com/4"},
        {"marketplace": "IndiaHandmade", "title": "Handmade Cane Stool", "price": 1199,
         "product_type": "Stool", "material": "Cane", "features": ["Handmade", "Cane work"], "url": "https://example.com/5"},
        {"marketplace": "Gaatha", "title": "Handmade Brass Diya", "price": 450,
         "product_type": "Diya", "material": "Brass", "features": ["Traditional"], "url": "https://example.com/6"},
    ]

    engine = SimilarityEngine()
    results = engine.match_marketplace_listings(artisan, listings, top_k=10)

    print("\nTOP SIMILAR MARKETPLACE PRODUCTS")
    print("=" * 90)
    for i, result in enumerate(results, 1):
        b = result.breakdown
        print(f"{i}. {result.title} | ₹{result.price:.0f} | score={result.similarity_score:.4f} | {result.match_level}")
        print(f"   type={b.product_type:.2f} material={b.material:.2f} size={b.size:.2f} dimensions={b.dimensions:.2f} features={b.features:.2f} color={b.color:.2f} text={b.text:.2f}")
        print(f"   evidence: {', '.join(b.evidence_fields or [])}")
        print(f"   reason: {b.comparable_reason}")
