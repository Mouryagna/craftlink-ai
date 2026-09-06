"""
Similarity Engine for SIH26090 Dynamic Pricing.

Responsibilities
----------------
1. Compare an artisan product against marketplace products.
2. Perform multi-factor similarity matching.
3. Prioritize:
       - product type
       - material
       - size
       - dimensions
       - craft/features
       - color
       - product text
4. Produce a normalized similarity score in [0, 1].
5. Rank marketplace comparables.
6. Return structured comparable-product information.

This module does NOT:
- calculate prices
- train CatBoost
- decide ML readiness
- scrape marketplaces
- modify historical marketplace data
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import re
import logging


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------------------------

@dataclass
class ProductAttributes:
    """
    Normalized product representation used by the similarity engine.
    """

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

    def __post_init__(self) -> None:

        self.product_name = self._clean_text(
            self.product_name
        )

        self.product_type = self._clean_text(
            self.product_type
        )

        self.material = self._clean_text(
            self.material
        )

        self.color = self._clean_text(
            self.color
        )

        self.size = self._clean_text(
            self.size
        )

        self.description = self._clean_text(
            self.description
        )

        if self.additional_materials is None:
            self.additional_materials = []

        if self.features is None:
            self.features = []

        self.additional_materials = self._clean_list(
            self.additional_materials
        )

        self.features = self._clean_list(
            self.features
        )

        self.dimensions_length = self._clean_dimension(
            self.dimensions_length
        )

        self.dimensions_width = self._clean_dimension(
            self.dimensions_width
        )

        self.dimensions_height = self._clean_dimension(
            self.dimensions_height
        )

    @staticmethod
    def _clean_text(value: Any) -> str:

        if value is None:
            return ""

        return str(value).strip()

    @staticmethod
    def _clean_list(
        values: Iterable[Any],
    ) -> List[str]:

        result = []

        for value in values:

            value = str(value).strip()

            if value:
                result.append(value)

        return result

    @staticmethod
    def _clean_dimension(
        value: Any,
    ) -> Optional[float]:

        if value is None:
            return None

        try:

            value = float(value)

            if not math.isfinite(value):
                return None

            if value <= 0:
                return None

            return value

        except (TypeError, ValueError):

            return None


@dataclass
class SimilarityBreakdown:
    """
    Individual similarity components.
    """

    product_type: float
    material: float
    size: float
    dimensions: float
    features: float
    color: float
    text: float

    total_score: float

    match_level: str

    def to_dict(self) -> Dict[str, Any]:

        return asdict(self)


@dataclass
class SimilarProduct:
    """
    A marketplace product matched against the artisan product.
    """

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

        data = asdict(self)

        return data


# ---------------------------------------------------------------------------
# SIMILARITY ENGINE
# ---------------------------------------------------------------------------

class SimilarityEngine:
    """
    Multi-factor similarity engine.

    Matching hierarchy
    ------------------

    A:
        Same product type
        + same material
        + similar size/dimensions
        + similar craft features

        -> strongest comparable

    B:
        Same product type
        + same material
        + slightly different size

        -> strong comparable

    C:
        Same product type
        + different material

        -> fallback comparable with penalty

    D:
        Different product type

        -> excluded
    """

    # -----------------------------------------------------------------------
    # WEIGHTS
    # -----------------------------------------------------------------------

    WEIGHTS = {
        "product_type": 0.25,
        "material": 0.20,
        "size": 0.10,
        "dimensions": 0.10,
        "features": 0.15,
        "color": 0.05,
        "text": 0.15,
    }

    # -----------------------------------------------------------------------
    # THRESHOLDS
    # -----------------------------------------------------------------------

    MINIMUM_MATCH_SCORE = 0.45

    STRONG_MATCH_SCORE = 0.75

    MEDIUM_MATCH_SCORE = 0.60

    DIMENSION_TOLERANCE = 0.25

    # -----------------------------------------------------------------------
    # INITIALIZATION
    # -----------------------------------------------------------------------

    def __init__(
        self,
        minimum_score: float = MINIMUM_MATCH_SCORE,
    ) -> None:

        if not 0 <= minimum_score <= 1:

            raise ValueError(
                "minimum_score must be between 0 and 1"
            )

        self.minimum_score = minimum_score

    # -----------------------------------------------------------------------
    # TEXT NORMALIZATION
    # -----------------------------------------------------------------------

    @staticmethod
    def normalize_text(
        text: Optional[str],
    ) -> str:

        if not text:

            return ""

        text = str(text).lower()

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # -----------------------------------------------------------------------
    # TOKENIZATION
    # -----------------------------------------------------------------------

    @classmethod
    def tokenize(
        cls,
        text: Optional[str],
    ) -> set:

        normalized = cls.normalize_text(text)

        if not normalized:

            return set()

        return set(normalized.split())

    # -----------------------------------------------------------------------
    # STRING SIMILARITY
    # -----------------------------------------------------------------------

    @classmethod
    def text_similarity(
        cls,
        first: Optional[str],
        second: Optional[str],
    ) -> float:

        first_tokens = cls.tokenize(first)
        second_tokens = cls.tokenize(second)

        if not first_tokens or not second_tokens:

            return 0.0

        intersection = (
            first_tokens & second_tokens
        )

        union = (
            first_tokens | second_tokens
        )

        if not union:

            return 0.0

        return len(intersection) / len(union)

    # -----------------------------------------------------------------------
    # EXACT ATTRIBUTE MATCH
    # -----------------------------------------------------------------------

    @classmethod
    def attribute_similarity(
        cls,
        first: Optional[str],
        second: Optional[str],
    ) -> float:

        first = cls.normalize_text(first)
        second = cls.normalize_text(second)

        if not first or not second:

            return 0.0

        if first == second:

            return 1.0

        return cls.text_similarity(
            first,
            second,
        )

    # -----------------------------------------------------------------------
    # PRODUCT TYPE
    # -----------------------------------------------------------------------

    def product_type_similarity(
        self,
        product: ProductAttributes,
        candidate: ProductAttributes,
    ) -> float:

        first = self.normalize_text(
            product.product_type
        )

        second = self.normalize_text(
            candidate.product_type
        )

        if not first or not second:

            return 0.0

        if first == second:

            return 1.0

        # Product type mismatch is handled separately.
        return 0.0

    # -----------------------------------------------------------------------
    # MATERIAL
    # -----------------------------------------------------------------------

    def material_similarity(
        self,
        product: ProductAttributes,
        candidate: ProductAttributes,
    ) -> float:

        first = self.normalize_text(
            product.material
        )

        second = self.normalize_text(
            candidate.material
        )

        if not first or not second:

            return 0.0

        if first == second:

            return 1.0

        # Check additional materials.
        candidate_materials = {
            self.normalize_text(second)
        }

        for material in candidate.additional_materials:

            candidate_materials.add(
                self.normalize_text(material)
            )

        if first in candidate_materials:

            return 1.0

        # Partial semantic/token overlap.
        return self.text_similarity(
            first,
            second,
        )

    # -----------------------------------------------------------------------
    # SIZE
    # -----------------------------------------------------------------------

    def size_similarity(
        self,
        product: ProductAttributes,
        candidate: ProductAttributes,
    ) -> float:

        first = self.normalize_text(
            product.size
        )

        second = self.normalize_text(
            candidate.size
        )

        if not first or not second:

            return 0.0

        if first == second:

            return 1.0

        # Common size hierarchy.
        size_order = {
            "tiny": 1,
            "small": 2,
            "medium": 3,
            "large": 4,
            "extra large": 5,
            "xl": 5,
            "extra small": 1,
            "xs": 1,
            "s": 2,
            "m": 3,
            "l": 4,
        }

        first_value = size_order.get(first)
        second_value = size_order.get(second)

        if first_value is None or second_value is None:

            return self.text_similarity(
                first,
                second,
            )

        difference = abs(
            first_value - second_value
        )

        if difference == 0:

            return 1.0

        if difference == 1:

            return 0.75

        if difference == 2:

            return 0.40

        return 0.0

    # -----------------------------------------------------------------------
    # DIMENSIONS
    # -----------------------------------------------------------------------

    @staticmethod
    def _dimension_vector(
        product: ProductAttributes,
    ) -> Optional[Tuple[float, ...]]:

        values = (
            product.dimensions_length,
            product.dimensions_width,
            product.dimensions_height,
        )

        available = [
            value
            for value in values
            if value is not None
        ]

        if not available:

            return None

        return tuple(available)

    def dimensions_similarity(
        self,
        product: ProductAttributes,
        candidate: ProductAttributes,
    ) -> float:

        first = self._dimension_vector(
            product
        )

        second = self._dimension_vector(
            candidate
        )

        if first is None or second is None:

            # If structured dimensions are absent,
            # compare dimension text if available.
            return self.text_similarity(
                product.dimensions,
                candidate.dimensions,
            )

        if len(first) != len(second):

            return 0.0

        similarities = []

        for first_value, second_value in zip(
            first,
            second,
        ):

            denominator = max(
                abs(first_value),
                abs(second_value),
                1e-9,
            )

            relative_difference = (
                abs(first_value - second_value)
                / denominator
            )

            if relative_difference <= 0.10:

                score = 1.0

            elif relative_difference <= 0.25:

                score = 0.75

            elif relative_difference <= 0.40:

                score = 0.40

            else:

                score = 0.0

            similarities.append(score)

        if not similarities:

            return 0.0

        return sum(similarities) / len(
            similarities
        )

    # -----------------------------------------------------------------------
    # FEATURES
    # -----------------------------------------------------------------------

    def features_similarity(
        self,
        product: ProductAttributes,
        candidate: ProductAttributes,
    ) -> float:

        first = {
            self.normalize_text(feature)
            for feature in product.features
            if self.normalize_text(feature)
        }

        second = {
            self.normalize_text(feature)
            for feature in candidate.features
            if self.normalize_text(feature)
        }

        if not first or not second:

            return 0.0

        intersection = first & second
        union = first | second

        if not union:

            return 0.0

        return len(intersection) / len(union)

    # -----------------------------------------------------------------------
    # COLOR
    # -----------------------------------------------------------------------

    def color_similarity(
        self,
        product: ProductAttributes,
        candidate: ProductAttributes,
    ) -> float:

        return self.attribute_similarity(
            product.color,
            candidate.color,
        )

    # -----------------------------------------------------------------------
    # PRODUCT TEXT
    # -----------------------------------------------------------------------

    def product_text_similarity(
        self,
        product: ProductAttributes,
        candidate: ProductAttributes,
    ) -> float:

        first_text = " ".join(
            [
                product.product_name,
                product.description,
            ]
        )

        second_text = " ".join(
            [
                candidate.product_name,
                candidate.description,
            ]
        )

        return self.text_similarity(
            first_text,
            second_text,
        )

    # -----------------------------------------------------------------------
    # MATCH LEVEL
    # -----------------------------------------------------------------------

    @staticmethod
    def determine_match_level(
        product_type_score: float,
        material_score: float,
        size_score: float,
        dimensions_score: float,
        features_score: float,
        total_score: float,
    ) -> str:

        # Different product types must never become comparables.
        if product_type_score < 1.0:

            return "EXCLUDED"

        # Strong primary comparable.
        if (
            product_type_score == 1.0
            and material_score >= 0.90
            and (
                size_score >= 0.75
                or dimensions_score >= 0.75
            )
            and features_score >= 0.50
        ):

            return "PRIMARY"

        # Same type/material but weaker size/features.
        if (
            product_type_score == 1.0
            and material_score >= 0.90
            and total_score >= 0.60
        ):

            return "SECONDARY"

        # Same product type, different material.
        if (
            product_type_score == 1.0
            and material_score < 0.90
            and total_score >= 0.45
        ):

            return "FALLBACK"

        return "WEAK"

    # -----------------------------------------------------------------------
    # SCORE ONE CANDIDATE
    # -----------------------------------------------------------------------

    def calculate_similarity(
        self,
        product: ProductAttributes,
        candidate: ProductAttributes,
    ) -> SimilarityBreakdown:

        product_type_score = (
            self.product_type_similarity(
                product,
                candidate,
            )
        )

        # ---------------------------------------------------------------
        # HARD EXCLUSION
        #
        # Different product types are not comparables.
        # ---------------------------------------------------------------

        if product_type_score < 1.0:

            return SimilarityBreakdown(
                product_type=product_type_score,
                material=0.0,
                size=0.0,
                dimensions=0.0,
                features=0.0,
                color=0.0,
                text=0.0,
                total_score=0.0,
                match_level="EXCLUDED",
            )

        material_score = (
            self.material_similarity(
                product,
                candidate,
            )
        )

        size_score = (
            self.size_similarity(
                product,
                candidate,
            )
        )

        dimensions_score = (
            self.dimensions_similarity(
                product,
                candidate,
            )
        )

        features_score = (
            self.features_similarity(
                product,
                candidate,
            )
        )

        color_score = (
            self.color_similarity(
                product,
                candidate,
            )
        )

        text_score = (
            self.product_text_similarity(
                product,
                candidate,
            )
        )

        # ---------------------------------------------------------------
        # WEIGHTED SCORE
        # ---------------------------------------------------------------

        total_score = (
            product_type_score
            * self.WEIGHTS["product_type"]
            +
            material_score
            * self.WEIGHTS["material"]
            +
            size_score
            * self.WEIGHTS["size"]
            +
            dimensions_score
            * self.WEIGHTS["dimensions"]
            +
            features_score
            * self.WEIGHTS["features"]
            +
            color_score
            * self.WEIGHTS["color"]
            +
            text_score
            * self.WEIGHTS["text"]
        )

        total_score = max(
            0.0,
            min(
                1.0,
                total_score,
            ),
        )

        match_level = self.determine_match_level(
            product_type_score=product_type_score,
            material_score=material_score,
            size_score=size_score,
            dimensions_score=dimensions_score,
            features_score=features_score,
            total_score=total_score,
        )

        return SimilarityBreakdown(
            product_type=round(
                product_type_score,
                4,
            ),
            material=round(
                material_score,
                4,
            ),
            size=round(
                size_score,
                4,
            ),
            dimensions=round(
                dimensions_score,
                4,
            ),
            features=round(
                features_score,
                4,
            ),
            color=round(
                color_score,
                4,
            ),
            text=round(
                text_score,
                4,
            ),
            total_score=round(
                total_score,
                4,
            ),
            match_level=match_level,
        )

    # -----------------------------------------------------------------------
    # CANDIDATE CONVERSION
    # -----------------------------------------------------------------------

    @staticmethod
    def candidate_to_attributes(
        candidate: Dict[str, Any],
    ) -> ProductAttributes:

        if not isinstance(candidate, dict):

            raise TypeError(
                "candidate must be a dictionary"
            )

        features = candidate.get(
            "features",
            [],
        )

        if isinstance(features, str):

            features = [
                item.strip()
                for item in features.split(",")
                if item.strip()
            ]

        return ProductAttributes(
            product_id=candidate.get(
                "product_id"
            ),

            product_name=candidate.get(
                "title",
                candidate.get(
                    "product_name",
                    "",
                ),
            ),

            product_type=candidate.get(
                "product_type",
                "",
            ),

            material=candidate.get(
                "material",
                "",
            ),

            additional_materials=candidate.get(
                "additional_materials",
                [],
            ),

            color=candidate.get(
                "color",
                "",
            ),

            size=candidate.get(
                "size",
                "",
            ),

            dimensions_length=candidate.get(
                "dimensions_length"
            ),

            dimensions_width=candidate.get(
                "dimensions_width"
            ),

            dimensions_height=candidate.get(
                "dimensions_height"
            ),

            dimensions=candidate.get(
                "dimensions"
            ),

            features=features,

            description=candidate.get(
                "description",
                "",
            ),
        )

    # -----------------------------------------------------------------------
    # RANK CANDIDATES
    # -----------------------------------------------------------------------

    def find_similar_products(
        self,
        product: ProductAttributes,
        candidates: Iterable[
            Dict[str, Any] | ProductAttributes
        ],
        top_k: int = 10,
    ) -> List[SimilarProduct]:

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero"
            )

        results: List[
            SimilarProduct
        ] = []

        for candidate_raw in candidates:

            try:

                if isinstance(
                    candidate_raw,
                    ProductAttributes,
                ):

                    candidate = candidate_raw

                elif isinstance(
                    candidate_raw,
                    dict,
                ):

                    candidate = (
                        self.candidate_to_attributes(
                            candidate_raw
                        )
                    )

                else:

                    logger.warning(
                        "Skipping unsupported candidate: %s",
                        type(candidate_raw).__name__,
                    )

                    continue

                breakdown = (
                    self.calculate_similarity(
                        product,
                        candidate,
                    )
                )

                # Different product type.
                if (
                    breakdown.match_level
                    == "EXCLUDED"
                ):

                    continue

                if (
                    breakdown.total_score
                    < self.minimum_score
                ):

                    continue

                if not hasattr(
                    candidate,
                    "_marketplace"
                ):

                    marketplace = ""

                else:

                    marketplace = (
                        candidate._marketplace
                    )

                # Candidate marketplace data is attached
                # by the database/marketplace layer.
                price = getattr(
                    candidate,
                    "_price",
                    None,
                )

                if price is None:

                    continue

                result = SimilarProduct(
                    marketplace=marketplace,
                    title=candidate.product_name,
                    price=float(price),

                    similarity_score=round(
                        breakdown.total_score,
                        4,
                    ),

                    match_level=(
                        breakdown.match_level
                    ),

                    listing_id=getattr(
                        candidate,
                        "_listing_id",
                        None,
                    ),

                    url=getattr(
                        candidate,
                        "_url",
                        None,
                    ),

                    product_type=(
                        candidate.product_type
                        or None
                    ),

                    material=(
                        candidate.material
                        or None
                    ),

                    color=(
                        candidate.color
                        or None
                    ),

                    size=(
                        candidate.size
                        or None
                    ),

                    dimensions=(
                        candidate.dimensions
                        or None
                    ),

                    features=(
                        candidate.features
                        or []
                    ),

                    breakdown=breakdown,
                )

                results.append(result)

            except (
                TypeError,
                ValueError,
                AttributeError,
            ) as exc:

                logger.warning(
                    "Skipping similarity candidate: %s",
                    exc,
                )

        # Highest similarity first.
        results.sort(
            key=lambda item: (
                item.similarity_score,
                1
                if item.match_level == "PRIMARY"
                else 0,
                1
                if item.match_level == "SECONDARY"
                else 0,
            ),
            reverse=True,
        )

        return results[:top_k]

    # -----------------------------------------------------------------------
    # MARKETPLACE LISTING SUPPORT
    # -----------------------------------------------------------------------

    def marketplace_listing_to_candidate(
        self,
        listing: Dict[str, Any],
    ) -> ProductAttributes:

        candidate = (
            self.candidate_to_attributes(
                listing
            )
        )

        # Keep marketplace metadata attached to the
        # normalized object without polluting the
        # ProductAttributes public schema.
        candidate._marketplace = (
            listing.get(
                "marketplace",
                "",
            )
        )

        candidate._price = listing.get(
            "price"
        )

        candidate._listing_id = listing.get(
            "listing_id"
        )

        candidate._url = listing.get(
            "url"
        )

        return candidate

    # -----------------------------------------------------------------------
    # MARKETPLACE MATCHING
    # -----------------------------------------------------------------------

    def match_marketplace_listings(
        self,
        product: ProductAttributes,
        listings: Sequence[
            Dict[str, Any]
        ],
        top_k: int = 10,
    ) -> List[SimilarProduct]:

        candidates = []

        for listing in listings:

            try:

                candidate = (
                    self.marketplace_listing_to_candidate(
                        listing
                    )
                )

                candidates.append(candidate)

            except (
                TypeError,
                ValueError,
            ) as exc:

                logger.warning(
                    "Invalid marketplace listing: %s",
                    exc,
                )

        return self.find_similar_products(
            product=product,
            candidates=candidates,
            top_k=top_k,
        )

    # -----------------------------------------------------------------------
    # APPLICATION OUTPUT
    # -----------------------------------------------------------------------

    def build_comparable_market(
        self,
        product: ProductAttributes,
        listings: Sequence[
            Dict[str, Any]
        ],
        top_k: int = 10,
    ) -> Dict[str, Any]:

        matches = (
            self.match_marketplace_listings(
                product=product,
                listings=list(listings),
                top_k=top_k,
            )
        )

        return {
            "product_id": product.product_id,
            "comparable_count": len(matches),
            "comparables": [
                match.to_dict()
                for match in matches
            ],
        }


# ---------------------------------------------------------------------------
# DEFAULT ENGINE
# ---------------------------------------------------------------------------

similarity_engine = SimilarityEngine()


# ---------------------------------------------------------------------------
# APPLICATION HELPER
# ---------------------------------------------------------------------------

def find_similar_products(
    product: ProductAttributes,
    marketplace_listings: Sequence[
        Dict[str, Any]
    ],
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Application-level similarity API.
    """

    return similarity_engine.build_comparable_market(
        product=product,
        listings=marketplace_listings,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# SELF TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO
    )

    artisan_product = ProductAttributes(
        product_id="ART-000001",

        product_name=(
            "Handmade Bamboo Basket"
        ),

        product_type="Basket",

        material="Bamboo",

        color="Natural Brown",

        size="Medium",

        dimensions_length=30,
        dimensions_width=20,

        features=[
            "Handwoven",
            "Traditional weaving",
            "Natural bamboo",
            "Round shape",
            "Handcrafted",
        ],

        description=(
            "Handcrafted round bamboo basket "
            "made using traditional weaving."
        ),
    )

    marketplace_listings = [

        {
            "marketplace": "Amazon",

            "listing_id": "AMZ-001",

            "title": (
                "SAI BALAJI Natural Bamboo "
                "Medium Round Basket"
            ),

            "price": 699,

            "product_type": "Basket",

            "material": "Bamboo",

            "color": "Natural Brown",

            "size": "Medium",

            "features": [
                "Handwoven",
                "Traditional weaving",
                "Round shape",
                "Handcrafted",
            ],

            "description": (
                "Natural bamboo handwoven "
                "round basket."
            ),
        },

        {
            "marketplace": "Amazon",

            "listing_id": "AMZ-002",

            "title": (
                "Handwoven Bamboo Basket"
            ),

            "price": 159,

            "product_type": "Basket",

            "material": "Bamboo",

            "color": "Natural Brown",

            "size": "Medium",

            "features": [
                "Handwoven",
            ],

            "description": (
                "Handwoven bamboo basket."
            ),
        },

        {
            "marketplace": "Amazon",

            "listing_id": "AMZ-003",

            "title": (
                "NISKANET Natural Bamboo "
                "Medium Round Basket"
            ),

            "price": 599,

            "product_type": "Basket",

            "material": "Bamboo",

            "color": "Natural Brown",

            "size": "Medium",

            "features": [
                "Handcrafted",
                "Round shape",
            ],

            "description": (
                "Natural bamboo medium round basket."
            ),
        },

        {
            "marketplace": "Amazon",

            "listing_id": "AMZ-004",

            "title": (
                "Wicker Storage Basket"
            ),

            "price": 2999,

            "product_type": "Basket",

            "material": "Wicker",

            "color": "Brown",

            "size": "Large",

            "features": [
                "Storage",
            ],

            "description": (
                "Large wicker storage basket."
            ),
        },

        {
            "marketplace": "Amazon",

            "listing_id": "AMZ-005",

            "title": (
                "Bamboo Wall Decor"
            ),

            "price": 999,

            "product_type": "Wall Decor",

            "material": "Bamboo",

            "color": "Natural Brown",

            "size": "Medium",

            "features": [
                "Handcrafted",
            ],

            "description": (
                "Decorative bamboo wall hanging."
            ),
        },
    ]

    engine = SimilarityEngine(
        minimum_score=0.45
    )

    matches = engine.match_marketplace_listings(
        product=artisan_product,
        listings=marketplace_listings,
        top_k=10,
    )

    print("\n" + "=" * 70)
    print("SIMILARITY ENGINE TEST")
    print("=" * 70)

    print(
        f"\nFound {len(matches)} comparable products."
    )

    for index, match in enumerate(
        matches,
        start=1,
    ):

        print(
            f"\n{index}. {match.title}"
        )

        print(
            f"   Marketplace : "
            f"{match.marketplace}"
        )

        print(
            f"   Price       : "
            f"₹{match.price:.2f}"
        )

        print(
            f"   Score       : "
            f"{match.similarity_score:.4f}"
        )

        print(
            f"   Match level : "
            f"{match.match_level}"
        )

        if match.breakdown:

            print(
                f"   Type        : "
                f"{match.breakdown.product_type:.2f}"
            )

            print(
                f"   Material    : "
                f"{match.breakdown.material:.2f}"
            )

            print(
                f"   Size        : "
                f"{match.breakdown.size:.2f}"
            )

            print(
                f"   Dimensions  : "
                f"{match.breakdown.dimensions:.2f}"
            )

            print(
                f"   Features    : "
                f"{match.breakdown.features:.2f}"
            )

            print(
                f"   Color       : "
                f"{match.breakdown.color:.2f}"
            )

            print(
                f"   Text        : "
                f"{match.breakdown.text:.2f}"
            )

    print("\n" + "=" * 70)
    print("SIMILARITY ENGINE TEST PASSED")
    print("=" * 70)