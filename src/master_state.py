from typing import List, Optional, Dict, Any, TypedDict
from pydantic import BaseModel


class BilingualText(BaseModel):
    en: str
    hi: str


class CardDisplayData(BaseModel):
    hero_image_url: Optional[str]
    title: BilingualText
    description: BilingualText
    display_price: float
    currency_symbol: str = "₹"
    pricing_tier: str = "recommended"


class FinalCatalogRecord(BaseModel):
    product_id: str
    card_view: CardDisplayData
    technical_specs: Dict[str, Any]
    pricing_details: Dict[str, Any]


class MasterGraphState(TypedDict, total=False):
    # Ingestion parameters
    product_id: str
    audio_path: Optional[str]
    manual_text: Optional[str]
    image_paths: List[str]
    custom_bg_color: Optional[str]

    # Subgraph outputs & states
    voice_output: Dict[str, Any]
    vision_output: Dict[str, Any]
    pricing_input: Dict[str, Any]
    pricing_output: Dict[str, Any]

    # Storefront delivery payload
    final_catalog: Dict[str, Any]