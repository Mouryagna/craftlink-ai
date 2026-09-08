from typing import List, Dict, Optional, TypedDict
from pydantic import BaseModel, Field


# ==========================================
# 1. Product Information & Catalog Schemas
# ==========================================

class ProductInformation(BaseModel):
    product_type: str
    material: str
    craft_method: str
    production_time: str


class CatalogLanguage(BaseModel):
    title: str
    description: str
    bullet_points: List[str]
    seo_keywords: List[str]


class Catalog(BaseModel):
    english: CatalogLanguage
    hindi: CatalogLanguage


# ==========================================
# 2. Pricing Features Schema
# ==========================================

class DimensionsSchema(BaseModel):
    length: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    unit: str = "cm"


class PricingInputFeatures(BaseModel):
    product_id: str
    product_name: str
    product_type: str
    material: str
    additional_materials: List[str] = Field(default_factory=list)
    color: Optional[str] = None
    size: Optional[str] = None
    dimensions: DimensionsSchema = Field(default_factory=DimensionsSchema)
    features: List[str] = Field(default_factory=list)
    description: str
    material_cost: Optional[float] = None
    labour_cost: Optional[float] = None
    production_cost: Optional[float] = None
    time_worked_hours: Optional[float] = None


# ==========================================
# 3. Overall NLP Module Output Contract
# ==========================================

class NLPModuleOutput(BaseModel):
    product_id: str
    transcription: str
    translation: str
    product_information: ProductInformation
    catalog: Catalog
    pricing_input_features: PricingInputFeatures


# ==========================================
# 4. LangGraph Subgraph Working State
# ==========================================

class VoiceState(TypedDict, total=False):
    # Inputs
    product_id: str
    audio_path: str

    # Optional inputs from upstream or defaults
    vision_context: Optional[Dict]

    # Extracted fields
    transcription: str
    translation: str
    product_information: Dict
    catalog: Dict
    pricing_input_features: Dict

    # Final consolidated output matching the required contract
    final_output: Dict