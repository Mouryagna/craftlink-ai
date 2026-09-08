from typing import List, Dict, Optional, TypedDict
from pydantic import BaseModel, Field

# ==========================================
# Pricing Feature Contract (For Module 3)
# ==========================================

class PricingFeaturesInput(BaseModel):
    """Normalized feature vector ready for tabular ML / XGBoost dynamic pricing models."""
    craft_category: str = Field(description="Normalized craft sector")
    visual_complexity_score: int = Field(ge=1, le=5, description="Visual intricacy from Module 1")
    production_time_days: float = Field(description="Total artisan days invested")
    estimated_volume_cm3: Optional[float] = Field(default=None, description="Bounding volume for shipping/material scale")
    primary_material: str = Field(default="terracotta", description="Dominant raw material")
    materials_count: int = Field(default=1, description="Number of distinct raw materials used")
    estimated_height_cm: Optional[float] = None
    estimated_width_cm: Optional[float] = None
    region_state: str = Field(default="Telangana", description="Artisan location for minimum wage floor")
    base_material_cost_inr: Optional[float] = Field(default=None, description="Artisan self-reported material expense if stated")

# ==========================================
# Output Schemas (Dual JSON Contracts)
# ==========================================

class CatalogCopy(BaseModel):
    title: str
    tagline: str
    story_description: str
    bullet_features: List[str]
    tags: List[str]

class BilingualCatalog(BaseModel):
    english: CatalogCopy
    hindi: CatalogCopy

class ExtractedArtisanSlots(BaseModel):
    artisan_name: Optional[str] = None
    craft_category: str
    materials_used: List[str]
    production_time_days: int
    dimensions: Dict[str, Optional[float]]
    technique_details: str
    reconciled_from_vision: List[str] = Field(default_factory=list)

class VoiceUserDisplayOutput(BaseModel):
    status: str = "success"
    transcribed_text: str
    detected_language: str
    summary_slots: Dict[str, str]
    catalog_preview: Dict[str, str]
    ui_prompt: str = "Voice captured and catalog drafted! Confirm details to proceed to pricing."

class VoicePipelinePayloadOutput(BaseModel):
    raw_transcript: str
    detected_language: str
    slots: ExtractedArtisanSlots
    bilingual_catalog: BilingualCatalog
    pricing_features: PricingFeaturesInput  # <-- Direct hook for Module 3


# ==========================================
# LangGraph Working State
# ==========================================

class VoiceState(TypedDict, total=False):
    audio_path: str
    vision_context: Optional[Dict]
    raw_transcript: str
    detected_language: str
    extracted_slots: Dict
    user_display: Dict
    pipeline_payload: Dict