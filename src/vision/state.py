from pathlib import Path
from typing import List, Dict, Optional, TypedDict
from pydantic import BaseModel, Field


# ==========================================
# 1. Output Schemas (JSON-Exportable via Pydantic)
# ==========================================

class UserDisplayOutput(BaseModel):
    """Payload formatted specifically for mobile app UI rendering."""
    status: str = Field(default="success", description="Status code for the UI")
    hero_image_url: str = Field(description="Primary enhanced studio photo path/URL")
    gallery_image_urls: List[str] = Field(default_factory=list, description="All enhanced angle shots")
    visual_badge: str = Field(description="Clean, low-literacy label e.g. 'Terracotta Pot • High Detail (4/5)'")
    detected_colors: List[str] = Field(default_factory=list, description="Visual color palette tags")
    ui_message: str = Field(
        default="Photos enhanced successfully! Tap the microphone to tell us about your craft.",
        description="Friendly instruction for the artisan"
    )


class DimensionsEstimate(BaseModel):
    height_cm: Optional[float] = None
    width_cm: Optional[float] = None
    depth_cm: Optional[float] = None
    confidence: str = Field(default="estimated", description="Confidence level of scale estimation")


class PipelinePayloadOutput(BaseModel):
    """Strict data payload passed downstream to LangGraph parent state, NLP reconciler, and Pricing engine."""
    primary_image_path: str
    gallery_image_paths: List[str]
    visual_complexity_score: int = Field(ge=1, le=5, description="Integer rating from 1 to 5 for XGBoost")
    detected_craft_type: str = Field(description="Backup craft classification if omitted in voice")
    materials_detected: List[str] = Field(default_factory=list, description="Backup materials if omitted in voice")
    color_palette: List[str] = Field(default_factory=list)
    dimensions_estimate: DimensionsEstimate


# ==========================================
# 2. LangGraph Subgraph State
# ==========================================

class VisionState(TypedDict, total=False):
    # Inputs
    raw_image_paths: List[str]

    # Intermediate working artifacts
    enhanced_image_paths: List[str]
    contour_count: int
    edge_density_pct: float

    # Final Output Contracts
    user_display: Dict  # Serialized UserDisplayOutput JSON
    pipeline_payload: Dict  # Serialized PipelinePayloadOutput JSON