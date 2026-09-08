from typing import List, Dict, Optional, TypedDict
from pydantic import BaseModel, Field


# ==========================================
# 1. Image Enhancement Output Schemas
# ==========================================

class ProcessedImageItem(BaseModel):
    original_path: str
    upscaled_path: Optional[str] = None
    cutout_path: Optional[str] = None
    studio_path: Optional[str] = None


class ProcessedImages(BaseModel):
    items: List[ProcessedImageItem] = Field(default_factory=list)
    hero_studio_path: Optional[str] = None
    background_color: str = "#FFFFFF"


# ==========================================
# 2. Visual Analysis Output Schemas
# ==========================================

class DimensionsEstimate(BaseModel):
    length: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    unit: str = "cm"


class VisualAnalysis(BaseModel):
    detected_craft_type: str
    primary_color: str
    detected_colors: List[str] = Field(default_factory=list)
    size_category: str = Field(description="'Small', 'Medium', or 'Large'")
    dimensions_estimate: DimensionsEstimate = Field(default_factory=DimensionsEstimate)
    visual_complexity_score: int = Field(default=3, ge=1, le=5)
    surface_detailing: str


# ==========================================
# 3. Vision Module Master Output Contract
# ==========================================

class VisionModuleOutput(BaseModel):
    product_id: str
    processed_images: ProcessedImages
    visual_analysis: VisualAnalysis


# ==========================================
# 4. LangGraph Subgraph Working State
# ==========================================

class VisionState(TypedDict, total=False):
    # Inputs
    product_id: str
    image_paths: List[str]  # Max 4 to 5 images
    custom_bg_color: Optional[str]  # e.g., "#FFFFFF", "#F4F1EA", or any hex

    # Intermediate / Processed data
    processed_images: Dict
    visual_analysis: Dict

    # Final consolidated output payload
    final_output: Dict