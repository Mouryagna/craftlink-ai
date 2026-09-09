import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from src.master_state import MasterGraphState, FinalCatalogRecord, CardDisplayData, BilingualText
from src.vision.graph import vision_subgraph
from src.voice_catalog.graph import nlp_subgraph
from src.pricing.graph import pricing_subgraph

# Base URL for static hosting (e.g. "http://localhost:8000" or production URL)
BASE_URL = os.getenv("BACKEND_BASE_URL", "").rstrip("/")


def to_web_url(local_path_str: Optional[str]) -> Optional[str]:
    """
    Converts a local filesystem path into an accessible static web URL.
    Example input:  'E:\\craftlink-ai\\outputs\\ART-44B913\\ART-44B913_1_studio.jpg'
    Example output: '/outputs/ART-44B913/ART-44B913_1_studio.jpg'
    """
    if not local_path_str:
        return None

    p = Path(local_path_str)
    product_folder = p.parent.name
    filename = p.name

    relative_path = f"/outputs/{product_folder}/{filename}"
    return f"{BASE_URL}{relative_path}" if BASE_URL else relative_path


# -------------------------------------------------------------------------
# Node 1: Vision Subgraph Execution
# -------------------------------------------------------------------------
def run_vision_node(state: MasterGraphState) -> Dict[str, Any]:
    vision_input = {
        "product_id": state.get("product_id", "ART-000001"),
        "image_paths": state.get("image_paths", []),
        "custom_bg_color": state.get("custom_bg_color", "#FFFFFF")
    }
    result = vision_subgraph.invoke(vision_input)
    return {"vision_output": result.get("final_output", {})}


# -------------------------------------------------------------------------
# Node 2: Voice / NLP Subgraph Execution (Audio OR Manual Text)
# -------------------------------------------------------------------------
def run_voice_node(state: MasterGraphState) -> Dict[str, Any]:
    voice_input = {
        "product_id": state.get("product_id", "ART-000001"),
        "audio_path": state.get("audio_path"),
        "manual_text": state.get("manual_text")
    }
    result = nlp_subgraph.invoke(voice_input)
    return {"voice_output": result.get("final_output", {})}


# -------------------------------------------------------------------------
# Node 3: Modality Reconciliation & Pricing Vector Assembly
# -------------------------------------------------------------------------
def reconcile_features_node(state: MasterGraphState) -> Dict[str, Any]:
    voice = state.get("voice_output", {})
    vision = state.get("vision_output", {})

    prod_info = voice.get("product_information", {})
    pricing_feats = voice.get("pricing_input_features", {})
    v_analysis = vision.get("visual_analysis", {})

    # Craft type: NLP extraction takes precedence, fallback to vision analysis
    craft_type = pricing_feats.get("product_type") or v_analysis.get("detected_craft_type", "terracotta pottery")

    # Sizing & complexity: extracted directly from multi-image analysis
    size_category = str(v_analysis.get("size_category", "medium")).lower().strip()
    complexity_score = int(v_analysis.get("visual_complexity_score", 3))

    # Labor & materials: extracted by Gemini from audio transcription or manual text
    worked_hours = float(pricing_feats.get("time_worked_hours", 6.0))
    stated_cost = pricing_feats.get("material_cost")
    artisan_floor = prod_info.get("artisan_floor_price")

    pricing_payload = {
        "product_id": state.get("product_id", "ART-000001"),
        "craft_type": craft_type,
        "size_category": size_category,
        "time_worked_hours": worked_hours,
        "stated_material_cost": stated_cost,
        "artisan_floor_price": artisan_floor,
        "visual_complexity_score": complexity_score
    }
    return {"pricing_input": pricing_payload}


# -------------------------------------------------------------------------
# Node 4: Pricing Subgraph Execution (CatBoost + Guardrails)
# -------------------------------------------------------------------------
def run_pricing_node(state: MasterGraphState) -> Dict[str, Any]:
    pricing_in = state["pricing_input"]
    result = pricing_subgraph.invoke(pricing_in)
    return {"pricing_output": result.get("final_output", {})}


# -------------------------------------------------------------------------
# Node 5: Storefront Card Assembly (Bilingual UI Data & Web URL Conversion)
# -------------------------------------------------------------------------
def assemble_catalog_node(state: MasterGraphState) -> Dict[str, Any]:
    product_id = state.get("product_id", "ART-000001")
    voice = state.get("voice_output", {})
    vision = state.get("vision_output", {})
    pricing = state.get("pricing_output", {})

    catalog = voice.get("catalog", {})
    en_cat = catalog.get("english", {})
    hi_cat = catalog.get("hindi", {})

    v_analysis = vision.get("visual_analysis", {})
    studio_images = vision.get("processed_images", {})
    tiers = pricing.get("retail_tiers", {})
    recommended_price = float(tiers.get("recommended_price", 499.0))

    # 1. Convert hero studio path to web URL
    hero_local_path = studio_images.get("hero_studio_path")
    hero_url = to_web_url(hero_local_path)

    # 2. Convert all studio image gallery paths to web URLs
    raw_gallery_items: List[Dict[str, Any]] = studio_images.get("items", [])
    formatted_gallery: List[Dict[str, Any]] = []

    for item in raw_gallery_items:
        orig_filename = Path(item["original_path"]).name if item.get("original_path") else None
        original_url = f"{BASE_URL}/uploads/{product_id}/{orig_filename}" if (BASE_URL and orig_filename) else (f"/uploads/{product_id}/{orig_filename}" if orig_filename else None)

        formatted_gallery.append({
            "original_url": original_url,
            "upscaled_url": to_web_url(item.get("upscaled_path")),
            "cutout_url": to_web_url(item.get("cutout_path")),
            "studio_url": to_web_url(item.get("studio_path"))
        })

    # 3. Assemble frontend Card view structure
    card_data = CardDisplayData(
        hero_image_url=hero_url,
        title=BilingualText(
            en=en_cat.get("title", "Handcrafted Artisan Craft"),
            hi=hi_cat.get("title", "हस्तनिर्मित कारीगरी उत्पाद")
        ),
        description=BilingualText(
            en=en_cat.get("description", "Authentic handcrafted heritage product."),
            hi=hi_cat.get("description", "पारंपरिक हस्तनिर्मित शिल्प।")
        ),
        display_price=recommended_price,
        currency_symbol="₹",
        pricing_tier="recommended"
    )

    # 4. Create final catalog payload
    record = FinalCatalogRecord(
        product_id=product_id,
        card_view=card_data,
        technical_specs={
            "craft_type": pricing.get("craft_type") or v_analysis.get("detected_craft_type"),
            "primary_color": v_analysis.get("primary_color", "Natural"),
            "detected_colors": v_analysis.get("detected_colors", []),
            "dimensions": v_analysis.get("dimensions_estimate", {}),
            "visual_complexity_score": v_analysis.get("visual_complexity_score", 3),
            "surface_detailing": v_analysis.get("surface_detailing", ""),
            "bullet_points": {
                "en": en_cat.get("bullet_points", []),
                "hi": hi_cat.get("bullet_points", [])
            },
            "seo_keywords": {
                "en": en_cat.get("seo_keywords", []),
                "hi": hi_cat.get("seo_keywords", [])
            },
            "studio_images": formatted_gallery
        },
        pricing_details=pricing
    )

    return {"final_catalog": record.model_dump()}


# -------------------------------------------------------------------------
# Multi-Step Execution Helper (Step 2: Voice -> NLP -> Auto Pricing)
# -------------------------------------------------------------------------
def execute_nlp_and_pricing(
    product_id: str,
    vision_output: Dict[str, Any],
    audio_path: Optional[str],
    manual_text: Optional[str]
) -> Dict[str, Any]:
    """
    Executes Voice/Catalog subgraph, reconciles it with Vision, triggers
    CatBoost Pricing, and returns the final unified FinalCatalogRecord.
    """
    # 1. Run Voice Subgraph
    voice_input = {
        "product_id": product_id,
        "audio_path": audio_path,
        "manual_text": manual_text
    }
    voice_res = nlp_subgraph.invoke(voice_input)
    voice_output = voice_res.get("final_output", {})

    # 2. Reconcile Features for Pricing
    mock_state: MasterGraphState = {
        "product_id": product_id,
        "image_paths": [],
        "vision_output": vision_output,
        "voice_output": voice_output
    }
    pricing_payload = reconcile_features_node(mock_state)["pricing_input"]

    # 3. Auto-Trigger Pricing Subgraph
    pricing_res = pricing_subgraph.invoke(pricing_payload)
    pricing_output = pricing_res.get("final_output", {})

    # 4. Assemble identical Final Catalog
    mock_state["pricing_output"] = pricing_output
    final_record = assemble_catalog_node(mock_state)["final_catalog"]
    return final_record