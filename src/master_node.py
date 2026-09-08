from pathlib import Path
from typing import Dict, Any

from src.master_state import MasterGraphState, FinalCatalogRecord, CardDisplayData, BilingualText
from src.vision.graph import vision_subgraph
from src.voice_catalog.graph import nlp_subgraph
from src.pricing.graph import pricing_subgraph


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

    # Labor & materials: extracted by Gemini from audio transcription/manual text
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
# Node 5: Storefront Card Assembly (Bilingual UI Data)
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

    # Convert local studio path to web-served asset URL
    hero_path = studio_images.get("hero_studio_path")
    hero_url = f"/outputs/{Path(hero_path).name}" if hero_path else None

    # Construct the exact bilingual mobile card structure
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
            "studio_images": studio_images.get("items", [])
        },
        pricing_details=pricing
    )

    return {"final_catalog": record.model_dump()}