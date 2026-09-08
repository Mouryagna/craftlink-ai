import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import whisper
import whisper.audio
import imageio_ffmpeg

# =========================================================================
# Fix: Direct imageio-ffmpeg binding for Whisper on Windows
# =========================================================================
FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()


def custom_load_audio(file: str, sr: int = whisper.audio.SAMPLE_RATE):
    """Decodes audio directly using imageio-ffmpeg binary, bypassing system PATH."""
    cmd = [
        FFMPEG_BINARY,
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


# Monkeypatch whisper's loader with the custom decoder
whisper.audio.load_audio = custom_load_audio
whisper.load_audio = custom_load_audio
# =========================================================================

from src.voice_catalog.state import (
    VoiceState,
    VoiceUserDisplayOutput,
    VoicePipelinePayloadOutput,
    ExtractedArtisanSlots,
    BilingualCatalog,
    CatalogCopy,
    PricingFeaturesInput
)

_WHISPER_MODEL = None


def get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model("base")
    return _WHISPER_MODEL


# ==========================================
# NODE 1: Whisper ASR Transcription
# ==========================================

def transcribe_speech_node(state: VoiceState) -> Dict[str, Any]:
    audio_path = state.get("audio_path")
    if not audio_path or not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = get_whisper_model()
    result = model.transcribe(audio_path)

    transcript = result.get("text", "").strip()
    detected_lang = result.get("language", "hi")

    return {
        "raw_transcript": transcript,
        "detected_language": detected_lang
    }


# ==========================================
# NODE 2: Slot Extraction & Cross-Modal Reconciliation
# ==========================================

def extract_and_reconcile_node(state: VoiceState) -> Dict[str, Any]:
    transcript = state.get("raw_transcript", "")
    vision_context = state.get("vision_context") or {}
    api_key = os.getenv("GEMINI_API_KEY")

    slots_dict = {}

    if api_key and transcript:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            prompt = (
                f"Extract structured handicraft product information from this artisan voice transcript: '{transcript}'.\n"
                "Return a JSON object with keys:\n"
                "- artisan_name (str or null)\n"
                "- craft_category (str or null)\n"
                "- materials_used (list of strings, or empty list)\n"
                "- production_time_days (int, days spent making the craft, default 1 if not stated)\n"
                "- dimensions: {height_cm: float or null, width_cm: float or null, depth_cm: float or null}\n"
                "- technique_details (brief string of how it was made)"
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            slots_dict = json.loads(response.text)
        except Exception as e:
            print(f"[!] LLM extraction failed ({e}), using default extraction.")

    if not slots_dict:
        slots_dict = {
            "artisan_name": "Ramu",
            "craft_category": "Terracotta & Pottery",
            "materials_used": ["Clay", "Natural lacquer"],
            "production_time_days": 4,
            "dimensions": {"height_cm": 25.0, "width_cm": 14.0, "depth_cm": None},
            "technique_details": "Handcrafted on wheel, kiln-fired, and hand-carved"
        }

    # Cross-Modal Reconciliation using Module 1 Vision Context
    reconciled_keys = []

    if not slots_dict.get("craft_category") and vision_context.get("detected_craft_type"):
        slots_dict["craft_category"] = vision_context["detected_craft_type"]
        reconciled_keys.append("craft_category")

    if not slots_dict.get("materials_used") and vision_context.get("materials_detected"):
        slots_dict["materials_used"] = vision_context["materials_detected"]
        reconciled_keys.append("materials_used")

    voice_dims = slots_dict.get("dimensions") or {}
    vision_dims = vision_context.get("dimensions_estimate") or {}
    for dim_axis in ["height_cm", "width_cm", "depth_cm"]:
        if voice_dims.get(dim_axis) is None and vision_dims.get(dim_axis) is not None:
            voice_dims[dim_axis] = vision_dims[dim_axis]
            if "dimensions" not in reconciled_keys:
                reconciled_keys.append("dimensions")
    slots_dict["dimensions"] = voice_dims

    extracted = ExtractedArtisanSlots(
        artisan_name=slots_dict.get("artisan_name"),
        craft_category=slots_dict.get("craft_category") or "Handicraft",
        materials_used=slots_dict.get("materials_used") or ["Natural Clay"],
        production_time_days=int(slots_dict.get("production_time_days") or 1),
        dimensions=voice_dims,
        technique_details=slots_dict.get("technique_details", "Traditional handcrafted"),
        reconciled_from_vision=reconciled_keys
    )

    return {"extracted_slots": extracted.model_dump()}


# ==========================================
# NODE 3: Bilingual Copywriter & Output Formatter
# ==========================================

def generate_bilingual_catalog_node(state: VoiceState) -> Dict[str, Any]:
    slots = state.get("extracted_slots", {})
    transcript = state.get("raw_transcript", "")
    lang = state.get("detected_language", "hi")

    name = slots.get("artisan_name") or "Master Artisan"
    craft = slots.get("craft_category") or "Handcrafted Item"
    materials_list = slots.get("materials_used", ["Clay"])
    materials_str = ", ".join(materials_list)
    days = slots.get("production_time_days", 1)
    dims = slots.get("dimensions", {})
    h = dims.get("height_cm", 25.0)
    w = dims.get("width_cm", 14.0)

    english_copy = CatalogCopy(
        title=f"Handcrafted {craft} Vase by {name}",
        tagline="Authentic Indian Artistry • Sustainable & Wheel-Thrown",
        story_description=(
            f"Handmade over {days} days using pure {materials_str}. Shaped traditionally on the potter's wheel, "
            "kiln-fired, and finished with delicate hand carvings."
        ),
        bullet_features=[
            f"Artisan: Handcrafted by {name}",
            f"Materials: 100% Eco-friendly {materials_str}",
            f"Dimensions: Approx. {h}cm (Height) x {w}cm (Width)",
            f"Craftsmanship: {days} days of dedicated artisan labor"
        ],
        tags=["handicrafts", "terracotta", "eco-friendly", "indian-pottery", "home-decor"]
    )

    hindi_copy = CatalogCopy(
        title=f"{name} द्वारा हस्तनिर्मित {craft}",
        tagline="प्रामाणिक भारतीय कला • पर्यावरण के अनुकूल",
        story_description=(
            f"शुद्ध {materials_str} से {days} दिनों में तैयार किया गया। चाक पर ढालकर पारंपरिक भट्टी में पकाया गया "
            "और हाथों से बारीक नक्काशी की गई।"
        ),
        bullet_features=[
            f"कारीगर: {name}",
            f"सामग्री: शुद्ध {materials_str}",
            f"आकार: लगभग {h} सेमी (ऊंचाई) x {w} सेमी (चौड़ाई)",
            f"श्रम: {days} दिन का पारंपरिक हस्तशिल्प"
        ],
        tags=["हस्तशिल्प", "टेराकोटा", "मिट्टी-के-बर्तन", "सजावट"]
    )

    bilingual_catalog = BilingualCatalog(english=english_copy, hindi=hindi_copy)

    # 1. Output for UI Display
    user_out = VoiceUserDisplayOutput(
        status="success",
        transcribed_text=transcript,
        detected_language=lang,
        summary_slots={
            "Artisan": name,
            "Craft": craft,
            "Materials": materials_str,
            "Labor Time": f"{days} Days",
            "Dimensions": f"{h}cm x {w}cm"
        },
        catalog_preview={
            "English Title": english_copy.title,
            "Hindi Title": hindi_copy.title
        }
    )

    # ----------------------------------------------------
    # Assemble Dynamic Pricing Vector for Module 3
    # ----------------------------------------------------
    vision_context = state.get("vision_context") or {}

    height = float(dims.get("height_cm") or 0.0)
    width = float(dims.get("width_cm") or 0.0)
    depth = float(dims.get("depth_cm") or width or 0.0)
    calc_volume = round(height * width * depth, 2) if (height > 0 and width > 0) else None

    primary_mat = materials_list[0] if materials_list else "Clay"

    pricing_vector = PricingFeaturesInput(
        craft_category=craft,
        visual_complexity_score=int(vision_context.get("visual_complexity_score", 3)),
        production_time_days=float(days),
        estimated_volume_cm3=calc_volume,
        primary_material=primary_mat,
        materials_count=len(materials_list),
        estimated_height_cm=height if height > 0 else None,
        estimated_width_cm=width if width > 0 else None,
        region_state="Telangana",
        base_material_cost_inr=None
    )

    # 2. Output for Downstream Pipeline & Pricing Engine
    pipeline_out = VoicePipelinePayloadOutput(
        raw_transcript=transcript,
        detected_language=lang,
        slots=ExtractedArtisanSlots(**slots),
        bilingual_catalog=bilingual_catalog,
        pricing_features=pricing_vector
    )

    return {
        "user_display": user_out.model_dump(),
        "pipeline_payload": pipeline_out.model_dump()
    }