import os
import re
import json
import uuid
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# =========================================================================
# Whisper Conditional Import & Binding
# =========================================================================
try:
    import whisper
    import whisper.audio
    import imageio_ffmpeg

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

    whisper.audio.load_audio = custom_load_audio
    whisper.load_audio = custom_load_audio
except (ImportError, ModuleNotFoundError, Exception):
    whisper = None

from src.voice_catalog.state import (
    VoiceState,
    ProductInformation,
    Catalog,
    DimensionsSchema,
    PricingInputFeatures,
    NLPModuleOutput
)

_WHISPER_MODEL = None


def get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None and whisper is not None:
        _WHISPER_MODEL = whisper.load_model("base")
    return _WHISPER_MODEL


# ==========================================
# NODE 1: Ingestion (Manual Text OR Speech)
# ==========================================

def transcribe_node(state: VoiceState) -> Dict[str, Any]:
    """
    Handles dual input pathways:
    1. If user typed manual text -> routes directly to downstream cataloging.
    2. If user recorded audio -> passes through Whisper ASR (or Gemini fallback).
    """
    manual_text = state.get("manual_text", "").strip() if state.get("manual_text") else ""
    audio_path = state.get("audio_path")
    product_id = state.get("product_id") or f"ART-{uuid.uuid4().hex[:6].upper()}"

    # Path A: User provided manual input
    if manual_text:
        print(f"[-] Manual text received for product {product_id}. Bypassing Whisper.")
        return {
            "product_id": product_id,
            "transcription": manual_text
        }

    # Path B: User recorded audio
    if audio_path and Path(audio_path).exists():
        # Option 1: Local Whisper available
        if whisper is not None:
            print(f"[-] Transcribing voice note via local Whisper for product {product_id}...")
            try:
                model = get_whisper_model()
                result = model.transcribe(
                    audio_path,
                    language="hi",
                    initial_prompt="यह हस्तनिर्मित भारतीय शिल्पकला उत्पाद का विवरण, लागत और समय है।"
                )
                return {
                    "product_id": product_id,
                    "transcription": result.get("text", "").strip()
                }
            except Exception as e:
                print(f"[!] Local Whisper error: {e}. Falling back to Gemini.")

        # Option 2: Cloud Fallback using Gemini (Lightweight for Render)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            print(f"[-] Transcribing voice note via Gemini Flash for product {product_id}...")
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                uploaded_audio = client.files.upload(file=audio_path)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        uploaded_audio,
                        "Transcribe this artisan audio note verbatim in its spoken language (Hindi, vernacular, or English). Return only the transcribed text."
                    ]
                )
                return {
                    "product_id": product_id,
                    "transcription": response.text.strip()
                }
            except Exception as e:
                print(f"[!] Gemini audio transcription failed: {e}")

    # Path C: Fallback baseline if audio failed or was not provided
    print("[!] No audio or manual text provided. Using default artisan template.")
    return {
        "product_id": product_id,
        "transcription": "हमने यह लाल मिट्टी का बर्तन हाथ से चाक पर बनाया है। इसमें 6 घंटे का समय लगा और 150 रुपये की सामग्री लगी।"
    }


# ==========================================
# NODE 2: Translation & Bilingual Catalog Node
# ==========================================

def extract_and_catalog_node(state: VoiceState) -> Dict[str, Any]:
    transcription = state.get("transcription", "")
    api_key = os.getenv("GEMINI_API_KEY")

    extracted_payload = None

    if api_key and transcription:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            prompt = f"""You are an expert Indian artisan e-commerce cataloger.
Given this artisan's description (Hindi or English):
"{transcription}"

Return a valid JSON object matching this exact schema:
{{
  "translation": "Precise English translation of the transcription",
  "product_information": {{
    "product_type": "terracotta pottery | bamboo basket | wooden craft | handloom textile",
    "material": "Comma-separated list of raw materials used",
    "craft_method": "Artisanal technique and crafting methods used",
    "production_time": "Production duration string (e.g. '6 hours' or '2 days')",
    "time_worked_hours": float (total actual labor hours spent, default 6.0),
    "stated_material_cost": float or null (extracted cost in INR if stated, else null),
    "artisan_floor_price": float or null (minimum selling price if stated, else null)
  }},
  "catalog": {{
    "english": {{
      "title": "SEO-optimized e-commerce title",
      "description": "Rich artisanal narrative description",
      "bullet_points": [
        "4 detailed, compelling e-commerce highlight bullets"
      ],
      "seo_keywords": [
        "7 relevant search tags and keywords"
      ]
    }},
    "hindi": {{
      "title": "हिंदी में शीर्षक",
      "description": "हिंदी में विस्तृत विवरण",
      "bullet_points": [
        "4 विस्तृत और स्पष्ट बिंदु"
      ],
      "seo_keywords": [
        "7 प्रमुख खोज कीवर्ड"
      ]
    }}
  }}
}}"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            extracted_payload = json.loads(response.text)
        except Exception as e:
            print(f"[!] Gemini extraction failed ({e}), using default fallback.")

    # High quality domain fallback if offline or API failure
    if not extracted_payload:
        extracted_payload = {
            "translation": "We made this terracotta vessel by hand on the potter's wheel. It took 6 hours of labor and 150 rupees in raw materials.",
            "product_information": {
                "product_type": "terracotta pottery",
                "material": "Pure Riverbed Clay, Natural Terracotta",
                "craft_method": "Wheel-thrown & Sun-dried Kiln Fired",
                "production_time": "6 hours",
                "time_worked_hours": 6.0,
                "stated_material_cost": 150.0,
                "artisan_floor_price": 400.0
            },
            "catalog": {
                "english": {
                    "title": "Handcrafted Traditional Terracotta Clay Pot",
                    "description": "Elevate your home decor and festive rituals with this authentic wheel-thrown terracotta pot. Sourced from natural riverbed clay and fired in traditional kilns, it embodies centuries of heritage craftsmanship.",
                    "bullet_points": [
                        "AUTHENTIC CLAY: Made from 100% natural, unadulterated riverbed clay.",
                        "WHEEL-THROWN: Individually shaped on traditional potters' wheels.",
                        "ECO-FRIENDLY & SUSTAINABLE: Biodegradable, chemical-free artisanal finish.",
                        "CULTURAL HERITAGE: Direct from rural Indian pottery master craftsmen."
                    ],
                    "seo_keywords": [
                        "Terracotta Pot", "Clay Matka", "Handmade Pottery",
                        "Indian Handicrafts", "Diwali Decor", "Eco friendly planter", "Traditional Vessel"
                    ]
                },
                "hindi": {
                    "title": "हाथ से बना पारंपरिक टेराकोटा मिट्टी का बर्तन",
                    "description": "अपने घर और पूजा स्थल को इस प्रामाणिक हस्तनिर्मित टेराकोटा बर्तन से सजाएं। नदी की शुद्ध चिकनी मिट्टी से चाक पर निर्मित और पारंपरिक भट्टी में पकाया गया यह बर्तन भारतीय शिल्पकला का अनूठा उदाहरण है।",
                    "bullet_points": [
                        "प्राकृतिक नदी मिट्टी: 100% शुद्ध और रासायनिक रंगों से रहित प्राकृतिक मिट्टी।",
                        "चाक पर निर्मित: कुशल कारीगरों द्वारा पारंपरिक चाक पर हस्तनिर्मित।",
                        "पर्यावरण अनुकूल: पूर्णतः पर्यावरण-हितैषी और पारंपरिक शैली में निर्मित।",
                        "त्योहारों के लिए उत्तम: गृह सज्जा और पारंपरिक पूजा-अर्चना हेतु श्रेष्ठ।"
                    ],
                    "seo_keywords": [
                        "टेराकोटा बर्तन", "मिट्टी का गमला", "हस्तशिल्प मटका",
                        "भारतीय मिट्टी कला", "पूजा सामग्री", "हस्तनिर्मित शिल्प", "पर्यावरण अनुकूल"
                    ]
                }
            }
        }

    return {
        "translation": extracted_payload["translation"],
        "product_information": extracted_payload["product_information"],
        "catalog": extracted_payload["catalog"]
    }


# ==========================================
# NODE 3: Pricing Features Normalization
# ==========================================

def assemble_pricing_features_node(state: VoiceState) -> Dict[str, Any]:
    prod_info = state["product_information"]
    catalog_en = state["catalog"]["english"]
    product_id = state.get("product_id", "ART-000001")

    # 1. Parse raw materials list
    materials_list = [m.strip() for m in prod_info.get("material", "Clay").split(",") if m.strip()]
    primary_material = materials_list[0] if materials_list else "Clay"
    additional_materials = materials_list[1:] if len(materials_list) > 1 else []

    # 2. Extract or calculate time_worked_hours
    worked_hours = prod_info.get("time_worked_hours")
    if worked_hours is None:
        time_str = str(prod_info.get("production_time", "1"))
        matched_digits = re.findall(r"\d+", time_str)
        if "hour" in time_str.lower():
            worked_hours = float(matched_digits[0]) if matched_digits else 6.0
        else:
            days = float(matched_digits[0]) if matched_digits else 1.0
            worked_hours = days * 8.0
    else:
        worked_hours = float(worked_hours)

    # 3. Compile distinctive feature tags
    features: List[str] = []
    if prod_info.get("craft_method"):
        features.extend([m.strip() for m in prod_info["craft_method"].split("&")])
    features.append("Handcrafted")
    features = list(dict.fromkeys(features))

    # 4. Standardize craft type for market benchmark lookup
    raw_type = str(prod_info.get("product_type", "terracotta pottery")).lower()
    if any(k in raw_type for k in ["pot", "terracotta", "clay"]):
        standardized_craft = "terracotta pottery"
    elif any(k in raw_type for k in ["bamboo", "basket", "cane"]):
        standardized_craft = "bamboo basket"
    elif any(k in raw_type for k in ["wood", "carving", "timber"]):
        standardized_craft = "wooden craft"
    elif any(k in raw_type for k in ["textile", "saree", "handloom", "shawl", "stole"]):
        standardized_craft = "handloom textile"
    else:
        standardized_craft = "terracotta pottery"

    # 5. Build Pricing Input Features
    pricing_features = PricingInputFeatures(
        product_id=product_id,
        product_name=catalog_en.get("title", "Handmade Craft"),
        product_type=standardized_craft,
        material=primary_material,
        additional_materials=additional_materials,
        color=None,
        size=None,
        dimensions=DimensionsSchema(length=None, width=None, height=None, unit="cm"),
        features=features,
        description=catalog_en.get("description", ""),
        material_cost=prod_info.get("stated_material_cost"),
        labour_cost=None,
        production_cost=None,
        time_worked_hours=worked_hours
    )

    # 6. Build Final NLP Output schema
    final_output = NLPModuleOutput(
        product_id=product_id,
        transcription=state["transcription"],
        translation=state["translation"],
        product_information=ProductInformation(**prod_info),
        catalog=Catalog(**state["catalog"]),
        pricing_input_features=pricing_features
    )

    return {
        "pricing_input_features": pricing_features.model_dump(),
        "final_output": final_output.model_dump()
    }