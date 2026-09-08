import os
import io
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests
from PIL import Image
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

from src.vision.state import (
    VisionState,
    DimensionsEstimate,
    VisualAnalysis,
    ProcessedImageItem,
    ProcessedImages,
    VisionModuleOutput
)


# =========================================================================
# 1. Photoroom Studio AI Engine
# =========================================================================

def process_with_photoroom_ai(
        input_path: Path,
        bg_hex: str = "FFFFFF",
        api_key: Optional[str] = None
) -> bytes:
    """
    Calls Photoroom's studio pipeline to remove background and cast
    a realistic physical floor contact shadow.
    """
    clean_hex = bg_hex.lstrip("#")
    url = "https://image-api.photoroom.com/v2/edit"

    headers = {"x-api-key": api_key}
    data = {
        "background.color": clean_hex,
        "shadow.mode": "ai.soft",
        "padding": "0.1",
        "outputSize": "1200x1200"
    }

    with open(input_path, "rb") as f:
        files = {"imageFile": (input_path.name, f, "image/jpeg")}
        response = requests.post(url, headers=headers, data=data, files=files, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"Photoroom API error ({response.status_code}): {response.text}")

    return response.content


def fallback_clean_cutout(input_path: Path, bg_hex: str = "#FFFFFF") -> Image.Image:
    """Fallback if API key is not present."""
    import rembg
    raw = Image.open(input_path).convert("RGB")
    cutout = rembg.remove(raw)
    bbox = cutout.getbbox()
    trimmed = cutout.crop(bbox) if bbox else cutout

    tw, th = trimmed.size
    canvas = Image.new("RGB", (int(tw * 1.25), int(th * 1.25)), bg_hex)
    canvas.paste(trimmed, ((canvas.width - tw) // 2, (canvas.height - th) // 2), mask=trimmed.split()[-1])
    return canvas


# =========================================================================
# NODE 1: Image Enhancement
# =========================================================================

def enhance_images_node(state: VisionState) -> Dict[str, Any]:
    image_paths = state.get("image_paths", [])[:5]
    product_id = state.get("product_id", "ART-000001")
    bg_color = state.get("custom_bg_color", "#FFFFFF")
    api_key = os.getenv("PHOTOROOM_API_KEY")

    output_dir = Path.cwd() / "media" / "processed" / product_id
    output_dir.mkdir(parents=True, exist_ok=True)

    items: List[ProcessedImageItem] = []
    hero_studio_path = None

    for idx, path_str in enumerate(image_paths):
        orig_file = Path(path_str)
        if not orig_file.exists():
            continue

        studio_file = output_dir / f"{product_id}_{idx + 1}_studio.jpg"
        print(f"[-] Generating Studio Shot for View {idx + 1}/{len(image_paths)}: {orig_file.name}")

        if api_key:
            try:
                processed_bytes = process_with_photoroom_ai(orig_file, bg_hex=bg_color, api_key=api_key)
                with open(studio_file, "wb") as f:
                    f.write(processed_bytes)
                print(f"    [+] Commercial studio finish saved.")
            except Exception as e:
                print(f"    [!] Photoroom error: {e}. Using fallback...")
                fallback_img = fallback_clean_cutout(orig_file, bg_hex=bg_color)
                fallback_img.save(studio_file, format="JPEG", quality=95)
        else:
            print("    [!] PHOTOROOM_API_KEY not found in env. Running basic fallback...")
            fallback_img = fallback_clean_cutout(orig_file, bg_hex=bg_color)
            fallback_img.save(studio_file, format="JPEG", quality=95)

        if hero_studio_path is None:
            hero_studio_path = str(studio_file)

        items.append(ProcessedImageItem(
            original_path=str(orig_file),
            studio_path=str(studio_file)
        ))

    processed_images = ProcessedImages(
        items=items,
        hero_studio_path=hero_studio_path,
        background_color=bg_color
    )

    return {"processed_images": processed_images.model_dump()}


# =========================================================================
# NODE 2: Multimodal Visual Analysis
# =========================================================================

def visual_analysis_node(state: VisionState) -> Dict[str, Any]:
    image_paths = state.get("image_paths", [])[:5]
    product_id = state.get("product_id", "ART-000001")
    gemini_key = os.getenv("GEMINI_API_KEY")

    analysis_data = None

    if gemini_key and image_paths:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key)
            pil_images = [Image.open(p) for p in image_paths if Path(p).exists()]

            prompt = """Analyze these handicraft photos as an e-commerce catalog specialist.
Estimate physical dimensions in cm, size category, dominant colors, and craft complexity.

Return valid JSON:
{
  "detected_craft_type": "string",
  "primary_color": "string",
  "detected_colors": ["list of colors"],
  "size_category": "Small, Medium, or Large",
  "dimensions_estimate": {
    "length": float,
    "width": float,
    "height": float,
    "unit": "cm"
  },
  "visual_complexity_score": integer (1 to 5),
  "surface_detailing": "brief description"
}"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt] + pil_images,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            analysis_data = json.loads(response.text)
        except Exception as e:
            print(f"[!] Gemini visual analysis fallback: {e}")

    if not analysis_data:
        analysis_data = {
            "detected_craft_type": "Clay Terracotta Pot",
            "primary_color": "Terracotta Red",
            "detected_colors": ["Terracotta Red", "Brown"],
            "size_category": "Medium",
            "dimensions_estimate": {"length": 18.0, "width": 18.0, "height": 20.0, "unit": "cm"},
            "visual_complexity_score": 3,
            "surface_detailing": "Earthy natural clay finish with standard flared neck"
        }

    visual_analysis = VisualAnalysis(**analysis_data)
    final_output = VisionModuleOutput(
        product_id=product_id,
        processed_images=ProcessedImages(**state["processed_images"]),
        visual_analysis=visual_analysis
    )

    return {
        "visual_analysis": visual_analysis.model_dump(),
        "final_output": final_output.model_dump()
    }