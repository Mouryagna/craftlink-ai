import os
import io
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests
from PIL import Image
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from src.vision.state import (
    VisionState,
    VisualAnalysis,
    ProcessedImageItem,
    ProcessedImages,
    VisionModuleOutput
)


# =========================================================================
# 1. Studio Finish & Fallback Engines
# =========================================================================

def fallback_clean_cutout(input_path: Path, bg_hex: str = "#FFFFFF") -> Image.Image:
    """
    Robust local background cutout using rembg with zero-crash fallback.
    If rembg model weights or ONNX fails, centers the original image on the canvas.
    """
    clean_hex = f"#{bg_hex.lstrip('#')}"
    try:
        import rembg
        raw = Image.open(input_path).convert("RGBA")
        cutout = rembg.remove(raw)
        bbox = cutout.getbbox()
        trimmed = cutout.crop(bbox) if bbox else cutout

        tw, th = trimmed.size
        canvas = Image.new("RGBA", (int(tw * 1.2), int(th * 1.2)), clean_hex)
        canvas.paste(trimmed, ((canvas.width - tw) // 2, (canvas.height - th) // 2), mask=trimmed.split()[-1])
        return canvas.convert("RGB")
    except Exception as err:
        print(f"    [!] rembg local processing error: {err}. Using canvas padding fallback.")
        raw = Image.open(input_path).convert("RGB")
        w, h = raw.size
        canvas = Image.new("RGB", (int(w * 1.15), int(h * 1.15)), clean_hex)
        canvas.paste(raw, ((canvas.width - w) // 2, (canvas.height - h) // 2))
        return canvas


def process_with_photoroom_ai(
    input_path: Path,
    bg_hex: str = "FFFFFF",
    api_key: Optional[str] = None
) -> bytes:
    """
    Calls Photoroom v2 API using the strict 'imageFile' binary key.
    Falls back gracefully to local processing without throwing unhandled exceptions.
    """
    clean_hex = bg_hex.lstrip("#")

    if api_key:
        url = "https://image-api.photoroom.com/v2/edit"
        headers = {
            "x-api-key": api_key,
            "Accept": "image/png, application/json"
        }
        data = {
            "background.color": clean_hex,
            "shadow.mode": "ai.soft",
            "padding": "0.1",
            "outputSize": "1200x1200"
        }

        try:
            with open(input_path, "rb") as f:
                # Key MUST strictly be 'imageFile' in camelCase
                files = {
                    "imageFile": (input_path.name, f.read(), "image/jpeg")
                }
                response = requests.post(url, headers=headers, data=data, files=files, timeout=30)

            if response.status_code == 200:
                print("    [+] Commercial studio finish generated via Photoroom.")
                return response.content

            print(f"    [!] Photoroom API returned HTTP {response.status_code}: {response.text}")
        except Exception as e:
            print(f"    [!] Photoroom network request error: {e}")

    # Fallback when API key is missing or request fails
    print("    [!] Running local rembg cutout fallback...")
    fallback_img = fallback_clean_cutout(input_path, bg_hex=clean_hex)
    buf = io.BytesIO()
    fallback_img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# =========================================================================
# NODE 1: Image Enhancement
# =========================================================================

def enhance_images_node(state: VisionState) -> Dict[str, Any]:
    image_paths = state.get("image_paths", [])[:5]
    product_id = state.get("product_id", "ART-000001")
    bg_color = state.get("custom_bg_color", "#FFFFFF")
    api_key = os.getenv("PHOTOROOM_API_KEY")

    # Save to outputs/<product_id> matching the FastAPI static files mount
    output_dir = Path.cwd() / "outputs" / product_id
    output_dir.mkdir(parents=True, exist_ok=True)

    items: List[ProcessedImageItem] = []
    hero_studio_path = None

    if not image_paths:
        print("[!] No image paths provided in state.")

    for idx, path_str in enumerate(image_paths):
        orig_file = Path(path_str)
        if not orig_file.exists():
            print(f"    [!] Skipping missing file: {path_str}")
            continue

        studio_file = output_dir / f"{product_id}_{idx + 1}_studio.jpg"
        print(f"[-] Processing view {idx + 1}/{len(image_paths)}: {orig_file.name}")

        processed_bytes = process_with_photoroom_ai(orig_file, bg_hex=bg_color, api_key=api_key)
        with open(studio_file, "wb") as f:
            f.write(processed_bytes)

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
    valid_images = [p for p in image_paths if Path(p).exists()]

    if gemini_key and valid_images:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key)
            pil_images = [Image.open(p).convert("RGB") for p in valid_images]

            prompt = """Analyze these handicraft photos as an e-commerce catalog specialist.
Estimate physical dimensions in cm, size category, dominant colors, and craft complexity.

Return strictly raw JSON without markdown formatting:
{
  "detected_craft_type": "terracotta pottery | bamboo basket | wooden craft | handloom textile",
  "primary_color": "string",
  "detected_colors": ["list of colors"],
  "size_category": "small, medium, or large",
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

            raw_text = response.text.strip()
            clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.DOTALL).strip()
            analysis_data = json.loads(clean_json)

        except Exception as e:
            print(f"[!] Gemini visual analysis error: {e}. Using baseline handicraft schema.")

    # Default fallback values if analysis fails or API is unavailable
    if not analysis_data:
        analysis_data = {
            "detected_craft_type": "terracotta pottery",
            "primary_color": "Terracotta Red",
            "detected_colors": ["Terracotta Red", "Brown"],
            "size_category": "medium",
            "dimensions_estimate": {"length": 18.0, "width": 18.0, "height": 20.0, "unit": "cm"},
            "visual_complexity_score": 3,
            "surface_detailing": "Earthy natural clay finish with traditional hand-crafted contours"
        }

    # Normalize size category for pricing benchmarks
    raw_size = str(analysis_data.get("size_category", "medium")).lower().strip()
    if raw_size not in ["small", "medium", "large"]:
        raw_size = "medium"
    analysis_data["size_category"] = raw_size

    # Clamp visual complexity score to 1-5
    try:
        raw_complexity = int(analysis_data.get("visual_complexity_score", 3))
    except (ValueError, TypeError):
        raw_complexity = 3
    analysis_data["visual_complexity_score"] = max(1, min(5, raw_complexity))

    visual_analysis = VisualAnalysis(**analysis_data)

    processed_imgs_dict = state.get("processed_images") or {
        "items": [],
        "hero_studio_path": None,
        "background_color": state.get("custom_bg_color", "#FFFFFF")
    }

    final_output = VisionModuleOutput(
        product_id=product_id,
        processed_images=ProcessedImages(**processed_imgs_dict),
        visual_analysis=visual_analysis
    )

    return {
        "visual_analysis": visual_analysis.model_dump(),
        "final_output": final_output.model_dump()
    }