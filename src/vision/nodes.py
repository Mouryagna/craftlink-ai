import os
import io
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests
import numpy as np
from PIL import Image, ImageEnhance
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
# 1. Surface Polishing & Defect Clean-Up Engine
# =========================================================================

def apply_local_surface_polish(pil_image: Image.Image) -> Image.Image:
    """
    Edge-preserving surface smoothing fallback using OpenCV / Pillow.
    Removes raw micro-speckles, dust dots, and rough kiln spots while
    keeping boundaries and traditional wheel ridges crisp.
    """
    try:
        import cv2

        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 1. Bilateral filter smooths flat textures while strictly preserving sharp contours
        polished_bgr = cv2.bilateralFilter(cv_img, d=9, sigmaColor=75, sigmaSpace=75)

        # 2. Subtle unsharp mask to restore crisp artisan boundary highlights
        gaussian = cv2.GaussianBlur(polished_bgr, (0, 0), 2.0)
        sharpened_bgr = cv2.addWeighted(polished_bgr, 1.25, gaussian, -0.25, 0)

        polished_rgb = cv2.cvtColor(sharpened_bgr, cv2.COLOR_BGR2RGB)
        result_img = Image.fromarray(polished_rgb)

        # 3. Slight color richness adjustment to bring out authentic terracotta warmth
        enhancer = ImageEnhance.Color(result_img)
        return enhancer.enhance(1.08)
    except Exception as err:
        print(f"    [!] Local polish filter failed: {err}. Returning unpolished base.")
        return pil_image


def polish_and_enhance_craft(
    image_bytes: bytes,
    gemini_key: Optional[str] = None
) -> bytes:
    """
    Polishes raw artisan artifacts using Google Imagen / Gemini visual prompt.
    Removes clay blemishes, surface dirt, and speckles while enriching natural texture.
    """
    base_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Prompt engineered specifically for handicraft defect correction and premium polish
    POLISH_PROMPT = (
        "Professional commercial e-commerce product photograph of this handcrafted artifact. "
        "Perform luxury catalog retouching: clean away all tiny dust speckles, rough clay blemishes, "
        "irregular dark stains, and surface dirt dots. "
        "Smooth out the surface finish with a refined, natural terracotta earthen sheen. "
        "Maintain 100% of the original shape, lip rim geometry, and traditional potter wheel contours. "
        "Ensure studio-grade lighting, rich warm earthen colors, and pristine clean texture."
    )

    if gemini_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key)

            # Route to Imagen 3 edit / image-to-image pipeline
            response = client.models.generate_images(
                model="imagen-3.0-generate-002",
                prompt=POLISH_PROMPT,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    output_mime_type="image/jpeg",
                    aspect_ratio="1:1"
                )
            )

            if response.generated_images:
                print("    [+] Commercial surface polish applied via AI Generative Clean-Up.")
                return response.generated_images[0].image.image_bytes

        except Exception as e:
            print(f"    [!] AI generative polish unavailable ({e}). Using edge-preserving local polish filter.")

    # Fallback to smart local bilateral filter
    polished_pil = apply_local_surface_polish(base_img)
    buffer = io.BytesIO()
    polished_pil.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


# =========================================================================
# 2. Studio Finish & Cutout Engine
# =========================================================================

def fallback_clean_cutout(input_path: Path, bg_hex: str = "#FFFFFF") -> Image.Image:
    """
    Robust local background cutout using rembg with zero-crash fallback.
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
    except BaseException as err:
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
    Calls Photoroom v2 API using strict 'imageFile' binary key with drop shadows.
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
                files = {
                    "imageFile": (input_path.name, f.read(), "image/jpeg")
                }
                response = requests.post(url, headers=headers, data=data, files=files, timeout=30)

            if response.status_code == 200:
                print("    [+] Commercial studio cutout generated via Photoroom.")
                return response.content

            print(f"    [!] Photoroom API returned HTTP {response.status_code}: {response.text}")
        except Exception as e:
            print(f"    [!] Photoroom network request error: {e}")

    # Fallback to local rembg cutout
    print("    [!] Running local rembg cutout fallback...")
    fallback_img = fallback_clean_cutout(input_path, bg_hex=clean_hex)
    buf = io.BytesIO()
    fallback_img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# =========================================================================
# NODE 1: Image Enhancement & Polishing
# =========================================================================

def enhance_images_node(state: VisionState) -> Dict[str, Any]:
    image_paths = state.get("image_paths", [])[:5]
    product_id = state.get("product_id", "ART-000001")
    bg_color = state.get("custom_bg_color", "#FFFFFF")
    photoroom_key = os.getenv("PHOTOROOM_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    # Output directory matching FastAPI static mount /outputs/{product_id}/...
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

        # Step 1: Background extraction & centering
        cutout_bytes = process_with_photoroom_ai(orig_file, bg_hex=bg_color, api_key=photoroom_key)

        # Step 2: Surface Retouch & Polishing (Cleans dots, marks, and blemishes)
        polished_bytes = polish_and_enhance_craft(cutout_bytes, gemini_key=gemini_key)

        with open(studio_file, "wb") as f:
            f.write(polished_bytes)

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

    if not analysis_data:
        analysis_data = {
            "detected_craft_type": "terracotta pottery",
            "primary_color": "Terracotta Red",
            "detected_colors": ["Terracotta Red", "Earthen Brown"],
            "size_category": "medium",
            "dimensions_estimate": {"length": 18.0, "width": 18.0, "height": 20.0, "unit": "cm"},
            "visual_complexity_score": 3,
            "surface_detailing": "Earthy natural clay finish with traditional hand-crafted contours"
        }

    raw_size = str(analysis_data.get("size_category", "medium")).lower().strip()
    if raw_size not in ["small", "medium", "large"]:
        raw_size = "medium"
    analysis_data["size_category"] = raw_size

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