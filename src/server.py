import uuid
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.vision.graph import vision_subgraph
from src.master_node import execute_nlp_and_pricing, to_web_url
from src.db_service import (
    init_db,
    save_initial_job,
    save_vision_step,
    get_vision_step,
    complete_job,
    fail_job,
    get_catalog_by_id,
    get_catalogs_by_user
)

app = FastAPI(
    title="CraftLink AI Backend",
    description="Multimodal mobile backend for automated studio photography, multilingual cataloging, and fair-trade pricing.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path.cwd() / "uploads"
OUTPUT_DIR = Path.cwd() / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

init_db()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "craftlink-ai-backend"}


# -------------------------------------------------------------------------
# STEP 1: Process Images Only (Studio Cutout + Visual Analysis)
# -------------------------------------------------------------------------
@app.post("/api/v1/catalog/step1-vision")
async def step1_vision(
    image_1: UploadFile = File(..., description="Hero product image"),
    image_2: Optional[UploadFile] = File(None),
    image_3: Optional[UploadFile] = File(None),
    image_4: Optional[UploadFile] = File(None),
    user_id: str = Form("default_artisan", description="Artisan identifier"),
    custom_bg_color: str = Form("#FFFFFF")
):
    uploaded = [img for img in [image_1, image_2, image_3, image_4] if img and img.filename]
    if not uploaded:
        raise HTTPException(status_code=422, detail="At least one image is required.")

    product_id = f"ART-{uuid.uuid4().hex[:6].upper()}"
    prod_dir = UPLOAD_DIR / product_id
    prod_dir.mkdir(parents=True, exist_ok=True)

    saved_images = []
    for idx, img in enumerate(uploaded):
        ext = Path(img.filename).suffix or ".jpg"
        target = prod_dir / f"input_{idx}{ext}"
        with open(target, "wb") as buf:
            shutil.copyfileobj(img.file, buf)
        saved_images.append(str(target))

    save_initial_job(product_id, user_id=user_id)

    try:
        vision_result = vision_subgraph.invoke({
            "product_id": product_id,
            "image_paths": saved_images,
            "custom_bg_color": custom_bg_color
        })
        vision_output = vision_result.get("final_output", {})
        save_vision_step(product_id, vision_output)

        processed_imgs = vision_output.get("processed_images", {})
        studio_urls = [
            to_web_url(item.get("studio_path"))
            for item in processed_imgs.get("items", [])
        ]

        return {
            "status": "VISION_COMPLETED",
            "product_id": product_id,
            "user_id": user_id,
            "hero_image_url": to_web_url(processed_imgs.get("hero_studio_path")),
            "studio_images": studio_urls,
            "detected_visuals": vision_output.get("visual_analysis", {})
        }

    except Exception as e:
        fail_job(product_id, str(e))
        raise HTTPException(status_code=500, detail=f"Vision processing failed: {e}")


# -------------------------------------------------------------------------
# STEP 2: Ingest Speech/Text -> Auto-Trigger Pricing -> Final Catalog
# -------------------------------------------------------------------------
def process_nlp_and_pricing_task(
    product_id: str,
    vision_output: dict,
    audio_path: Optional[str],
    manual_text: Optional[str]
):
    try:
        final_catalog = execute_nlp_and_pricing(
            product_id=product_id,
            vision_output=vision_output,
            audio_path=audio_path,
            manual_text=manual_text
        )
        complete_job(product_id, final_catalog)
        print(f"[+] Multi-step cataloging completed successfully: {product_id}")
    except Exception as e:
        print(f"[!] Step 2 pipeline error: {e}")
        fail_job(product_id, str(e))


@app.post("/api/v1/catalog/{product_id}/step2-nlp")
async def step2_nlp_and_pricing(
    product_id: str,
    background_tasks: BackgroundTasks,
    audio: Optional[UploadFile] = File(None),
    manual_text: Optional[str] = Form(None)
):
    vision_data = get_vision_step(product_id)
    if not vision_data:
        raise HTTPException(status_code=404, detail="Product ID not found or Step 1 not completed.")

    clean_text = manual_text.strip() if manual_text else None
    has_audio = audio is not None and bool(audio.filename)

    if not has_audio and not clean_text:
        raise HTTPException(status_code=422, detail="Provide either an audio recording or manual text.")

    saved_audio = None
    if has_audio:
        ext = Path(audio.filename).suffix or ".wav"
        audio_path = UPLOAD_DIR / product_id / f"artisan_voice{ext}"
        with open(audio_path, "wb") as buf:
            shutil.copyfileobj(audio.file, buf)
        saved_audio = str(audio_path)

    background_tasks.add_task(
        process_nlp_and_pricing_task,
        product_id=product_id,
        vision_output=vision_data,
        audio_path=saved_audio,
        manual_text=clean_text
    )

    return {
        "status": "QUEUED",
        "product_id": product_id,
        "message": "Generating bilingual descriptions and ML fair-trade pricing..."
    }


# -------------------------------------------------------------------------
# GET A SINGLE PRODUCT'S STATUS & FULL DETAILS
# -------------------------------------------------------------------------
@app.get("/api/v1/catalog/{product_id}")
async def get_catalog_status(product_id: str):
    record = get_catalog_by_id(product_id)
    if not record:
        raise HTTPException(status_code=404, detail="Product ID not found")

    if record["status"] in ["PROCESSING", "VISION_COMPLETED"]:
        return {
            "status": record["status"],
            "product_id": product_id,
            "message": "Processing in progress..."
        }

    if record["status"] == "FAILED":
        return {"status": "FAILED", "product_id": product_id, "error": record.get("error")}

    return {
        "status": "COMPLETED",
        "product_id": product_id,
        "data": record.get("catalog")
    }


# -------------------------------------------------------------------------
# GET ALL COMPLETED PRODUCTS (Filtered by User or Global)
# -------------------------------------------------------------------------
@app.get("/api/v1/catalogs")
async def list_completed_catalogs(
    user_id: Optional[str] = Query(None, description="Optional user/artisan ID to filter products"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Returns a ready-to-render list of completed products for the storefront/portfolio screen.
    """
    records = get_catalogs_by_user(user_id=user_id, limit=limit, offset=offset)

    catalog_cards = []
    for item in records:
        cat_data = item.get("catalog", {})
        card_view = cat_data.get("card_view", {})
        tech_specs = cat_data.get("technical_specs", {})

        catalog_cards.append({
            "product_id": item["product_id"],
            "user_id": item.get("user_id"),
            "hero_image_url": card_view.get("hero_image_url"),
            "title_en": card_view.get("title", {}).get("en", "Handcrafted Art"),
            "title_hi": card_view.get("title", {}).get("hi", "हस्तशिल्प उत्पाद"),
            "display_price": card_view.get("display_price"),
            "currency": card_view.get("currency_symbol", "₹"),
            "craft_type": tech_specs.get("craft_type", "Handicraft"),
            "created_at": item.get("created_at")
        })

    return {
        "count": len(catalog_cards),
        "products": catalog_cards
    }