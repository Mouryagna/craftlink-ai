import uuid
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.master_graph import master_graph
from src.master_state import MasterGraphState
from src.db_service import (
    save_initial_job,
    complete_job,
    fail_job,
    get_catalog_by_id,
    get_all_completed_catalogs
)

app = FastAPI(
    title="CraftLink AI - Virtual Artisan Manager Backend",
    description="Multimodal mobile backend for automated studio photography, multilingual cataloging, and fair-trade pricing.",
    version="1.0.0"
)

# Allow Flutter, React Native, or Web frontends to communicate
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

# Mount both static asset directories
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


def process_catalog_pipeline(
    product_id: str,
    image_paths: List[str],
    audio_path: Optional[str],
    manual_text: Optional[str],
    bg_color: str
):
    """Executes the master LangGraph pipeline asynchronously in the background."""
    try:
        graph_input: MasterGraphState = {
            "product_id": product_id,
            "image_paths": image_paths,
            "audio_path": audio_path,
            "manual_text": manual_text,
            "custom_bg_color": bg_color
        }
        result = master_graph.invoke(graph_input)
        complete_job(product_id, result["final_catalog"])
        print(f"[+] Successfully digitized and cataloged: {product_id}")
    except Exception as e:
        print(f"[!] Pipeline error for {product_id}: {e}")
        fail_job(product_id, str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "craftlink-ai-backend"}


@app.post("/api/v1/catalog/create")
async def create_catalog_entry(
    background_tasks: BackgroundTasks,
    image_1: UploadFile = File(..., description="Primary hero product photo"),
    image_2: Optional[UploadFile] = File(None, description="Optional angle view 2"),
    image_3: Optional[UploadFile] = File(None, description="Optional angle view 3"),
    image_4: Optional[UploadFile] = File(None, description="Optional detail/back view 4"),
    audio: Optional[UploadFile] = File(None, description="Optional spoken voice note"),
    manual_text: Optional[str] = Form(None, description="Artisan text description"),
    custom_bg_color: str = Form("#FFFFFF", description="Studio background hex code")
):
    """
    Ingests 1 to 4 raw product photos along with speech audio or typed text.
    Renders clean 'Choose File' pickers in Swagger UI.
    """
    clean_manual_text = manual_text.strip() if manual_text else None
    has_audio = audio is not None and bool(audio.filename)

    if not has_audio and not clean_manual_text:
        raise HTTPException(
            status_code=422,
            detail="You must provide either an audio recording or a manual text description."
        )

    # Collect provided images and verify at least one is valid
    uploaded_images: List[UploadFile] = [
        img for img in [image_1, image_2, image_3, image_4] if img is not None and img.filename
    ]

    if not uploaded_images:
        raise HTTPException(status_code=422, detail="At least one valid image file is required.")

    product_id = f"ART-{uuid.uuid4().hex[:6].upper()}"
    saved_images: List[str] = []

    prod_dir = UPLOAD_DIR / product_id
    prod_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save uploaded images
    for idx, img in enumerate(uploaded_images):
        suffix = Path(img.filename).suffix or ".jpg"
        target_path = prod_dir / f"input_{idx}{suffix}"
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(img.file, buffer)
        saved_images.append(str(target_path))

    # 2. Save voice recording if provided
    saved_audio = None
    if has_audio:
        suffix = Path(audio.filename).suffix or ".wav"
        audio_path = prod_dir / f"artisan_voice{suffix}"
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        saved_audio = str(audio_path)

    # 3. Create tracking entry in SQLite
    save_initial_job(product_id)

    # 4. Dispatch the Master LangGraph pipeline to the background task queue
    background_tasks.add_task(
        process_catalog_pipeline,
        product_id=product_id,
        image_paths=saved_images,
        audio_path=saved_audio,
        manual_text=clean_manual_text,
        bg_color=custom_bg_color
    )

    return {
        "status": "QUEUED",
        "product_id": product_id,
        "input_mode": "manual_text" if clean_manual_text and not saved_audio else "speech_audio",
        "message": "Product digitization in progress."
    }


@app.get("/api/v1/catalog/{product_id}")
async def get_catalog_status(product_id: str):
    """
    Mobile client polls this endpoint every 1-2 seconds until status == 'COMPLETED'.
    """
    record = get_catalog_by_id(product_id)
    if not record:
        raise HTTPException(status_code=404, detail="Product ID not found")

    if record["status"] == "PROCESSING":
        return {
            "status": "PROCESSING",
            "product_id": product_id,
            "message": "AI is generating studio images, translations, and fair-trade pricing..."
        }

    if record["status"] == "FAILED":
        return {
            "status": "FAILED",
            "product_id": product_id,
            "error": record.get("error") or record.get("catalog")
        }

    return {
        "status": "COMPLETED",
        "product_id": product_id,
        "execution_time_seconds": record.get("execution_time_seconds"),
        "data": record.get("catalog")
    }


@app.get("/api/v1/catalogs")
async def list_completed_catalogs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Returns a paginated list of all digitized artisan products for the mobile gallery.
    """
    catalogs = get_all_completed_catalogs(limit=limit, offset=offset)
    return {
        "count": len(catalogs),
        "items": catalogs
    }