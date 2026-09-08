import json
from pathlib import Path
from langgraph.graph import StateGraph, START, END

from src.vision.state import VisionState
from src.vision.nodes import (
    enhance_images_node,
    visual_analysis_node
)

# 1. Build Vision Subgraph
builder = StateGraph(VisionState)

builder.add_node("enhance_images", enhance_images_node)
builder.add_node("visual_analysis", visual_analysis_node)

builder.add_edge(START, "enhance_images")
builder.add_edge("enhance_images", "visual_analysis")
builder.add_edge("visual_analysis", END)

vision_subgraph = builder.compile()

if __name__ == "__main__":
    base_dir = Path.cwd()
    media_dir = base_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    # Collect available test images in media directory or project root (up to 5)
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    sample_images = []
    for ext in extensions:
        sample_images.extend([str(p) for p in media_dir.glob(ext)])
        if len(sample_images) >= 5:
            break

    if not sample_images:
        for ext in extensions:
            sample_images.extend([str(p) for p in base_dir.glob(ext)])
            if len(sample_images) >= 5:
                break

    sample_images = sample_images[:5]

    print("[-] Running Vision Module Subgraph...")
    print(f"[-] Input Images Count: {len(sample_images)}")

    initial_input: VisionState = {
        "product_id": "ART-000001",
        "image_paths": sample_images,
        "custom_bg_color": "#FFFFFF"
    }

    result = vision_subgraph.invoke(initial_input)
    final_output = result.get("final_output", {})

    print("\n==========================================")
    print("1. PROCESSED STUDIO IMAGES")
    print("==========================================")
    print(json.dumps(final_output.get("processed_images", {}), indent=2, ensure_ascii=False))

    print("\n==========================================")
    print("2. VISUAL ANALYSIS & DIMENSIONS")
    print("==========================================")
    print(json.dumps(final_output.get("visual_analysis", {}), indent=2, ensure_ascii=False))

    print("\n==========================================")
    print("3. FULL VISION MODULE OUTPUT")
    print("==========================================")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))