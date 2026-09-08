import json
from pathlib import Path
from langgraph.graph import StateGraph, START, END

from src.vision.state import VisionState
from src.vision.nodes import enhance_and_studio_node, visual_analysis_node

# 1. Build the Vision Subgraph
builder = StateGraph(VisionState)

# Add processing nodes
builder.add_node("enhance_and_studio", enhance_and_studio_node)
builder.add_node("visual_analysis", visual_analysis_node)

# Define linear flow
builder.add_edge(START, "enhance_and_studio")
builder.add_edge("enhance_and_studio", "visual_analysis")
builder.add_edge("visual_analysis", END)

# Compile into reusable subgraph
vision_subgraph = builder.compile()


if __name__ == "__main__":
    base_dir = Path.cwd()
    media_dir = base_dir / "media"

    # Explicitly list the real multi-angle images from your media folder
    target_filenames = [
        "front-view.png",
        "side-view.png",
        "top-view.png",
        "45.png"
    ]

    real_images = [
        str(media_dir / filename)
        for filename in target_filenames
        if (media_dir / filename).exists()
    ]

    if not real_images:
        raise FileNotFoundError(f"None of the target images {target_filenames} were found in {media_dir}")

    print(f"[-] Executing Vision Subgraph on {len(real_images)} multi-angle captures:")
    for img in real_images:
        print(f"    • {img}")

    initial_input: VisionState = {
        "raw_image_paths": real_images
    }

    result = vision_subgraph.invoke(initial_input)

    print("\n==========================================")
    print("OUTPUT 1: USER DISPLAY (Mobile App UI)")
    print("==========================================")
    print(json.dumps(result["user_display"], indent=2, ensure_ascii=False))

    print("\n==========================================")
    print("OUTPUT 2: PIPELINE PAYLOAD (Downstream)")
    print("==========================================")
    print(json.dumps(result["pipeline_payload"], indent=2, ensure_ascii=False))