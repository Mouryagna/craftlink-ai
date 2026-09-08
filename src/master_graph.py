import json
from pathlib import Path
from langgraph.graph import StateGraph, START, END

from src.master_state import MasterGraphState
from src.master_node import (
    run_vision_node,
    run_voice_node,
    reconcile_features_node,
    run_pricing_node,
    assemble_catalog_node
)

# -------------------------------------------------------------------------
# Compile Master Graph
# -------------------------------------------------------------------------
builder = StateGraph(MasterGraphState)

builder.add_node("vision_module", run_vision_node)
builder.add_node("voice_module", run_voice_node)
builder.add_node("reconcile_features", reconcile_features_node)
builder.add_node("pricing_module", run_pricing_node)
builder.add_node("assemble_catalog", assemble_catalog_node)

# Execute Vision and Voice modules concurrently
builder.add_edge(START, "vision_module")
builder.add_edge(START, "voice_module")

# Converge both outputs into reconciliation
builder.add_edge("vision_module", "reconcile_features")
builder.add_edge("voice_module", "reconcile_features")

# Sequential pricing and catalog generation
builder.add_edge("reconcile_features", "pricing_module")
builder.add_edge("pricing_module", "assemble_catalog")
builder.add_edge("assemble_catalog", END)

master_graph = builder.compile()


# -------------------------------------------------------------------------
# Local Verification Block
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("[-] Executing Full Master Catalog Pipeline...")

    base_dir = Path.cwd()
    media_dir = base_dir / "media"

    # Search for sample test images
    sample_images = [str(p) for p in media_dir.glob("*.jpg")][:4]
    if not sample_images:
        sample_images = [str(p) for p in base_dir.glob("*.png")][:4]

    # Test payload with manual Hindi text input (voice bypass)
    test_input: MasterGraphState = {
        "product_id": "ART-000001",
        "image_paths": sample_images,
        "audio_path": None,
        "manual_text": "हमने यह लाल मिट्टी का मटका चाक पर 6 घंटे लगाकर बनाया है। इसमें 150 रुपये की कच्ची सामग्री लगी है और हम इसे कम से कम 400 में बेचना चाहते हैं।",
        "custom_bg_color": "#FFFFFF"
    }

    output = master_graph.invoke(test_input)

    print("\n" + "=" * 55)
    print("STOREFRONT BILINGUAL CARD OUTPUT")
    print("=" * 55)
    print(json.dumps(output["final_catalog"], ensure_ascii=False, indent=2))