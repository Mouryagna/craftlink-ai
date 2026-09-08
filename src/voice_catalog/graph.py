import json
from pathlib import Path
from langgraph.graph import StateGraph, START, END

from src.voice_catalog.state import VoiceState
from src.voice_catalog.nodes import (
    transcribe_node,
    extract_and_catalog_node,
    assemble_pricing_features_node
)

# Build NLP / Voice Subgraph
builder = StateGraph(VoiceState)

builder.add_node("transcribe", transcribe_node)
builder.add_node("extract_and_catalog", extract_and_catalog_node)
builder.add_node("assemble_pricing_features", assemble_pricing_features_node)

builder.add_edge(START, "transcribe")
builder.add_edge("transcribe", "extract_and_catalog")
builder.add_edge("extract_and_catalog", "assemble_pricing_features")
builder.add_edge("assemble_pricing_features", END)

nlp_subgraph = builder.compile()


if __name__ == "__main__":
    base_dir = Path.cwd()
    sample_audio = base_dir / "media" / "artisan_sample.mp3"

    # Ensure audio exists for local verification
    if not sample_audio.exists():
        sample_audio = base_dir / "media" / "artisan_sample.mp3"

        if not sample_audio.exists():
            raise FileNotFoundError(f"Audio file not found at: {sample_audio}")

    print("[-] Running NLP Module Subgraph...")
    initial_input: VoiceState = {
        "product_id": "ART-000001",
        "audio_path": str(sample_audio)
    }

    result = nlp_subgraph.invoke(initial_input)
    final_output = result["final_output"]

    print("\n==========================================")
    print("1. PRICING INPUT FEATURES (NLP EXTRACTED)")
    print("==========================================")
    print(json.dumps(final_output["pricing_input_features"], indent=2, ensure_ascii=False))

    print("\n==========================================")
    print("2. FULL NLP MODULE OUTPUT (SCHEMA VERIFIED)")
    print("==========================================")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))