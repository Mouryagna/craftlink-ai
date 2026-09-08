import json
from pathlib import Path
from langgraph.graph import StateGraph, START, END

from src.voice_catalog.state import VoiceState
from src.voice_catalog.nodes import (
    transcribe_speech_node,
    extract_and_reconcile_node,
    generate_bilingual_catalog_node
)

# 1. Build the Voice & Catalog Subgraph
builder = StateGraph(VoiceState)

# Add processing nodes
builder.add_node("transcribe_speech", transcribe_speech_node)
builder.add_node("extract_and_reconcile", extract_and_reconcile_node)
builder.add_node("generate_bilingual_catalog", generate_bilingual_catalog_node)

# Define linear flow
builder.add_edge(START, "transcribe_speech")
builder.add_edge("transcribe_speech", "extract_and_reconcile")
builder.add_edge("extract_and_reconcile", "generate_bilingual_catalog")
builder.add_edge("generate_bilingual_catalog", END)

# Compile into reusable subgraph
voice_subgraph = builder.compile()


if __name__ == "__main__":
    base_dir = Path.cwd()
    media_dir = base_dir / "media"
    sample_audio = media_dir / "artisan_sample.mp3"

    # Verify test audio exists
    if not sample_audio.exists():
        from src.voice_catalog.audio import create_sample_audio
        create_sample_audio(sample_audio)

    # Simulated pipeline payload received from Module 1 (Vision Subgraph)
    mock_vision_context = {
        "visual_complexity_score": 3,
        "detected_craft_type": "Terracotta & Pottery",
        "materials_detected": ["Riverbed Clay", "Natural Pigment"],
        "dimensions_estimate": {
            "height_cm": 25.0,
            "width_cm": 14.0,
            "depth_cm": 14.0
        }
    }

    print("[-] Executing Voice & NLP Subgraph...")
    initial_input: VoiceState = {
        "audio_path": str(sample_audio),
        "vision_context": mock_vision_context
    }

    result = voice_subgraph.invoke(initial_input)

    print("\n==========================================")
    print("OUTPUT 1: USER DISPLAY (Mobile App UI)")
    print("==========================================")
    print(json.dumps(result["user_display"], indent=2, ensure_ascii=False))

    print("\n==========================================")
    print("OUTPUT 2: PIPELINE PAYLOAD (For Module 3 Pricing)")
    print("==========================================")
    print(json.dumps(result["pipeline_payload"], indent=2, ensure_ascii=False))