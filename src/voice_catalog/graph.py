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
    media_dir = base_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    # Look for any available audio sample
    audio_extensions = ("*.mp3", "*.wav", "*.m4a", "*.ogg")
    sample_audio = None
    for ext in audio_extensions:
        found = list(media_dir.glob(ext))
        if found:
            sample_audio = found[0]
            break

    print("[-] Running NLP / Voice Module Subgraph Self-Test...")

    if sample_audio and sample_audio.exists():
        print(f"[-] Testing with audio file: {sample_audio.name}")
        initial_input: VoiceState = {
            "product_id": "ART-000001",
            "audio_path": str(sample_audio),
            "manual_text": None
        }
    else:
        print("[-] No audio file detected in media/ folder. Testing with manual text input...")
        initial_input: VoiceState = {
            "product_id": "ART-000001",
            "audio_path": None,
            "manual_text": "हमने यह लाल मिट्टी का बर्तन हाथ से चाक पर बनाया है। इसमें 6 घंटे का समय लगा और 150 रुपये की कच्ची सामग्री लगी। हम इसे कम से कम 400 में बेचना चाहते हैं।"
        }

    result = nlp_subgraph.invoke(initial_input)
    final_output = result.get("final_output", {})

    print("\n==========================================")
    print("1. PRICING INPUT FEATURES (NLP EXTRACTED)")
    print("==========================================")
    print(json.dumps(final_output.get("pricing_input_features", {}), indent=2, ensure_ascii=False))

    print("\n==========================================")
    print("2. FULL NLP MODULE OUTPUT (SCHEMA VERIFIED)")
    print("==========================================")
    print(json.dumps(final_output, indent=2, ensure_ascii=False))