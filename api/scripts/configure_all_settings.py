#!/usr/bin/env python3
import json
import os

MODELS_TO_ADD = [
    {"model": "models/gemini-3.6-flash", "displayName": "Gemini 3.6 Flash"},
    {"model": "models/gemini-3.5-flash", "displayName": "Gemini 3.5 Flash"},
    {"model": "models/gemini-3.5-flash-lite", "displayName": "Gemini 3.5 Flash Lite"},
    {"model": "models/gemini-omni-flash-preview", "displayName": "Gemini Omni Flash (Preview)"},
    {"model": "models/lyria-3-clip-preview", "displayName": "Lyria 3 Clip (Preview)"},
    {"model": "models/lyria-3-pro-preview", "displayName": "Lyria 3 Pro (Preview)"},
    {"model": "models/gemini-3.1-flash-tts-preview", "displayName": "Gemini 3.1 Flash TTS (Preview)"},
    {"model": "models/gemini-robotics-er-1.5-preview", "displayName": "Gemini Robotics-ER 1.5 (Preview)"},
    {"model": "models/gemini-robotics-er-1.6-preview", "displayName": "Gemini Robotics-ER 1.6 (Preview)"},
    {"model": "models/gemini-2.5-computer-use-preview-10-2025", "displayName": "Gemini 2.5 Computer Use (Preview)"},
    {"model": "models/antigravity-preview-05-2026", "displayName": "Antigravity Preview"},
    {"model": "models/deep-research-max-preview-04-2026", "displayName": "Deep Research Max (Preview)"},
    {"model": "models/deep-research-preview-04-2026", "displayName": "Deep Research (Preview)"},
    {"model": "models/deep-research-pro-preview-12-2025", "displayName": "Deep Research Pro (Preview)"},
    {"model": "models/gemini-embedding-001", "displayName": "Gemini Embedding 001"},
    {"model": "models/gemini-embedding-2-preview", "displayName": "Gemini Embedding 2 (Preview)"},
    {"model": "models/gemini-embedding-2", "displayName": "Gemini Embedding 2"},
    {"model": "models/aqa", "displayName": "Attributed Question Answering"},
    {"model": "models/imagen-4.0-generate-001", "displayName": "Imagen 4"},
    {"model": "models/imagen-4.0-ultra-generate-001", "displayName": "Imagen 4 Ultra"},
    {"model": "models/imagen-4.0-fast-generate-001", "displayName": "Imagen 4 Fast"},
    {"model": "models/veo-3.1-generate-preview", "displayName": "Veo 3.1"},
    {"model": "models/veo-3.1-fast-generate-preview", "displayName": "Veo 3.1 Fast"},
    {"model": "models/veo-3.1-lite-generate-preview", "displayName": "Veo 3.1 Lite"},
    {"model": "models/gemini-2.5-flash-native-audio-latest", "displayName": "Gemini 2.5 Flash Native Audio Latest"},
    {"model": "models/gemini-2.5-flash-native-audio-preview-09-2025", "displayName": "Gemini 2.5 Flash Native Audio Preview 09-25"},
    {"model": "models/gemini-2.5-flash-native-audio-preview-12-2025", "displayName": "Gemini 2.5 Flash Native Audio Preview 12-25"},
    {"model": "models/gemini-3.1-flash-live-preview", "displayName": "Gemini 3.1 Flash Live Preview"},
    {"model": "models/gemini-3.5-live-translate-preview", "displayName": "Gemini 3.5 Live Translate Preview"},
    {"model": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash"},
    {"model": "models/gemini-2.5-pro", "displayName": "Gemini 2.5 Pro"},
    {"model": "models/gemini-2.0-flash", "displayName": "Gemini 2.0 Flash"},
    {"model": "models/gemini-2.0-flash-001", "displayName": "Gemini 2.0 Flash 001"},
    {"model": "models/gemini-2.0-flash-lite", "displayName": "Gemini 2.0 Flash-Lite"},
    {"model": "models/gemini-2.0-flash-lite-001", "displayName": "Gemini 2.0 Flash-Lite 001"}
]

def update_file(filepath):
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist.")
        return
    
    with open(filepath, "r") as f:
        data = json.load(f)
    
    # 1. Update customModels
    custom_models = data.setdefault("customModels", [])
    existing_model_ids = {m["model"] for m in custom_models}
    
    for m in MODELS_TO_ADD:
        if m["model"] not in existing_model_ids:
            custom_models.append({
                "model": m["model"],
                "displayName": m["displayName"],
                "provider": "google-ai"
            })
            existing_model_ids.add(m["model"])
    
    # 2. Update modelConfigs.customAliases
    model_configs = data.setdefault("modelConfigs", {})
    custom_aliases = model_configs.setdefault("customAliases", {})
    
    for m in MODELS_TO_ADD:
        if m["displayName"] not in custom_aliases:
            custom_aliases[m["displayName"]] = {
                "extends": m["model"]
            }
            
    # 3. Set default model to Gemini 3.6 Flash (unless it's workspace-specific where we don't have model at root)
    if "antigravity-cli" in filepath or "settings.json" == os.path.basename(filepath) and "/.gemini/" not in filepath:
         data["model"] = "Gemini 3.6 Flash"
    elif "settings.json" == os.path.basename(filepath) and "/.gemini/" in filepath and ".gemini/settings.json" not in filepath:
         # Under ~/.gemini/settings.json
         pass
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Successfully updated {filepath}!")

def main():
    update_file("/home/aaryn/.gemini/settings.json")
    update_file("/home/aaryn/.gemini/antigravity-cli/settings.json")
    update_file("/mnt/storage/projects/xcelsior/.gemini/settings.json")

if __name__ == "__main__":
    main()
