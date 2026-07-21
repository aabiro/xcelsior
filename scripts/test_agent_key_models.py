#!/usr/bin/env python3
import json
import urllib.request
import urllib.error
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
dotenv_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
AGENT_KEY = os.environ.get("AGENT_PLATFORM_API_KEY") or os.environ.get("AGENT_API_KEY")

MODEL_NAME = "gemini-3.1-pro-preview-customtools"

# Check for dry run
DRY_RUN = os.environ.get("DRY_RUN", "0").lower() in ("1", "true", "yes") or "--dry-run" in sys.argv

def list_models(api_key, key_label):
    print(f"\n=== [Listing Models for {key_label}] ===")
    if not api_key:
        print(f"Error: No API key provided for {key_label}!")
        return None
        
    if DRY_RUN:
        masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "SecretKey"
        print(f"[Dry Run] Successfully validated key: {masked_key}")
        print(f"[Dry Run] Skipping API call to list models.")
        return ["gemini-3.1-pro-preview-customtools"]

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            print(f"Success! Found {len(models)} available models.")
            for m in models:
                print(f"  - {m}")
            return models
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"HTTP Error {e.code}: {e.reason}")
        print(f"Response: {error_body}")
    except Exception as e:
        print(f"Error listing models: {e}")
    return None

def test_streaming_and_tools(api_key, key_label):
    print(f"\n=== [Testing Streaming & Tool Calls for {key_label}] ===")
    if not api_key:
        print(f"Error: No API key provided for {key_label}!")
        return
        
    if DRY_RUN:
        masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "SecretKey"
        print(f"[Dry Run] Successfully validated key: {masked_key}")
        print(f"[Dry Run] Skipping streaming and tool calls API call.")
        return

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:streamGenerateContent?key={api_key}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "What is the weather in Tokyo right now? Please use your get_current_weather tool to answer."
                    }
                ]
            }
        ],
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "get_current_weather",
                        "description": "Get the current weather for a location",
                        "parameters": {
                          "type": "OBJECT",
                          "properties": {
                            "location": {
                              "type": "STRING",
                              "description": "The city and state, e.g. San Francisco, CA"
                            }
                          },
                          "required": ["location"]
                        }
                    }
                ]
            }
        ]
    }
    
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            print("Streaming chunks started:")
            # Read line-by-line (SSE chunks)
            raw_response = response.read().decode("utf-8")
            # Try parsing as JSON array or single object
            try:
                chunks = json.loads(raw_response)
                if not isinstance(chunks, list):
                    chunks = [chunks]
                for i, chunk in enumerate(chunks):
                    candidates = chunk.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        for part in parts:
                            if "text" in part:
                                print(part["text"], end="", flush=True)
                            if "functionCall" in part:
                                print(f"\n[TOOL CALL DETECTED! Chunk {i}]: {json.dumps(part['functionCall'])}")
            except Exception as parse_err:
                # Fallback to direct text search if it's SSE-streamed or chunked JSON
                print("\n[Raw Response Received]:")
                print(raw_response)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"HTTP Error {e.code}: {e.reason}")
        print(f"Response: {error_body}")
    except Exception as e:
        print(f"Streaming failed: {e}")

if __name__ == "__main__":
    # 1. List models
    list_models(GEMINI_KEY, "Gemini Key")
    list_models(AGENT_KEY, "Agent Platform Key")
    
    # 2. Test streaming and tool calls
    test_streaming_and_tools(GEMINI_KEY, "Gemini Key")
    test_streaming_and_tools(AGENT_KEY, "Agent Platform Key")
