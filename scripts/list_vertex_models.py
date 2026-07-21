#!/usr/bin/env python3
import os
import json
import urllib.request
import urllib.error
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
dotenv_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "xcelsior-502014")
REGION = "us-central1"
# Enterprise / Agent Platform key
AGENT_KEY = os.environ.get("AGENT_PLATFORM_API_KEY") or os.environ.get("AGENT_API_KEY")

# Check for dry run
DRY_RUN = os.environ.get("DRY_RUN", "0").lower() in ("1", "true", "yes") or "--dry-run" in sys.argv

def list_vertex_models_with_key(api_key, project_id, region):
    print(f"\n=== [Listing Enterprise Vertex AI Models for Project: {project_id}] ===")
    if not api_key:
        print("Error: AGENT_PLATFORM_API_KEY not found in environment or .env!", file=sys.stderr)
        sys.exit(1)

    if DRY_RUN:
        masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "SecretKey"
        print(f"[Dry Run] Successfully validated AGENT_PLATFORM_API_KEY: {masked_key}")
        print("[Dry Run] Skipping API list Vertex models call.")
        return ["vertex-model-placeholder"]

    # Vertex AI foundational models publisher URL supporting API keys on authorized projects
    url = f"https://{region}-aiplatform.googleapis.com/v1/locations/{region}/publishers/google/models?key={api_key}"
    
    try:
        req = urllib.request.Request(url, method="GET")
        req.add_header("Content-Type", "application/json")
        
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            models = data.get("models", [])
            print(f"Success! Found {len(models)} foundational models on Vertex AI endpoint.")
            for m in models:
                name = m.get("name", "unknown")
                display_name = m.get("displayName", "unknown")
                print(f"  - Model: {name} (Display: {display_name})")
            return models
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"HTTP Error {e.code}: {e.reason}")
        print(f"Response Body: {error_body}")
    except Exception as e:
        print(f"Request failed: {e}")
    return None

if __name__ == "__main__":
    list_vertex_models_with_key(AGENT_KEY, PROJECT_ID, REGION)
