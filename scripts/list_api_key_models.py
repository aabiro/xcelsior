#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai

def main():
    # Load .env file
    dotenv_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(dotenv_path=dotenv_path)

    # Check for dry run
    DRY_RUN = os.environ.get("DRY_RUN", "0").lower() in ("1", "true", "yes") or "--dry-run" in sys.argv

    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY not found in environment or .env!", file=sys.stderr)
            sys.exit(1)

        if DRY_RUN:
            masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "SecretKey"
            print(f"[Dry Run] Successfully validated GEMINI_API_KEY: {masked_key}")
            print("[Dry Run] Skipping API list models call.")
            return

        client = genai.Client(api_key=api_key)
        
        print("Querying all available models for the provided API key...")
        models = client.models.list()
        
        print("\n=== [AVAILABLE MODELS] ===")
        for model in models:
            # Let's dynamically print attributes that exist
            name = getattr(model, 'name', 'N/A')
            display_name = getattr(model, 'display_name', 'N/A')
            print(f"- Name: {name}")
            print(f"  Display Name: {display_name}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Failed to list models: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
