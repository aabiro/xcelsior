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
            print("[Dry Run] Skipping API generation content call.")
            return

        # Initialize Google GenAI client (Developer/AI Studio API)
        client = genai.Client(api_key=api_key)
        
        print("Making exactly 1 API Test Call using the provided API Key with gemini-3.6-flash...")
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents='Hello! This is a test call verifying connection with the provided API key. Respond with "API Key Connection verified successfully!".',
        )
        print("\n=== [API Call Response] ===")
        print(response.text.strip())
        print("============================\n")
        
    except Exception as e:
        print(f"API Key test call failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
