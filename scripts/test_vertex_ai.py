#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.oauth2.credentials import Credentials

def main():
    # Load .env file
    dotenv_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(dotenv_path=dotenv_path)

    # Check for dry run
    DRY_RUN = os.environ.get("DRY_RUN", "0").lower() in ("1", "true", "yes") or "--dry-run" in sys.argv

    try:
        token = os.environ.get("VERTEX_OAUTH_TOKEN")
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "xcelsior-502014")

        if not token:
            print("Error: VERTEX_OAUTH_TOKEN not found in environment or .env!", file=sys.stderr)
            sys.exit(1)

        if DRY_RUN:
            masked_token = f"{token[:10]}...{token[-10:]}" if len(token) > 20 else "SecretToken"
            print(f"[Dry Run] Successfully validated VERTEX_OAUTH_TOKEN: {masked_token}")
            print(f"[Dry Run] Project ID: {project}")
            print("[Dry Run] Skipping Vertex Client initialization and model listing.")
            return

        # Instantiate Credentials with quota_project_id set to 'xcelsior-502014'
        creds = Credentials(token, quota_project_id=project)
        print(f"Loaded explicit OAuth Credentials with quota project: {project}")
        
        # Initialize Vertex GenAI client
        client = genai.Client(vertexai=True, project=project, location="us-central1", credentials=creds)
        
        print("\n=== [Retrieving Vertex AI Foundational Models via GenAI SDK] ===")
        models = client.models.list()
        count = 0
        for m in models:
            count += 1
            print(f"Model {count:02d}:")
            print(f"  Name: {m.name}")
            print(f"  Supported Actions: {m.supported_actions}")
            print(f"  Input Token Limit: {m.input_token_limit}")
            print("-" * 50)
            
        print(f"\nSuccessfully listed {count} foundational models.")
        
    except Exception as e:
        print(f"Error querying Vertex AI: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
