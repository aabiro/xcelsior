"""Shared pytest configuration for Xcelsior test suite.

Ensures the project root is on sys.path so test files can import
source modules (api, scheduler, billing, etc.) directly.

Loads .env.test so tests always use the test database and config.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path so `import scheduler`, `from api import app`, etc. work
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load test environment BEFORE any module imports touch os.environ
from dotenv import load_dotenv

_env_test = os.path.join(PROJECT_ROOT, ".env.test")
if os.path.exists(_env_test):
    load_dotenv(_env_test, override=True)
else:
    # Fall back to .env if .env.test doesn't exist
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)
