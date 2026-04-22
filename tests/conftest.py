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

# B1 — agent auth bypass is now an explicit opt-in (see routes/agent.py).
# Tests that hit /agent/* endpoints without a bearer token need this flag
# set regardless of which .env was loaded. Also pin XCELSIOR_ENV=test so
# helpers that branch on it (e.g. logging) still behave.
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_ALLOW_UNAUTH_AGENT", "1")

# Exclude live E2E test scripts from pytest collection
collect_ignore = ["test_e2e_live.py"]
