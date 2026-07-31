"""AI assistant configuration and safety constants (modularized from ai_assistant.py)."""

from __future__ import annotations

import hashlib
import hmac
import os

BASE_URL = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca")
SSH_HOST = os.environ.get("XCELSIOR_SSH_HOST", "connect.xcelsior.ca")
BASE_DOMAIN = BASE_URL.replace("https://", "").replace("http://", "").rstrip("/")
API_DOMAIN = f"api.{BASE_DOMAIN}"

# Back-compat aliases for ai_assistant imports
_BASE_URL = BASE_URL
_SSH_HOST = SSH_HOST
_BASE_DOMAIN = BASE_DOMAIN
_API_DOMAIN = API_DOMAIN

FEATURE_AI_ASSISTANT = os.environ.get("FEATURE_AI_ASSISTANT", "false").lower() in (
    "true",
    "1",
    "yes",
)
AI_PROVIDER = os.environ.get("AI_ASSISTANT_PROVIDER", "xai").strip().lower() or "xai"
AI_FALLBACK_PROVIDERS = os.environ.get("AI_ASSISTANT_FALLBACK_PROVIDERS", "anthropic,openai")
AI_ENABLE_LIVE_CALLS = os.environ.get("AI_ASSISTANT_ENABLE_LIVE_CALLS", "false").lower() in (
    "true",
    "1",
    "yes",
)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
AI_MODEL = os.environ.get("AI_ASSISTANT_MODEL", "claude-haiku-4-20250414")
AI_MAX_TOKENS = int(os.environ.get("AI_ASSISTANT_MAX_TOKENS") or "4096")
AI_RATE_LIMIT = int(os.environ.get("AI_ASSISTANT_RATE_LIMIT", "20"))
CONFIRMATION_TTL_SEC = 300
RAG_CHUNK_SIZE = 400
RAG_CHUNK_OVERLAP = 50

# ── Support documentation corpus ──────────────────────────────────────
# The corpus Xcel AI can search is an allowlist, never a directory scan.
# docs/ holds internal material — business strategy, financial proposals,
# the control-plane blueprint, security and site audits — and any signed-up
# user who can reach the assistant could surface it by asking the right
# question. Only what is already published, plus supplemental user-facing
# pages written for support, is eligible to be ingested.
#
# fern/pages is the published documentation site, so it is public by
# construction: anything added there is already visible at docs.xcelsior.ca.
# support_docs holds user-facing pages covering features the published site
# does not yet document. Adding a directory here is a disclosure decision.
PUBLIC_DOC_GLOBS: tuple[tuple[str, str], ...] = (
    ("fern/pages", "*.mdx"),
    ("support_docs", "*.mdx"),
)
PUBLIC_DOC_FILES: tuple[str, ...] = ("llms.txt",)

# Repository roots that must never contribute to the corpus, at any depth.
# Enforced at ingest time and asserted by tests/test_ai_assistant.py.
FORBIDDEN_DOC_ROOTS: frozenset[str] = frozenset({"docs"})
MAX_TOOL_ROUNDS = 5
AI_TEMPERATURE = 0.2
TEXT_PROVIDER_TIMEOUT = 60.0
TOOL_PROVIDER_TIMEOUT = 120.0
ERROR_BODY_PREVIEW_LEN = 200
RATE_LIMIT_WINDOW_SEC = 60
MAX_HISTORY_ROWS = 30
DEFAULT_CONVERSATION_LIMIT = 30
DEFAULT_MESSAGE_LIMIT = 50
DEFAULT_DOC_SEARCH_LIMIT = 5
DEFAULT_JOB_LIST_LIMIT = 10
MAX_MARKETPLACE_RESULTS = 10
MAX_GPU_RECOMMENDATIONS = 5
MAX_GPU_COUNT = 64
MAX_JOB_HOURS = 8760
MAX_DOCKER_IMAGE_LEN = 256
MAX_WIZARD_KV_VALUE_LEN = 500
MOCK_PREVIEW_MAX_LEN = 140
CONVERSATION_TITLE_MAX_LEN = 80
SUMMARY_MAX_TOKENS = 512
RECENT_USAGE_LIMIT = 5
DEFAULT_GPU_VRAM_GB = 24
DEFAULT_BASE_RATE_CAD = 0.45
KEY_PREVIEW_PREFIX_LEN = 9
KEY_PREVIEW_SUFFIX_LEN = 4

_CONFIRMATION_SECRET = os.environ.get("AI_CONFIRMATION_SECRET", "").encode() or os.urandom(32)

TEXT_PROVIDERS = {
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-4",
        "api_key": os.environ.get("AI_ASSISTANT_XAI_API_KEY", ""),
        "model": os.environ.get("AI_ASSISTANT_XAI_MODEL", ""),
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "api_key": os.environ.get("AI_ASSISTANT_OPENAI_API_KEY", ""),
        "model": os.environ.get("AI_ASSISTANT_OPENAI_MODEL", ""),
    },
}
SUPPORTED_AI_PROVIDERS = {"anthropic", *TEXT_PROVIDERS.keys()}
TOOL_CAPABLE_PROVIDERS = {"anthropic", "openai"}
WRITE_TOOLS = {"launch_job", "stop_job", "create_api_key", "revoke_api_key", "add_ssh_key"}


def sign_confirmation_id(cid: str) -> str:
    sig = hmac.new(_CONFIRMATION_SECRET, cid.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{cid}:{sig}"


def verify_confirmation_token(token: str) -> str | None:
    parts = token.split(":", 1)
    if len(parts) != 2:
        return None
    cid, sig = parts
    expected = hmac.new(_CONFIRMATION_SECRET, cid.encode(), hashlib.sha256).hexdigest()[:16]
    if not hmac.compare_digest(sig, expected):
        return None
    return cid