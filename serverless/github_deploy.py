# Xcelsior — GitHub source resolution for custom serverless workers (Phase 15)

from __future__ import annotations

import re

_GITHUB_HTTPS_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)
_GITHUB_SSH_RE = re.compile(
    r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$",
    re.IGNORECASE,
)


class GitHubSourceError(ValueError):
    pass


def parse_github_repo(url: str) -> tuple[str, str]:
    """Return (owner, repo) from a GitHub HTTPS or git@ URL."""
    raw = (url or "").strip()
    if not raw:
        raise GitHubSourceError("GitHub repository URL is required")
    for pattern in (_GITHUB_HTTPS_RE, _GITHUB_SSH_RE):
        match = pattern.match(raw)
        if match:
            return match.group("owner"), match.group("repo")
    raise GitHubSourceError("Unsupported GitHub URL — use https://github.com/owner/repo")


def resolve_github_image(repo_url: str, *, ref: str = "main") -> str:
    """
    Map a GitHub repo to a container image reference.

    Convention: GHCR image at ghcr.io/<owner>/<repo>:<ref> (user builds via CI).
    """
    owner, repo = parse_github_repo(repo_url)
    tag = (ref or "main").strip() or "main"
    tag = re.sub(r"[^a-zA-Z0-9._-]+", "-", tag)[:64]
    return f"ghcr.io/{owner.lower()}/{repo.lower()}:{tag}"


def apply_github_source(
    *,
    mode: str,
    image_ref: str,
    source_type: str,
    source_ref: str,
    source_ref_branch: str = "main",
) -> tuple[str, dict[str, str]]:
    """
    When custom mode uses GitHub deploy, derive image_ref if omitted.

    Returns (image_ref, env_overrides).
    """
    if mode != "custom" or (source_type or "").strip().lower() != "github":
        return image_ref, {}
    repo = (source_ref or "").strip()
    if not repo:
        raise GitHubSourceError("source_ref is required for GitHub deploy")
    resolved = image_ref.strip() or resolve_github_image(repo, ref=source_ref_branch)
    env = {
        "XCELSIOR_SOURCE_TYPE": "github",
        "XCELSIOR_SOURCE_REF": repo,
        "XCELSIOR_SOURCE_REF_BRANCH": (source_ref_branch or "main").strip() or "main",
    }
    return resolved, env