"""What the first deploy after the Vultr outage did, and what stops it now.

On 2026-10-05 `scripts/deploy.sh` ran against production and, in order:

1. Deleted `/opt/xcelsior/.env`. The code sync rebuilds `/opt/xcelsior_new`
   from nothing and moves it over `/opt/xcelsior`; `.env` was never carried
   across, because the env push always re-created it afterwards. When the push
   was stopped (so the dev machine's sandbox Stripe mode would not replace
   production's live one), nothing re-created it.
2. Installed vhosts in the SNI-passthrough layout — listening on 8444 behind
   `stream-tls-router.conf` — onto a host whose nginx.conf has no stream block.
   xcelsior.ca left port 443, landed on the agent gateway's mTLS server, and
   answered "400 No required SSL certificate was sent" for about twelve minutes.
3. Was stopped by `deploy_docker`'s `.env` check before any container moved.

Recovery restored the previous vhosts and the `.env` from the deploy's own
backup, verified key-for-key by digest.

Found while recovering: rsync excluded only `.env` exactly, and `.dockerignore`
listed three names, so `.env.bak.*` backups of production's env and
`.env.staging.secrets` were in `/opt/xcelsior` *and inside the running API
image*.

These tests run the real shell and the real rsync rather than reading the
script, because each of these defects was a script that read fine.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "scripts" / "deploy.sh"


def _function(name: str) -> str:
    """The text of one shell function from deploy.sh."""
    text = DEPLOY.read_text()
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.M | re.S)
    assert match, f"{name} not found in deploy.sh"
    return match.group(0)


def _bash(script: str, **kw) -> subprocess.CompletedProcess:  # noqa: ANN003
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60, **kw)


# ── The ingress cutover is never half-installed ───────────────────────────

PROD_NGINX_CONF = """user www-data;
include /etc/nginx/modules-enabled/*.conf;
events { worker_connections 768; }
http {
    include /etc/nginx/mime.types;
    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;
}
"""


def _router_check(conf: str) -> bool:
    script = _function("nginx_conf_includes_tls_router") + "nginx_conf_includes_tls_router"
    return _bash(script, input=conf).returncode == 0


def test_the_repo_vhosts_need_the_router() -> None:
    """If this stops being true the guard is inert, not satisfied."""
    script = _function("vhosts_need_tls_router") + f'vhosts_need_tls_router "{ROOT}/nginx"'
    assert _bash(script).returncode == 0


def test_production_shape_nginx_conf_does_not_count_as_routed() -> None:
    assert not _router_check(PROD_NGINX_CONF)


def test_a_commented_include_is_the_state_before_the_cutover() -> None:
    assert not _router_check(PROD_NGINX_CONF + "# include /etc/nginx/stream-tls-router.conf;\n")


def test_an_active_top_level_include_counts() -> None:
    assert _router_check(PROD_NGINX_CONF + "include /etc/nginx/stream-tls-router.conf;\n")


def test_a_held_install_is_not_recorded_as_done() -> None:
    """Otherwise the next deploy sees nginx as unchanged and never retries."""
    body = _function("store_remote_deploy_hashes")
    assert "DEPLOY_NGINX_HELD" in body
    install = _function("install_nginx_configs")
    held = install.index("DEPLOY_NGINX_HELD=1")
    assert held < install.index("sudo cp"), "the hold must return before any vhost is copied"


# ── The production .env survives the directory swap ──────────────────────


def _remote_body(function: str) -> str:
    """The heredoc a function sends over SSH, pointed at a sandbox instead of /opt."""
    text = _function(function)
    body = text.split("<< 'EOF'\n", 1)[1].rsplit("\nEOF\n", 1)[0]
    return body.replace("sudo ", "")


@pytest.fixture
def opt(tmp_path: Path, monkeypatch) -> Path:  # noqa: ANN001
    root = tmp_path / "opt"
    (root / "xcelsior").mkdir(parents=True)
    (root / "xcelsior" / ".env").write_text("XCELSIOR_STRIPE_MODE=live\n")
    (root / "xcelsior" / ".env").chmod(0o600)
    return root


def _run_remote(function: str, opt: Path) -> None:
    # Placeholders first: the sandbox path itself lives under /tmp, so replacing
    # /tmp/ after /opt/ would rewrite the substitution.
    body = _remote_body(function).replace("/tmp/", "\0TMP\0").replace("/opt/", "\0OPT\0")
    body = body.replace("\0TMP\0", f"{opt}/tmp_").replace("\0OPT\0", f"{opt}/")
    body = body.replace('-o "$USER" -g "$USER" ', "")
    result = _bash(body)
    assert result.returncode == 0, result.stderr


def test_the_env_is_carried_into_the_new_tree(opt: Path) -> None:
    _run_remote("sync_code_preserve_remote_files_host", opt)
    # What the sync does: a fresh tree that never contained .env, then the swap.
    shutil.rmtree(opt / "xcelsior")
    (opt / "xcelsior_new").mkdir()
    _run_remote("sync_code_restore_preserved_files_host", opt)
    (opt / "xcelsior_new").rename(opt / "xcelsior")

    env = opt / "xcelsior" / ".env"
    assert env.read_text() == "XCELSIOR_STRIPE_MODE=live\n", "production .env lost in the swap"
    assert oct(env.stat().st_mode & 0o777) == "0o600"
    assert not (opt / "xcelsior_env_keep" / ".env").exists(), "a second copy was left behind"


def test_the_kept_copy_is_private_while_it_exists(opt: Path) -> None:
    _run_remote("sync_code_preserve_remote_files_host", opt)
    keep = opt / "xcelsior_env_keep"
    assert oct(keep.stat().st_mode & 0o777) == "0o700"
    assert oct((keep / ".env").stat().st_mode & 0o777) == "0o600"


def test_a_missing_env_stops_the_deploy_where_it_is_found() -> None:
    body = _function("sync_code_push_env_host")
    skip = re.search(r'if \[\[ -n "\$\{SKIP_PROD_ENV_PUSH:-\}" \]\]; then\n(.*?)\n    fi\n', body, re.S).group(1)
    assert "test -f /opt/xcelsior/.env" in skip and "|| error" in skip


# ── No env file ships from a developer machine ────────────────────────────

ENV_FILES = [
    ".env", ".env.bak.secrets-20260803-182458", ".env.staging.secrets", ".env.worker",
    ".env.test", ".env.audit", ".env.example", ".env.paypal-live.example",
]


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
def test_rsync_ships_only_env_templates(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    for name in ENV_FILES + ["api.py"]:
        (src / name).write_text("x")
    text = DEPLOY.read_text()
    opts = re.search(r"^RSYNC_EXCLUDES=\((.*?)^\)", text, re.M | re.S).group(1)
    filters = re.findall(r"--(?:include|exclude)='[^']*'", opts)
    result = _bash(f"rsync -a {' '.join(filters)} '{src}/' '{dst}/'")
    assert result.returncode == 0, result.stderr
    shipped = sorted(p.name for p in dst.iterdir())
    assert shipped == sorted([".env.example", ".env.paypal-live.example", "api.py"]), shipped


def _dockerignored(name: str) -> bool:
    """Docker's rule: the last matching pattern wins; `!` re-includes."""
    ignored = False
    for raw in (ROOT / ".dockerignore").read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        pattern = line[1:] if negate else line
        if fnmatch.fnmatch(name, pattern.lstrip("/")):
            ignored = not negate
    return ignored


@pytest.mark.parametrize("name", [n for n in ENV_FILES if not n.endswith(".example")])
def test_no_env_file_enters_the_image(name: str) -> None:
    assert _dockerignored(name), f"{name} would be baked into the API image"


def test_the_templates_still_do() -> None:
    assert not _dockerignored(".env.example")
