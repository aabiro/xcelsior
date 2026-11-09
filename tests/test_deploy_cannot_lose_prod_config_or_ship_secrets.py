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
DEPLOY = ROOT / "api" / "scripts" / "deploy.sh"


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


def test_production_shape_nginx_conf_does_not_count_as_routed() -> None:
    assert not _router_check(PROD_NGINX_CONF)


def test_a_commented_include_is_the_state_before_the_cutover() -> None:
    assert not _router_check(PROD_NGINX_CONF + "# include /etc/nginx/stream-tls-router.conf;\n")


def test_an_active_top_level_include_counts() -> None:
    assert _router_check(PROD_NGINX_CONF + "include /etc/nginx/stream-tls-router.conf;\n")


def test_the_install_renders_for_the_detected_ingress_and_can_roll_back() -> None:
    """The router check now picks the rendering instead of holding the install.

    Holding left production's vhosts drifting from the repo for every release;
    tests/test_nginx_vhosts_install_safely.py runs the renderer and the
    installer for real.
    """
    install = _function("install_nginx_configs")
    assert "nginx_conf_includes_tls_router" in install
    assert "render_nginx_vhosts.py" in install and "install_nginx_vhosts.sh" in install
    assert "read_remote_api_colour" in install and "read_remote_mcp_colour" in install
    assert "sudo cp" not in install, "vhosts are copied by the installer, which can restore them"


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


def test_host_release_state_survives_the_swap(opt: Path) -> None:
    # Systemd units execute frozen operator scripts from here; the 2026-10-25
    # deploy deleted the directory and the nightly backup failed (203/EXEC).
    frozen = opt / "xcelsior" / "releases" / "cp" / "source" / "backup-db.sh"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("#!/bin/sh\n")
    frozen.chmod(0o700)
    _run_remote("sync_code_preserve_remote_files_host", opt)
    (opt / "xcelsior_new").mkdir()
    _run_remote("sync_code_restore_preserved_files_host", opt)
    shutil.rmtree(opt / "xcelsior")
    (opt / "xcelsior_new").rename(opt / "xcelsior")
    assert frozen.read_text() == "#!/bin/sh\n", "release state lost in the swap"
    assert oct(frozen.stat().st_mode & 0o777) == "0o700"


def test_the_kept_copy_is_private_while_it_exists(opt: Path) -> None:
    _run_remote("sync_code_preserve_remote_files_host", opt)
    keep = opt / "xcelsior_env_keep"
    assert oct(keep.stat().st_mode & 0o777) == "0o700"
    assert oct((keep / ".env").stat().st_mode & 0o777) == "0o600"


def test_a_missing_env_stops_the_deploy_where_it_is_found() -> None:
    body = _function("sync_code_push_env_host")
    skip = re.search(
        r'if \[\[ -n "\$\{SKIP_PROD_ENV_PUSH:-\}" \]\]; then\n(.*?)\n    fi\n', body, re.S
    ).group(1)
    assert "test -f /opt/xcelsior/.env" in skip and "|| error" in skip


# ── No env file ships from a developer machine ────────────────────────────

ENV_FILES = [
    ".env",
    ".env.bak.secrets-20260803-182458",
    ".env.staging.secrets",
    ".env.worker",
    ".env.test",
    ".env.audit",
    ".env.example",
    ".env.paypal-live.example",
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


@pytest.mark.parametrize("transport", ["rsync", "tar"])
def test_worker_signing_private_key_stays_on_the_signing_machine(
    tmp_path: Path, transport: str
) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "scripts").mkdir(parents=True)
    dst.mkdir()
    for name in ("agent-signing.key", "agent-signing.pub", "worker_agent.py.sig"):
        (src / "scripts" / name).write_text("fixture, not a key")
    if transport == "rsync":
        options = re.search(r"^RSYNC_EXCLUDES=\((.*?)^\)", DEPLOY.read_text(), re.M | re.S).group(1)
        filters = re.findall(r"--(?:include|exclude)='[^']*'", options)
        result = _bash(f"rsync -a {' '.join(filters)} '{src}/' '{dst}/'")
    else:
        filters = re.findall(r"--exclude='[^']*'", _function("sync_code_tarball"))
        result = _bash(f"tar -cf - {' '.join(filters)} -C '{src}' . | tar -xf - -C '{dst}'")
    assert result.returncode == 0, result.stderr
    assert not (dst / "scripts" / "agent-signing.key").exists()
    assert (dst / "scripts" / "agent-signing.pub").exists()
    assert (dst / "scripts" / "worker_agent.py.sig").exists()
    assert _dockerignored("scripts/agent-signing.key")
    assert not _dockerignored("scripts/agent-signing.pub")


# ── Local-only material stays on the machine it was written on ───────────
#
# Research notes and reports written into the checkout, and the Slurm kit's
# git-ignored site values, built packages, run results and rehearsal work dir
# (which holds a throwaway SSH private key), were all inside the tree deploy
# ships with rsync. git ignoring them meant nothing to rsync.

LOCAL_ONLY = [
    "reports/Slurm setup blueprint for Xcelsior.md",
    "research_notes/Slurm setup blueprint for Xcelsior/notes.md",
    "hpc/slurm-local/site.env",
    "hpc/slurm-local/debs/slurm-smd_25.11.8-1_amd64.deb",
    "hpc/slurm-local/results/r1/results.json",
    "hpc/slurm-local/rehearsal/.work/adapter_key",
]
STILL_SHIPPED = [
    "api.py",
    "hpc/slurm-local/README.md",
    "hpc/slurm-local/conf/xcelsior-profiles.json",
    "hpc/slurm-local/site.env.example",
    "hpc/slurm-local/evidence/r1/results.json",
    "hpc/slurm-local/rehearsal/run-rehearsal.sh",
]


def _ship(transport: str, src: Path, dst: Path) -> subprocess.CompletedProcess:
    """Copy src to dst with deploy.sh's own filters for that transport."""
    if transport == "rsync":
        options = re.search(r"^RSYNC_EXCLUDES=\((.*?)^\)", DEPLOY.read_text(), re.M | re.S).group(1)
        filters = re.findall(r"--(?:include|exclude)='[^']*'", options)
        return _bash(f"rsync -a {' '.join(filters)} '{src}/' '{dst}/'")
    filters = re.findall(r"--exclude='[^']*'", _function("sync_code_tarball"))
    env_list = src.parent / "env-excludes"
    script = (
        _function("_tar_env_excludes")
        + f"_tar_env_excludes '{src}' >'{env_list}'\n"
        + f"tar -cf - --exclude-from='{env_list}' {' '.join(filters)} -C '{src}' . | tar -xf - -C '{dst}'"
    )
    return _bash(script)


@pytest.mark.parametrize("transport", ["rsync", "tar"])
def test_local_only_material_never_ships(tmp_path: Path, transport: str) -> None:
    if transport == "rsync" and shutil.which("rsync") is None:
        pytest.skip("rsync not installed")
    src, dst = tmp_path / "src", tmp_path / "dst"
    dst.mkdir()
    for rel in LOCAL_ONLY + STILL_SHIPPED:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text("fixture")
    result = _ship(transport, src, dst)
    assert result.returncode == 0, result.stderr
    leaked = [rel for rel in LOCAL_ONLY if (dst / rel).exists()]
    assert not leaked, f"{transport} would ship {leaked}"
    missing = [rel for rel in STILL_SHIPPED if not (dst / rel).exists()]
    assert not missing, f"{transport} no longer ships {missing}"


@pytest.mark.parametrize(
    "rel",
    [
        "reports",
        "research_notes",
        "hpc/slurm-local/site.env",
        "hpc/slurm-local/debs",
        "hpc/slurm-local/results",
        "hpc/slurm-local/rehearsal/.work",
    ],
)
def test_local_only_material_stays_out_of_images(rel: str) -> None:
    assert _dockerignored(rel), f"{rel} would be baked into an image built from a working checkout"


def test_the_tarball_fallback_ships_only_env_templates(tmp_path: Path) -> None:
    """rsync re-includes the templates; tar can't, so it gets the explicit list."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "api").mkdir(parents=True)
    dst.mkdir()
    for name in ENV_FILES + ["api.py"]:
        (src / name).write_text("x")
    (src / "api" / ".env.bak.20261027").write_text("x")
    (src / "api" / ".env.example").write_text("x")
    result = _ship("tar", src, dst)
    assert result.returncode == 0, result.stderr
    shipped = sorted(p.name for p in dst.iterdir() if p.is_file())
    assert shipped == sorted([".env.example", ".env.paypal-live.example", "api.py"]), shipped
    assert sorted(p.name for p in (dst / "api").iterdir()) == [".env.example"]


# ── Nothing on the production path starts the bundled redis ───────────────
#
# The redeploy that followed stopped at the migration: `docker compose run --rm
# api …` without `--no-deps` started `api`'s `depends_on`, including the bundled
# redis, which cannot bind 127.0.0.1:6379 beside the host's own redis-server. It
# went unhealthy, the migration failed, and the container was left crash-looping
# with `restart: unless-stopped`, ready to take the port whenever the host's
# redis restarted.


def _compose_deps() -> dict[str, set[str]]:
    import yaml

    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
    direct = {}
    for name, svc in services.items():
        deps = svc.get("depends_on") or {}
        direct[name] = set(deps if isinstance(deps, list) else deps.keys())

    def closure(name: str, seen: set[str]) -> set[str]:
        for dep in direct.get(name, set()) - seen:
            seen.add(dep)
            closure(dep, seen)
        return seen

    return {name: closure(name, set()) for name in direct}


def test_the_premise_api_does_depend_on_the_bundled_redis() -> None:
    """If this stops being true, the guard below is guarding nothing."""
    assert "redis" in _compose_deps()["api"]


def _production_compose_calls() -> list[str]:
    calls = []
    for line in DEPLOY.read_text().splitlines():
        if line.lstrip().startswith("#") or "/opt/xcelsior" not in line:
            continue
        calls += re.findall(r"docker compose(?: [^;&|\"]*)? (?:run|up)\b[^;&|\"]*", line)
    return calls


def test_no_production_compose_call_can_start_redis() -> None:
    deps = _compose_deps()
    offenders = []
    for call in _production_compose_calls():
        if "--no-deps" in call:
            continue
        named = [w for w in call.split() if w in deps]
        if not named:
            offenders.append(f"{call!r} names no service, so it starts all of them")
            continue
        for service in named:
            if "redis" in deps[service] or service == "redis":
                offenders.append(f"{call!r} starts redis through {service}")
    assert not offenders, offenders
    assert _production_compose_calls(), "the scan found no compose calls — it is broken"


def test_rollback_cannot_take_production_down() -> None:
    """`compose down` then a bare `up -d` was a full outage in this layout."""
    body = _function("rollback")
    code = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#"))
    assert "compose" not in code or "down" not in code, "rollback still stops every service"
    assert not re.search(r"docker compose[^\n\"]*up -d\"", code), "rollback still runs a bare up"


# ── Production's env is kept unless pushing it is asked for ───────────────
#
# b8c6d21f went back to copying the local `.env` over production's. On
# 2026-11-02 that file lacked 49 production keys and pointed the database at
# the developer's own, so the next deploy would have taken production down.

GUARD = ("guard_prod_env", "env_digests", "env_drift_remote_py", "frontend_build_arg_names")
BUILD_ARGS = (
    "services:\n  frontend:\n    build:\n      args:\n"
    "        NEXT_PUBLIC_API_URL: ${NEXT_PUBLIC_API_URL:-}\n"
    "        NEXT_PUBLIC_POSTHOG_PROJECT_TOKEN: ${NEXT_PUBLIC_POSTHOG_PROJECT_TOKEN:-}\n"
)


def _guard(tmp_path: Path, local: str, prod: str, **env: str) -> subprocess.CompletedProcess:
    (tmp_path / "local.env").write_text(local)
    (tmp_path / "prod.env").write_text(prod)
    (tmp_path / "docker-compose.yml").write_text(BUILD_ARGS)
    stubs = (
        'log() { echo "LOG $*"; }; warn() { echo "WARN $*"; }; success() { echo "OK $*"; }\n'
        'error() { echo "ERROR $*"; exit 1; }\n'
        # The comparison runs on the host; here, against a stand-in for its file.
        f'ssh_cmd() {{ eval "${{1//\\/opt\\/xcelsior\\/.env/{tmp_path}/prod.env}}"; }}\n'
    )
    script = stubs + "".join(_function(name) for name in GUARD)
    script += 'guard_prod_env\necho "SKIP=${SKIP_PROD_ENV_PUSH:-}"\n'
    base = {k: v for k, v in os.environ.items() if k != "PROD_ENV_GUARDED"}
    return _bash(
        script,
        env={
            **base,
            "TARGET_ENV": "prod",
            "ENV_FILE": str(tmp_path / "local.env"),
            "PROJECT_DIR": str(tmp_path),
            **env,
        },
    )


def test_a_prod_deploy_keeps_production_env_and_reports_names_only(tmp_path: Path) -> None:
    result = _guard(
        tmp_path,
        "XCELSIOR_STRIPE_MODE=sandbox\n",
        "XCELSIOR_STRIPE_MODE=live\nXCELSIOR_REDIS_REQUIREPASS=not-a-real-password\n",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines()[-1] == "SKIP=1"
    assert "differ              XCELSIOR_STRIPE_MODE" in result.stdout
    assert "only in production  XCELSIOR_REDIS_REQUIREPASS" in result.stdout
    assert "not-a-real-password" not in result.stdout + result.stderr


def test_replacing_production_env_is_an_explicit_choice(tmp_path: Path) -> None:
    result = _guard(tmp_path, "A=1\n", "A=2\n", XCELSIOR_DEPLOY_PUSH_LOCAL_ENV="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines()[-1] == "SKIP="
    assert "WARN XCELSIOR_DEPLOY_PUSH_LOCAL_ENV=1" in result.stdout


@pytest.mark.parametrize(
    "local,prod",
    [
        (
            "NEXT_PUBLIC_API_URL=http://localhost:9500\n",
            "NEXT_PUBLIC_API_URL=https://xcelsior.ca\n",
        ),
        ("NEXT_PUBLIC_ONLY_HERE=1\n", ""),
    ],
)
def test_public_values_the_bundle_would_ship_are_refused(
    tmp_path: Path, local: str, prod: str
) -> None:
    result = _guard(tmp_path, local, prod)
    assert result.returncode == 1
    assert "ERROR NEXT_PUBLIC_* values differ from production" in result.stdout


def test_a_public_key_only_production_has_is_built_from_production(tmp_path: Path) -> None:
    built = _guard(tmp_path, "", "NEXT_PUBLIC_POSTHOG_PROJECT_TOKEN=phc_x\n")
    assert built.returncode == 0, built.stdout + built.stderr
    unbuilt = _guard(tmp_path, "", "NEXT_PUBLIC_NOT_A_BUILD_ARG=1\n")
    assert unbuilt.returncode == 1
    assert "NEXT_PUBLIC_NOT_A_BUILD_ARG is only in production" in unbuilt.stdout


def test_the_real_frontend_build_takes_productions_posthog_keys() -> None:
    script = _function("frontend_build_arg_names") + "frontend_build_arg_names\n"
    result = _bash(script, env={**os.environ, "PROJECT_DIR": str(ROOT)})
    assert result.returncode == 0, result.stderr
    assert {"NEXT_PUBLIC_POSTHOG_PROJECT_TOKEN", "NEXT_PUBLIC_POSTHOG_HOST"} <= set(
        result.stdout.split()
    )


# ── Slurm: placement is a flag, the profile ships on every deploy ─────────


def _rewrite(tmp_path: Path, text: str, flag: str | None = None) -> tuple:
    path = tmp_path / "prod.env"
    path.write_text(text)
    env = {k: v for k, v in os.environ.items() if k != "XCELSIOR_DEPLOY_SLURM_PLACEMENT"}
    if flag is not None:
        env["XCELSIOR_DEPLOY_SLURM_PLACEMENT"] = flag
    script = _function("rewrite_slurm_placement_env")
    script += f"rewrite_slurm_placement_env '{path}'\n"
    return _bash(script, env=env), path.read_text()


def test_a_normal_deploy_removes_placement_and_leaves_every_other_line(tmp_path: Path) -> None:
    original = (
        "XCELSIOR_POSTGRES_PASSWORD=kept-secret\n"
        "XCELSIOR_SLURM_PLACEMENT=true\n"
        "export XCELSIOR_SLURM_PLACEMENT=yes\n"
        "XCELSIOR_SLURM_KEY_XCL_LOCAL=kept-secret\n"
    )
    result, text = _rewrite(tmp_path, original)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "slurm_placement off"
    assert "kept-secret" not in result.stdout + result.stderr
    assert text == (
        "XCELSIOR_POSTGRES_PASSWORD=kept-secret\n"
        "XCELSIOR_SLURM_KEY_XCL_LOCAL=kept-secret\n"
    )
    again, same = _rewrite(tmp_path, text)
    assert again.returncode == 0
    assert same == text


def test_the_placement_flag_writes_one_line_and_nothing_else(tmp_path: Path) -> None:
    result, text = _rewrite(
        tmp_path,
        "XCELSIOR_SLURM_KEY_XCL_LOCAL=kept-secret\nXCELSIOR_SLURM_PLACEMENT=true\n",
        "1",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "slurm_placement on"
    assert "kept-secret" not in result.stdout + result.stderr
    assert text == (
        "XCELSIOR_SLURM_KEY_XCL_LOCAL=kept-secret\n"
        "XCELSIOR_SLURM_PLACEMENT=1\n"
    )


def test_a_bad_placement_flag_is_refused(tmp_path: Path) -> None:
    result, text = _rewrite(tmp_path, "A=1\n", "banana")
    assert result.returncode == 2
    assert text == "A=1\n"
    assert "must be 0 or 1" in result.stderr


def test_a_missing_env_file_is_left_missing(tmp_path: Path) -> None:
    script = _function("rewrite_slurm_placement_env")
    script += f"rewrite_slurm_placement_env '{tmp_path}/absent.env'\n"
    env = {k: v for k, v in os.environ.items() if k != "XCELSIOR_DEPLOY_SLURM_PLACEMENT"}
    result = _bash(script, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "slurm_placement missing_env"
    assert not (tmp_path / "absent.env").exists()


def test_prod_paths_apply_the_policy_before_hashing_the_runtime() -> None:
    text = DEPLOY.read_text()
    branches = {
        "--quick)": "--rollback)",
        "--post-merge)": "--systemd)",
        '""|--docker)': "\n        *)",
    }
    for start, end in branches.items():
        body = text.split(start, 1)[1].split(end, 1)[0]
        assert body.index("apply_slurm_placement_policy") < body.index("detect_deploy_inputs"), start
    systemd = text.split("--systemd)", 1)[1].split("--drain-host)", 1)[0]
    assert "apply_slurm_placement_policy" in systemd
    assert 'text += "XCELSIOR_SLURM_PLACEMENT=1\\n"' in text


def test_compose_forwards_slurm_and_leaves_placement_empty() -> None:
    import yaml

    services = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]
    for name in ("api", "api-blue", "bg-worker", "scheduler-worker"):
        assert services[name]["environment"]["XCELSIOR_SLURM_PLACEMENT"] == (
            "${XCELSIOR_SLURM_PLACEMENT:-}"
        ), name
    worker = services["bg-worker"]["environment"]
    assert worker["XCELSIOR_SLURM_PROFILES_FILE"] == (
        "${XCELSIOR_SLURM_PROFILES_FILE:-/app/hpc/slurm-local/conf/xcelsior-profiles.json}"
    )
    assert worker["XCELSIOR_SLURM_KEY_XCL_LOCAL"] == "${XCELSIOR_SLURM_KEY_XCL_LOCAL:-}"
    assert "XCELSIOR_SLURM_KEY_XCL_LOCAL" not in services["api"]["environment"]
    assert "XCELSIOR_SLURM_KEY_XCL_LOCAL" not in services["scheduler-worker"]["environment"]
    hashed = _function("api_build_hash")
    assert "hpc/slurm-local/conf/xcelsior-profiles.json" in hashed
