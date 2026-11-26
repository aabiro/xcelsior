#!/usr/bin/env python3
"""Tag a component release, write its notes, and checksum and sign its files.

Xcelsior ships six things on their own version lines. Until now only the
platform's 1.0.0 was ever tagged, so nothing recorded which commit became which
release, and nothing a downloader could check came with the npm packages. This
script is the release path for every component, and it runs locally: hosted
GitHub Actions does not run on this repository.

    python3 api/scripts/release.py status
    python3 api/scripts/release.py notes wizard
    python3 api/scripts/release.py tag wizard
    python3 api/scripts/release.py pack wizard
    python3 api/scripts/release.py verify dist/release/wizard-v0.1.11

``tag`` makes an annotated tag on HEAD, signed when git has a signing key, with
the release notes as its message. It refuses unless the tree is clean and HEAD is
``origin/main``: the deploy ships the working tree, so a tag has to name the
commit that is actually out there. It never pushes; it prints the push.

``pack`` builds an npm component the way ``npm publish`` would, packs it, writes
a CycloneDX SBOM, and writes ``SHA256SUMS``. With the release signing key it also
writes ``SHA256SUMS.sig``: a raw Ed25519 signature over ``SHA256SUMS`` that
``openssl`` can check against ``api/scripts/release-signing.pub``.

The release key is not the worker agent's key (``agent-signing.key``). The worker
trusts that one to replace its own code on providers' hosts; it signs nothing
else, so a leak of the release key cannot be used to push code to the fleet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RELEASE_DIR = ROOT / "dist" / "release"
PUBLIC_KEY = HERE / "release-signing.pub"
DEFAULT_PRIVATE_KEY = Path.home() / ".config" / "xcelsior" / "release-signing.key"
SUMS = "SHA256SUMS"


# ── Components ────────────────────────────────────────────────────────────────


def _package_json(path: str) -> Callable[[], str]:
    return lambda: json.loads((ROOT / path / "package.json").read_text(encoding="utf-8"))["version"]


def _pyproject() -> str:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def _agent_version() -> str:
    match = re.search(
        r'^VERSION = "(\d+\.\d+\.\d+)"$',
        (ROOT / "workers" / "worker_agent.py").read_text(encoding="utf-8"),
        re.M,
    )
    if match is None:
        raise SystemExit("workers/worker_agent.py no longer defines VERSION")
    return match.group(1)


def _agent_paths() -> tuple[str, ...]:
    """The worker distribution: exactly the files the agent signer signs."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_xcelsior_agent_signer", HERE / "sign-static-agent.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(str(module.source(name).relative_to(ROOT)) for name in module.SIGNABLE)


@dataclass(frozen=True)
class Component:
    name: str
    tag_prefix: str
    version: Callable[[], str]
    paths: Callable[[], tuple[str, ...]]
    #: Directory ``npm pack`` runs in, for components published to npm.
    npm_dir: str | None = None

    def tag(self, version: str | None = None) -> str:
        return f"{self.tag_prefix}{version or self.version()}"


# Tag prefixes are fixed by the workflows that react to them: ``v*`` publishes
# the Python package, images and SDK; ``mcp-v*`` publishes ``@xcelsior-gpu/mcp``.
# Every other prefix is chosen so it matches neither pattern.
COMPONENTS: dict[str, Component] = {
    c.name: c
    for c in (
        Component("platform", "v", _pyproject, lambda: (".",)),
        Component("agent", "agent-v", _agent_version, _agent_paths),
        Component("mcp-server", "mcp-server-v", _package_json("api/mcp"), lambda: ("api/mcp",)),
        Component(
            "mcp",
            "mcp-v",
            _package_json("api/mcp-cli"),
            lambda: ("api/mcp-cli", "api/mcp"),
            npm_dir="api/mcp-cli",
        ),
        Component(
            "wizard",
            "wizard-v",
            _package_json("api/wizard"),
            lambda: ("api/wizard",),
            npm_dir="api/wizard",
        ),
        Component("desktop", "desktop-v", _package_json("api/desktop"), lambda: ("api/desktop",)),
    )
}


# ── Git ───────────────────────────────────────────────────────────────────────


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def previous_tag(component: Component, before: str = "HEAD") -> str | None:
    """The newest tag of this component that is an ancestor of ``before``."""
    pattern = re.compile(rf"^{re.escape(component.tag_prefix)}(\d+\.\d+\.\d+)$")
    versions = {}
    for tag in git("tag", "--list", f"{component.tag_prefix}*").splitlines():
        match = pattern.match(tag)
        if match:
            versions[tag] = _version_key(match.group(1))
    for tag in sorted(versions, key=versions.__getitem__, reverse=True):
        if (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", tag, before], cwd=ROOT, capture_output=True
            ).returncode
            == 0
        ):
            return tag
    return None


# ── Release notes ─────────────────────────────────────────────────────────────

_CONVENTIONAL = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?: (?P<subject>.+)$"
)

SECTIONS = (
    ("breaking", "Breaking changes"),
    ("feat", "Added"),
    ("fix", "Fixed"),
    ("perf", "Faster"),
)


@dataclass(frozen=True)
class Commit:
    sha: str
    subject: str
    body: str


def commits_between(since: str | None, until: str, paths: tuple[str, ...]) -> list[Commit]:
    revision = f"{since}..{until}" if since else until
    raw = git("log", "--no-merges", "--format=%H%x1f%s%x1f%b%x1e", revision, "--", *paths)
    commits = []
    for record in raw.split("\x1e"):
        record = record.strip("\n")
        if not record:
            continue
        sha, subject, body = (record.split("\x1f") + ["", ""])[:3]
        commits.append(Commit(sha, subject, body))
    return commits


def render_notes(title: str, commits: list[Commit]) -> str:
    """Group conventional commits into a changelog section.

    A commit is breaking when its type carries ``!`` or its body has a
    ``BREAKING CHANGE:`` footer. Types other than feat, fix and perf (docs,
    chore, test, refactor, style) are left out of the notes; they are still in
    the tag's history.
    """
    grouped: dict[str, list[str]] = {key: [] for key, _ in SECTIONS}
    for commit in commits:
        match = _CONVENTIONAL.match(commit.subject)
        if match is None:
            continue
        line = match.group("subject")
        if match.group("scope"):
            line = f"**{match.group('scope')}:** {line}"
        line = f"- {line} ({commit.sha[:8]})"
        if match.group("bang") or "BREAKING CHANGE:" in commit.body:
            grouped["breaking"].append(line)
        if match.group("type") in grouped:
            grouped[match.group("type")].append(line)
    out = [f"## {title}", ""]
    for key, heading in SECTIONS:
        if grouped[key]:
            out += [f"### {heading}", "", *grouped[key], ""]
    if len(out) == 2:
        out += ["No user-facing changes.", ""]
    return "\n".join(out)


def notes_for(component: Component, since: str | None = None) -> str:
    since = since or previous_tag(component)
    commits = commits_between(since, "HEAD", component.paths())
    title = f"{component.name} {component.version()}"
    return render_notes(title, commits)


# ── Checksums and signatures ──────────────────────────────────────────────────


def write_checksums(directory: Path) -> Path:
    """``sha256sum``-format checksums of every file in ``directory``."""
    lines = []
    for path in sorted(
        p for p in directory.iterdir() if p.is_file() and p.name not in (SUMS, f"{SUMS}.sig")
    ):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    sums = directory / SUMS
    sums.write_text("".join(lines), encoding="utf-8")
    return sums


def _load_private_key(path: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise SystemExit(f"{path} is not an Ed25519 private key")
    return key


def _load_public_key(path: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    key = serialization.load_pem_public_key(path.read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise SystemExit(f"{path} is not an Ed25519 public key")
    return key


def sign_checksums(sums: Path, private_key: Path, public_key: Path = PUBLIC_KEY) -> Path:
    key = _load_private_key(private_key)
    if public_key.exists():
        published = _load_public_key(public_key).public_bytes_raw()
        if key.public_key().public_bytes_raw() != published:
            raise SystemExit(f"{private_key} does not match the published {public_key.name}")
    signature = sums.with_name(f"{SUMS}.sig")
    signature.write_bytes(key.sign(sums.read_bytes()))
    return signature


def verify_directory(directory: Path, public_key: Path = PUBLIC_KEY) -> list[str]:
    """Check every checksum, and the signature when there is one. Returns problems."""
    problems = []
    sums = directory / SUMS
    if not sums.exists():
        return [f"{sums} is missing"]
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        target = directory / name
        if not target.exists():
            problems.append(f"{name}: listed but missing")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            problems.append(f"{name}: checksum does not match")
    signature = directory / f"{SUMS}.sig"
    if signature.exists():
        from cryptography.exceptions import InvalidSignature

        try:
            _load_public_key(public_key).verify(signature.read_bytes(), sums.read_bytes())
        except InvalidSignature:
            problems.append(f"{SUMS}.sig does not verify against {public_key}")
    else:
        problems.append(f"{SUMS}.sig is missing: the checksums are unsigned")
    return problems


# ── Commands ──────────────────────────────────────────────────────────────────


def _component(name: str) -> Component:
    if name not in COMPONENTS:
        raise SystemExit(f"unknown component {name!r}; one of: {', '.join(COMPONENTS)}")
    return COMPONENTS[name]


def cmd_status(_: argparse.Namespace) -> int:
    print(
        f"{'component':<11} {'version':<8} {'tag':<20} {'tagged':<7} {'previous tag':<20} unreleased commits"
    )
    for component in COMPONENTS.values():
        tag = component.tag()
        tagged = bool(git("tag", "--list", tag))
        previous = previous_tag(component)
        pending = len(commits_between(previous, "HEAD", component.paths())) if previous else "-"
        print(
            f"{component.name:<11} {component.version():<8} {tag:<20} {'yes' if tagged else 'no':<7} "
            f"{previous or '(none)':<20} {pending}"
        )
    return 0


def cmd_notes(args: argparse.Namespace) -> int:
    component = _component(args.component)
    if args.since is None and previous_tag(component) is None:
        raise SystemExit(
            f"{component.name} has never been tagged, so there is no previous release to start from. "
            "Pass --since <commit or tag> to say where these notes begin."
        )
    print(notes_for(component, args.since))
    return 0


def _signing_configured() -> bool:
    return bool(git("config", "--get", "user.signingkey", check=False))


def cmd_tag(args: argparse.Namespace) -> int:
    component = _component(args.component)
    tag = component.tag()
    if git("status", "--porcelain"):
        raise SystemExit(
            "the working tree has uncommitted changes; a tag must name a commit that was shipped"
        )
    if git("rev-parse", "--abbrev-ref", "HEAD") != "main":
        raise SystemExit("releases are tagged on main")
    git("fetch", "--quiet", "origin", "main")
    if git("rev-parse", "HEAD") != git("rev-parse", "origin/main"):
        raise SystemExit(
            "HEAD is not origin/main; push first, so the tag names the commit everyone has"
        )
    if git("tag", "--list", tag):
        raise SystemExit(f"{tag} already exists; bump {component.name}'s version for a new release")
    previous = previous_tag(component)
    if previous and _version_key(component.version()) <= _version_key(
        previous[len(component.tag_prefix) :]
    ):
        raise SystemExit(f"{tag} is not newer than {previous}")
    if args.since is None and previous is None:
        raise SystemExit(
            f"{component.name} has never been tagged; pass --since <commit or tag> for the notes"
        )
    notes = notes_for(component, args.since)
    signed = _signing_configured()
    if args.require_signature and not signed:
        raise SystemExit(
            "no git signing key is configured (user.signingkey); refusing an unsigned tag"
        )
    git(
        "tag",
        "-s" if signed else "-a",
        tag,
        "-m",
        f"{component.name} {component.version()}\n\n{notes}",
    )
    print(
        f"Created {'signed' if signed else 'UNSIGNED annotated'} tag {tag} at {git('rev-parse', '--short', 'HEAD')}."
    )
    print(f"Push it with:  git push origin refs/tags/{tag}")
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    component = _component(args.component)
    if component.npm_dir is None:
        raise SystemExit(
            f"{component.name} is not an npm package; pack covers {', '.join(c.name for c in COMPONENTS.values() if c.npm_dir)}"
        )
    out = (Path(args.out) if args.out else RELEASE_DIR / component.tag()).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; a release directory holds exactly one release")
    out.mkdir(parents=True, exist_ok=True)
    package = ROOT / component.npm_dir
    if component.name == "mcp":
        # What .github/workflows/mcp-publish.yml does: the CLI package ships the
        # server's build output as its own dist/.
        subprocess.run(["npm", "run", "build"], cwd=ROOT / "api" / "mcp", check=True)
        shutil.rmtree(package / "dist", ignore_errors=True)
        shutil.copytree(ROOT / "api" / "mcp" / "dist", package / "dist")
    # `npm pack` does not run prepublishOnly, which is where these packages build.
    subprocess.run(["npm", "run", "build"], cwd=package, check=True)
    subprocess.run(["npm", "pack", "--pack-destination", str(out)], cwd=package, check=True)
    sbom = subprocess.run(
        ["npm", "sbom", "--sbom-format", "cyclonedx", "--omit", "dev"],
        cwd=package,
        capture_output=True,
        text=True,
        check=True,
    )
    (out / f"{component.tag()}.cdx.json").write_text(sbom.stdout, encoding="utf-8")
    sums = write_checksums(out)
    key = Path(os.environ.get("XCELSIOR_RELEASE_SIGNING_KEY", DEFAULT_PRIVATE_KEY))
    if key.exists():
        sign_checksums(sums, key)
        print(f"Signed {sums.name} with {key}")
    else:
        print(
            f"No release signing key at {key}: {SUMS} is NOT signed. Run `release.py keygen` once."
        )
    print(f"Release files are in {out}")
    print(
        f"Attach them with:  gh release create {component.tag()} {out}/* "
        f"--title '{component.name} {component.version()}' --notes-file <(python3 api/scripts/release.py notes {component.name})"
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    problems = verify_directory(Path(args.directory), Path(args.public_key))
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        print(f"OK   every file in {args.directory} matches {SUMS}, and its signature verifies")
    return 1 if problems else 0


def cmd_keygen(args: argparse.Namespace) -> int:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Path(args.private_key)
    if private.exists() or PUBLIC_KEY.exists():
        raise SystemExit(
            "a release key already exists; rotating it is a deliberate act, not a keygen rerun"
        )
    key = Ed25519PrivateKey.generate()
    private.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    PUBLIC_KEY.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    print(f"Private key: {private} (keep it off this repository and back it up)")
    print(f"Public key:  {PUBLIC_KEY} (commit it; downloaders verify against it)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "status", help="each component's version, tag and unreleased commits"
    ).set_defaults(func=cmd_status)
    notes = sub.add_parser("notes", help="release notes since the component's previous tag")
    notes.add_argument("component")
    notes.add_argument("--since", help="start the notes here instead of at the previous tag")
    notes.set_defaults(func=cmd_notes)
    tag = sub.add_parser("tag", help="tag HEAD as the component's current version")
    tag.add_argument("component")
    tag.add_argument("--since", help="start the notes here instead of at the previous tag")
    tag.add_argument(
        "--require-signature", action="store_true", help="refuse to make an unsigned tag"
    )
    tag.set_defaults(func=cmd_tag)
    pack = sub.add_parser("pack", help="pack an npm component with SBOM, checksums and signature")
    pack.add_argument("component")
    pack.add_argument(
        "--out", help=f"output directory (default {RELEASE_DIR.relative_to(ROOT)}/<tag>)"
    )
    pack.set_defaults(func=cmd_pack)
    verify = sub.add_parser("verify", help="check a release directory's checksums and signature")
    verify.add_argument("directory")
    verify.add_argument("--public-key", default=str(PUBLIC_KEY))
    verify.set_defaults(func=cmd_verify)
    keygen = sub.add_parser("keygen", help="create the release signing key (once)")
    keygen.add_argument("--private-key", default=str(DEFAULT_PRIVATE_KEY))
    keygen.set_defaults(func=cmd_keygen)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
