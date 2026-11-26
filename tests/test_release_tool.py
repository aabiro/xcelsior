"""The release tool tags the right things and its signatures mean something.

`api/scripts/release.py` is how a component release gets a tag, notes, an SBOM,
checksums and a signature. Two of its choices are load-bearing in ways a later
edit could break without noticing:

* Tag prefixes trigger workflows. `v*` publishes the Python package, images and
  SDK; `mcp-v*` publishes `@xcelsior-gpu/mcp`. A wizard or agent tag that
  matched either would publish the wrong thing the day hosted Actions runs.
* A signature that is not checked is worse than none. The Bastion's escrow
  script "signed" session recordings for months while writing no signature
  (CVE-2025-59339). These tests make a tampered file and a tampered checksum
  list both fail verification.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "_xcelsior_release", ROOT / "api" / "scripts" / "release.py"
)
assert _spec and _spec.loader
release = importlib.util.module_from_spec(_spec)
# Registered before it runs: its dataclasses look their module up by name.
sys.modules[_spec.name] = release
_spec.loader.exec_module(release)


@pytest.mark.parametrize("name", sorted(release.COMPONENTS))
def test_every_component_has_a_semver_version(name):
    assert re.fullmatch(r"\d+\.\d+\.\d+", release.COMPONENTS[name].version())


def test_only_the_platform_tag_triggers_the_v_star_publish_workflow():
    triggering = [c.name for c in release.COMPONENTS.values() if fnmatch.fnmatch(c.tag(), "v*")]
    assert triggering == ["platform"]


def test_only_the_mcp_package_tag_triggers_the_mcp_publish_workflow():
    triggering = [c.name for c in release.COMPONENTS.values() if fnmatch.fnmatch(c.tag(), "mcp-v*")]
    assert triggering == ["mcp"]


def test_the_workflows_still_listen_for_the_tags_this_tool_assumes():
    workflows = ROOT / ".github" / "workflows"
    assert 'tags: ["v*"]' in (workflows / "publish.yml").read_text(encoding="utf-8")
    assert 'tags: ["mcp-v*"]' in (workflows / "mcp-publish.yml").read_text(encoding="utf-8")


def test_agent_release_notes_cover_exactly_the_files_the_agent_signer_signs():
    paths = release.COMPONENTS["agent"].paths()
    assert "workers/worker_agent.py" in paths
    assert all((ROOT / path).is_file() for path in paths), paths


def _commit(subject: str, body: str = "", sha: str = "0123456789abcdef") -> object:
    return release.Commit(sha, subject, body)


def test_notes_group_features_fixes_and_breaking_changes_and_drop_chores():
    notes = release.render_notes(
        "wizard 1.2.0",
        [
            _commit("feat(sdk): regenerate the SDK"),
            _commit("fix: a refresh no longer signs everyone out"),
            _commit("feat!: drop the old flag"),
            _commit("fix(api): rename a field", body="BREAKING CHANGE: the field is renamed"),
            _commit("chore: bump a pin"),
            _commit("Merge something"),
        ],
    )
    added = notes.split("### Added")[1].split("###")[0]
    fixed = notes.split("### Fixed")[1].split("###")[0]
    breaking = notes.split("### Breaking changes")[1].split("###")[0]
    assert "**sdk:** regenerate the SDK (01234567)" in added
    assert "drop the old flag" in added and "drop the old flag" in breaking
    assert "rename a field" in fixed and "rename a field" in breaking
    assert "a refresh no longer signs everyone out" in fixed
    assert "bump a pin" not in notes and "Merge something" not in notes


def test_notes_with_nothing_user_facing_say_so():
    assert "No user-facing changes." in release.render_notes("x 1.0.1", [_commit("chore: tidy")])


@pytest.fixture
def keys(tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    private = tmp_path / "release.key"
    public = tmp_path / "release.pub"
    private.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    return private, public


@pytest.fixture
def release_dir(tmp_path):
    directory = tmp_path / "wizard-v9.9.9"
    directory.mkdir()
    (directory / "xcelsior-gpu-wizard-9.9.9.tgz").write_bytes(b"tarball bytes")
    (directory / "wizard-v9.9.9.cdx.json").write_text('{"bomFormat": "CycloneDX"}')
    return directory


def test_signed_checksums_verify(keys, release_dir):
    private, public = keys
    sums = release.write_checksums(release_dir)
    release.sign_checksums(sums, private, public)
    assert release.verify_directory(release_dir, public) == []
    assert sums.read_text().count("\n") == 2


def test_a_changed_file_fails_verification(keys, release_dir):
    private, public = keys
    release.sign_checksums(release.write_checksums(release_dir), private, public)
    (release_dir / "xcelsior-gpu-wizard-9.9.9.tgz").write_bytes(b"swapped by an attacker")
    assert release.verify_directory(release_dir, public) == [
        "xcelsior-gpu-wizard-9.9.9.tgz: checksum does not match"
    ]


def test_a_rewritten_checksum_list_fails_the_signature(keys, release_dir):
    import hashlib

    private, public = keys
    sums = release.write_checksums(release_dir)
    release.sign_checksums(sums, private, public)
    # The attack the October 2026 ovhcloud-cli hijack allowed: swap the file and
    # rewrite the checksum next to it. Only the signature catches this.
    tarball = release_dir / "xcelsior-gpu-wizard-9.9.9.tgz"
    tarball.write_bytes(b"swapped by an attacker")
    forged = sums.read_text().splitlines()
    forged[1] = f"{hashlib.sha256(tarball.read_bytes()).hexdigest()}  {tarball.name}"
    sums.write_text("\n".join(forged) + "\n")
    problems = release.verify_directory(release_dir, public)
    assert len(problems) == 1 and "does not verify" in problems[0]


def test_unsigned_checksums_are_reported_not_passed(keys, release_dir):
    _, public = keys
    release.write_checksums(release_dir)
    assert release.verify_directory(release_dir, public) == [
        "SHA256SUMS.sig is missing: the checksums are unsigned"
    ]


def test_signing_refuses_a_key_that_is_not_the_published_one(keys, release_dir, tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    _, public = keys
    other = tmp_path / "other.key"
    other.write_bytes(
        Ed25519PrivateKey.generate().private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    with pytest.raises(SystemExit, match="does not match"):
        release.sign_checksums(release.write_checksums(release_dir), other, public)
