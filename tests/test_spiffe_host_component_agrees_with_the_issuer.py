"""The SPIFFE host component must encode identically in Python and in shell.

Two implementations of one encoding, on opposite sides of the trust boundary:

* `control_plane.identity.spiffe_host_component` — the **verifier**. Decides
  whether a presented SVID belongs to the host it claims.
* `infra/spire/register-host.sh` — the **issuer**. Decides what SPIFFE ID the
  host is registered under in the first place.

If they disagree, a host is registered under one ID and checked against
another, so it is refused an identity it genuinely holds — and the failure
appears at cutover, as an authentication problem, on a system where every
individual piece looks correct.

They did disagree. `str.isalnum()` is Unicode-aware and kept characters that
`sed 's/[^A-Za-z0-9_-]/-/g'` replaced (`日本-gpu` → `日本-gpu` vs `---gpu`).
Worse, the shell side was locale-dependent: the identical command yields
`café-gpu` under `en_US.UTF-8` and `caf---gpu` under `C`, so the answer
depended on the operator's terminal.

This test runs *both* implementations over the same inputs. Asserting about the
source text of either one would not have caught the locale bug.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control_plane.identity import spiffe_host_component  # noqa: E402

SCRIPT = ROOT / "infra" / "spire" / "register-host.sh"

CASES = [
    "aaryn-tuf-rtx2060",
    "5f41ab8b-126",
    "host.with.dots",
    "UPPER_Case",
    "trailing-",
    "_leading",
    "spaces in it",
    "weird id!#",
    "café-gpu",
    "münchen-01",
    "日本-gpu",
    "emoji-🎮-gpu",
    "mixed.é日!_-",
]


def _shell_component(host_id: str, *, locale: str) -> str:
    """Run the issuer's own function, in a given locale."""
    script = SCRIPT.read_text(encoding="utf-8")
    start = script.index("sanitize_host_component() {")
    end = script.index("}", start) + 1
    fn = script[start:end]
    proc = subprocess.run(
        ["bash", "-c", f'{fn}\nsanitize_host_component "$1"', "_", host_id],
        capture_output=True,
        text=True,
        timeout=15,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": locale},
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
@pytest.mark.parametrize("host_id", CASES)
@pytest.mark.parametrize("locale", ["C", "en_US.UTF-8", "C.UTF-8"])
def test_both_sides_encode_a_host_id_identically(host_id: str, locale: str) -> None:
    assert _shell_component(host_id, locale=locale) == spiffe_host_component(host_id), (
        f"issuer and verifier disagree on {host_id!r} under LC_ALL={locale}: the host "
        "would be registered under one SPIFFE ID and verified against another"
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
@pytest.mark.parametrize("host_id", CASES)
def test_the_issuer_does_not_depend_on_the_operators_locale(host_id: str) -> None:
    """Registering the same host from two terminals must give one SPIFFE ID."""
    results = {loc: _shell_component(host_id, locale=loc) for loc in ("C", "en_US.UTF-8")}
    assert len(set(results.values())) == 1, (
        f"{host_id!r} encodes differently per locale: {results}"
    )
