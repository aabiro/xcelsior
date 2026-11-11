"""NFS exports name the admitted hosts, never the whole mesh (CVP-01 T-01 step 3)."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "render_nfs_exports", Path(__file__).parent.parent / "api/scripts/render_nfs_exports.py"
)
exports = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exports)


def test_linux_line_names_each_host_and_no_subnet():
    line = exports.render("linux", "/exports/volumes", ["100.64.0.6", "100.64.0.4", "100.64.0.6"])
    assert line == (
        "/exports/volumes 100.64.0.4(rw,sync,no_subtree_check,no_root_squash) "
        "100.64.0.6(rw,sync,no_subtree_check,no_root_squash)\n"
    )
    assert "/10" not in line and "100.64.0.0" not in line


def test_mac_line_names_each_host_and_no_network():
    line = exports.render("mac", "/Volumes/InferenceData", ["100.64.0.6", "100.64.0.4"])
    assert line == "/Volumes/InferenceData -mapall=501:20 -alldirs 100.64.0.4 100.64.0.6\n"
    assert "-network" not in line and "-mask" not in line


@pytest.mark.parametrize("bad", ["100.64.0.0/10", "0.0.0.0", "192.168.1.5", "*", "host name"])
def test_a_subnet_wildcard_or_outside_address_is_refused(bad):
    with pytest.raises(ValueError):
        exports.render("linux", "/exports/volumes", ["100.64.0.6", bad])


def test_an_empty_host_list_is_refused():
    with pytest.raises(ValueError):
        exports.render("linux", "/exports/volumes", ["", " "])


def test_a_base_that_could_add_an_export_is_refused():
    with pytest.raises(ValueError):
        exports.render("linux", "/exports/volumes *(rw)", ["100.64.0.6"])


def test_cli_exit_codes(capsys):
    assert exports.main(["--format", "mac", "--base", "/Volumes/X", "100.64.0.6"]) == 0
    assert capsys.readouterr().out.startswith("/Volumes/X ")
    assert exports.main(["--format", "mac", "--base", "/Volumes/X", "10.0.0.1"]) == 2
