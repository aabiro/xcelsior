"""Unit tests for SSH credential protection and presentation."""

import pytest
from instance_credentials import present_ssh_status, protect_ssh_status
from routes.instances import _enrich_instance


def test_protect_ssh_status_encrypts_password():
    raw_status = {
        "sshd_started": True,
        "key_count": 1,
        "root_password": "secret_password_123!",
        "summary": "SSH daemon running",
    }
    protected = protect_ssh_status(raw_status)
    assert "root_password" not in protected
    assert "root_password_encrypted" in protected
    assert protected["root_password_encrypted"] != "secret_password_123!"
    assert protected["sshd_started"] is True
    assert protected["key_count"] == 1


def test_protect_ssh_status_handles_empty_password():
    raw_status = {
        "sshd_started": True,
        "root_password": "",
    }
    protected = protect_ssh_status(raw_status)
    assert "root_password" not in protected
    assert "root_password_encrypted" not in protected


def test_present_ssh_status_masked_by_default():
    raw_status = {"sshd_started": True, "root_password": "secret_password_123!"}
    protected = protect_ssh_status(raw_status)

    # reveal=False should strip both root_password and root_password_encrypted
    presented = present_ssh_status(protected, reveal=False)
    assert "root_password" not in presented
    assert "root_password_encrypted" not in presented
    assert presented["sshd_started"] is True


def test_present_ssh_status_strips_legacy_plaintext_by_default():
    legacy_status = {"sshd_started": True, "root_password": "plaintext_legacy"}
    presented = present_ssh_status(legacy_status, reveal=False)
    assert "root_password" not in presented
    assert "root_password_encrypted" not in presented


def test_present_ssh_status_reveals_when_authorized():
    raw_status = {"sshd_started": True, "root_password": "secret_password_123!"}
    protected = protect_ssh_status(raw_status)

    revealed = present_ssh_status(protected, reveal=True)
    assert revealed["root_password"] == "secret_password_123!"
    assert "root_password_encrypted" not in revealed


def test_present_ssh_status_reveals_legacy_plaintext_when_authorized():
    legacy_status = {"sshd_started": True, "root_password": "plaintext_legacy"}
    revealed = present_ssh_status(legacy_status, reveal=True)
    assert revealed["root_password"] == "plaintext_legacy"


def test_enrich_instance_ssh_status_masking():
    raw_status = {"sshd_started": True, "root_password": "secret_password_123!"}
    protected = protect_ssh_status(raw_status)

    instance = {
        "job_id": "inst-test-123",
        "status": "running",
        "ssh_status": protected,
    }
    host_map = {}

    # List or telemetry default: password must be masked
    enriched_list = _enrich_instance(dict(instance), host_map, reveal_ssh_password=False)
    assert "root_password" not in enriched_list["ssh_status"]
    assert "root_password_encrypted" not in enriched_list["ssh_status"]

    # Detail GET /instance/{job_id}: password must be revealed
    enriched_detail = _enrich_instance(dict(instance), host_map, reveal_ssh_password=True)
    assert enriched_detail["ssh_status"]["root_password"] == "secret_password_123!"
    assert "root_password_encrypted" not in enriched_detail["ssh_status"]


def test_ssh_status_handles_none_and_invalid_types():
    assert protect_ssh_status(None) == {}
    assert protect_ssh_status("not-a-dict") == {}
    assert present_ssh_status(None) == {}
    assert present_ssh_status([]) == {}
