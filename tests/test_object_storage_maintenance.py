from hashlib import sha256
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import control_plane_watchdog, with_runtime_secrets
from scripts.object_storage_ops import fetch_latest_postgres
from workers.object_storage import ObjectStorageError, ObjectStoreBackend, StoredObject


def test_a_bad_object_leaves_no_cache_file(tmp_path):
    client = Mock()
    client.head_object.return_value = {
        "ContentLength": 5,
        "Metadata": {"sha256": sha256(b"hello").hexdigest()},
    }
    client.download_fileobj.side_effect = lambda _bucket, _key, dest: dest.write(b"wrong")
    with pytest.raises(ObjectStorageError, match="checksum"):
        ObjectStoreBackend(client, "bucket").download("volume/key", tmp_path / "cache")
    assert list(tmp_path.iterdir()) == []


def test_latest_postgres_ignores_other_databases_and_non_dump_objects(tmp_path, monkeypatch):
    monkeypatch.delenv("XCELSIOR_OBJECT_STORE_BACKUP_DB", raising=False)
    store = Mock()
    older = "backups/postgres/xcelsior_20260101_123400-" + "a" * 32 + ".dump"
    newer = "backups/postgres/xcelsior_20260102_123400-" + "b" * 32 + ".dump"
    store.keys.return_value = [newer, older, newer + ".partial", newer.replace("xcelsior", "other")]
    store.download.return_value = StoredObject(newer, 5, sha256(b"hello").hexdigest())
    downloaded = fetch_latest_postgres(store, tmp_path)
    store.download.assert_called_once_with(newer, downloaded)
    assert downloaded.with_suffix(".dump.sha256").read_text().endswith(downloaded.name + "\n")


def test_external_postgres_password_is_encoded_for_compose_and_raw_for_pg_dump():
    from cryptography.fernet import Fernet

    result = with_runtime_secrets._validate(
        {
            "XCELSIOR_POSTGRES_PASSWORD": "x%@/:",
            "XCELSIOR_SECRETS_KEY": Fernet.generate_key().decode(),
        }
    )
    assert result["XCELSIOR_POSTGRES_PASSWORD"] == "x%@/:"
    assert result["XCELSIOR_POSTGRES_DSN_PASSWORD"] == "x%25%40%2F%3A"


@pytest.mark.parametrize("status", ["ready", "unready"])
def test_watchdog_requires_both_probes_before_sending_a_success(monkeypatch, status):
    requests = []

    class Reply:
        def __init__(self, body):
            self.status = 200
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size):
            return json.dumps(self.body).encode()

    def open_url(request, timeout):
        requests.append(request.full_url)
        if request.full_url.endswith("/readyz"):
            return Reply({"status": status, "ok": True})
        return Reply({"status": "pass"})

    monkeypatch.setenv("XCELSIOR_HEALTHCHECKS_PING_URL", "https://example.test/fixture")
    monkeypatch.setenv("XCELSIOR_WATCHDOG_API_URL", "https://example.test/readyz")
    monkeypatch.setenv("XCELSIOR_WATCHDOG_HEADSCALE_URL", "https://example.test/health")
    monkeypatch.setattr(
        control_plane_watchdog, "build_opener", lambda _: SimpleNamespace(open=open_url)
    )
    assert control_plane_watchdog.main() == int(status != "ready")
    assert len(requests) == 3
    assert requests[-1] == "https://example.test/fixture" + ("/fail" if status != "ready" else "")
