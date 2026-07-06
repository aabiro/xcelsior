"""B2 model-sync env contract."""

import os

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from serverless.b2_model_sync import b2_model_sync_config, default_b2_bucket


class TestB2ModelSyncConfig:
    def test_defaults_to_pixelenhance_models_bucket(self, monkeypatch):
        monkeypatch.delenv("B2_MODEL_SYNC_BUCKET", raising=False)
        assert default_b2_bucket() == "pixelenhance-models"

    def test_reads_model_sync_env(self, monkeypatch):
        monkeypatch.setenv("B2_MODEL_SYNC_BUCKET", "my-bucket")
        monkeypatch.setenv("B2_MODEL_SYNC_KEY_ID", "kid")
        monkeypatch.setenv("B2_MODEL_SYNC_KEY", "ksec")
        cfg = b2_model_sync_config()
        assert cfg.bucket == "my-bucket"
        assert cfg.key_id == "kid"
        assert cfg.app_key == "ksec"

    def test_falls_back_to_b2_application_keys(self, monkeypatch):
        monkeypatch.delenv("B2_MODEL_SYNC_KEY_ID", raising=False)
        monkeypatch.delenv("B2_MODEL_SYNC_KEY", raising=False)
        monkeypatch.setenv("B2_APPLICATION_KEY_ID", "app-id")
        monkeypatch.setenv("B2_APPLICATION_KEY", "app-key")
        cfg = b2_model_sync_config()
        assert cfg.key_id == "app-id"
        assert cfg.app_key == "app-key"