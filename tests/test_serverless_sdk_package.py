"""The SDK-only image must import without API/database modules installed."""

from pathlib import Path
import shutil
import subprocess
import sys
import json
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest


def test_sdk_imports_from_the_minimal_worker_image_package(tmp_path):
    source = Path(__file__).resolve().parents[1] / "api" / "serverless"
    package = tmp_path / "serverless"
    package.mkdir()
    shutil.copy2(source / "__init__.py", package / "__init__.py")
    shutil.copytree(
        source / "worker_sdk", package / "worker_sdk", ignore=shutil.ignore_patterns("__pycache__")
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from serverless.worker_sdk import WorkerClient, run_worker; "
            "assert callable(run_worker); "
            "assert not hasattr(WorkerClient, 'signal_ready'); "
            "assert 'serverless.repo' not in sys.modules; "
            "assert 'serverless.service' not in sys.modules",
            str(tmp_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_service_exports_remain_available():
    import serverless
    from serverless.repo import ServerlessRepo
    from serverless.service import ServerlessService, get_serverless_service

    assert serverless.ServerlessRepo is ServerlessRepo
    assert serverless.ServerlessService is ServerlessService
    assert serverless.get_serverless_service is get_serverless_service


@pytest.mark.parametrize("state,status", [(True, 200), (False, 503), (None, 503)])
def test_queue_sdk_health_checks_run_without_bound_handler_arguments(state, status):
    from serverless.worker_sdk.health import start_health_server

    def checker():
        if state is None:
            raise RuntimeError("model not loaded")
        return {"ok": state}

    server = start_health_server(0, checker=checker)
    try:
        try:
            response = urlopen(f"http://127.0.0.1:{server.server_port}/healthz", timeout=3)
        except HTTPError as exc:
            response = exc
        with response:
            assert response.status == status
            assert json.loads(response.read())["ok"] is (state is True)
    finally:
        server.shutdown()
        server.server_close()
