"""Worker exits must not turn authenticated compatibility calls into false zeros."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import signal
import socket
import urllib.request

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from api import app

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "api/scripts/start_api.sh"


def child_env():
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "LC_ALL") if key in os.environ}
    # Tests are run by the isolated gate; copy its test DSNs, never dotenv or
    # unrelated credentials. API imports need the migrated disposable schema.
    for key in (
        "XCELSIOR_POSTGRES_DSN",
        "XCELSIOR_TEST_POSTGRES_DSN",
        "XCELSIOR_PG_DSN",
        "DATABASE_URL",
    ):
        if key in os.environ:
            env[key] = os.environ[key]
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                str(ROOT / part)
                for part in (
                    "archive",
                    "billing",
                    "workers",
                    "auth",
                    "hosts",
                    "privacy",
                    "api",
                    "core",
                    "",
                )
            ),
            "XCELSIOR_SKIP_DOTENV": "1",
            "PYTHON_DOTENV_DISABLED": "1",
            "XCELSIOR_ENV": "test",
            "XCELSIOR_BG_TASKS": "false",
            "XCELSIOR_DB_BACKEND": "postgres",
            "XCELSIOR_API_TOKEN": "isolated-metric-test-token",
            "XCELSIOR_AUTH_CACHE_BACKEND": "memory",
            "XCELSIOR_AUTH_REQUIRED": "false",
            "XCELSIOR_AGENT_HOST_TOKENS": "off",
            "XCELSIOR_ALLOW_UNAUTH_AGENT": "1",
            "XCELSIOR_SCHEDULER_MODE": "active",
            "XCELSIOR_RATE_LIMIT_REQUESTS": "5000",
            "XCELSIOR_AUTH_RATE_LIMIT_REQUESTS": "5000",
        }
    )
    return env


def run_child(code, env, *, wrapper=False):
    argv = [sys.executable, "-c", textwrap.dedent(code)]
    if wrapper:
        argv = ["/bin/sh", str(START), *argv]
    result = subprocess.run(argv, env=env, cwd=ROOT, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr[-5000:]
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.parametrize("kill_before_scrape", [False, True])
def test_http_scrape_keeps_requests_from_exited_workers(tmp_path, kill_before_scrape):
    env = child_env()
    env["PROMETHEUS_MULTIPROC_DIR"] = str(tmp_path)
    env["XCELSIOR_METRICS_GENERATION_STARTED"] = str(time.time())
    env["KILL_BEFORE_SCRAPE"] = str(int(kill_before_scrape))
    result = run_child(
        """
        import json,os,signal
        from fastapi.testclient import TestClient
        from prometheus_client import Gauge,Histogram
        from prometheus_client.parser import text_string_to_metric_families
        from api import app
        from control_plane.legacy_execution import record_use
        g=Gauge('aggregation_callback_probe','Callback must remain live')
        g.set_function(lambda:42)
        h=Histogram('aggregation_histogram_probe','Worker observations')
        children=[]
        for amount in (3,4):
            read_fd,write_fd=os.pipe()
            pid=os.fork()
            if pid==0:
                os.close(read_fd)
                record_use('status_patch','requested',amount)
                h.observe(amount)
                os.write(write_fd,b'1')
                if os.environ['KILL_BEFORE_SCRAPE']=='1':
                    signal.pause()
                os._exit(0)
            os.close(write_fd)
            assert os.read(read_fd,1)==b'1'
            os.close(read_fd)
            if os.environ['KILL_BEFORE_SCRAPE']=='1':
                os.kill(pid,signal.SIGKILL)
            os.waitpid(pid,0)
            children.append(pid)
        response=TestClient(app).get('/metrics/prometheus')
        assert response.status_code==200,response.text
        samples=[s for f in text_string_to_metric_families(response.text) for s in f.samples]
        def value(name,labels=None):
            matches=[s.value for s in samples if s.name==name and (labels is None or s.labels==labels)]
            assert len(matches)==1,(name,matches)
            return matches[0]
        print(json.dumps({
          'requests':value('xcelsior_legacy_execution_total',{'path':'status_patch','outcome':'requested'}),
          'histogram_count':value('aggregation_histogram_probe_count'),
          'histogram_sum':value('aggregation_histogram_probe_sum'),
          'callback':value('aggregation_callback_probe'),
          'work_retired':value('xcelsior_legacy_path_retired',{'path':'work_poll'}),
          'status_retired':value('xcelsior_legacy_path_retired',{'path':'status_patch'}),
          'aggregation':value('xcelsior_metrics_aggregation_enabled'),
          'generation':value('xcelsior_metrics_generation_started_timestamp_seconds'),
        }))
    """,
        env,
    )
    assert result.pop("generation") == float(env["XCELSIOR_METRICS_GENERATION_STARTED"])
    assert result == {
        "requests": 7,
        "histogram_count": 2,
        "histogram_sum": 7,
        "callback": 42,
        "work_retired": 1,
        "status_retired": 0,
        "aggregation": 1,
    }


def test_registry_failure_returns_unavailable_instead_of_partial_success(monkeypatch):
    def unavailable():
        raise RuntimeError("synthetic collector failure")

    monkeypatch.setattr(REGISTRY, "collect", unavailable)
    response = TestClient(app, raise_server_exceptions=False).get("/metrics/prometheus")
    assert response.status_code == 503


@pytest.mark.parametrize("damage", ["missing_directory", "empty_directory", "invalid_file"])
def test_missing_or_invalid_shared_storage_never_falls_back_to_local_zero(tmp_path, damage):
    directory = tmp_path / "metrics"
    directory.mkdir()
    env = child_env()
    env["PROMETHEUS_MULTIPROC_DIR"] = str(directory)
    env["XCELSIOR_METRICS_GENERATION_STARTED"] = str(time.time())
    env["DAMAGE"] = damage
    result = run_child(
        """
        import json,os
        from pathlib import Path
        from fastapi.testclient import TestClient
        from api import app
        directory=Path(os.environ['PROMETHEUS_MULTIPROC_DIR'])
        if os.environ['DAMAGE']=='invalid_file':
            (directory/'counter_invalid.db').write_bytes(b'broken')
        else:
            directory.rename(directory.with_name('old-metrics'))
            if os.environ['DAMAGE']=='empty_directory':
                directory.mkdir()
        response=TestClient(app,raise_server_exceptions=False).get('/metrics/prometheus')
        print(json.dumps({'http_status':response.status_code}))
    """,
        env,
    )
    assert result["http_status"] == 503


def test_start_wrapper_gives_each_master_a_fresh_private_generation(tmp_path):
    env = child_env()
    env["TMPDIR"] = str(tmp_path)
    code = """
        import json,os,stat
        from pathlib import Path
        from prometheus_client import REGISTRY,values
        from control_plane.legacy_execution import record_use
        directory=Path(os.environ['PROMETHEUS_MULTIPROC_DIR'])
        labels={'path':'status_patch','outcome':'requested'}
        before=REGISTRY.get_sample_value('xcelsior_legacy_execution_total',labels)
        record_use('status_patch','requested',9)
        print(json.dumps({'directory':str(directory),'mode':stat.S_IMODE(directory.stat().st_mode),
            'multiprocess':values.ValueClass._multiprocess,'before':before,
            'started':float(os.environ['XCELSIOR_METRICS_GENERATION_STARTED'])}))
    """
    first = run_child(code, env, wrapper=True)
    second = run_child(code, env, wrapper=True)
    assert first["directory"] != second["directory"]
    assert first["started"] != second["started"]
    for result in (first, second):
        assert result["multiprocess"] is True
        assert result["before"] == 0
        assert result["mode"] == 0o700
        assert result["started"] > 0


@pytest.mark.parametrize("late_environment", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
def test_gunicorn_refuses_missing_or_late_metric_initialization(
    tmp_path, late_environment, workers
):
    env = child_env()
    env["GUNICORN_WORKERS"] = str(workers)
    env["LATE_ENV"] = str(int(late_environment))
    env["PROBE_DIRECTORY"] = str(tmp_path)
    result = run_child(
        """
        import json,os,runpy
        from types import SimpleNamespace
        import prometheus_client
        if os.environ['LATE_ENV']=='1':
            os.environ['PROMETHEUS_MULTIPROC_DIR']=os.environ['PROBE_DIRECTORY']
        cfg=runpy.run_path('gunicorn.conf.py')
        try:
            cfg['on_starting'](SimpleNamespace(cfg=SimpleNamespace(workers=int(os.environ['GUNICORN_WORKERS']))))
        except RuntimeError as exc:
            assert 'start_api.sh' in str(exc)
            print(json.dumps({'refused':True}))
        else:
            print(json.dumps({'refused':False}))
    """,
        env,
    )
    assert result["refused"] is True


def test_real_gunicorn_replacement_keeps_unscraped_worker_requests(tmp_path):
    """The actual master/preload/exit hooks and API scrape, not just fork math."""
    module = tmp_path / "aggregation_fixture.py"
    module.write_text(textwrap.dedent("""
        import os
        from api import app
        from control_plane.legacy_execution import record_use
        @app.get('/__aggregation_probe')
        def probe(increment: bool=True):
            if increment:
                record_use('status_patch','requested')
            return {'pid':os.getpid(), 'generation':os.environ['XCELSIOR_METRICS_GENERATION_STARTED']}
    """))
    env = child_env()
    env.update(
        {
            "TMPDIR": str(tmp_path),
            "GUNICORN_WORKERS": "2",
            "XCELSIOR_PERSISTENT_AUTH": "true",
            "XCELSIOR_SHARED_RUNTIME_LIMITS": "true",
        }
    )
    env["PYTHONPATH"] += os.pathsep + str(tmp_path)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    base = f"http://127.0.0.1:{sock.getsockname()[1]}"
    argv = [
        "/bin/sh",
        str(START),
        sys.executable,
        "-m",
        "gunicorn",
        "aggregation_fixture:app",
        "-c",
        str(ROOT / "gunicorn.conf.py"),
        "--bind",
        f"fd://{sock.fileno()}",
    ]
    count, seen, generations = 0, set(), set()
    log_path = tmp_path / "gunicorn.log"

    def request(path):
        req = urllib.request.Request(
            base + path, headers={"Authorization": "Bearer isolated-metric-test-token"}
        )
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.read().decode()

    with log_path.open("w") as log:
        process = subprocess.Popen(
            argv,
            env=env,
            cwd=ROOT,
            stdout=log,
            stderr=log,
            pass_fds=(sock.fileno(),),
            start_new_session=True,
        )
    sock.close()
    try:
        deadline = time.monotonic() + 35
        while len(seen) < 2 and time.monotonic() < deadline:
            assert process.poll() is None, log_path.read_text()[-5000:]
            try:
                body = json.loads(request("/__aggregation_probe?increment=false"))
            except (OSError, ValueError):
                time.sleep(0.05)
                continue
            seen.add(body["pid"])
            generations.add(body["generation"])
        assert len(seen) == 2, log_path.read_text()[-5000:]
        counted_workers = set()
        for _ in range(100):
            body = json.loads(request("/__aggregation_probe"))
            count += 1
            counted_workers.add(body["pid"])
            if len(counted_workers) == 2:
                break
        assert counted_workers == seen
        victim = min(seen)
        assert victim != process.pid
        os.kill(victim, signal.SIGKILL)
        replacement = None
        deadline = time.monotonic() + 25
        while replacement is None and time.monotonic() < deadline:
            try:
                body = json.loads(request("/__aggregation_probe?increment=false"))
            except (OSError, ValueError):
                time.sleep(0.05)
                continue
            generations.add(body["generation"])
            if body["pid"] not in seen:
                replacement = body["pid"]
            else:
                time.sleep(0.05)
        assert replacement is not None, log_path.read_text()[-5000:]
        assert len(generations) == 1
        from prometheus_client.parser import text_string_to_metric_families

        for _ in range(6):
            samples = [
                s
                for family in text_string_to_metric_families(request("/metrics/prometheus"))
                for s in family.samples
            ]
            values = [
                s.value
                for s in samples
                if s.name == "xcelsior_legacy_execution_total"
                and s.labels == {"path": "status_patch", "outcome": "requested"}
            ]
            assert values == [count], (count, values)
            assert [
                s.value for s in samples if s.name == "xcelsior_metrics_aggregation_enabled"
            ] == [1]
            assert [
                s.value
                for s in samples
                if s.name == "xcelsior_metrics_generation_started_timestamp_seconds"
            ] == [float(next(iter(generations)))]
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
