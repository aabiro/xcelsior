"""Real spool durability and delivery races; Docker and HTTP remain isolated."""

import json
import os
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import requests

import worker_agent as worker


@pytest.fixture
def log_state(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "_V2_JOURNAL_PATH", str(tmp_path / "attempts.json"))
    monkeypatch.setattr(worker.LogForwarder, "_backoff_wait", lambda _: None)
    request = mock.Mock(return_value=SimpleNamespace(status_code=200))
    monkeypatch.setattr(worker, "_api_request", request)
    forwarder = worker.LogForwarder("job", "xcl-job-attempt-1")
    entry = {"message": "retained output", "level": "info", "timestamp": 100.25}
    forwarder._buffer.append(entry)
    return forwarder, entry, request


@pytest.mark.parametrize("failure", [SimpleNamespace(status_code=503), requests.ConnectionError()])
def test_failed_delivery_survives_process_restart(log_state, failure):
    forwarder, entry, request = log_state
    request.side_effect = failure if isinstance(failure, Exception) else None
    request.return_value = failure
    assert forwarder._flush_batch() is False
    assert forwarder._buffer == [entry]
    saved = json.loads(Path(forwarder._spool_path).read_text())
    assert saved["lines"] == [entry] and saved["cursor"] == 0
    restarted = worker.LogForwarder(forwarder.job_id, forwarder.container_name)
    assert restarted._buffer == [entry]
    request.side_effect = None
    request.return_value = SimpleNamespace(status_code=200)
    assert restarted._flush_batch() is True
    saved = json.loads(Path(restarted._spool_path).read_text())
    assert saved["lines"] == [] and saved["cursor"] == entry["timestamp"]


def test_shutdown_does_not_discard_unacknowledged_output(log_state):
    forwarder, entry, request = log_state
    request.return_value = SimpleNamespace(status_code=503)
    forwarder.stop()
    restarted = worker.LogForwarder(forwarder.job_id, forwarder.container_name)
    assert restarted._buffer == [entry]


def test_a_replacement_attempt_cannot_consume_prior_logs(log_state):
    forwarder, entry, request = log_state
    request.return_value = SimpleNamespace(status_code=503)
    forwarder._flush_batch()
    replacement = worker.LogForwarder(forwarder.job_id, "xcl-job-attempt-2")
    assert replacement._buffer == []
    assert replacement._spool_path != forwarder._spool_path
    assert json.loads(Path(forwarder._spool_path).read_text())["lines"] == [entry]


def test_parallel_flush_delivers_one_batch(log_state):
    forwarder, _, request = log_state
    entered, release = threading.Event(), threading.Event()

    def post(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(status_code=200)

    request.side_effect = post
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(forwarder._flush_batch)
        assert entered.wait(5)
        second = pool.submit(forwarder._flush_batch)
        release.set()
        assert first.result() and second.result()
    assert request.call_count == 1


def test_new_lines_arriving_during_delivery_are_retained(log_state):
    forwarder, entry, request = log_state
    later = dict(entry, message="next output", timestamp=101)

    def post(*args, **kwargs):
        with forwarder._lock:
            forwarder._buffer.append(later)
        return SimpleNamespace(status_code=200)

    request.side_effect = post
    assert forwarder._flush_batch()
    assert forwarder._buffer == [later]
    assert worker.LogForwarder(forwarder.job_id, forwarder.container_name)._buffer == [later]


def test_disk_failure_before_delivery_retains_memory_and_does_not_send(log_state, monkeypatch):
    forwarder, entry, request = log_state
    monkeypatch.setattr(forwarder, "_persist_spool", mock.Mock(side_effect=OSError("disk")))
    assert forwarder._flush_batch() is False
    assert forwarder._buffer == [entry]
    request.assert_not_called()


def test_lost_ack_persistence_replays_instead_of_losing_output(log_state, monkeypatch):
    forwarder, entry, _ = log_state
    persist = forwarder._persist_spool
    calls = 0

    def save():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk after delivery")
        persist()

    monkeypatch.setattr(forwarder, "_persist_spool", save)
    assert forwarder._flush_batch() is False
    assert forwarder._buffer == [entry]
    assert worker.LogForwarder(forwarder.job_id, forwarder.container_name)._buffer == [entry]


def test_restart_tails_from_acknowledged_cursor(log_state, monkeypatch):
    forwarder, entry, _ = log_state
    assert forwarder._flush_batch()
    restarted = worker.LogForwarder(forwarder.job_id, forwarder.container_name)
    process = mock.Mock(stdout=[])
    popen = mock.Mock(return_value=process)
    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    restarted._tail_loop()
    assert popen.call_args.args[0] == [
        "docker",
        "logs",
        "--follow",
        "--timestamps",
        "--since",
        str(entry["timestamp"]),
        forwarder.container_name,
    ]


def test_corrupt_spool_is_retained(log_state):
    forwarder, _, _ = log_state
    assert forwarder._flush_batch()
    path = Path(forwarder._spool_path)
    path.write_text("broken-record")
    with pytest.raises(ValueError):
        worker.LogForwarder(forwarder.job_id, forwarder.container_name)
    assert path.read_text() == "broken-record"


def test_spool_capacity_is_applied_before_batch_selection(log_state, monkeypatch):
    forwarder, entry, request = log_state
    monkeypatch.setattr(worker, "_LOG_SPOOL_MAX_BYTES", 1024)
    request.return_value = SimpleNamespace(status_code=503)
    forwarder._buffer = [dict(entry, message=str(i) + "x" * 256, timestamp=i) for i in range(100)]
    assert forwarder._flush_batch() is False
    assert os.path.getsize(forwarder._spool_path) <= 1024
    assert forwarder._buffer[-1]["timestamp"] == 99
    sent = json.loads(request.call_args.kwargs["data"])["lines"]
    assert sent == forwarder._buffer[: worker._LOG_BATCH_SIZE]


def test_docker_utc_timestamp_is_independent_of_host_timezone(log_state, monkeypatch):
    forwarder, _, _ = log_state
    forwarder._buffer = []
    stamp = "2026-10-22T12:34:56.123456789Z"
    process = mock.Mock(stdout=[stamp + " output\n"])
    monkeypatch.setattr(worker.subprocess, "Popen", mock.Mock(return_value=process))
    try:
        with monkeypatch.context() as scoped:
            scoped.setenv("TZ", "America/Toronto")
            time.tzset()
            forwarder._tail_loop()
            assert (
                forwarder._buffer[0]["timestamp"]
                == datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
            )
    finally:
        time.tzset()


def test_terminal_log_backlog_is_replayed_without_a_live_attempt(log_state, monkeypatch):
    forwarder, entry, request = log_state
    request.return_value = SimpleNamespace(status_code=503)
    assert not forwarder._flush_batch()
    request.return_value = SimpleNamespace(status_code=200)
    monkeypatch.setattr(worker, "_shutdown", threading.Event())
    monkeypatch.setattr(worker, "_v2_attempts", {})
    monkeypatch.setattr(worker, "_log_replay_running", threading.Lock())
    with mock.patch.object(worker.threading, "Thread") as thread:
        worker._resume_pending_log_delivery()
    thread.call_args.kwargs["target"]()
    assert not Path(forwarder._spool_path).exists()
    assert request.call_count == 2


def test_only_confirmed_adoption_starts_and_stops_one_forwarder(monkeypatch):
    auth = dict(
        job_id="job",
        container_name="xcl-attempt",
        attempt_id="attempt",
        lease_id="lease",
        fencing_token=1,
        stop=threading.Event(),
    )
    forwarder = mock.Mock()
    factory = mock.Mock(return_value=forwarder)
    monkeypatch.setattr(worker, "LogForwarder", factory)
    monkeypatch.setattr(worker, "_active_containers", {})
    monkeypatch.setattr(worker, "_adopted_containers", set())
    monkeypatch.setattr(worker, "_v2_attempts", {"job": auth})
    monkeypatch.setattr(worker, "_v2_journal_save", lambda: True)
    worker._v2_track_adopted(auth, confirmed=False)
    factory.assert_not_called()
    worker._v2_track_adopted(auth)
    worker._v2_track_adopted(auth)
    factory.assert_called_once_with("job", "xcl-attempt")
    forwarder.start.assert_called_once()
    worker._v2_forget_attempt("job", auth)
    forwarder.stop.assert_called_once()
