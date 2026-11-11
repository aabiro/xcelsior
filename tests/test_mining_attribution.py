"""Mining detection must only act on job containers that actually use the flagged GPU.

Regression: the detector read whole-GPU telemetry and, after 5 hits, killed
*every* active job container and reported it failed. A host running its own
GPU service (e.g. pixelspark-gpu on the ASUS worker) tripped it continuously,
which would have killed tenants' jobs for load they did not cause.
"""

from types import SimpleNamespace
from unittest.mock import Mock, mock_open

import pytest

import worker_agent as worker

JOB_CID = "a" * 64
OTHER_CID = "b" * 64
MINING = {"index": 0, "utilization": 100, "memory_util": 19, "pcie_tx_mb_s": 0.5}


def _run_loop(monkeypatch, iterations, telemetry, responder="mock"):
    """Run mining_detection_loop for `iterations` passes with IO isolated.

    Returns the responder the loop drove (a Mock unless one is given).
    """
    calls = {"n": 0}

    def fake_telemetry():
        calls["n"] += 1
        if calls["n"] > iterations:
            worker._shutdown.set()
            return []
        return telemetry

    if responder == "mock":
        responder = Mock()
        responder.episodes = {}
    monkeypatch.setattr(worker, "_build_mining_responder", lambda: responder)
    monkeypatch.setattr(worker, "MINING_CHECK_INTERVAL", 0)
    monkeypatch.setattr(worker, "get_gpu_telemetry", fake_telemetry)
    monkeypatch.setattr(worker, "get_gpu_compute_pids", lambda idx: set())
    for name in ("report_mining_alert", "_kill_container", "report_job_status"):
        monkeypatch.setattr(worker, name, Mock())
    worker._shutdown.clear()
    try:
        worker.mining_detection_loop()
    finally:
        worker._shutdown.clear()
    return responder


def _observed(responder):
    """{container name: gpu_keys} for the last pass over each container."""
    return {c.args[1]: c.kwargs["gpu_keys"] for c in responder.observe.call_args_list}


@pytest.fixture
def active_jobs(monkeypatch):
    jobs = {"job-on-gpu": "xcl-on-gpu", "job-elsewhere": "xcl-elsewhere"}
    monkeypatch.setattr(worker, "_active_containers", dict(jobs))
    return jobs


def test_cgroup_v2_and_cgroupfs_paths_resolve_to_the_container_id(monkeypatch):
    for content in (f"0::/system.slice/docker-{JOB_CID}.scope\n", f"0::/docker/{JOB_CID}\n"):
        monkeypatch.setattr("builtins.open", mock_open(read_data=content))
        assert worker._container_id_for_pid(123) == JOB_CID
    monkeypatch.setattr("builtins.open", mock_open(read_data="0::/user.slice/session-2.scope\n"))
    assert worker._container_id_for_pid(123) is None


def test_job_containers_on_gpu_matches_by_container_id(monkeypatch):
    monkeypatch.setattr(worker, "get_gpu_compute_pids", lambda idx: {11, 22})
    monkeypatch.setattr(worker, "_container_id_for_pid", {11: JOB_CID, 22: None}.get)
    ids = {"xcl-on-gpu": JOB_CID, "xcl-elsewhere": OTHER_CID}
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda cmd, **kw: SimpleNamespace(returncode=0, stdout=ids[cmd[-1]] + "\n"),
    )
    active = {"job-on-gpu": "xcl-on-gpu", "job-elsewhere": "xcl-elsewhere"}
    assert worker._job_containers_on_gpu(0, active) == {"job-on-gpu": "xcl-on-gpu"}


def test_attribution_unavailable_is_distinct_from_no_job_on_gpu(monkeypatch):
    monkeypatch.setattr(worker, "get_gpu_compute_pids", lambda idx: None)
    assert worker._job_containers_on_gpu(0, {"j": "c"}) is None
    monkeypatch.setattr(worker, "get_gpu_compute_pids", lambda idx: {99})
    monkeypatch.setattr(worker, "_container_id_for_pid", lambda pid: None)
    assert worker._job_containers_on_gpu(0, {"j": "c"}) == {}


def test_host_workload_gives_no_container_the_gpu(monkeypatch, active_jobs):
    monkeypatch.setattr(worker, "_job_containers_on_gpu", lambda idx, active: {})
    responder = _run_loop(monkeypatch, iterations=8, telemetry=[MINING])
    worker.report_mining_alert.assert_not_called()
    worker._kill_container.assert_not_called()
    assert _observed(responder) == {"xcl-on-gpu": [], "xcl-elsewhere": []}
    assert worker._active_containers == active_jobs


def test_only_the_container_on_the_gpu_is_given_its_samples(monkeypatch, active_jobs):
    monkeypatch.setattr(
        worker, "_job_containers_on_gpu", lambda idx, active: {"job-on-gpu": "xcl-on-gpu"}
    )
    responder = _run_loop(monkeypatch, iterations=3, telemetry=[MINING])
    assert _observed(responder) == {"xcl-on-gpu": ["0"], "xcl-elsewhere": []}
    on_gpu = [c for c in responder.observe.call_args_list if c.args[1] == "xcl-on-gpu"][-1]
    assert on_gpu.args[0] == "job-on-gpu"
    assert on_gpu.kwargs["gpu_indices"] == {"0": 0}
    assert responder.record_gpu_samples.call_count == 3


def test_the_loop_itself_ends_nothing(monkeypatch, active_jobs):
    """Five hits used to be a kill. Ending a container is the responder's decision,
    taken on a score, never the loop's on a count."""
    monkeypatch.setattr(
        worker, "_job_containers_on_gpu", lambda idx, active: {"job-on-gpu": "xcl-on-gpu"}
    )
    _run_loop(monkeypatch, iterations=12, telemetry=[MINING])
    worker._kill_container.assert_not_called()
    worker.report_job_status.assert_not_called()
    assert worker._active_containers == active_jobs


def test_without_attribution_nothing_is_acted_on_and_the_operator_is_told_once(
    monkeypatch, active_jobs
):
    # CVP brief 01, T-16 collateral: the old fallback killed every active job
    # on the host and reported each as mining when NVML could not list the
    # GPU's processes. A response may only act on a workload it can name.
    monkeypatch.setattr(worker, "_job_containers_on_gpu", lambda idx, active: None)
    responder = _run_loop(monkeypatch, iterations=12, telemetry=[MINING])
    worker._kill_container.assert_not_called()
    worker.report_job_status.assert_not_called()
    assert worker._active_containers == active_jobs
    assert worker.report_mining_alert.call_count == 1
    assert "attribution unavailable" in worker.report_mining_alert.call_args.args[2]
    assert _observed(responder) == {"xcl-on-gpu": [], "xcl-elsewhere": []}


def test_attribution_coming_back_gives_the_container_its_gpu_again(monkeypatch, active_jobs):
    answers = iter([None, None] + [{"job-on-gpu": "xcl-on-gpu"}] * 5)
    monkeypatch.setattr(worker, "_job_containers_on_gpu", lambda idx, active: next(answers))
    responder = _run_loop(monkeypatch, iterations=7, telemetry=[MINING])
    per_pass = [
        c.kwargs["gpu_keys"] for c in responder.observe.call_args_list if c.args[1] == "xcl-on-gpu"
    ]
    assert per_pass == [[], []] + [["0"]] * 5


def test_a_gpu_the_host_also_uses_is_marked_shared(monkeypatch, active_jobs):
    monkeypatch.setattr(
        worker, "_job_containers_on_gpu", lambda idx, active: {"job-on-gpu": "xcl-on-gpu"}
    )
    responder = _run_loop(monkeypatch, iterations=1, telemetry=[MINING])
    exclusive = [c for c in responder.observe.call_args_list if c.args[1] == "xcl-on-gpu"][-1]
    assert exclusive.kwargs["shared_gpu"] is False

    monkeypatch.setattr(worker, "_container_id_for_pid", lambda pid: None)
    assert worker._gpu_shared_with_host(0) is False  # no processes at all
    monkeypatch.setattr(worker, "get_gpu_compute_pids", lambda idx: {7})
    assert worker._gpu_shared_with_host(0) is True  # a host process holds a context
    monkeypatch.setattr(worker, "get_gpu_compute_pids", lambda idx: None)
    assert worker._gpu_shared_with_host(0) is True  # unknown counts as shared


def test_one_containers_failed_pass_does_not_stop_the_others(monkeypatch, active_jobs):
    monkeypatch.setattr(worker, "_job_containers_on_gpu", lambda idx, active: {})
    responder = Mock()
    responder.episodes = {}
    seen = []

    def observe(job_id, cname, **kwargs):
        seen.append(cname)
        if cname == "xcl-on-gpu":
            raise RuntimeError("docker inspect timed out")

    responder.observe.side_effect = observe
    _run_loop(monkeypatch, iterations=2, telemetry=[MINING], responder=responder)
    assert seen.count("xcl-elsewhere") == 2


def test_an_episode_whose_container_is_gone_is_forgotten(monkeypatch, active_jobs):
    monkeypatch.setattr(worker, "_job_containers_on_gpu", lambda idx, active: {})
    responder = Mock()
    responder.episodes = {"xcl-gone": object(), "xcl-on-gpu": object()}
    _run_loop(monkeypatch, iterations=1, telemetry=[MINING], responder=responder)
    responder.forget.assert_called_once_with("xcl-gone")
    responder.restore_after_restart.assert_called_once_with({"xcl-on-gpu", "xcl-elsewhere"})


def test_a_host_without_the_response_module_reports_and_ends_nothing(monkeypatch, active_jobs):
    monkeypatch.setattr(worker, "_job_containers_on_gpu", lambda idx, active: None)
    _run_loop(monkeypatch, iterations=3, telemetry=[MINING], responder=None)
    worker._kill_container.assert_not_called()
    assert worker.report_mining_alert.call_count == 1


def test_the_worker_builds_a_responder_on_its_own_seams():
    import mining_response

    responder = worker._build_mining_responder()
    assert isinstance(responder, mining_response.Responder)
    assert responder.docker is worker._mining_docker
    assert responder.privd is worker._mining_privd
    assert responder.report is worker._mining_report
    assert responder.terminate is worker._mining_terminate
    assert 3333 in responder.pool_ports


def test_a_pool_url_in_one_containers_process_suspends_that_container_only(
    monkeypatch, tmp_path, active_jobs
):
    """The real responder driven by the loop: the named container is paused, its
    neighbour is untouched, and nothing is killed without a second incident or a decision."""
    import mining_response

    paused = []

    def docker(args, timeout):
        cname = args[1] if args[0] == "top" else args[-1]
        if args[0] == "pause":
            paused.append(cname)
            return 0, ""
        if args[0] == "inspect":
            return 0, f"{cname[-1] * 64}|4000000000|false|bridge=172.17.0.2 "
        if args[0] == "top":
            if cname == "xcl-on-gpu":
                return (
                    0,
                    "PID RSS COMMAND COMMAND\n41 90000 worker worker -o stratum+tcp://pool.example.test:3333\n",
                )
            return 0, "PID RSS COMMAND COMMAND\n42 90000 python python train.py\n"
        return 0, "3.0%|4"

    reports = []
    notes = []
    responder = mining_response.Responder(
        docker=docker,
        privd=lambda op, **a: {"ok": False},
        report=lambda ep, ev: reports.append((ep.container_name, ep.rung)) or {},
        notify=lambda job, text: notes.append(job),
        terminate=worker._mining_terminate,
        pool_ports=(3333,),
        state_path=str(tmp_path / "state.json"),
        signatures={"names": (), "arg_patterns": list(mining_response.DEFAULT_ARG_PATTERNS)},
    )
    monkeypatch.setattr(worker, "_job_containers_on_gpu", lambda idx, active: {})
    _run_loop(monkeypatch, iterations=2, telemetry=[MINING], responder=responder)
    assert set(responder.episodes) == {"xcl-on-gpu"}
    assert responder.episodes["xcl-on-gpu"].rung == mining_response.SUSPEND
    assert paused == ["xcl-on-gpu"]
    assert notes and set(notes) == {"job-on-gpu"}
    assert {name for name, _ in reports} == {"xcl-on-gpu"}
    worker._kill_container.assert_not_called()
    assert worker._active_containers == active_jobs
