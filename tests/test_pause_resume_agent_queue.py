"""P3/A3 — symmetric pause/resume via agent command queue.

Ensures:
  • worker_agent.py dispatches a new `pause_container` handler that does
    `docker stop` but NEVER `docker rm` (so resume is a cheap restart).
  • routes/agent.py _AGENT_COMMAND_ALLOWED includes `pause_container`.
  • billing.py pause_instance enqueues `pause_container` (not stop_container).
  • billing.py resume_instance enqueues `start_container` via the agent
    queue and no longer calls scheduler.run_job (SSH-based) — i.e. the
    resume path is fully CGNAT-safe now.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("AGENT_SECRET", "")
os.environ.setdefault("XCELSIOR_ALLOW_UNAUTH_AGENT", "1")


REPO = Path(__file__).resolve().parent.parent


def test_pause_container_in_agent_allowlist():
    from routes.agent import _AGENT_COMMAND_ALLOWED

    assert "pause_container" in _AGENT_COMMAND_ALLOWED
    assert "start_container" in _AGENT_COMMAND_ALLOWED
    assert "stop_container" in _AGENT_COMMAND_ALLOWED


def _cmd(name: str, **args):
    return {
        "id": f"cmd-{name}",
        "command": name,
        "args": args,
        "created_by": "pytest",
    }


def test_pause_container_stops_without_removing():
    """pause_container must call `docker stop` but NOT `docker rm`."""
    import worker_agent

    worker_agent.HOST_ID = "host-x"
    captured_args: list[list[str]] = []

    def fake_run(argv, *a, **kw):
        captured_args.append(list(argv))
        m = MagicMock()
        m.returncode = 0
        m.stdout = ""
        m.stderr = ""
        return m

    get_mock = MagicMock()
    get_mock.status_code = 200
    get_mock.json.return_value = {
        "commands": [_cmd("pause_container", container_name="xcl-job1", job_id="job1")]
    }

    with patch("worker_agent.requests.get", return_value=get_mock), \
         patch("worker_agent.subprocess.run", side_effect=fake_run):
        worker_agent.drain_agent_commands()

    # Exactly one docker invocation, and it must be `docker stop`.
    docker_calls = [c for c in captured_args if c and c[0] == "docker"]
    assert any(c[:2] == ["docker", "stop"] for c in docker_calls), (
        f"pause_container did not call `docker stop`: {docker_calls}"
    )
    assert not any(c[:2] == ["docker", "rm"] for c in docker_calls), (
        f"pause_container must NOT call `docker rm`: {docker_calls}"
    )


def test_billing_pause_uses_pause_container():
    """Source grep: pause_instance should enqueue pause_container."""
    src = (REPO / "billing.py").read_text()
    pause_region = src[src.index("def pause_instance"):src.index("def resume_instance")]
    assert '"pause_container"' in pause_region, (
        "billing.pause_instance must enqueue pause_container (state-preserving)"
    )
    assert '"stop_container"' not in pause_region, (
        "billing.pause_instance must NOT enqueue stop_container (destroys state)"
    )


def test_billing_resume_uses_agent_queue_not_ssh():
    """Source grep: resume_instance must enqueue start_container and must
    not call scheduler.run_job (the old SSH-based restart path)."""
    src = (REPO / "billing.py").read_text()
    resume_region = src[src.index("def resume_instance"):]
    # Cut at next def to stay within the method body.
    nxt = resume_region.find("\n    def ", 10)
    if nxt > 0:
        resume_region = resume_region[:nxt]
    assert '"start_container"' in resume_region, (
        "billing.resume_instance must enqueue start_container via agent queue"
    )
    assert "run_job(" not in resume_region, (
        "billing.resume_instance must not use scheduler.run_job (SSH path)"
    )
