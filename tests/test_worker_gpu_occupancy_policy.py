from unittest import mock

import pytest

from tests.test_worker_agent_v2 import _start_cmd, worker_agent


@pytest.mark.parametrize(
    ("required", "outside", "reason"),
    [
        (True, None, "gpu_outside_use_unavailable"),
        (False, None, ""),
        (True, {}, ""),
        (True, {"GPU-other": {"processes": 1, "vram_mb": 256}}, ""),
        (True, {"GPU-test-1": {"processes": 1, "vram_mb": 0}}, "gpu_outside_use"),
        (False, {"GPU-test-1": {"processes": 1, "vram_mb": 0}}, "gpu_outside_use"),
    ],
)
def test_start_uses_the_fenced_grant_occupancy_policy(tmp_path, required, outside, reason):
    with (
        mock.patch.object(
            worker_agent,
            "v2_claim_lease_fenced",
            return_value={
                "gpu_device_uuids": ["GPU-test-1"],
                "gpu_occupancy_required": required,
            },
        ),
        mock.patch.object(worker_agent, "_V2_JOURNAL_PATH", str(tmp_path / "attempts.json")),
        mock.patch.object(worker_agent, "_v2_attempts", {}),
        mock.patch.object(worker_agent, "gpu_outside_use", return_value=outside),
        mock.patch.object(worker_agent, "_v2_abort_claimed_start") as abort,
        mock.patch.object(worker_agent, "v2_nack_command") as nack,
        mock.patch.object(worker_agent, "v2_report_attempt_status", return_value="ok"),
        mock.patch.object(worker_agent, "_v2_record_and_ack", return_value=True),
        mock.patch.object(worker_agent.threading, "Thread"),
        mock.patch.object(worker_agent, "run_job") as run,
    ):
        worker_agent.handle_start_attempt(_start_cmd())
    if reason:
        assert abort.call_args.args[1] == reason
        nack.assert_called_once_with("cmd-1", reason, retryable=False)
        run.assert_not_called()
    else:
        abort.assert_not_called()
        run.assert_called_once()
