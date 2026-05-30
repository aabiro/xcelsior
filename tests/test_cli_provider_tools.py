from types import SimpleNamespace
from unittest.mock import patch

import cli


def test_cmd_run_passes_required_gpu_model(capsys):
    job = {
        "job_id": "job-1",
        "name": "cuda-smoke",
        "vram_needed_gb": 2,
        "tier": "free",
        "priority": 0,
        "num_gpus": 1,
        "gpu_model": "RTX 3060",
    }
    args = SimpleNamespace(
        model="cuda-smoke",
        vram=2,
        priority=0,
        tier=None,
        gpus=1,
        gpu="RTX 3060",
        nfs_server=None,
        nfs_path=None,
        image=None,
        no_assign=True,
    )

    with patch("cli.submit_job", return_value=job) as submit:
        cli.cmd_run(args)

    submit.assert_called_once()
    assert submit.call_args.kwargs["gpu_model"] == "RTX 3060"
    assert "gpu=RTX 3060" in capsys.readouterr().out
