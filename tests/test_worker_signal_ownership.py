"""Importing worker helpers must not own another process's termination signals."""

import subprocess
import sys
import ast
from pathlib import Path


def test_import_preserves_the_callers_signal_handlers():
    code = """import signal
before={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
import worker_agent
assert {s:signal.getsignal(s) for s in before}==before
worker_agent._install_signal_handlers()
assert all(signal.getsignal(s) is worker_agent._signal_handler for s in before)
worker_agent._signal_handler(signal.SIGTERM,None)
assert worker_agent._shutdown.is_set()
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr


def test_real_script_entrypoint_defines_all_dispatch_helpers_before_startup(tmp_path):
    import worker_agent

    tree = ast.parse(Path(worker_agent.__file__).read_text())
    # Stop at the first startup validation boundary, before credentials, GPU
    # probing or network calls. Keep the actual main and script entrypoint.
    checks = ast.parse("""assert callable(_promote_artifacts)
assert callable(_report_environment_fingerprint)
assert callable(_copy_promoted_artifact)
assert callable(_cleanup_part)
assert all(signal.getsignal(s) is _signal_handler for s in (signal.SIGINT,signal.SIGTERM))
raise SystemExit(0)
""").body
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "validate_config":
            node.body = checks
    script = tmp_path / "worker_entrypoint.py"
    script.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
