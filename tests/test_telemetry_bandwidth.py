"""P2.4 — net/disk IO delta calculation in worker_agent._sample_io_delta.

Covers: first-sample priming returns zeros, second sample computes Mbps/MB/s
deltas, counter monotonicity clamps to zero, missing psutil falls back cleanly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import worker_agent  # noqa: E402


class _FakePsutil:
    """Minimal psutil stand-in returning scripted counter values."""

    def __init__(self, samples: list[tuple[int, int, int, int]]):
        self._samples = list(samples)

    def _pop(self):
        if not self._samples:
            raise RuntimeError("out of samples")
        return self._samples.pop(0)

    def net_io_counters(self, pernic=False):  # noqa: ARG002
        rx, tx, _, _ = self._peek()
        return SimpleNamespace(bytes_recv=rx, bytes_sent=tx)

    def disk_io_counters(self, perdisk=False):  # noqa: ARG002
        _, _, r, w = self._pop()
        return SimpleNamespace(read_bytes=r, write_bytes=w)

    def _peek(self):
        # Return current without popping — net called before disk
        return self._samples[0]


def test_first_sample_returns_zeros(monkeypatch):
    fake = _FakePsutil([(1000, 2000, 500, 700)])
    monkeypatch.setattr(worker_agent, "psutil", fake)
    state: dict = {"t": None, "net": None, "disk": None}
    out = worker_agent._sample_io_delta(state)
    assert out == {
        "net_rx_mbps": 0.0,
        "net_tx_mbps": 0.0,
        "disk_read_mb_s": 0.0,
        "disk_write_mb_s": 0.0,
    }
    # State must now be primed
    assert state["t"] is not None


def test_second_sample_computes_deltas(monkeypatch):
    # First sample primes, second produces deltas.
    fake = _FakePsutil(
        [
            (0, 0, 0, 0),
            # 1.25 MB rx, 2.5 MB tx, 10 MB read, 20 MB write
            (1_250_000, 2_500_000, 10_000_000, 20_000_000),
        ]
    )
    monkeypatch.setattr(worker_agent, "psutil", fake)

    times = iter([1000.0, 1001.0])  # 1 second apart
    monkeypatch.setattr(worker_agent.time, "monotonic", lambda: next(times))

    state: dict = {"t": None, "net": None, "disk": None}
    worker_agent._sample_io_delta(state)  # prime
    out = worker_agent._sample_io_delta(state)

    # 1.25 MB/s rx → 10 Mbps
    assert out["net_rx_mbps"] == 10.0
    # 2.5 MB/s tx → 20 Mbps
    assert out["net_tx_mbps"] == 20.0
    assert out["disk_read_mb_s"] == 10.0
    assert out["disk_write_mb_s"] == 20.0


def test_counter_rollback_clamps_to_zero(monkeypatch):
    # If counters decrease (e.g. NIC reset) we must not emit negative values.
    fake = _FakePsutil([(10_000_000, 10_000_000, 10_000_000, 10_000_000), (0, 0, 0, 0)])
    monkeypatch.setattr(worker_agent, "psutil", fake)
    times = iter([1000.0, 1001.0])
    monkeypatch.setattr(worker_agent.time, "monotonic", lambda: next(times))

    state: dict = {"t": None, "net": None, "disk": None}
    worker_agent._sample_io_delta(state)
    out = worker_agent._sample_io_delta(state)
    assert out["net_rx_mbps"] == 0.0
    assert out["net_tx_mbps"] == 0.0
    assert out["disk_read_mb_s"] == 0.0
    assert out["disk_write_mb_s"] == 0.0


def test_missing_psutil_returns_zeros(monkeypatch):
    monkeypatch.setattr(worker_agent, "psutil", None)
    state: dict = {"t": None, "net": None, "disk": None}
    out = worker_agent._sample_io_delta(state)
    assert out == {
        "net_rx_mbps": 0.0,
        "net_tx_mbps": 0.0,
        "disk_read_mb_s": 0.0,
        "disk_write_mb_s": 0.0,
    }
