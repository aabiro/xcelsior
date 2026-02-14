# Tests for nvml_telemetry.py — NVML-based GPU monitoring
# Tests use mocking since pynvml requires actual NVIDIA hardware.

import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock


# ── Mock pynvml Module ──────────────────────────────────────────────
# Create a comprehensive mock of the pynvml module so we can test
# the NVML telemetry code without actual NVIDIA hardware.


class MockPynvml:
    """Mock pynvml module for testing."""
    NVML_TEMPERATURE_GPU = 0
    NVML_SINGLE_BIT_ECC = 0
    NVML_DOUBLE_BIT_ECC = 1
    NVML_VOLATILE_ECC = 0
    NVML_PCIE_UTIL_TX_BYTES = 0
    NVML_PCIE_UTIL_RX_BYTES = 1

    def nvmlInit(self): pass
    def nvmlShutdown(self): pass

    def nvmlSystemGetDriverVersion(self):
        return "560.35.03"

    def nvmlSystemGetCudaDriverVersion_v2(self):
        return 12040  # CUDA 12.4

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return MagicMock(name=f"gpu_handle_{index}")

    def nvmlDeviceGetName(self, handle):
        return "NVIDIA GeForce RTX 4090"

    def nvmlDeviceGetSerial(self, handle):
        return "GPU-SN-4090-001"

    def nvmlDeviceGetUUID(self, handle):
        return "GPU-uuid-4090-001"

    def nvmlDeviceGetPciInfo(self, handle):
        pci = MagicMock()
        pci.busId = b"0000:01:00.0"
        pci.pciDeviceId = 0x268410de
        pci.pciSubSystemId = 0x16e810de
        return pci

    def nvmlDeviceGetUtilizationRates(self, handle):
        rates = MagicMock()
        rates.gpu = 65
        rates.memory = 42
        return rates

    def nvmlDeviceGetMemoryInfo(self, handle):
        mem = MagicMock()
        mem.total = 24 * 1024**3   # 24 GB
        mem.used = 8 * 1024**3     # 8 GB used
        mem.free = 16 * 1024**3    # 16 GB free
        return mem

    def nvmlDeviceGetTemperature(self, handle, sensor_type):
        return 72

    def nvmlDeviceGetPowerUsage(self, handle):
        return 285_000  # 285W in milliwatts

    def nvmlDeviceGetEnforcedPowerLimit(self, handle):
        return 450_000  # 450W limit

    def nvmlDeviceGetCurrPcieLinkGeneration(self, handle):
        return 4

    def nvmlDeviceGetCurrPcieLinkWidth(self, handle):
        return 16

    def nvmlDeviceGetPcieThroughput(self, handle, counter):
        if counter == self.NVML_PCIE_UTIL_TX_BYTES:
            return 512_000  # 512 MB/s in KB/s
        return 384_000  # 384 MB/s

    def nvmlDeviceGetTotalEccErrors(self, handle, error_type, counter_type):
        return 0

    def nvmlDeviceGetCudaComputeCapability(self, handle):
        return (8, 9)  # Compute capability 8.9 (Ada Lovelace)


mock_pynvml = MockPynvml()


class TestNVMLInit(unittest.TestCase):
    """Test NVML lifecycle management."""

    def setUp(self):
        # Clear module state before each test
        import nvml_telemetry as mod
        mod._nvml_initialized = False
        mod._nvml_available = False
        mod._thermal_history.clear()

    def test_init_without_pynvml_returns_false(self):
        """NVML init should return False when pynvml is not available."""
        import nvml_telemetry as mod
        mod._nvml_available = False
        assert mod.nvml_init() is False

    def test_init_with_pynvml_returns_true(self):
        """NVML init should return True when pynvml is available."""
        import nvml_telemetry as mod
        mod._nvml_available = True
        mod.pynvml = mock_pynvml
        assert mod.nvml_init() is True
        assert mod._nvml_initialized is True

    def test_double_init_is_safe(self):
        """Calling nvml_init() twice should be idempotent."""
        import nvml_telemetry as mod
        mod._nvml_available = True
        mod.pynvml = mock_pynvml
        mod.nvml_init()
        mod.nvml_init()
        assert mod._nvml_initialized is True

    def test_shutdown(self):
        """nvml_shutdown should clear initialized flag."""
        import nvml_telemetry as mod
        mod._nvml_available = True
        mod.pynvml = mock_pynvml
        mod.nvml_init()
        mod.nvml_shutdown()
        assert mod._nvml_initialized is False

    def test_is_nvml_available(self):
        """is_nvml_available reflects initialization state."""
        import nvml_telemetry as mod
        assert mod.is_nvml_available() is False
        mod._nvml_available = True
        mod.pynvml = mock_pynvml
        mod.nvml_init()
        assert mod.is_nvml_available() is True


class TestCollectGPUTelemetry(unittest.TestCase):
    """Test GPU telemetry collection via mocked NVML."""

    def setUp(self):
        import nvml_telemetry as mod
        self.mod = mod
        mod._nvml_available = True
        mod._nvml_initialized = True
        mod.pynvml = mock_pynvml
        mod._thermal_history.clear()

    def tearDown(self):
        self.mod._nvml_initialized = False
        self.mod._nvml_available = False

    def test_collect_returns_dict(self):
        """collect_gpu_telemetry should return a dict with all expected keys."""
        data = self.mod.collect_gpu_telemetry(0)
        assert data is not None
        assert isinstance(data, dict)

    def test_gpu_model(self):
        """Should report correct GPU model name."""
        data = self.mod.collect_gpu_telemetry(0)
        assert data["gpu_model"] == "NVIDIA GeForce RTX 4090"

    def test_utilization(self):
        """Should report GPU SM utilization percentage."""
        data = self.mod.collect_gpu_telemetry(0)
        assert data["utilization"] == 65
        assert data["memory_util"] == 42

    def test_memory_info(self):
        """Should report memory from nvmlDeviceGetMemoryInfo."""
        data = self.mod.collect_gpu_telemetry(0)
        assert data["memory_total_gb"] == 24.0
        assert data["memory_used_gb"] == 8.0
        assert data["memory_free_gb"] == 16.0
        assert data["memory_total_bytes"] == 24 * 1024**3

    def test_temperature(self):
        """Should report GPU temperature."""
        data = self.mod.collect_gpu_telemetry(0)
        assert data["temperature_c"] == 72

    def test_temperature_rolling_avg(self):
        """Should maintain rolling temperature average."""
        self.mod.collect_gpu_telemetry(0)
        self.mod.collect_gpu_telemetry(0)
        data = self.mod.collect_gpu_telemetry(0)
        assert data["temperature_avg_c"] == 72.0  # All same temp

    def test_power(self):
        """Should report power draw and limit in watts."""
        data = self.mod.collect_gpu_telemetry(0)
        assert data["power_draw_w"] == 285.0
        assert data["power_limit_w"] == 450.0

    def test_pcie_link(self):
        """Should report PCIe generation and width."""
        data = self.mod.collect_gpu_telemetry(0)
        assert data["pcie_gen"] == 4
        assert data["pcie_width"] == 16

    def test_pcie_throughput(self):
        """Should report actual PCIe throughput in MB/s."""
        data = self.mod.collect_gpu_telemetry(0)
        assert data["pcie_tx_mb_s"] == 500.0   # 512000 KB/s → 500 MB/s
        assert data["pcie_rx_mb_s"] == 375.0   # 384000 KB/s → 375 MB/s

    def test_serial(self):
        """Should report GPU serial for anti-spoofing."""
        data = self.mod.collect_gpu_telemetry(0)
        assert data["serial"] == "GPU-SN-4090-001"

    def test_pci_info(self):
        """Should report PCI bus info."""
        data = self.mod.collect_gpu_telemetry(0)
        assert data["pci_info"]["bus_id"] == "0000:01:00.0"

    def test_compute_capability(self):
        """Should report CUDA compute capability."""
        data = self.mod.collect_gpu_telemetry(0)
        assert data["compute_capability"] == "8.9"

    def test_driver_version(self):
        """Should report NVIDIA driver version."""
        data = self.mod.collect_gpu_telemetry(0)
        assert data["driver_version"] == "560.35.03"

    def test_cuda_version(self):
        """Should convert CUDA version int to string."""
        data = self.mod.collect_gpu_telemetry(0)
        assert data["cuda_version"] == "12.4"

    def test_ecc_errors(self):
        """Should report ECC memory errors."""
        data = self.mod.collect_gpu_telemetry(0)
        assert data["memory_errors"] == 0

    def test_not_initialized_returns_none(self):
        """collect_gpu_telemetry returns None when NVML not initialized."""
        self.mod._nvml_initialized = False
        assert self.mod.collect_gpu_telemetry(0) is None

    def test_timestamp_present(self):
        """Should include a timestamp."""
        data = self.mod.collect_gpu_telemetry(0)
        assert "timestamp" in data
        assert data["timestamp"] > 0


class TestCollectAllGPUs(unittest.TestCase):
    """Test collecting telemetry for all GPUs."""

    def setUp(self):
        import nvml_telemetry as mod
        self.mod = mod
        mod._nvml_available = True
        mod._nvml_initialized = True
        mod.pynvml = mock_pynvml
        mod._thermal_history.clear()

    def tearDown(self):
        self.mod._nvml_initialized = False
        self.mod._nvml_available = False

    def test_returns_list(self):
        """collect_all_gpus should return a list."""
        result = self.mod.collect_all_gpus()
        assert isinstance(result, list)
        assert len(result) == 1  # Mock returns count=1

    def test_fallback_when_not_initialized(self):
        """Falls back to nvidia-smi subprocess when NVML not available."""
        self.mod._nvml_initialized = False
        # The fallback will fail since no nvidia-smi in test env
        result = self.mod.collect_all_gpus()
        assert isinstance(result, list)


class TestGetGPUInfoNVML(unittest.TestCase):
    """Test GPU info convenience function."""

    def setUp(self):
        import nvml_telemetry as mod
        self.mod = mod
        mod._nvml_available = True
        mod._nvml_initialized = True
        mod.pynvml = mock_pynvml

    def tearDown(self):
        self.mod._nvml_initialized = False
        self.mod._nvml_available = False

    def test_returns_gpu_info(self):
        """get_gpu_info_nvml should return model, VRAM, serial."""
        info = self.mod.get_gpu_info_nvml()
        assert info is not None
        assert info["gpu_model"] == "NVIDIA GeForce RTX 4090"
        assert info["total_vram_gb"] == 24.0
        assert info["serial"] == "GPU-SN-4090-001"

    def test_returns_none_when_not_initialized(self):
        """Returns None when NVML not initialized."""
        self.mod._nvml_initialized = False
        assert self.mod.get_gpu_info_nvml() is None


class TestBuildVerificationReport(unittest.TestCase):
    """Test building verification reports from NVML data."""

    def setUp(self):
        import nvml_telemetry as mod
        self.mod = mod
        mod._nvml_available = True
        mod._nvml_initialized = True
        mod.pynvml = mock_pynvml
        mod._thermal_history.clear()

    def tearDown(self):
        self.mod._nvml_initialized = False
        self.mod._nvml_available = False

    def test_report_has_required_fields(self):
        """Verification report should have all fields needed by verification.py."""
        report = self.mod.build_verification_report(0)
        assert report is not None
        assert "gpu_model" in report
        assert "serial" in report
        assert "total_vram_gb" in report
        assert "gpu_temp_celsius" in report
        assert "compute_capability" in report
        assert "driver_version" in report
        assert "memory_fragmentation_pct" in report

    def test_report_values_match_telemetry(self):
        """Report values should match raw telemetry."""
        report = self.mod.build_verification_report(0)
        assert report["gpu_model"] == "NVIDIA GeForce RTX 4090"
        assert report["total_vram_gb"] == 24.0
        assert report["gpu_temp_celsius"] == 72
        assert report["compute_capability"] == "8.9"


class TestThermalHistory(unittest.TestCase):
    """Test rolling thermal average tracking."""

    def setUp(self):
        import nvml_telemetry as mod
        self.mod = mod
        mod._thermal_history.clear()

    def test_single_reading(self):
        """Single reading = that reading as average."""
        avg = self.mod._get_thermal_avg(0, 75.0)
        assert avg == 75.0

    def test_multiple_readings_averaged(self):
        """Multiple readings produce correct average."""
        self.mod._get_thermal_avg(0, 70.0)
        self.mod._get_thermal_avg(0, 80.0)
        avg = self.mod._get_thermal_avg(0, 75.0)
        assert avg == 75.0  # (70 + 80 + 75) / 3

    def test_separate_gpu_indices(self):
        """Each GPU index has its own thermal history."""
        self.mod._get_thermal_avg(0, 70.0)
        avg1 = self.mod._get_thermal_avg(1, 90.0)
        assert avg1 == 90.0  # GPU 1's own history

    def test_rolling_window_size_respected(self):
        """History is capped at THERMAL_HISTORY_SIZE."""
        old_size = self.mod.THERMAL_HISTORY_SIZE
        self.mod.THERMAL_HISTORY_SIZE = 3
        self.mod._thermal_history.clear()

        from collections import deque
        self.mod._thermal_history[0] = deque(maxlen=3)

        self.mod._get_thermal_avg(0, 60.0)
        self.mod._get_thermal_avg(0, 70.0)
        self.mod._get_thermal_avg(0, 80.0)
        avg = self.mod._get_thermal_avg(0, 90.0)
        # deque(maxlen=3) should have [70, 80, 90]
        assert avg == 80.0
        self.mod.THERMAL_HISTORY_SIZE = old_size


class TestNVMLFallback(unittest.TestCase):
    """Test nvidia-smi fallback path."""

    def test_fallback_returns_empty_list_without_nvidia_smi(self):
        """Fallback should return empty list when nvidia-smi not found."""
        import nvml_telemetry as mod
        result = mod._fallback_nvidia_smi()
        assert isinstance(result, list)
        # Will be empty in test env (no nvidia-smi)

    def test_fallback_with_mocked_nvidia_smi(self):
        """Fallback parses nvidia-smi output correctly."""
        import nvml_telemetry as mod
        mock_output = (
            "0, 65, 42, 4, 16, 285.0, 72, NVIDIA GeForce RTX 4090, "
            "GPU-SN-001, GPU-uuid-001, 24576, 8192, 16384"
        )
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = mock_output
            mock_run.return_value = mock_result

            result = mod._fallback_nvidia_smi()
            assert len(result) == 1
            assert result[0]["gpu_model"] == "NVIDIA GeForce RTX 4090"
            assert result[0]["utilization"] == 65
            assert result[0]["temperature_c"] == 72
            assert result[0]["memory_total_gb"] == 24.0


if __name__ == "__main__":
    unittest.main()
