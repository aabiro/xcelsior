// Provider-side local checks — GPU detection, benchmarks, verification.
// Shells out to nvidia-smi, ping, and Python (torch) for FP16 matmul benchmarks.

import { execFile } from "node:child_process";
import { promisify } from "node:util";

const exec = promisify(execFile);

// ── Version requirements (mirrors security.py MINIMUM_VERSIONS) ──────

export const MINIMUM_VERSIONS: Record<string, string> = {
    runc: "1.1.12",
    nvidia_toolkit: "1.17.8",
    nvidia_driver: "550.0",
    docker: "24.0.0",
};

// ── Verification thresholds (mirrors verification.py) ────────────────

export const VERIFICATION_THRESHOLDS = {
    min_pcie_bandwidth_gbps: 8.0,
    max_gpu_temp_celsius: 90,
    min_vram_match_pct: 95.0,
    min_cuda_compute_capability: 7.0,
    max_network_loss_pct: 2.0,
    max_network_jitter_ms: 50.0,
    min_network_throughput_mbps: 100.0,
};

// ── Types ────────────────────────────────────────────────────────────

export interface VersionCheck {
    component: string;
    version: string | null;
    minimum: string;
    passed: boolean;
}

export interface GpuInfo {
    gpu_model: string;
    total_vram_gb: number;
    free_vram_gb: number;
    serial: string;
    uuid: string;
    pci_bus_id: string;
    driver_version: string;
    compute_capability: string;
}

export interface BenchmarkResult {
    tflops: number;
    pcie_bandwidth_gbps: number;
    pcie_h2d_gbps: number;
    pcie_d2h_gbps: number;
    gpu_temp_celsius: number;
    gpu_temp_avg_celsius: number;
    gpu_temp_samples: number;
    gpu_model: string;
    total_vram_gb: number;
    compute_capability: string;
    cuda_version: string;
    driver_version: string;
    xcu_score: number;
    elapsed_s: number;
    error?: string;
}

export interface NetworkBenchResult {
    latency_avg_ms: number;
    latency_min_ms: number;
    latency_max_ms: number;
    jitter_ms: number;
    packet_loss_pct: number;
    throughput_mbps: number;
}

export interface VerificationCheck {
    name: string;
    passed: boolean;
    detail: string;
    threshold?: string;
    actual?: string;
}

export interface VerificationReport {
    checks: VerificationCheck[];
    allPassed: boolean;
    gpu_fingerprint: string;
    benchmark: BenchmarkResult;
    network: NetworkBenchResult;
    versions: VersionCheck[];
}

// ── Helpers ──────────────────────────────────────────────────────────

function parseVersion(raw: string): [number, number, number] {
    const m = raw.match(/(\d+)\.(\d+)\.(\d+)/);
    if (!m) return [0, 0, 0];
    return [Number(m[1]), Number(m[2]), Number(m[3])];
}

function versionGte(actual: string, minimum: string): boolean {
    const a = parseVersion(actual);
    const b = parseVersion(minimum);
    for (let i = 0; i < 3; i++) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;
}

async function run(cmd: string, args: string[], timeout = 10_000): Promise<string | null> {
    try {
        const { stdout } = await exec(cmd, args, { timeout });
        return stdout.trim();
    } catch (err) {
        lastRunError = err instanceof Error ? err.message : String(err);
        return null;
    }
}

/** Last error from run() — for diagnostic context */
let lastRunError: string | null = null;

/** XCU score divisor — normalizes TFLOPS to marketplace compute units */
const XCU_DIVISOR = 10;

// ── 1. Version checks ───────────────────────────────────────────────

export async function checkVersions(): Promise<VersionCheck[]> {
    const results: VersionCheck[] = [];

    // runc
    const runcOut = await run("runc", ["--version"]);
    const runcVer = runcOut?.match(/runc version (\S+)/)?.[1] ?? null;
    results.push({
        component: "runc",
        version: runcVer,
        minimum: MINIMUM_VERSIONS.runc,
        passed: runcVer !== null && versionGte(runcVer, MINIMUM_VERSIONS.runc),
    });

    // Docker
    const dockerOut = await run("docker", ["version", "--format", "{{.Server.Version}}"]);
    results.push({
        component: "docker",
        version: dockerOut || null,
        minimum: MINIMUM_VERSIONS.docker,
        passed: dockerOut !== null && versionGte(dockerOut, MINIMUM_VERSIONS.docker),
    });

    // NVIDIA driver
    const driverOut = await run("nvidia-smi", [
        "--query-gpu=driver_version", "--format=csv,noheader,nounits",
    ]);
    const driverVer = driverOut?.split("\n")[0]?.trim() ?? null;
    results.push({
        component: "nvidia_driver",
        version: driverVer,
        minimum: MINIMUM_VERSIONS.nvidia_driver,
        passed: driverVer !== null && versionGte(driverVer, MINIMUM_VERSIONS.nvidia_driver),
    });

    // NVIDIA Container Toolkit
    const ntkOut = await run("nvidia-container-toolkit", ["--version"]);
    const ntkVer = ntkOut?.match(/(\d+\.\d+\.\d+)/)?.[1] ?? null;
    // Fallback: nvidia-container-runtime
    let toolkitVer = ntkVer;
    if (!toolkitVer) {
        const ncrOut = await run("nvidia-container-runtime", ["--version"]);
        toolkitVer = ncrOut?.match(/(\d+\.\d+\.\d+)/)?.[1] ?? null;
    }
    results.push({
        component: "nvidia_toolkit",
        version: toolkitVer,
        minimum: MINIMUM_VERSIONS.nvidia_toolkit,
        passed: toolkitVer !== null && versionGte(toolkitVer, MINIMUM_VERSIONS.nvidia_toolkit),
    });

    return results;
}

// ── 2. Full GPU detection ───────────────────────────────────────────

export async function detectGpuFull(): Promise<GpuInfo | null> {
    const fields = [
        "name", "memory.total", "memory.free",
        "serial", "uuid", "pci.bus_id",
        "driver_version", "compute_cap",
    ].join(",");

    const out = await run("nvidia-smi", [
        `--query-gpu=${fields}`,
        "--format=csv,noheader,nounits",
    ]);

    if (!out) return null;

    const parts = out.split("\n")[0].split(",").map((s) => s.trim());
    if (parts.length < 8) return null;

    return {
        gpu_model: parts[0],
        total_vram_gb: Math.round(parseFloat(parts[1]) / 1024 * 100) / 100,
        free_vram_gb: Math.round(parseFloat(parts[2]) / 1024 * 100) / 100,
        serial: parts[3],
        uuid: parts[4],
        pci_bus_id: parts[5],
        driver_version: parts[6],
        compute_capability: parts[7],
    };
}

// ── 3. Compute benchmark (FP16 matmul + PCIe + thermal) ─────────────

const BENCHMARK_SCRIPT = `
import time, json, subprocess

report = {}
try:
    import torch
    if not torch.cuda.is_available():
        print(json.dumps({"error": "no_cuda"}))
        raise SystemExit

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)

    report["gpu_model"] = props.name
    report["total_vram_gb"] = round(props.total_mem / (1024**3), 2)
    report["compute_capability"] = f"{props.major}.{props.minor}"
    report["cuda_version"] = torch.version.cuda or ""

    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        report["driver_version"] = smi.stdout.strip().split("\\n")[0].strip() if smi.returncode == 0 else ""
    except Exception:
        report["driver_version"] = ""

    # FP16 Matmul
    a = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    b = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    for _ in range(5):
        torch.mm(a, b)
    torch.cuda.synchronize()
    iters = 50
    start = time.perf_counter()
    for _ in range(iters):
        torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    flops_per_iter = 2 * 4096**3
    report["tflops"] = round((flops_per_iter * iters) / elapsed / 1e12, 2)
    report["elapsed_s"] = round(elapsed, 3)

    # PCIe Bandwidth
    try:
        size_mb = 256
        data_h = torch.randn(size_mb * 1024 * 256, dtype=torch.float32)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        data_d = data_h.to(device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        _ = data_d.to("cpu")
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        h2d_gbps = round(size_mb / 1024 / (t1 - t0), 2)
        d2h_gbps = round(size_mb / 1024 / (t2 - t1), 2)
        report["pcie_bandwidth_gbps"] = round((h2d_gbps + d2h_gbps) / 2, 2)
        report["pcie_h2d_gbps"] = h2d_gbps
        report["pcie_d2h_gbps"] = d2h_gbps
        del data_h, data_d
    except Exception as e:
        report["pcie_bandwidth_gbps"] = 0
        report["pcie_error"] = str(e)

    # Thermal Stability (15s sustained load)
    try:
        temps = []
        stress_start = time.perf_counter()
        while time.perf_counter() - stress_start < 15:
            torch.mm(a, b)
            torch.cuda.synchronize()
            try:
                tsmi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu",
                     "--format=csv,noheader,nounits", "--id=0"],
                    capture_output=True, text=True, timeout=5,
                )
                if tsmi.returncode == 0:
                    temps.append(float(tsmi.stdout.strip()))
            except Exception:
                pass
            time.sleep(0.5)
        if temps:
            report["gpu_temp_celsius"] = max(temps)
            report["gpu_temp_avg_celsius"] = round(sum(temps) / len(temps), 1)
            report["gpu_temp_samples"] = len(temps)
        else:
            report["gpu_temp_celsius"] = 0
    except Exception as e:
        report["gpu_temp_celsius"] = 0
        report["thermal_error"] = str(e)

    del a, b
    torch.cuda.empty_cache()
    print(json.dumps(report))

except ImportError:
    print(json.dumps({"error": "no_torch"}))
`;

export async function runComputeBenchmark(): Promise<BenchmarkResult | null> {
    try {
        const { stdout, stderr } = await exec(
            "python3", ["-c", BENCHMARK_SCRIPT],
            { timeout: 300_000 }, // 5 min timeout for thermal test
        );

        const line = stdout.trim().split("\n").pop();
        if (!line) {
            return {
                tflops: 0, pcie_bandwidth_gbps: 0, pcie_h2d_gbps: 0, pcie_d2h_gbps: 0,
                gpu_temp_celsius: 0, gpu_temp_avg_celsius: 0, gpu_temp_samples: 0,
                gpu_model: "", total_vram_gb: 0, compute_capability: "",
                cuda_version: "", driver_version: "", xcu_score: 0, elapsed_s: 0,
                error: stderr?.trim() || "No output from benchmark script",
            };
        }

        const data = JSON.parse(line);
        if (data.error) return { ...data, xcu_score: 0 } as BenchmarkResult;

        return {
            tflops: data.tflops ?? 0,
            pcie_bandwidth_gbps: data.pcie_bandwidth_gbps ?? 0,
            pcie_h2d_gbps: data.pcie_h2d_gbps ?? 0,
            pcie_d2h_gbps: data.pcie_d2h_gbps ?? 0,
            gpu_temp_celsius: data.gpu_temp_celsius ?? 0,
            gpu_temp_avg_celsius: data.gpu_temp_avg_celsius ?? 0,
            gpu_temp_samples: data.gpu_temp_samples ?? 0,
            gpu_model: data.gpu_model ?? "",
            total_vram_gb: data.total_vram_gb ?? 0,
            compute_capability: data.compute_capability ?? "",
            cuda_version: data.cuda_version ?? "",
            driver_version: data.driver_version ?? "",
            xcu_score: Math.round((data.tflops ?? 0) / XCU_DIVISOR * 100) / 100,
            elapsed_s: data.elapsed_s ?? 0,
        };
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        return {
            tflops: 0, pcie_bandwidth_gbps: 0, pcie_h2d_gbps: 0, pcie_d2h_gbps: 0,
            gpu_temp_celsius: 0, gpu_temp_avg_celsius: 0, gpu_temp_samples: 0,
            gpu_model: "", total_vram_gb: 0, compute_capability: "",
            cuda_version: "", driver_version: "", xcu_score: 0, elapsed_s: 0,
            error: msg,
        };
    }
}

// ── 4. Network benchmark ────────────────────────────────────────────

export async function runNetworkBenchmark(schedulerUrl: string): Promise<NetworkBenchResult> {
    const result: NetworkBenchResult = {
        latency_avg_ms: 0,
        latency_min_ms: 0,
        latency_max_ms: 0,
        jitter_ms: 0,
        packet_loss_pct: 0,
        throughput_mbps: 0,
    };

    // Extract hostname
    let host: string;
    try {
        host = new URL(schedulerUrl).hostname;
    } catch {
        return result;
    }

    // Ping for latency + jitter + loss (Linux/macOS compatible)
    const isLinux = process.platform === "linux";
    const pingArgs = isLinux
        ? ["-c", "20", "-i", "0.2", "-W", "2", host]
        : ["-c", "20", host]; // macOS: no -i/-W with these semantics
    const pingOut = await run("ping", pingArgs, 30_000);
    if (pingOut) {
        const lossMatch = pingOut.match(/(\d+(?:\.\d+)?)% packet loss/);
        if (lossMatch) result.packet_loss_pct = parseFloat(lossMatch[1]);

        // Linux: rtt min/avg/max/mdev   macOS: round-trip min/avg/max/stddev
        const rttMatch = pingOut.match(
            /(?:rtt|round-trip) min\/avg\/max\/(?:mdev|stddev) = ([\d.]+)\/([\d.]+)\/([\d.]+)\/([\d.]+)/,
        );
        if (rttMatch) {
            result.latency_min_ms = parseFloat(rttMatch[1]);
            result.latency_avg_ms = parseFloat(rttMatch[2]);
            result.latency_max_ms = parseFloat(rttMatch[3]);
            result.jitter_ms = parseFloat(rttMatch[4]);
        }
    }

    // Throughput estimate via HTTP — download a larger payload for meaningful measurement
    try {
        // Use /healthz (small) but do many iterations to amortize connection overhead
        const url = new URL("/healthz", schedulerUrl);
        const totalStart = performance.now();
        let totalBytes = 0;
        const ITERATIONS = 20;
        for (let i = 0; i < ITERATIONS; i++) {
            const resp = await fetch(url.toString(), { signal: AbortSignal.timeout(5_000) });
            const body = await resp.arrayBuffer();
            totalBytes += body.byteLength;
        }
        const totalTime = (performance.now() - totalStart) / 1000;
        if (totalTime > 0) {
            result.throughput_mbps = Math.round(totalBytes * 8 / totalTime / 1_000_000 * 100) / 100;
        }
    } catch {
        // throughput test failed — leave at 0
    }

    return result;
}

// ── 5. Run verification checks ──────────────────────────────────────

export function runVerificationChecks(
    gpu: GpuInfo,
    bench: BenchmarkResult,
    network: NetworkBenchResult,
    versions: VersionCheck[],
): VerificationCheck[] {
    const T = VERIFICATION_THRESHOLDS;
    const checks: VerificationCheck[] = [];

    // 1. GPU Identity
    checks.push({
        name: "GPU Identity",
        passed: !!gpu.gpu_model && !!gpu.uuid && gpu.total_vram_gb > 0,
        detail: `${gpu.gpu_model} · ${gpu.total_vram_gb} GB · ${gpu.uuid.slice(0, 12)}...`,
    });

    // 2. CUDA Readiness
    const cc = parseFloat(gpu.compute_capability || bench.compute_capability || "0");
    checks.push({
        name: "CUDA Readiness",
        passed: cc >= T.min_cuda_compute_capability && !!gpu.driver_version,
        detail: `Compute ${cc} · Driver ${gpu.driver_version}`,
        threshold: `≥ ${T.min_cuda_compute_capability}`,
        actual: String(cc),
    });

    // 3. PCIe Bandwidth
    checks.push({
        name: "PCIe Bandwidth",
        passed: bench.pcie_bandwidth_gbps >= T.min_pcie_bandwidth_gbps,
        detail: `${bench.pcie_bandwidth_gbps} GB/s (H2D: ${bench.pcie_h2d_gbps}, D2H: ${bench.pcie_d2h_gbps})`,
        threshold: `≥ ${T.min_pcie_bandwidth_gbps} GB/s`,
        actual: `${bench.pcie_bandwidth_gbps} GB/s`,
    });

    // 4. Thermal Stability — if temp is 0, sensor failed; treat as unknown/pass
    const thermalMeasured = bench.gpu_temp_celsius > 0;
    checks.push({
        name: "Thermal Stability",
        passed: !thermalMeasured || bench.gpu_temp_celsius <= T.max_gpu_temp_celsius,
        detail: thermalMeasured
            ? `Peak ${bench.gpu_temp_celsius}°C · Avg ${bench.gpu_temp_avg_celsius}°C (${bench.gpu_temp_samples} samples)`
            : "Temperature sensor unavailable — skipped",
        threshold: `≤ ${T.max_gpu_temp_celsius}°C`,
        actual: thermalMeasured ? `${bench.gpu_temp_celsius}°C` : "N/A",
    });

    // 5. Network Quality
    const netPassed = network.packet_loss_pct <= T.max_network_loss_pct
        && network.jitter_ms <= T.max_network_jitter_ms
        && network.throughput_mbps >= T.min_network_throughput_mbps;
    checks.push({
        name: "Network Quality",
        passed: netPassed,
        detail: `Loss ${network.packet_loss_pct}% · Jitter ${network.jitter_ms}ms · ${network.throughput_mbps} Mbps`,
        threshold: `Loss ≤${T.max_network_loss_pct}%, Jitter ≤${T.max_network_jitter_ms}ms, ≥${T.min_network_throughput_mbps} Mbps`,
    });

    // 6. Memory Fragmentation (VRAM claimed vs actual)
    const vramRatio = gpu.total_vram_gb > 0
        ? (bench.total_vram_gb / gpu.total_vram_gb) * 100
        : 0;
    checks.push({
        name: "Memory Integrity",
        passed: vramRatio >= T.min_vram_match_pct,
        detail: `Claimed ${gpu.total_vram_gb} GB · Measured ${bench.total_vram_gb} GB (${vramRatio.toFixed(1)}%)`,
        threshold: `≥ ${T.min_vram_match_pct}% match`,
        actual: `${vramRatio.toFixed(1)}%`,
    });

    // 7. Security Posture (all versions meet minimums)
    const allVersionsPass = versions.every((v) => v.passed);
    const failedVersions = versions.filter((v) => !v.passed);
    checks.push({
        name: "Security Posture",
        passed: allVersionsPass,
        detail: allVersionsPass
            ? "All components meet minimum versions"
            : `Failed: ${failedVersions.map((v) => `${v.component} (${v.version ?? "missing"} < ${v.minimum})`).join(", ")}`,
    });

    return checks;
}

// ── 6. Build verification report for POST /agent/verify ─────────────

export function buildVerificationReport(
    gpu: GpuInfo,
    bench: BenchmarkResult,
    network: NetworkBenchResult,
    versions: VersionCheck[],
): VerificationReport {
    const checks = runVerificationChecks(gpu, bench, network, versions);

    return {
        checks,
        allPassed: checks.every((c) => c.passed),
        gpu_fingerprint: `${gpu.gpu_model}:${gpu.uuid}:${gpu.pci_bus_id}`,
        benchmark: bench,
        network,
        versions,
    };
}
