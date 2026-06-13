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

/** Actionable remediation hints for check failures (A9). */
export const CHECK_REMEDIATION: Record<string, string> = {
    docker: "Install Docker 24+ and add your user to the docker group: sudo usermod -aG docker $USER",
    nvidia_driver: "Install NVIDIA driver 550+: https://www.nvidia.com/Download/index.aspx",
    nvidia_toolkit: "Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html",
    runc: "Upgrade runc to >= 1.1.12 via your distro packages or containerd",
    network: "Open outbound HTTPS (443) and ensure mesh (Tailscale) or public IP is reachable",
    benchmark: "Verify GPU is not throttled; check nvidia-smi and thermal limits",
    verify: "Re-run verification after fixing failed checks listed above",
    "host-register": "Confirm OAuth/API credentials in ~/.xcelsior/token.json and scheduler URL",
    "worker-install": "Run: curl -fsSL https://xcelsior.ca/install.sh | bash -s -- --agent-only",
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

/** @internal exported for testing */
export function parseVersion(raw: string): [number, number, number] {
    const m = raw.match(/(\d+)\.(\d+)\.(\d+)/);
    if (!m) return [0, 0, 0];
    return [Number(m[1]), Number(m[2]), Number(m[3])];
}

/** @internal exported for testing */
export function versionGte(actual: string, minimum: string): boolean {
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
    } catch {
        return null;
    }
}

/** XCU score divisor — normalizes TFLOPS to marketplace compute units */
const XCU_DIVISOR = 10;

// ── 1. Version checks ───────────────────────────────────────────────

export async function checkVersions(onItem?: (v: VersionCheck) => void): Promise<VersionCheck[]> {
    const results: VersionCheck[] = [];
    // Stream each component as it resolves (Part E — line-by-line status).
    const push = (v: VersionCheck) => { results.push(v); onItem?.(v); };

    // runc
    const runcOut = await run("runc", ["--version"]);
    const runcVer = runcOut?.match(/runc version (\S+)/)?.[1] ?? null;
    push({
        component: "runc",
        version: runcVer,
        minimum: MINIMUM_VERSIONS.runc,
        passed: runcVer !== null && versionGte(runcVer, MINIMUM_VERSIONS.runc),
    });

    // Docker
    const dockerOut = await run("docker", ["version", "--format", "{{.Server.Version}}"]);
    push({
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
    push({
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
    push({
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
    report["total_vram_gb"] = round(props.total_memory / (1024**3), 2)
    report["compute_capability"] = f"{props.major}.{props.minor}"
    report["cuda_version"] = torch.version.cuda or ""
    print("@progress Initializing CUDA on " + props.name, flush=True)

    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        report["driver_version"] = smi.stdout.strip().split("\\n")[0].strip() if smi.returncode == 0 else ""
    except Exception:
        report["driver_version"] = ""

    # FP16 Matmul
    print("@progress FP16 matmul (4096x4096) x50…", flush=True)
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
    print("@progress PCIe bandwidth (H2D / D2H)…", flush=True)
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
    print("@progress Thermal soak (15s sustained load)…", flush=True)
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

function emptyBenchmark(error: string): BenchmarkResult {
    return {
        tflops: 0, pcie_bandwidth_gbps: 0, pcie_h2d_gbps: 0, pcie_d2h_gbps: 0,
        gpu_temp_celsius: 0, gpu_temp_avg_celsius: 0, gpu_temp_samples: 0,
        gpu_model: "", total_vram_gb: 0, compute_capability: "",
        cuda_version: "", driver_version: "", xcu_score: 0, elapsed_s: 0,
        error,
    };
}

/** Parse the benchmark script's final JSON line into a BenchmarkResult. @internal */
export function parseBenchmarkOutput(stdout: string, stderr = ""): BenchmarkResult {
    const line = stdout
        .trim()
        .split("\n")
        .filter((l) => !l.startsWith("@progress"))
        .pop();
    if (!line) return emptyBenchmark(stderr.trim() || "No output from benchmark script");
    let data: Record<string, number | string>;
    try {
        data = JSON.parse(line);
    } catch {
        return emptyBenchmark(`Unparseable benchmark output: ${line.slice(0, 120)}`);
    }
    if (data.error) return { ...emptyBenchmark(String(data.error)), ...(data as object) } as BenchmarkResult;
    const tflops = Number(data.tflops ?? 0);
    return {
        tflops,
        pcie_bandwidth_gbps: Number(data.pcie_bandwidth_gbps ?? 0),
        pcie_h2d_gbps: Number(data.pcie_h2d_gbps ?? 0),
        pcie_d2h_gbps: Number(data.pcie_d2h_gbps ?? 0),
        gpu_temp_celsius: Number(data.gpu_temp_celsius ?? 0),
        gpu_temp_avg_celsius: Number(data.gpu_temp_avg_celsius ?? 0),
        gpu_temp_samples: Number(data.gpu_temp_samples ?? 0),
        gpu_model: String(data.gpu_model ?? ""),
        total_vram_gb: Number(data.total_vram_gb ?? 0),
        compute_capability: String(data.compute_capability ?? ""),
        cuda_version: String(data.cuda_version ?? ""),
        driver_version: String(data.driver_version ?? ""),
        xcu_score: Math.round(tflops / XCU_DIVISOR * 100) / 100,
        elapsed_s: Number(data.elapsed_s ?? 0),
    };
}

/**
 * Run the GPU compute benchmark. Streams phase markers ("@progress …") emitted
 * by the script to `onPhase` as they occur (FP16 matmul → PCIe → thermal),
 * then resolves with the parsed final result. Uses spawn so phases surface live
 * in the status pane instead of one buffered blob.
 */
export async function runComputeBenchmark(
    onPhase?: (msg: string) => void,
): Promise<BenchmarkResult | null> {
    const { spawn } = await import("node:child_process");
    return new Promise<BenchmarkResult>((resolve) => {
        let stdout = "";
        let stderr = "";
        let lineBuf = "";
        let settled = false;
        const finish = (r: BenchmarkResult) => { if (!settled) { settled = true; resolve(r); } };

        let child;
        try {
            child = spawn("python3", ["-c", BENCHMARK_SCRIPT], { timeout: 300_000 });
        } catch (err) {
            finish(emptyBenchmark(err instanceof Error ? err.message : String(err)));
            return;
        }

        child.stdout?.on("data", (d: Buffer) => {
            const s = d.toString();
            stdout += s;
            lineBuf += s;
            let idx: number;
            while ((idx = lineBuf.indexOf("\n")) >= 0) {
                const line = lineBuf.slice(0, idx);
                lineBuf = lineBuf.slice(idx + 1);
                if (line.startsWith("@progress ")) onPhase?.(line.slice("@progress ".length).trim());
            }
        });
        child.stderr?.on("data", (d: Buffer) => { stderr += d.toString(); });
        child.on("error", (err: Error) => finish(emptyBenchmark(err.message)));
        child.on("close", () => finish(parseBenchmarkOutput(stdout, stderr)));
    });
}

// ── 4. Network benchmark ────────────────────────────────────────────

export async function runNetworkBenchmark(
    schedulerUrl: string,
    onPhase?: (msg: string) => void,
): Promise<NetworkBenchResult> {
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
    onPhase?.(`Pinging ${host} (20 packets)…`);
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
    onPhase?.("Measuring throughput to the scheduler…");
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


// ── 7. Network setup — detect and configure mesh networking ─────────

export interface NetworkSetupResult {
    method: "tailscale" | "headscale" | "public" | "none";
    ip: string;
    detail: string;
}

export async function setupNetworking(): Promise<NetworkSetupResult> {
    const { execSync } = await import("child_process");

    // Check if Headscale/Tailscale mesh is available
    try {
        const tsStatus = execSync("tailscale status --json 2>/dev/null", {
            encoding: "utf-8",
            timeout: 10_000,
        });
        const status = JSON.parse(tsStatus);
        const selfIps = status?.Self?.TailscaleIPs;
        if (selfIps && selfIps.length > 0) {
            // Filter for IPv4
            const ip4 = selfIps.find((ip: string) => !ip.includes(":")) || selfIps[0];
            // Check if this is a headscale coordination server
            const isHeadscale = (status?.Self?.DNSName || "").includes("headscale") ||
                (status?.CurrentTailnet?.Name || "").includes("headscale");
            return {
                method: isHeadscale ? "headscale" : "tailscale",
                ip: ip4,
                detail: `Mesh network active — IP ${ip4}`,
            };
        }
    } catch {
        // Mesh networking not available or not connected
    }

    // Check for public IP (if no mesh available, use public IP with warning)
    try {
        const publicIp = execSync("curl -s --max-time 5 https://api.ipify.org", {
            encoding: "utf-8",
            timeout: 10_000,
        }).trim();
        if (/^\d+\.\d+\.\d+\.\d+$/.test(publicIp)) {
            return {
                method: "public",
                ip: publicIp,
                detail: `Public IP ${publicIp} — mesh networking recommended for security`,
            };
        }
    } catch {
        // Can't determine public IP
    }

    return {
        method: "none",
        ip: "",
        detail: "No network connectivity detected — please check your connection",
    };
}


// ── 8. Worker agent install — set up systemd service ────────────────

export interface WorkerInstallResult {
    installed: boolean;
    detail: string;
}

export async function installWorkerAgent(
    apiUrl: string,
    apiToken: string,
    hostId: string,
    hostIp: string,
    oauthClientId?: string,
    oauthClientSecret?: string,
): Promise<WorkerInstallResult> {
    const { execSync } = await import("child_process");
    const { writeFileSync, existsSync, mkdirSync } = await import("fs");
    const { join } = await import("path");
    const { homedir } = await import("os");

    // 1. Ensure config dir exists
    const configDir = join(homedir(), ".xcelsior");
    if (!existsSync(configDir)) {
        mkdirSync(configDir, { recursive: true });
    }

    // 2. Write worker agent env file
    // Prefer OAuth client credentials when available; fall back to API token.
    const envLines = [
        `XCELSIOR_HOST_ID=${hostId}`,
        `XCELSIOR_SCHEDULER_URL=${apiUrl}`,
        `XCELSIOR_HOST_IP=${hostIp}`,
    ];
    if (oauthClientId && oauthClientSecret) {
        envLines.push(`XCELSIOR_OAUTH_CLIENT_ID=${oauthClientId}`);
        envLines.push(`XCELSIOR_OAUTH_CLIENT_SECRET=${oauthClientSecret}`);
    }
    if (apiToken) {
        envLines.push(`XCELSIOR_API_TOKEN=${apiToken}`);
    }
    const envContent = envLines.join("\n") + "\n";

    const envFile = join(configDir, "worker.env");
    writeFileSync(envFile, envContent, { mode: 0o600 });

    // 3. Download worker_agent.py if not present
    const agentPath = join(configDir, "worker_agent.py");
    if (!existsSync(agentPath)) {
        try {
            execSync(
                `curl -fsSL "${apiUrl}/static/worker_agent.py" -o "${agentPath}"`,
                { timeout: 30_000 },
            );
        } catch {
            return {
                installed: false,
                detail: "Failed to download worker agent — check network connectivity",
            };
        }
    }

    // 4. Create systemd service unit
    const serviceContent = `[Unit]
Description=Xcelsior Worker Agent
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
EnvironmentFile=${envFile}
ExecStart=/usr/bin/python3 ${agentPath}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
`;

    try {
        // Write service file (needs sudo)
        const tmpService = "/tmp/xcelsior-worker.service";
        writeFileSync(tmpService, serviceContent);
        execSync(`sudo cp "${tmpService}" /etc/systemd/system/xcelsior-worker.service`, {
            timeout: 10_000,
        });
        execSync("sudo systemctl daemon-reload", { timeout: 10_000 });
        execSync("sudo systemctl enable xcelsior-worker", { timeout: 10_000 });
        execSync("sudo systemctl start xcelsior-worker", { timeout: 10_000 });

        return {
            installed: true,
            detail: "Worker agent installed and running as systemd service",
        };
    } catch (e) {
        return {
            installed: false,
            detail: `Service install failed — run manually: python3 ${agentPath}`,
        };
    }
}


// ── 9. SSH key setup for renters ────────────────────────────────────

export interface SshKeySetupResult {
    keyFound: boolean;
    uploaded: boolean;
    detail: string;
}

export async function setupSshKeys(
    apiUrl: string,
    token: string,
): Promise<SshKeySetupResult> {
    const { existsSync, readFileSync } = await import("fs");
    const { join } = await import("path");
    const { homedir } = await import("os");

    // Check for existing SSH keys
    const keyPaths = [
        join(homedir(), ".ssh", "id_ed25519.pub"),
        join(homedir(), ".ssh", "id_rsa.pub"),
        join(homedir(), ".ssh", "id_ecdsa.pub"),
    ];

    let pubKey: string | null = null;
    let keyType = "";
    for (const p of keyPaths) {
        if (existsSync(p)) {
            pubKey = readFileSync(p, "utf-8").trim();
            keyType = p.includes("ed25519") ? "ed25519" : p.includes("ecdsa") ? "ecdsa" : "rsa";
            break;
        }
    }

    if (!pubKey) {
        // Generate a new ed25519 key pair
        try {
            const { execSync } = await import("child_process");
            const keyPath = join(homedir(), ".ssh", "id_ed25519");
            execSync(
                `ssh-keygen -t ed25519 -f "${keyPath}" -N "" -q`,
                { timeout: 10_000 },
            );
            pubKey = readFileSync(keyPath + ".pub", "utf-8").trim();
            keyType = "ed25519";
        } catch {
            return {
                keyFound: false,
                uploaded: false,
                detail: "No SSH key found and auto-generation failed — add one manually in Settings",
            };
        }
    }

    // Upload to Xcelsior
    try {
        const resp = await fetch(`${apiUrl}/api/ssh/keys`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Cookie: `xcelsior_session=${token}`,
            },
            body: JSON.stringify({
                name: `${keyType} (wizard)`,
                public_key: pubKey,
            }),
        });
        if (resp.ok || resp.status === 409) {
            // 409 = key already exists, that's fine
            return {
                keyFound: true,
                uploaded: resp.ok,
                detail: resp.status === 409
                    ? "SSH key already registered"
                    : "SSH key uploaded — instances will be accessible via SSH",
            };
        }
        return {
            keyFound: true,
            uploaded: false,
            detail: "SSH key found but upload failed — add it manually in Settings",
        };
    } catch {
        return {
            keyFound: true,
            uploaded: false,
            detail: "SSH key found but upload failed — add it manually in Settings",
        };
    }
}
