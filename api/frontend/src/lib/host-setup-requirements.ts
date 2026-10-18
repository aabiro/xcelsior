// Shared host-onboarding verdicts. install.sh prints these same sentences.
// A row is pass or fail only after that fact was observed. Unmeasured rows
// stay unresolved so a browser hint cannot pass GPU, Docker, or systemd.

export const HOST_FAIL_LINUX_DARWIN =
    "This machine is macOS (Darwin), so it cannot host an Xcelsior worker yet. Use a Linux host. WSL2 counts only when systemd is actually running (the systemctl command is available).";

export const HOST_FAIL_LINUX_OTHER_SUFFIX =
    "not Linux, so it cannot host an Xcelsior worker yet. Use a Linux host. WSL2 counts only when systemd is actually running (the systemctl command is available).";

export const HOST_FAIL_ARCH_SUFFIX =
    "which this worker does not support. Use an amd64 (x86_64) or arm64 (aarch64) Linux machine.";

export const HOST_FAIL_SYSTEMD =
    "systemd is not available on this machine (systemctl was not found), so the worker service cannot be installed. Install systemd, or use a Linux distribution that ships it. On WSL2, set systemd=true in /etc/wsl.conf and restart WSL.";

export const HOST_FAIL_PYTHON3 =
    "python3 is not installed, so the worker cannot run. Install Python 3 from your distribution packages (for example: sudo apt-get install -y python3 python3-venv) and run the installer again.";

export const HOST_FAIL_CURL =
    "curl is not installed, so the installer cannot download the worker. Install curl (for example: sudo apt-get install -y curl) and run the installer again.";

export const HOST_FAIL_OPENSSL =
    "openssl is not installed, so the installer cannot verify the worker signature. Install openssl (for example: sudo apt-get install -y openssl) and run the installer again.";

export const HOST_FAIL_DOCKER_MISSING =
    "docker is not installed, so rented GPUs cannot start containers. Install Docker Engine, start it, and add your user to the docker group: https://docs.docker.com/engine/install/";

export const HOST_FAIL_NVIDIA_SMI =
    "nvidia-smi is not installed, so no NVIDIA driver was found. Install a supported NVIDIA driver (550 or newer), then run nvidia-smi once to confirm a GPU is listed: https://www.nvidia.com/Download/index.aspx";

export const HOST_FAIL_GPU =
    "nvidia-smi is installed but it did not list a usable NVIDIA GPU, so this machine cannot host yet. Seat a supported NVIDIA GPU, confirm the driver loaded (nvidia-smi prints a GPU name), and run the installer again.";

export const HOST_FAIL_DOCKER_DOWN =
    "Docker is installed but it is not running, or this user is not allowed to talk to the Docker daemon. Start Docker, then add your user to the docker group with sudo usermod -aG docker \"$USER\" and open a new login session.";

export const HOST_FAIL_NVIDIA_RUNTIME =
    "Docker is running but the NVIDIA container runtime is not installed, so GPU containers cannot start. Install the NVIDIA Container Toolkit and restart Docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html";

export const HOST_FAIL_HOST_IP =
    "No host IP could be detected, so renters would not know where to reach this machine. Check that this host can reach the public internet (curl https://api.ipify.org), or bring up Tailscale and run the installer again. Tailscale is optional; a public IP is enough.";

export const SDK_FAIL_PYTHON =
    "This is a Python project, so the TypeScript SDK cannot be installed in this folder yet.";

export const SDK_FAIL_NO_PACKAGE =
    "No package.json was found in this directory tree, so the TypeScript SDK cannot be installed yet.";

export const SDK_FAIL_INSTALL =
    "The package manager could not install @xcelsior-gpu/sdk, so this app is not connected yet.";

export const SDK_FAIL_CREDENTIALS =
    "SDK setup was started before credentials exist, so nothing can be saved yet. Finish sign-in, then create an API key or OAuth client and retry.";

export const SDK_FAIL_PACKAGE_MANAGER_PREFIX =
    "This project declares a package manager the TypeScript SDK setup does not support yet.";

export type Probe<T> = { status: "unmeasured" } | { status: "measured"; value: T };

export interface HostSetupObservation {
    os: Probe<string>;
    arch: Probe<string>;
    systemd: Probe<boolean>;
    python3: Probe<boolean>;
    curl: Probe<boolean>;
    openssl: Probe<boolean>;
    dockerCli: Probe<boolean>;
    dockerRunning: Probe<boolean>;
    nvidiaRuntime: Probe<boolean>;
    nvidiaSmi: Probe<boolean>;
    gpuUsable: Probe<boolean>;
    hostIp: Probe<string | null>;
}

export type HostSetupMark = "pass" | "fail" | "unresolved";

export interface HostSetupRow {
    id: string;
    label: string;
    mark: HostSetupMark;
    detail: string;
}

const unmeasured = <T,>(): Probe<T> => ({ status: "unmeasured" });

export function emptyObservation(): HostSetupObservation {
    return {
        os: unmeasured(),
        arch: unmeasured(),
        systemd: unmeasured(),
        python3: unmeasured(),
        curl: unmeasured(),
        openssl: unmeasured(),
        dockerCli: unmeasured(),
        dockerRunning: unmeasured(),
        nvidiaRuntime: unmeasured(),
        gpuUsable: unmeasured(),
        nvidiaSmi: unmeasured(),
        hostIp: unmeasured(),
    };
}

export function linuxRequiredMessage(os: string): string {
    if (os === "darwin" || os === "Darwin") return HOST_FAIL_LINUX_DARWIN;
    return `This machine is ${os}, ${HOST_FAIL_LINUX_OTHER_SUFFIX}`;
}

export function archRejectedMessage(arch: string): string {
    return `This CPU architecture is ${arch}, ${HOST_FAIL_ARCH_SUFFIX}`;
}

function measured<T>(probe: Probe<T>): T | undefined {
    return probe.status === "measured" ? probe.value : undefined;
}

function commandRow(id: string, label: string, probe: Probe<boolean>, fail: string): HostSetupRow {
    if (probe.status === "unmeasured") {
        return { id, label, mark: "unresolved", detail: "Not measured from this page. Run the installer on the GPU host to check this." };
    }
    return probe.value
        ? { id, label, mark: "pass", detail: `${label} is installed.` }
        : { id, label, mark: "fail", detail: fail };
}

const SUPPORTED_ARCH = new Set(["x86_64", "amd64", "aarch64", "arm64"]);

/** Verdicts in installer order. webgpu and user-agent never reach this function. */
export function evaluateHostSetup(obs: HostSetupObservation): HostSetupRow[] {
    const rows: HostSetupRow[] = [];

    const os = measured(obs.os);
    const systemd = measured(obs.systemd);
    if (os === undefined && systemd === undefined) {
        rows.push({
            id: "linux-systemd",
            label: "Linux with systemd",
            mark: "unresolved",
            detail: "Not measured from this page. Run the installer on the GPU host to check this.",
        });
    } else if (os !== undefined && os !== "Linux") {
        rows.push({
            id: "linux-systemd",
            label: "Linux with systemd",
            mark: "fail",
            detail: linuxRequiredMessage(os),
        });
    } else if (systemd === false) {
        rows.push({
            id: "linux-systemd",
            label: "Linux with systemd",
            mark: "fail",
            detail: HOST_FAIL_SYSTEMD,
        });
    } else if (os === "Linux" && systemd === true) {
        rows.push({
            id: "linux-systemd",
            label: "Linux with systemd",
            mark: "pass",
            detail: "Linux and systemd are available.",
        });
    } else {
        rows.push({
            id: "linux-systemd",
            label: "Linux with systemd",
            mark: "unresolved",
            detail: "Not measured from this page. Run the installer on the GPU host to check this.",
        });
    }

    const arch = measured(obs.arch);
    if (arch === undefined) {
        rows.push({
            id: "cpu",
            label: "Supported CPU",
            mark: "unresolved",
            detail: "Not measured from this page. Run the installer on the GPU host to check this.",
        });
    } else if (SUPPORTED_ARCH.has(arch)) {
        rows.push({ id: "cpu", label: "Supported CPU", mark: "pass", detail: `${arch} is supported.` });
    } else {
        rows.push({ id: "cpu", label: "Supported CPU", mark: "fail", detail: archRejectedMessage(arch) });
    }

    rows.push(commandRow("python3", "python3", obs.python3, HOST_FAIL_PYTHON3));
    rows.push(commandRow("curl", "curl", obs.curl, HOST_FAIL_CURL));
    rows.push(commandRow("openssl", "openssl", obs.openssl, HOST_FAIL_OPENSSL));

    const dockerCli = measured(obs.dockerCli);
    const dockerRunning = measured(obs.dockerRunning);
    const nvidiaRuntime = measured(obs.nvidiaRuntime);
    if (dockerCli === false) {
        rows.push({ id: "docker", label: "Docker with NVIDIA runtime", mark: "fail", detail: HOST_FAIL_DOCKER_MISSING });
    } else if (dockerRunning === false) {
        rows.push({ id: "docker", label: "Docker with NVIDIA runtime", mark: "fail", detail: HOST_FAIL_DOCKER_DOWN });
    } else if (nvidiaRuntime === false) {
        rows.push({ id: "docker", label: "Docker with NVIDIA runtime", mark: "fail", detail: HOST_FAIL_NVIDIA_RUNTIME });
    } else if (dockerCli === true && dockerRunning === true && nvidiaRuntime === true) {
        rows.push({
            id: "docker",
            label: "Docker with NVIDIA runtime",
            mark: "pass",
            detail: "Docker is running and the NVIDIA container runtime is installed.",
        });
    } else {
        rows.push({
            id: "docker",
            label: "Docker with NVIDIA runtime",
            mark: "unresolved",
            detail: "Not measured from this page. Run the installer on the GPU host to check this.",
        });
    }

    const nvidiaSmi = measured(obs.nvidiaSmi);
    const gpuUsable = measured(obs.gpuUsable);
    if (nvidiaSmi === false) {
        rows.push({ id: "gpu", label: "NVIDIA GPU", mark: "fail", detail: HOST_FAIL_NVIDIA_SMI });
    } else if (gpuUsable === false) {
        rows.push({ id: "gpu", label: "NVIDIA GPU", mark: "fail", detail: HOST_FAIL_GPU });
    } else if (nvidiaSmi === true && gpuUsable === true) {
        rows.push({ id: "gpu", label: "NVIDIA GPU", mark: "pass", detail: "nvidia-smi listed a usable NVIDIA GPU." });
    } else {
        rows.push({
            id: "gpu",
            label: "NVIDIA GPU",
            mark: "unresolved",
            detail: "Not measured from this page. A browser cannot see nvidia-smi.",
        });
    }

    const hostIp = obs.hostIp.status === "measured" ? obs.hostIp.value : undefined;
    if (obs.hostIp.status === "unmeasured") {
        rows.push({
            id: "host-ip",
            label: "Host IP",
            mark: "unresolved",
            detail: "Not measured from this page. Run the installer on the GPU host to check this.",
        });
    } else if (hostIp) {
        rows.push({ id: "host-ip", label: "Host IP", mark: "pass", detail: "A host IP was detected." });
    } else {
        rows.push({ id: "host-ip", label: "Host IP", mark: "fail", detail: HOST_FAIL_HOST_IP });
    }

    return rows;
}

/** Browser facts. webgpu is accepted and ignored so it cannot pass a GPU row. */
export function observeBrowser(input: { userAgent: string; webgpu?: boolean }): HostSetupObservation {
    const ua = input.userAgent || "";
    const obs = emptyObservation();
    if (/mac os x|macintosh/i.test(ua)) obs.os = { status: "measured", value: "Darwin" };
    else if (/linux/i.test(ua)) obs.os = { status: "measured", value: "Linux" };
    else if (/windows/i.test(ua)) obs.os = { status: "measured", value: "Windows" };
    else if (ua.trim()) obs.os = { status: "measured", value: "unknown" };

    if (/aarch64|arm64/i.test(ua)) obs.arch = { status: "measured", value: "arm64" };
    else if (/x86_64|amd64|win64|wow64/i.test(ua)) obs.arch = { status: "measured", value: "x86_64" };
    return obs;
}

export function explainInstallFailure(stdout: string, stderr: string, fallback: string): string {
    const clean = `${stdout}\n${stderr}`.replace(/\u001b\[[0-9;]*m/g, "");
    const line = clean
        .split("\n")
        .map((item) => item.replace(/^\[✗\]\s*/, "").trim())
        .find((item) =>
            item.includes("cannot host")
            || item.includes("not installed")
            || item.includes("does not support")
            || item.includes("not available")
            || item.includes("did not list")
            || item.includes("not running")
            || item.includes("could not be detected")
            || item.includes("container runtime is not installed"));
    return line || fallback;
}

export function sdkInstallFailure(raw: string): { detail: string; remediation: string } {
    const brief = raw
        .split("\n")
        .map((line) => line.trim())
        .find((line) => line && !line.startsWith("at ") && !line.startsWith("npm ERR!")) ?? "";
    const clipped = brief.slice(0, 180);
    return {
        detail: SDK_FAIL_INSTALL,
        remediation: clipped
            ? `Fix that package-manager error in this project, then retry this step. It reported: ${clipped}`
            : "Fix the package-manager error in this project, then retry this step.",
    };
}

export function unsupportedPackageManagerMessage(name: string): string {
    return `${SDK_FAIL_PACKAGE_MANAGER_PREFIX} ${name} is not one of npm, pnpm, yarn, or bun. Change package.json to one of those, then retry.`;
}
