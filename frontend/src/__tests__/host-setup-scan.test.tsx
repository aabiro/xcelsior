import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HostMachineSetupCard } from "@/components/hosts/host-setup-scan";
import {
  evaluateHostSetup,
  emptyObservation,
  observeBrowser,
  type HostSetupObservation,
  type Probe,
} from "@/lib/host-setup-requirements";

function measured<T>(value: T): Probe<T> {
  return { status: "measured", value };
}

function allMet(): HostSetupObservation {
  return {
    os: measured("Linux"),
    arch: measured("x86_64"),
    systemd: measured(true),
    python3: measured(true),
    curl: measured(true),
    openssl: measured(true),
    dockerCli: measured(true),
    dockerRunning: measured(true),
    nvidiaRuntime: measured(true),
    nvidiaSmi: measured(true),
    gpuUsable: measured(true),
    hostIp: measured("203.0.113.10"),
  };
}

function macWithoutGpu(): HostSetupObservation {
  const observation = emptyObservation();
  observation.os = measured("Darwin");
  observation.arch = measured("arm64");
  observation.nvidiaSmi = measured(true);
  observation.gpuUsable = measured(false);
  return observation;
}

async function scan(observation: HostSetupObservation) {
  render(<HostMachineSetupCard observe={async () => observation} />);
  fireEvent.click(screen.getByRole("button", { name: /test your machine setup/i }));
  expect(screen.getByTestId("host-setup-spinner")).toBeInTheDocument();
  expect(screen.queryAllByTestId("host-setup-row")).toHaveLength(0);
  await act(async () => {
    await Promise.resolve();
  });
  expect(screen.queryAllByTestId("host-setup-row")).toHaveLength(0);
  for (let index = 1; index <= 8; index += 1) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(280);
    });
    expect(screen.getAllByTestId("host-setup-row")).toHaveLength(index);
  }
  expect(screen.queryByTestId("host-setup-spinner")).not.toBeInTheDocument();
}

describe("host setup scan", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("reveals a check for every measured requirement, one row at a time", async () => {
    await scan(allMet());
    const rows = screen.getAllByTestId("host-setup-row");
    expect(rows.map((row) => row.getAttribute("data-requirement"))).toEqual([
      "linux-systemd",
      "cpu",
      "python3",
      "curl",
      "openssl",
      "docker",
      "gpu",
      "host-ip",
    ]);
    expect(rows.every((row) => row.getAttribute("data-mark") === "pass")).toBe(true);
    expect(screen.queryByText(/Not measured/)).not.toBeInTheDocument();
  });

  it("marks macOS and a missing GPU with an X and leaves unmeasured rows unmarked", async () => {
    await scan(macWithoutGpu());
    const linux = screen.getByText("Linux with systemd").closest("li");
    const gpu = screen.getByText("NVIDIA GPU").closest("li");
    const docker = screen.getByText("Docker with NVIDIA runtime").closest("li");
    expect(linux).toHaveAttribute("data-mark", "fail");
    expect(gpu).toHaveAttribute("data-mark", "fail");
    expect(linux?.textContent).toContain("Use a Linux host");
    expect(linux?.textContent).toContain("WSL2");
    expect(gpu?.textContent).toContain("Seat a supported NVIDIA GPU");
    expect(docker).toHaveAttribute("data-mark", "unresolved");
    expect(docker?.querySelector("svg")).toBeNull();
    expect(screen.getByText("Linux with systemd").closest("li")?.getAttribute("data-mark")).not.toBe("pass");
  });

  it("does not treat WebGPU or a Linux user agent as a GPU, Docker, or systemd pass", () => {
    const rows = evaluateHostSetup(observeBrowser({
      userAgent: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
      webgpu: true,
    }));
    for (const id of ["gpu", "docker", "linux-systemd"]) {
      expect(rows.find((row) => row.id === id)?.mark).not.toBe("pass");
    }
  });
});
