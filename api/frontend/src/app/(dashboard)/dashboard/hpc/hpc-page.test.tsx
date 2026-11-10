import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { HpcCluster, HpcJob } from "@/lib/api";
import { attentionItems, clusterStatus, formatDuration, jobStateLabel, summarise } from "./hpc-format";

// The operator console reads the v1 HPC read model (the reconciler's copy)
// and keeps the admin-only bridge for submit and cancel. The submit and
// cancel contracts were broken once (2026-11-07): priority went as "normal",
// and cancel passed the Xcelsior id where the route takes the Slurm id.

const replace = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn(), replace, back: vi.fn() }) }));
vi.mock("@/lib/locale", () => ({ useLocale: () => ({ t: (key: string) => key }) }));
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

let currentUser: { is_admin?: boolean; role?: string } | null = { is_admin: true };
vi.mock("@/lib/auth", () => ({ useAuth: () => ({ user: currentUser }) }));

const NOW = Date.now();
const ago = (seconds: number) => new Date(NOW - seconds * 1000).toISOString();

function cluster(over: Partial<HpcCluster> = {}): HpcCluster {
  return {
    cluster_id: "xcl-local", display_name: "Xcelsior local", profile: "xcl-local",
    enabled: true, research_only: false, marketplace: false, agent_access: true,
    poll_interval_seconds: 60,
    health: {
      last_poll_at: ago(20), transport_ok_at: ago(20), controller_ok_at: ago(20),
      accounting_ok_at: ago(20), consecutive_failures: 0, backing_off_until: null, last_error_code: null,
    },
    orphan_slurm_job_ids: [],
    transport: {
      kind: "registry", ssh_host: "10.0.0.5", ssh_port: 22, remote_user: "xclslurm",
      host_key_fingerprint: "SHA256:abc", credential_ref: "env:XCELSIOR_SLURM_KEY_LOCAL",
    },
    config_facts: { SLURM_VERSION: "25.11.8" },
    config_captured_at: ago(3600),
    ...over,
  };
}

function job(over: Partial<HpcJob> = {}): HpcJob {
  return {
    xcelsior_job_id: "a1b2c3d4", mapping_id: "m-1", submission_name: "train-a1", attempt_id: null,
    cluster_id: "xcl-local", slurm_job_id: "4242", observed_state: "running", slurm_state: "RUNNING",
    state_reason: "None", exit_code: null, derived_exit_code: null, verdict: null,
    desired_state: "running", settled: false,
    scheduler_progress: { elapsed_seconds: 1800, time_limit_seconds: 3600, fraction: 0.5 },
    awaiting_accounting_since: null, submitted_at: ago(2000), started_at: ago(1800), ended_at: null,
    settled_at: null, last_observed_at: ago(15), last_changed_at: ago(1800), missing_since: null,
    observed_age_seconds: 15, stale: false,
    ...over,
  };
}

const sick = cluster({
  cluster_id: "nibi", display_name: "SHARCNET Nibi", profile: "nibi",
  health: {
    last_poll_at: ago(900), transport_ok_at: ago(900), controller_ok_at: ago(900), accounting_ok_at: ago(900),
    consecutive_failures: 4, backing_off_until: new Date(NOW + 600_000).toISOString(),
    last_error_code: "host_key_mismatch",
  },
  orphan_slurm_job_ids: ["901"],
});

const fetchHpcClusters = vi.fn();
const fetchHpcJobs = vi.fn();
const fetchHpcJob = vi.fn();
const fetchHpcJobOutput = vi.fn();
const checkHpcClusterConnection = vi.fn();
const submitSlurmInstance = vi.fn();
const cancelSlurmJob = vi.fn();

vi.mock("@/lib/api", () => ({
  fetchHpcClusters: (...a: unknown[]) => fetchHpcClusters(...a),
  fetchHpcJobs: (...a: unknown[]) => fetchHpcJobs(...a),
  fetchHpcJob: (...a: unknown[]) => fetchHpcJob(...a),
  fetchHpcJobOutput: (...a: unknown[]) => fetchHpcJobOutput(...a),
  checkHpcClusterConnection: (...a: unknown[]) => checkHpcClusterConnection(...a),
  submitSlurmInstance: (...a: unknown[]) => submitSlurmInstance(...a),
  cancelSlurmJob: (...a: unknown[]) => cancelSlurmJob(...a),
  fetchSlurmProfiles: vi.fn(async () => ({
    profiles: { nibi: "SHARCNET Nibi", graham: "Compute Canada Graham (retired)" },
    details: {
      nibi: {
        name: "SHARCNET Nibi", cluster_id: "nibi", gpu_type: "h100", gpus_per_node: 8,
        partition_gpu: null, partition_cpu: null, max_walltime: "7-00:00:00",
        default_walltime: null, container_runtime: "apptainer", retired: false,
      },
      graham: {
        name: "Compute Canada Graham (retired)", cluster_id: "graham", gpu_type: "v100",
        gpus_per_node: 2, partition_gpu: "gpu", partition_cpu: "compute",
        max_walltime: "3-00:00:00", default_walltime: null, container_runtime: "apptainer", retired: true,
      },
    },
  })),
}));

import HpcPage from "./page";

const openTab = (name: RegExp) => fireEvent.click(screen.getByRole("tab", { name }));

describe("HPC operator console", () => {
  beforeEach(() => {
    currentUser = { is_admin: true };
    replace.mockReset();
    fetchHpcClusters.mockReset().mockResolvedValue({ ok: true, clusters: [cluster(), sick] });
    fetchHpcJobs.mockReset().mockResolvedValue({
      ok: true,
      jobs: [job(), job({ mapping_id: "m-2", xcelsior_job_id: "e5f6", submission_name: "queued-b2", slurm_job_id: "4243", observed_state: "pending", slurm_state: "PENDING", stale: true, scheduler_progress: { elapsed_seconds: null, time_limit_seconds: 3600, fraction: null } })],
      next_cursor: null,
    });
    fetchHpcJob.mockReset().mockImplementation(async () => ({ ok: true, job: job({ reported_progress: { source: "job", valid: true, fraction: 0.42, message: "epoch 3", fetched_at: 1, refresh_pending: false } }) }));
    fetchHpcJobOutput.mockReset().mockResolvedValue({
      ok: true,
      output: {
        xcelsior_job_id: "a1b2c3d4", cluster_id: "xcl-local", slurm_job_id: "4242", stream: "out",
        content: "step 1 ok\n", truncated: false, final: false, settled: false, fetched_at: ago(30),
        fetched_age_seconds: 30, refresh_pending: true, poll_interval_seconds: 60, fetch_error_code: null,
      },
    });
    checkHpcClusterConnection.mockReset().mockResolvedValue({ ok: true, check: { ok: false, error_code: "host_key_mismatch", error: "host key changed" } });
    submitSlurmInstance.mockReset().mockResolvedValue({ slurm_job_id: "77", cluster_id: "nibi", script: "#!/bin/bash\n#SBATCH --gpus-per-node=h100:2" });
    cancelSlurmJob.mockReset().mockResolvedValue({ cancelled: true, slurm_job_id: "4242" });
  });

  it("shows each cluster's health from the read model", async () => {
    render(<HpcPage />);
    const healthy = within(await screen.findByTestId("cluster-xcl-local"));
    expect(healthy.getByText("Healthy")).toBeTruthy();
    expect(healthy.getByText("xclslurm@10.0.0.5:22")).toBeTruthy();
    expect(healthy.getByText("env:XCELSIOR_SLURM_KEY_LOCAL")).toBeTruthy();

    const failing = within(screen.getByTestId("cluster-nibi"));
    expect(failing.getByText("Backing off")).toBeTruthy();
    expect(failing.getByText("host_key_mismatch")).toBeTruthy();
    expect(failing.getByText("901")).toBeTruthy();

    // 1 of 2 enabled clusters healthy; backing off, orphan, and a stale job.
    expect(screen.getByText("1/2")).toBeTruthy();
    expect(screen.getByText(/SHARCNET Nibi is backing off after 4 failed passes/)).toBeTruthy();
    expect(screen.getByText(/1 open job\(s\) have not been confirmed/)).toBeTruthy();
  });

  it("runs a connection check only after confirming, and shows a failure as a result", async () => {
    render(<HpcPage />);
    const card = within(await screen.findByTestId("cluster-nibi"));
    fireEvent.click(card.getByRole("button", { name: /Check connection/ }));
    expect(checkHpcClusterConnection).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Run check" }));
    await waitFor(() => expect(checkHpcClusterConnection).toHaveBeenCalledWith("nibi"));
    await card.findByText("Connection check failed: host_key_mismatch");
    expect(card.getByText("host key changed")).toBeTruthy();
  });

  it("lists jobs with state, time used and staleness, and refetches on a filter", async () => {
    render(<HpcPage />);
    await screen.findByTestId("cluster-xcl-local");
    openTab(/dash.hpc.tab_jobs/);
    await screen.findByText("train-a1");
    expect(screen.getByText("running (RUNNING)")).toBeTruthy();
    expect(screen.getByText("30m 0s of 1h 0m")).toBeTruthy();
    expect(screen.getByText("stale")).toBeTruthy();
    expect(screen.getByText("not started")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Settled" }));
    await waitFor(() =>
      expect(fetchHpcJobs).toHaveBeenCalledWith({ state: "settled", clusterId: undefined }),
    );

    fireEvent.change(screen.getByLabelText("Filter loaded jobs"), { target: { value: "4243" } });
    await waitFor(() => expect(screen.queryByText("train-a1")).toBeNull());
    expect(screen.getByText("queued-b2")).toBeTruthy();
  });

  it("opens a job with its cached output and the job's own progress", async () => {
    render(<HpcPage />);
    await screen.findByTestId("cluster-xcl-local");
    openTab(/dash.hpc.tab_jobs/);
    fireEvent.click(await screen.findByRole("button", { name: "View job train-a1" }));
    await waitFor(() => expect(fetchHpcJobOutput).toHaveBeenCalledWith("a1b2c3d4", "out"));
    await waitFor(() => expect(screen.getByTestId("hpc-job-output").textContent).toContain("step 1 ok"));
    await screen.findByText(/42% · epoch 3/);
    expect(screen.getByText(/A fresh copy is queued; reload after 1m 0s/)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "stderr" }));
    await waitFor(() => expect(fetchHpcJobOutput).toHaveBeenCalledWith("a1b2c3d4", "err"));
  });

  it("cancels by Slurm id on the job's cluster, after confirming", async () => {
    render(<HpcPage />);
    await screen.findByTestId("cluster-xcl-local");
    openTab(/dash.hpc.tab_jobs/);
    fireEvent.click(await screen.findByRole("button", { name: "Cancel job train-a1" }));
    expect(cancelSlurmJob).not.toHaveBeenCalled();
    const calls = fetchHpcJobs.mock.calls.length;
    fireEvent.click(screen.getByRole("button", { name: "Cancel job" }));
    await waitFor(() => expect(cancelSlurmJob).toHaveBeenCalledWith("4242", "xcl-local"));
    await waitFor(() => expect(fetchHpcJobs.mock.calls.length).toBeGreaterThan(calls));
  });

  it("submits the payload the API accepts", async () => {
    render(<HpcPage />);
    await screen.findByTestId("cluster-xcl-local");
    openTab(/dash.hpc.tab_submit/);
    fireEvent.change(await screen.findByPlaceholderText("my-training-run"), { target: { value: "train-1" } });
    fireEvent.click(screen.getByRole("button", { name: /Submit Job/ }));
    await waitFor(() => expect(submitSlurmInstance).toHaveBeenCalledTimes(1));
    const body = submitSlurmInstance.mock.calls[0][0];
    expect(body).toEqual({
      name: "train-1", profile: "nibi", priority: 0, num_gpus: 1,
      walltime: "01:00:00", command: "nvidia-smi", dry_run: false,
    });
    expect(typeof body.priority).toBe("number");
    expect(body).not.toHaveProperty("vram_needed_gb");
  });

  it("shows real profile specs and keeps retired clusters out of the picker", async () => {
    render(<HpcPage />);
    await screen.findByTestId("cluster-xcl-local");
    openTab(/dash.hpc.tab_submit/);
    await screen.findByText("h100 × 8/node");
    expect(screen.getAllByText("Partition: assigned by cluster").length).toBeGreaterThan(0);
    expect(screen.getByText("retired")).toBeTruthy();
    const options = Array.from(document.querySelectorAll("#hpc-profile option")).map((o) => o.textContent);
    expect(options).toEqual(["SHARCNET Nibi"]);
  });

  it("keeps the page up when the cluster read fails", async () => {
    fetchHpcClusters.mockRejectedValue(new Error("platform admin or a scoped machine principal is required"));
    render(<HpcPage />);
    await screen.findByText("platform admin or a scoped machine principal is required");
    openTab(/dash.hpc.tab_jobs/);
    await screen.findByText("train-a1");
  });

  it("is admin-only, like the routes behind it", async () => {
    currentUser = { role: "provider" };
    const { container } = render(<HpcPage />);
    expect(container.innerHTML).toBe("");
    await waitFor(() => expect(replace).toHaveBeenCalledWith("/dashboard"));
    expect(fetchHpcClusters).not.toHaveBeenCalled();
  });
});

describe("HPC console helpers", () => {
  it("classifies a cluster by what the reconciler last saw", () => {
    expect(clusterStatus(cluster(), NOW)).toBe("healthy");
    expect(clusterStatus(cluster({ enabled: false }), NOW)).toBe("disabled");
    expect(clusterStatus(sick, NOW)).toBe("backing_off");
    const neverPolled = cluster();
    neverPolled.health = { ...neverPolled.health, last_poll_at: null };
    expect(clusterStatus(neverPolled, NOW)).toBe("never_polled");
    // Accounting silent for more than three poll intervals.
    const lateAccounting = cluster();
    lateAccounting.health = { ...lateAccounting.health, accounting_ok_at: ago(181) };
    expect(clusterStatus(lateAccounting, NOW)).toBe("degraded");
  });

  it("labels a settled job by its verdict, an open one by its state", () => {
    expect(jobStateLabel(job())).toBe("running");
    expect(jobStateLabel(job({ settled: true, observed_state: "completed", verdict: "failed" }))).toBe("failed");
    expect(jobStateLabel(job({ settled: true, observed_state: "completed", verdict: "unknown" }))).toBe("completed");
  });

  it("summarises only enabled clusters and counts what needs attention", () => {
    const disabled = cluster({ cluster_id: "off", enabled: false });
    const summary = summarise([cluster(), sick, disabled], [job(), job({ stale: true, observed_state: "pending" })], NOW);
    expect(summary).toMatchObject({ clustersEnabled: 2, clustersHealthy: 1, open: 2, running: 1, pending: 1, stale: 1, orphans: 1 });
    expect(attentionItems([cluster(), disabled], [job()], NOW)).toEqual([]);
  });

  it("formats durations", () => {
    expect(formatDuration(42)).toBe("42s");
    expect(formatDuration(3725)).toBe("1h 2m");
    expect(formatDuration(200000)).toBe("2d 7h");
    expect(formatDuration(null)).toBe("—");
  });
});
