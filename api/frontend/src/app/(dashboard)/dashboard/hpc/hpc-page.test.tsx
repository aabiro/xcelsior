import { describe, expect, it, vi, beforeEach } from "vitest";
import { act, render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { HpcBilling, HpcCluster, HpcJob, HpcSubmission } from "@/lib/api";
import {
  attentionItems,
  billingRuleLabel,
  clusterStatus,
  formatDuration,
  formatGpuHours,
  formatMicros,
  jobStateLabel,
  rateLabel,
  submissionTone,
  summarise,
} from "./hpc-format";

// The operator console reads the v1 HPC read model (the reconciler's copy)
// and keeps the admin-only bridge for submit and cancel. The submit and
// cancel contracts were broken once (2026-11-07): priority went as "normal",
// and cancel passed the Xcelsior id where the route takes the Slurm id.

// The tab and the open job live in the query string, so the router mock keeps one.
const nav = vi.hoisted(() => ({
  search: "",
  listeners: new Set<() => void>(),
  replace: vi.fn(),
}));
vi.mock("next/navigation", async () => {
  const React = await import("react");
  const router = { push: vi.fn(), replace: nav.replace, back: vi.fn() };
  return {
    useRouter: () => router,
    usePathname: () => "/dashboard/hpc",
    useSearchParams: () => {
      const search = React.useSyncExternalStore(
        (notify: () => void) => {
          nav.listeners.add(notify);
          return () => nav.listeners.delete(notify);
        },
        () => nav.search,
      );
      return React.useMemo(() => new URLSearchParams(search), [search]);
    },
  };
});
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

const billing: HpcBilling = {
  ok: true, days: 14, tolerance_pct: 1,
  totals: {
    segments: 2, jobs: 1, gpu_seconds: 7200, billable_seconds: 7200, shadow_cost_micros: 700_000,
    charged_micros: 0, charged_segments: 0, unpriced_segments: 0,
  },
  drift: [
    {
      cluster_id: "xcl-local", day: "2026-11-13", sacct_gpu_seconds: 400, ledger_gpu_seconds: 380,
      sacct_records: 2, ledger_records: 2, drift_pct: 5, within_tolerance: false,
      unmatched: { only_sacct: ["77@1"] }, checked_at: ago(600),
    },
    {
      cluster_id: "xcl-local", day: "2026-11-12", sacct_gpu_seconds: 376, ledger_gpu_seconds: 376,
      sacct_records: 2, ledger_records: 2, drift_pct: 0, within_tolerance: true, unmatched: {}, checked_at: ago(900),
    },
  ],
  segments: [
    {
      segment_id: "s-2", mapping_id: "m-9", run_index: 1, cluster_id: "xcl-local", slurm_job_id: "298",
      xcelsior_job_id: "requeued-job", slurm_state: "COMPLETED", started_at: ago(4000), ended_at: ago(3700),
      elapsed_seconds: 211, suspended_seconds: 90, time_limit_seconds: 600, gpu_count: 1, cpu_count: 4,
      node_count: 1, billable_seconds: 301, billing_rule: "ran_to_end+suspended_gres_held", gpu_seconds: 301,
      rate_micros_per_gpu_hour: 350_000, shadow_cost_micros: 29_264, charged: false, meter_id: null,
    },
    {
      segment_id: "s-1", mapping_id: "m-9", run_index: 0, cluster_id: "xcl-local", slurm_job_id: "298",
      xcelsior_job_id: "requeued-job", slurm_state: "REQUEUED", started_at: ago(4200), ended_at: ago(4125),
      elapsed_seconds: 75, suspended_seconds: 0, time_limit_seconds: 600, gpu_count: 1, cpu_count: 4,
      node_count: 1, billable_seconds: 75, billing_rule: "requeued_run", gpu_seconds: 75,
      rate_micros_per_gpu_hour: 350_000, shadow_cost_micros: 7292, charged: false, meter_id: null,
    },
  ],
  next_cursor: null,
};

function submission(over: Partial<HpcSubmission> = {}): HpcSubmission {
  return {
    submission_name: "train-a1", cluster_id: "xcl-local", xcelsior_job_id: "a1b2c3d4", attempt_id: "att-1",
    state: "submitted", slurm_job_id: "4242", sbatch_calls: 1, checks: 0, error_code: null, error: null,
    created_at: ago(2010), submit_started_at: ago(2005), submitted_at: ago(2000), last_check_at: null,
    resolved_at: ago(2000),
    ...over,
  };
}

const fetchHpcClusters = vi.fn();
const fetchHpcJobs = vi.fn();
const fetchHpcJob = vi.fn();
const fetchHpcJobOutput = vi.fn();
const fetchHpcBilling = vi.fn();
const fetchHpcSubmissions = vi.fn();
const checkHpcClusterConnection = vi.fn();
const registerHpcCluster = vi.fn();
const submitSlurmInstance = vi.fn();
const cancelSlurmJob = vi.fn();

vi.mock("@/lib/api", () => ({
  fetchHpcClusters: (...a: unknown[]) => fetchHpcClusters(...a),
  fetchHpcJobs: (...a: unknown[]) => fetchHpcJobs(...a),
  fetchHpcJob: (...a: unknown[]) => fetchHpcJob(...a),
  fetchHpcJobOutput: (...a: unknown[]) => fetchHpcJobOutput(...a),
  fetchHpcBilling: (...a: unknown[]) => fetchHpcBilling(...a),
  fetchHpcSubmissions: (...a: unknown[]) => fetchHpcSubmissions(...a),
  checkHpcClusterConnection: (...a: unknown[]) => checkHpcClusterConnection(...a),
  registerHpcCluster: (...a: unknown[]) => registerHpcCluster(...a),
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

function navigate(url: string) {
  nav.search = url.split("?")[1] ?? "";
  for (const notify of [...nav.listeners]) notify();
}

describe("HPC operator console", () => {
  beforeEach(() => {
    currentUser = { is_admin: true };
    nav.search = "";
    nav.replace.mockReset().mockImplementation((url: string) => {
      if (url.startsWith("/dashboard/hpc")) act(() => navigate(url));
    });
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
    fetchHpcBilling.mockReset().mockResolvedValue(billing);
    fetchHpcSubmissions.mockReset().mockResolvedValue({
      ok: true,
      counts: { pending: 0, submitting: 0, submitted: 1, unknown: 0, rejected: 1, lost: 0 },
      submissions: [
        submission(),
        submission({
          submission_name: "bad-partition-a1", xcelsior_job_id: "dead0001", state: "rejected", slurm_job_id: null,
          error_code: "invalid_partition", error: "sbatch: error: invalid partition",
        }),
      ],
      next_cursor: null,
    });
    checkHpcClusterConnection.mockReset().mockResolvedValue({ ok: true, check: { ok: false, error_code: "host_key_mismatch", error: "host key changed" } });
    registerHpcCluster.mockReset().mockResolvedValue({ ok: true, cluster: { cluster_id: "nibi", profile_key: "nibi", research_only: true, transport: "registry" } });
    submitSlurmInstance.mockReset().mockResolvedValue({ slurm_job_id: "77", cluster_id: "nibi", script: "#!/bin/bash\n#SBATCH --gpus-per-node=h100:2" });
    cancelSlurmJob.mockReset().mockResolvedValue({ cancelled: true, slurm_job_id: "4242" });
  });

  it("shows each cluster's health from the read model", async () => {
    render(<HpcPage />);
    const healthyCard = await screen.findByTestId("cluster-xcl-local");
    const healthy = within(healthyCard);
    expect(healthy.getByText("Healthy")).toBeTruthy();
    // How it is reached is one click away, never the key itself.
    fireEvent.click(healthy.getByRole("button", { name: "Details" }));
    expect(healthy.getByText("xclslurm@10.0.0.5:22")).toBeTruthy();
    expect(healthy.getByText("env:XCELSIOR_SLURM_KEY_LOCAL")).toBeTruthy();
    expect(healthy.getByText("SHA256:abc")).toBeTruthy();

    const failing = within(screen.getByTestId("cluster-nibi"));
    expect(failing.getByText("Backing off")).toBeTruthy();
    expect(failing.getByText("host_key_mismatch")).toBeTruthy();
    expect(failing.getByText("901")).toBeTruthy();

    // 1 of 2 enabled clusters healthy; backing off, orphan, and a stale job.
    expect(screen.getByText("1/2")).toBeTruthy();
    expect(screen.getByText(/SHARCNET Nibi is backing off after 4 failed passes/)).toBeTruthy();
    expect(screen.getByText(/1 open job\(s\) have not been confirmed/)).toBeTruthy();
    expect(screen.getByText("Needs attention now")).toBeTruthy();
  });

  it("shows a cluster's exact open-job counts, rate and failures when the registry sends them", async () => {
    fetchHpcClusters.mockResolvedValue({
      ok: true,
      clusters: [
        cluster({
          open_jobs: { total: 250, running: 200, pending: 50, stale: 0, awaiting_accounting: 3 },
          rate_card: { gpu_hour_micros: 350_000, gpu_model: "RTX 3060" },
          failure_counts: { "pass:transport_timeout": 2, "output:fetch_failed": 1 },
          suspend_holds_gres: false,
        }),
      ],
    });
    render(<HpcPage />);
    const card = within(await screen.findByTestId("cluster-xcl-local"));
    expect(card.getByText("250")).toBeTruthy();
    expect(card.getByRole("img", { name: "200 running, 50 pending" })).toBeTruthy();
    expect(card.getByText("3 awaiting accounting")).toBeTruthy();
    expect(card.getByText("$0.35 per GPU hour · RTX 3060")).toBeTruthy();
    fireEvent.click(card.getByRole("button", { name: "Details" }));
    expect(card.getByText("transport_timeout")).toBeTruthy();
    expect(card.getByText(/Free: a suspended job releases its GPUs/)).toBeTruthy();
    // The header counts every open job, not the two the job list loaded.
    expect(screen.getByText("of 250 open")).toBeTruthy();
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

  it("registers a cluster disabled, and only with a pinned host key and a credential reference", async () => {
    render(<HpcPage />);
    await screen.findByTestId("cluster-xcl-local");
    fireEvent.click(screen.getByRole("button", { name: /Register cluster/ }));
    const dialog = within(await screen.findByRole("dialog"));
    await waitFor(() => expect((dialog.getByLabelText("Adapter profile") as HTMLSelectElement).value).toBe("nibi"));
    // The retired profile is not offered.
    expect(dialog.queryByRole("option", { name: /Graham/ })).toBeNull();
    const send = dialog.getByRole("button", { name: "Register disabled" }) as HTMLButtonElement;
    expect(send.disabled).toBe(true);

    fireEvent.change(dialog.getByLabelText("Login node"), { target: { value: "nibi.alliancecan.ca" } });
    fireEvent.change(dialog.getByLabelText("Host key to pin"), { target: { value: "nibi.alliancecan.ca ssh-ed25519 AAAA" } });
    // The key itself is refused: only a reference to where the API keeps it.
    fireEvent.change(dialog.getByLabelText("Credential reference"), { target: { value: "-----BEGIN OPENSSH PRIVATE KEY-----" } });
    fireEvent.blur(dialog.getByLabelText("Credential reference"));
    expect(send.disabled).toBe(true);
    expect(dialog.getByText(/env:XCELSIOR_SLURM_KEY_<NAME>, or file:<name>/)).toBeTruthy();

    fireEvent.change(dialog.getByLabelText("Credential reference"), { target: { value: "env:XCELSIOR_SLURM_KEY_NIBI" } });
    expect(send.disabled).toBe(false);
    const loads = fetchHpcClusters.mock.calls.length;
    fireEvent.click(send);
    await waitFor(() => expect(registerHpcCluster).toHaveBeenCalledTimes(1));
    expect(registerHpcCluster.mock.calls[0][0]).toEqual({
      profile_key: "nibi", enabled: false, ssh_host: "nibi.alliancecan.ca", ssh_port: 22,
      known_hosts_line: "nibi.alliancecan.ca ssh-ed25519 AAAA", credential_ref: "env:XCELSIOR_SLURM_KEY_NIBI",
    });
    await waitFor(() => expect(fetchHpcClusters.mock.calls.length).toBeGreaterThan(loads));
  });

  it("lists jobs with state, time used and staleness, and refetches on a filter", async () => {
    render(<HpcPage />);
    await screen.findByTestId("cluster-xcl-local");
    openTab(/dash.hpc.tab_jobs/);
    await screen.findAllByText("train-a1");
    expect(screen.getAllByText("30m 0s of 1h 0m").length).toBeGreaterThan(0);
    expect(screen.getAllByText("stale").length).toBeGreaterThan(0);
    expect(screen.getAllByText("not started").length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: "Settled" }));
    await waitFor(() =>
      expect(fetchHpcJobs).toHaveBeenCalledWith({ state: "settled", clusterId: undefined }),
    );

    fireEvent.change(screen.getByLabelText("Filter loaded jobs"), { target: { value: "4243" } });
    await waitFor(() => expect(screen.queryByText("train-a1")).toBeNull());
    expect(screen.getAllByText("queued-b2").length).toBeGreaterThan(0);
  });

  it("a cluster's View jobs opens the job list filtered to that cluster", async () => {
    render(<HpcPage />);
    const card = within(await screen.findByTestId("cluster-nibi"));
    fireEvent.click(card.getByRole("button", { name: /View jobs/ }));
    await waitFor(() => expect(fetchHpcJobs).toHaveBeenCalledWith({ state: "open", clusterId: "nibi" }));
    expect(nav.search).toBe("tab=jobs");
    expect((screen.getByLabelText("Cluster") as HTMLSelectElement).value).toBe("nibi");
  });

  it("opens a job with its cached output and the job's own progress, and puts it in the address", async () => {
    render(<HpcPage />);
    await screen.findByTestId("cluster-xcl-local");
    openTab(/dash.hpc.tab_jobs/);
    fireEvent.click((await screen.findAllByRole("button", { name: "View job train-a1" }))[0]);
    expect(nav.search).toBe("tab=jobs&job=a1b2c3d4");
    await waitFor(() => expect(fetchHpcJobOutput).toHaveBeenCalledWith("a1b2c3d4", "out"));
    await waitFor(() => expect(screen.getByTestId("hpc-job-output").textContent).toContain("step 1 ok"));
    await screen.findByText(/42% · epoch 3/);
    expect(screen.getByText(/A fresh copy is queued; reload after 1m 0s/)).toBeTruthy();
    const timeline = within(screen.getByRole("list", { name: "Job timeline" }));
    expect(timeline.getByText("queued 3m 20s")).toBeTruthy();
    expect(timeline.getByText("running 30m 0s")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "stderr" }));
    await waitFor(() => expect(fetchHpcJobOutput).toHaveBeenCalledWith("a1b2c3d4", "err"));

    // Closing it takes it out of the address and leaves the job list.
    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => expect(nav.search).toBe("tab=jobs"));
    expect(screen.queryByTestId("hpc-job-output")).toBeNull();
  });

  it("opens the job named in a link, even one the list has not loaded", async () => {
    nav.search = "job=zz99";
    fetchHpcJob.mockResolvedValue({ ok: true, job: job({ xcelsior_job_id: "zz99", mapping_id: "m-zz", submission_name: "linked-job" }) });
    render(<HpcPage />);
    await screen.findByRole("dialog", { name: "linked-job" });
    expect(fetchHpcJob).toHaveBeenCalledWith("zz99");
    expect(screen.getByRole("tab", { name: /dash.hpc.tab_jobs/ }).getAttribute("aria-selected")).toBe("true");
  });

  it("cancels by Slurm id on the job's cluster, after confirming", async () => {
    render(<HpcPage />);
    await screen.findByTestId("cluster-xcl-local");
    openTab(/dash.hpc.tab_jobs/);
    fireEvent.click((await screen.findAllByRole("button", { name: "Cancel job train-a1" }))[0]);
    expect(cancelSlurmJob).not.toHaveBeenCalled();
    const calls = fetchHpcJobs.mock.calls.length;
    fireEvent.click(screen.getByRole("button", { name: "Cancel job" }));
    await waitFor(() => expect(cancelSlurmJob).toHaveBeenCalledWith("4242", "xcl-local"));
    await waitFor(() => expect(fetchHpcJobs.mock.calls.length).toBeGreaterThan(calls));
  });

  it("shows the billing ledger: shadow cost, drift against accounting, and each run's rule", async () => {
    render(<HpcPage />);
    await screen.findByTestId("cluster-xcl-local");
    openTab(/dash.hpc.tab_ledger/);
    await screen.findByText("Shadow mode: nothing has been charged");
    expect(fetchHpcBilling).toHaveBeenCalledWith({ days: 14, clusterId: undefined });
    expect(screen.getByText("2.00")).toBeTruthy(); // 7200 GPU seconds
    expect(screen.getByText("$0.7000")).toBeTruthy();
    expect(screen.getByText("1 of 2 day(s) over 1%")).toBeTruthy();
    // The day over tolerance says what did not match.
    expect(screen.getByText("Over tolerance")).toBeTruthy();
    expect(screen.getByText("only sacct: 77@1")).toBeTruthy();
    expect(screen.getByText("Agrees")).toBeTruthy();
    // A requeued job is two rows, each with the rule that billed it.
    expect(screen.getByText("Ran to its end, suspended time billed (GPUs held)")).toBeTruthy();
    expect(screen.getByText("Requeued run")).toBeTruthy();
    expect(screen.getByText("+1m 30s suspended")).toBeTruthy();
    expect(screen.getAllByText("not charged").length).toBe(2);

    fireEvent.click(screen.getByRole("button", { name: "30 days" }));
    await waitFor(() => expect(fetchHpcBilling).toHaveBeenCalledWith({ days: 30, clusterId: undefined }));
  });

  it("shows the submit log with what never became a job, and filters by state", async () => {
    render(<HpcPage />);
    await screen.findByTestId("cluster-xcl-local");
    openTab(/dash.hpc.tab_submissions/);
    await screen.findByText("bad-partition-a1");
    expect(fetchHpcSubmissions).toHaveBeenCalledWith({ state: "all", clusterId: undefined });
    expect(screen.getByText("invalid_partition")).toBeTruthy();
    expect(screen.getByText("sbatch: error: invalid partition")).toBeTruthy();
    // A refused submission has no job to open; an accepted one does.
    expect((screen.getByRole("button", { name: "dead0001" }) as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: /^Rejected/ }));
    await waitFor(() => expect(fetchHpcSubmissions).toHaveBeenCalledWith({ state: "rejected", clusterId: undefined }));
    fireEvent.click(screen.getByRole("button", { name: "a1b2c3d4" }));
    expect(nav.search).toBe("tab=jobs&job=a1b2c3d4");
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

  it("previews the batch script as a dry run", async () => {
    render(<HpcPage />);
    await screen.findByTestId("cluster-xcl-local");
    openTab(/dash.hpc.tab_submit/);
    fireEvent.change(await screen.findByPlaceholderText("my-training-run"), { target: { value: "train-1" } });
    fireEvent.click(screen.getByRole("button", { name: /Preview script/ }));
    await waitFor(() => expect(screen.getByTestId("hpc-script-preview").textContent).toContain("#SBATCH --gpus-per-node=h100:2"));
    expect(submitSlurmInstance.mock.calls[0][0].dry_run).toBe(true);
  });

  it("shows real profile specs and keeps retired clusters out of the picker", async () => {
    render(<HpcPage />);
    await screen.findByTestId("cluster-xcl-local");
    openTab(/dash.hpc.tab_submit/);
    await screen.findByText("h100 × 8/node");
    expect(screen.getAllByText(/Partition: assigned by cluster/).length).toBeGreaterThan(0);
    expect(screen.getByText("retired")).toBeTruthy();
    const picker = within(screen.getByRole("radiogroup", { name: "Cluster profile" }));
    expect((picker.getByRole("radio", { name: /SHARCNET Nibi/ }) as HTMLButtonElement).getAttribute("aria-checked")).toBe("true");
    expect((picker.getByRole("radio", { name: /Graham/ }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("keeps the page up when the cluster read fails", async () => {
    fetchHpcClusters.mockRejectedValue(new Error("platform admin or a scoped machine principal is required"));
    render(<HpcPage />);
    await screen.findByText("platform admin or a scoped machine principal is required");
    expect(screen.getByText("Could not reach the API")).toBeTruthy();
    openTab(/dash.hpc.tab_jobs/);
    await screen.findAllByText("train-a1");
  });

  it("ignores a delayed older page after the job filter changes", async () => {
    let resolveOlder!: (value: unknown) => void;
    fetchHpcJobs.mockImplementation((options: { cursor?: string; state: string }) => {
      if (options.cursor) return new Promise((resolve) => { resolveOlder = resolve; });
      if (options.state === "all") return Promise.resolve({ ok: true, jobs: [job({ mapping_id: "all", submission_name: "all-filter-job" })], next_cursor: null });
      return Promise.resolve({ ok: true, jobs: [job()], next_cursor: "older" });
    });
    render(<HpcPage />);
    await screen.findByTestId("cluster-xcl-local");
    openTab(/dash.hpc.tab_jobs/);
    fireEvent.click(screen.getByRole("button", { name: "Load more" }));
    fireEvent.click(screen.getByRole("button", { name: /^All$/ }));
    await screen.findAllByText("all-filter-job");
    await act(async () => { resolveOlder({ ok: true, jobs: [job({ mapping_id: "old", submission_name: "old-filter-job" })], next_cursor: null }); });
    expect(screen.queryByText("old-filter-job")).not.toBeInTheDocument();
    expect(screen.getAllByText("all-filter-job").length).toBeGreaterThan(0);
  });

  it("is admin-only, like the routes behind it", async () => {
    currentUser = { role: "provider" };
    const { container } = render(<HpcPage />);
    expect(container.innerHTML).toBe("");
    await waitFor(() => expect(nav.replace).toHaveBeenCalledWith("/dashboard"));
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
    expect(summary).toMatchObject({ clustersEnabled: 2, clustersHealthy: 1, open: 2, running: 1, pending: 1, stale: 1, orphans: 1, exact: false });
    expect(attentionItems([cluster(), disabled], [job()], NOW)).toEqual([]);
  });

  it("takes exact counts from the registry over the loaded page", () => {
    const counted = cluster({ open_jobs: { total: 250, running: 200, pending: 50, stale: 4, awaiting_accounting: 3 } });
    expect(summarise([counted], [job()], NOW)).toMatchObject({
      open: 250, running: 200, pending: 50, stale: 4, awaitingAccounting: 3, exact: true,
    });
    // One cluster without counts (an older API) falls back to the loaded page for all.
    expect(summarise([counted, cluster({ cluster_id: "b" })], [job()], NOW)).toMatchObject({ open: 1, exact: false });
  });

  it("puts what is not being watched ahead of what can wait, and says where to act", () => {
    const items = attentionItems([sick], [job({ stale: true }), job({ missing_since: ago(60) })], NOW);
    expect(items.map((i) => i.severity)).toEqual(["critical", "critical", "warning", "warning"]);
    expect(items[0]).toMatchObject({ tab: "fleet", clusterId: "nibi" });
    expect(items[1]).toMatchObject({ tab: "jobs" });
    expect(items[1].text).toMatch(/missing from both the controller and accounting/);
  });

  it("formats durations", () => {
    expect(formatDuration(42)).toBe("42s");
    expect(formatDuration(3725)).toBe("1h 2m");
    expect(formatDuration(200000)).toBe("2d 7h");
    expect(formatDuration(null)).toBe("—");
  });

  it("formats money, GPU time and billing rules", () => {
    expect(formatMicros(389_083)).toBe("$0.3891");
    expect(formatMicros(12_500_000)).toBe("$12.50");
    expect(formatMicros(0)).toBe("$0.00");
    expect(formatMicros(null)).toBe("—");
    expect(formatGpuHours(4002)).toBe("1.11");
    expect(formatGpuHours(72_000)).toBe("20.0");
    expect(rateLabel(cluster({ rate_card: { gpu_hour_micros: 1_250_000 } }))).toBe("$1.25 per GPU hour");
    expect(rateLabel(cluster({ rate_card: {} }))).toBeNull();
    expect(rateLabel(cluster())).toBeNull();
    expect(billingRuleLabel("timeout_capped_to_limit")).toBe("Timed out, capped at the limit");
    // A rule this page has not heard of is still readable.
    expect(billingRuleLabel("some_new_rule")).toBe("some new rule");
    expect(submissionTone("unknown")).toBe("warning");
    expect(submissionTone("lost")).toBe("failed");
  });
});
