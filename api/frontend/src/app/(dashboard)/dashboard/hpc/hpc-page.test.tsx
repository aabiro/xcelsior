import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

// The page's three broken behaviours (2026-11-07): it sent priority "normal"
// and no vram, so every submit 422'd; cancel/refresh passed the Xcelsior id
// where the routes take the Slurm id; and it read profile fields the API
// never returned. These tests pin the fixed contract.

const replace = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn(), replace, back: vi.fn() }) }));
vi.mock("@/lib/locale", () => ({ useLocale: () => ({ t: (key: string) => key }) }));
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

let currentUser: { is_admin?: boolean; role?: string } | null = { is_admin: true };
vi.mock("@/lib/auth", () => ({ useAuth: () => ({ user: currentUser }) }));

const submitSlurmInstance = vi.fn();
const fetchSlurmJobStatus = vi.fn();
const cancelSlurmJob = vi.fn();
vi.mock("@/lib/api", () => ({
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
  fetchSlurmInstances: vi.fn(async () => ({
    ok: true,
    jobs: [{
      job_id: "a1b2c3d4", slurm_job_id: "4242", cluster_id: "nibi", observed_state: "running",
      desired_state: "running", submitted_at: 1794040946, last_observed_at: 1794040950, terminal_at: null,
    }],
  })),
  submitSlurmInstance: (...a: unknown[]) => submitSlurmInstance(...a),
  fetchSlurmJobStatus: (...a: unknown[]) => fetchSlurmJobStatus(...a),
  cancelSlurmJob: (...a: unknown[]) => cancelSlurmJob(...a),
}));

import HpcPage from "./page";

describe("HPC dashboard", () => {
  beforeEach(() => {
    currentUser = { is_admin: true };
    submitSlurmInstance.mockReset().mockResolvedValue({ slurm_job_id: "77", cluster_id: "nibi", script: "#!/bin/bash\n#SBATCH --gpus-per-node=h100:2" });
    fetchSlurmJobStatus.mockReset().mockResolvedValue({ slurm_job_id: "4242", slurm_state: "CANCELLED", xcelsior_state: "cancelled" });
    cancelSlurmJob.mockReset().mockResolvedValue({ cancelled: true, slurm_job_id: "4242" });
    replace.mockReset();
  });

  it("submits the payload the API accepts", async () => {
    render(<HpcPage />);
    await screen.findAllByText("SHARCNET Nibi");
    fireEvent.change(screen.getByPlaceholderText("my-training-run"), { target: { value: "train-1" } });
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
    await screen.findAllByText("SHARCNET Nibi");
    expect(screen.getByText("h100 × 8/node")).toBeTruthy();
    expect(screen.getAllByText("Partition: assigned by cluster").length).toBeGreaterThan(0);
    expect(screen.getByText("retired")).toBeTruthy();
    const options = Array.from(document.querySelectorAll("option")).map((o) => o.textContent);
    expect(options).toEqual(["SHARCNET Nibi"]);
  });

  it("cancels and refreshes by Slurm id on the job's cluster", async () => {
    render(<HpcPage />);
    await screen.findAllByText("SHARCNET Nibi");
    fireEvent.click(screen.getByRole("button", { name: /dash.hpc.tab_jobs/ }));
    await screen.findByText("Slurm job 4242");
    fireEvent.click(screen.getByTitle("Cancel job"));
    await waitFor(() => expect(cancelSlurmJob).toHaveBeenCalledWith("4242", "nibi"));
    await waitFor(() => expect(fetchSlurmJobStatus).toHaveBeenCalledWith("4242", "nibi"));
    await screen.findByText("cancelled (CANCELLED)");
  });

  it("is admin-only, like the routes behind it", async () => {
    currentUser = { role: "provider" };
    const { container } = render(<HpcPage />);
    expect(container.innerHTML).toBe("");
    await waitFor(() => expect(replace).toHaveBeenCalledWith("/dashboard"));
  });
});
