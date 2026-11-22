import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

vi.mock("@/lib/locale", () => ({
  useLocale: () => ({
    t: (key: string, params?: Record<string, unknown>) =>
      params ? `${key} ${JSON.stringify(params)}` : key,
  }),
}));
vi.mock("sonner", () => ({ toast: { error: vi.fn() } }));
const fetchShare = vi.fn();
const putShare = vi.fn();
vi.mock("@/lib/api", () => ({
  fetchReceiptShare: (...a: unknown[]) => fetchShare(...a),
  setReceiptShare: (...a: unknown[]) => putShare(...a),
}));

import { ReceiptShareControl } from "@/components/instances/receipt-share-control";

const state = (over: Record<string, unknown> = {}) => ({
  ok: true,
  job_id: "j1",
  shared: false,
  eligible: true,
  blocker: null,
  template_label: "vLLM",
  gpu_model: "RTX 3060",
  page: { slug: "vllm-on-rtx-3060", live: false, runs: null, min_runs: 3, gpu_has_supply: true },
  ...over,
});

describe("receipt share control", () => {
  beforeEach(() => {
    fetchShare.mockReset();
    putShare.mockReset();
  });

  it("is off until the owner turns it on, and saves at once", async () => {
    fetchShare.mockResolvedValue(state());
    putShare.mockResolvedValue(state({ shared: true }));
    render(<ReceiptShareControl jobId="j1" />);
    const toggle = await screen.findByRole("switch");
    expect(toggle.getAttribute("aria-checked")).toBe("false");
    fireEvent.click(toggle);
    await waitFor(() => expect(putShare).toHaveBeenCalledWith("j1", true));
    await waitFor(() => expect(toggle.getAttribute("aria-checked")).toBe("true"));
    expect(screen.getByTestId("receipt-share-page").textContent).toContain("waiting_runs");
  });

  it("links to the live page once it exists", async () => {
    fetchShare.mockResolvedValue(
      state({ shared: true, page: { slug: "vllm-on-rtx-3060", live: true, runs: 4, min_runs: 3, gpu_has_supply: true } }),
    );
    render(<ReceiptShareControl jobId="j1" />);
    const link = await screen.findByRole("link");
    expect(link.getAttribute("href")).toBe("/receipts/vllm-on-rtx-3060");
  });

  it("waits, disabled, while billing settles", async () => {
    fetchShare.mockResolvedValue(state({ eligible: false, blocker: "not_reconciled" }));
    render(<ReceiptShareControl jobId="j1" />);
    const toggle = await screen.findByRole("switch");
    expect(toggle).toHaveProperty("disabled", true);
    expect(screen.getByText("dash.receipt_share.settling")).toBeTruthy();
  });

  it("shows nothing for a run that can never count", async () => {
    fetchShare.mockResolvedValue(state({ eligible: false, blocker: "no_template", template_label: null }));
    const { container } = render(<ReceiptShareControl jobId="j1" />);
    await waitFor(() => expect(fetchShare).toHaveBeenCalled());
    await new Promise((r) => setTimeout(r, 0));
    expect(container.innerHTML).toBe("");
  });
});
