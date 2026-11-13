import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { UsageStatementDialog } from "@/components/billing/usage-statement-dialog";
import type { Invoice } from "@/lib/api";

const fetchUsageStatement = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api", () => ({ fetchUsageStatement }));
vi.mock("@/lib/locale", () => ({ useLocale: () => ({ t: (key: string) => key, locale: "en" }) }));

const invoice: Invoice = {
  invoice_id: "usage-october", period_start: 1790812800, period_end: 1793491200,
  total_cad: 12.5, subtotal_cad: 12.5, tax_cad: 0, tax_rate: 0, line_items: 1, status: "draft",
};
const response = {
  ok: true,
  invoice: {
    ...invoice, customer_id: "cust-1", currency: "CAD", tax_amount_cad: 0,
    line_items: [{ description: "GPU compute", category: "compute", quantity: 2.5, unit: "GPU-hours",
      unit_price_cad: 5, subtotal_cad: 12.5, gpu_model: "RTX 4090", job_id: "job-1" }],
  },
};

beforeEach(() => fetchUsageStatement.mockReset().mockResolvedValue(response));

describe("usage statement viewer", () => {
  it("loads the selected account and period and renders the API's line items and total", async () => {
    const download = vi.fn();
    render(<UsageStatementDialog customerId="cust-1" invoice={invoice} onClose={vi.fn()} onDownload={download} />);
    await screen.findByText("GPU compute");
    expect(fetchUsageStatement).toHaveBeenCalledWith("cust-1", invoice.period_start, invoice.period_end);
    expect(screen.getByText("GPU-hours")).toBeInTheDocument();
    expect(screen.getByText("RTX 4090")).toBeInTheDocument();
    expect(screen.getAllByText("$12.50")).toHaveLength(2);
    fireEvent.click(screen.getByRole("button", { name: "CSV" }));
    expect(download).toHaveBeenCalledOnce();
  });

  it("reports failure and retries the same statement", async () => {
    fetchUsageStatement.mockRejectedValueOnce(new Error("503"));
    render(<UsageStatementDialog customerId="cust-1" invoice={invoice} onClose={vi.fn()} onDownload={vi.fn()} />);
    await screen.findByRole("alert");
    expect(screen.queryByText("$12.50")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "dash.billing.statement_retry" }));
    await screen.findByText("GPU compute");
    expect(fetchUsageStatement).toHaveBeenCalledTimes(2);
  });

  it("ignores an old period's delayed response after switching statements", async () => {
    let finish!: (value: typeof response) => void;
    fetchUsageStatement.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
    const props = { customerId: "cust-1", onClose: vi.fn(), onDownload: vi.fn() };
    const { rerender } = render(<UsageStatementDialog {...props} invoice={invoice} />);
    const next = { ...invoice, period_start: invoice.period_end, period_end: invoice.period_end + 86400 };
    fetchUsageStatement.mockResolvedValueOnce({ ...response, invoice: { ...response.invoice, line_items: [] } });
    rerender(<UsageStatementDialog {...props} invoice={next} />);
    await screen.findByText("dash.billing.no_invoices");
    finish(response);
    await waitFor(() => expect(screen.queryByText("GPU compute")).not.toBeInTheDocument());
  });
});
