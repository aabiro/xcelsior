import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import EventsPage from "@/app/(dashboard)/dashboard/events/page";

const mocks = vi.hoisted(() => ({ apiFetch: vi.fn(), createEventSource: vi.fn(), isEventStreamAvailable: vi.fn() }));
vi.mock("@/lib/api", () => mocks);
vi.mock("@/lib/locale", () => ({ useLocale: () => ({ t: (key: string) => key }) }));
vi.mock("sonner", () => ({ toast: { error: vi.fn() } }));
const failure = { event_id: "failure-1", event_type: "job.failed", severity: "error", entity_type: "job", entity_id: "job-1", actor: "worker:tower", timestamp: 1794500000, data: { reason: "Container exited with code 1" } };

beforeEach(() => {
  mocks.apiFetch.mockReset().mockResolvedValue({ events: [failure], total: 1, next_cursor: null, event_types: ["job.failed", "job.lease.renewed"], severity_counts: { error: 1 } });
  mocks.isEventStreamAvailable.mockReset().mockResolvedValue(true);
  mocks.createEventSource.mockReset();
});

describe("EventsPage", () => {
  it("shows the failure reason, severity, actor, related instance and expandable details", async () => {
    render(<EventsPage />);
    await screen.findByText("Container exited with code 1");
    expect(screen.getByText("worker:tower", { exact: false })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "job: job-1" })).toHaveAttribute("href", "/dashboard/instances/job-1");
    expect(screen.getByText("Event details")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Event type" })).not.toHaveTextContent("job.lease.renewed");
  });

  it("sends severity and search filters to the server before pagination", async () => {
    render(<EventsPage />);
    await screen.findByText("Container exited with code 1");
    fireEvent.change(screen.getByRole("combobox", { name: "Severity" }), { target: { value: "error" } });
    fireEvent.change(screen.getByRole("textbox", { name: "Search events" }), { target: { value: "job-1" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(mocks.apiFetch).toHaveBeenLastCalledWith(expect.stringContaining("severity=error&query=job-1")));
  });

  it("receives named SSE events emitted by the actual stream protocol", async () => {
    const listeners = new Map<string, EventListener>();
    mocks.createEventSource.mockReturnValue({ addEventListener: (type: string, listener: EventListener) => listeners.set(type, listener), close: vi.fn() });
    render(<EventsPage />);
    await screen.findByText("Container exited with code 1");
    fireEvent.click(screen.getByRole("button", { name: /dash.events.connect/ }));
    await waitFor(() => expect(listeners.has("job_status")).toBe(true));
    await act(async () => { listeners.get("job_status")!(new MessageEvent("job_status", { data: JSON.stringify({ job_id: "job-2", status: "failed" }) })); });
    expect(screen.getByRole("button", { name: /1 new events/ })).toBeInTheDocument();
  });
});
