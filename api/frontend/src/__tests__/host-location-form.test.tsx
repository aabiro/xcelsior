import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useState } from "react";
import { HostLocationForm } from "@/components/ui/province-selector";
import type { Host } from "@/lib/api";

const initial = { host_id: "move-host", version: 1, country: "CA", province: "ON", region: "ca-on", status: "active", location_complete: true } as Host;
let host: Host;
let rejectSave: boolean;
const fetchMock = vi.fn();
const response = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });

function Screen() {
  const [value, setValue] = useState(initial);
  return <HostLocationForm host={value} onSaved={setValue} />;
}

describe("host relocation", () => {
  beforeEach(() => {
    host = { ...initial }; rejectSave = false;
    fetchMock.mockReset().mockImplementation(async (path: string, options?: RequestInit) => {
      if (path.startsWith("/api/locations")) return response({ countries: [{ code: "CA", name: "Canada" }, { code: "US", name: "United States" }],
        subdivisions: path.includes("country=US") ? [{ code: "NY", name: "New York", region: "us-ny" }] : [{ code: "ON", name: "Ontario", region: "ca-on" }],
        requires_subdivision: true, required_address_fields: ["province"] });
      if (path.endsWith("/location")) {
        if (rejectSave) return new Response(JSON.stringify({error:{message:"This host has active workloads. Save its new location after they finish."}}),{status:409});
        const body = JSON.parse(options!.body as string);
        expect(body.expected_version).toBe(host.version);
        host = { ...host, country: body.country, province: body.province, region: "us-ny", version: host.version! + 1 };
        return response({ host });
      }
      throw new Error(`Unexpected endpoint ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

  it("edits an active host's location with a single save", async () => {
    render(<Screen />);
    await screen.findByRole("option", { name: "United States" });
    expect(screen.getByRole("button", { name: "Save location" })).toBeDisabled();
    fireEvent.change(screen.getByRole("combobox", { name: "Country" }), { target: { value: "US" } });
    await screen.findByRole("option", { name: "New York" });
    fireEvent.change(screen.getByRole("combobox", { name: /state|province/i }), { target: { value: "NY" } });
    await waitFor(() => expect(screen.getByRole("button", { name: "Save location" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Save location" }));
    await screen.findByText("Host location saved.");
    expect(screen.getByRole("button", { name: "Save location" })).toBeDisabled();
    expect(host.region).toBe("us-ny");
    expect(host.status).toBe("active");
    expect(host.version).toBe(2);
    expect(screen.queryByRole("button", {name:/pause|resume/i})).not.toBeInTheDocument();
    const saves=fetchMock.mock.calls.filter(([path]) => path.endsWith("/location"));
    expect(saves).toHaveLength(1);
  });

  it("shows an inline error when a location cannot currently be saved", async () => {
    rejectSave = true;
    render(<Screen />);
    await screen.findByRole("option", { name: "United States" });
    fireEvent.change(screen.getByRole("combobox", { name: "Country" }), { target: { value: "US" } });
    await screen.findByRole("option", { name: "New York" });
    fireEvent.change(screen.getByRole("combobox", { name: /state|province/i }), { target: { value: "NY" } });
    await waitFor(() => expect(screen.getByRole("button", { name: "Save location" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "Save location" }));
    await screen.findByRole("alert");
    expect(screen.getByRole("alert")).toHaveTextContent("This host has active workloads.");
    expect(screen.getByRole("button", { name: "Save location" })).toBeEnabled();
    expect(host.region).toBe("ca-on");
  });

  it("shows the load error instead of crashing when the country list is missing", async () => {
    // Any 200 without `countries` (a proxy page, an older API) used to be
    // stored as the list, and the next render threw on `.some`.
    fetchMock.mockReset().mockImplementation(async () => response({ ok: true }));
    render(<Screen />);
    expect(await screen.findByText("Unable to load countries")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Country" })).toBeInTheDocument();
  });
});
