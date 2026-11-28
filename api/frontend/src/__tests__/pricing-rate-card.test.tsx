import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { loadPricing } from "@/app/(marketing)/pricing/rates";
import { SavingsCalculator } from "@/app/(marketing)/pricing/calculator";

// The pricing page read a hard-coded six-GPU list when the rate card was slow,
// and derived spot/reserved rates by fixed ratios. Its share image showed
// three invented prices. Everything now reads the published rate card
// (owner, 2026-11-23).

vi.mock("@/lib/locale", () => ({ useLocale: () => ({ t: (key: string) => key }) }));
const mocks = vi.hoisted(() => ({ apiFetch: vi.fn() }));
vi.mock("@/lib/api", () => mocks);

const card = {
  pricing: {
    "RTX 2060": { vram_gb: 6, base_rate_cad: 0.08, spot_cad: 0.032, reserved_1mo_cad: 0.064, reserved_1yr_cad: 0.044 },
    "RTX 3060": { vram_gb: 12, base_rate_cad: 0.12 },
    B200: { vram_gb: 192, base_rate_cad: 9.5, spot_cad: 3.8, reserved_1mo_cad: 7.6, reserved_1yr_cad: 5.225 },
    H100: { vram_gb: 80, base_rate_cad: 2.6, spot_cad: 1.04, reserved_1mo_cad: 2.08, reserved_1yr_cad: 1.43 },
    Mystery: { base_rate_cad: 1 },
  },
};

const inventory = [
  { gpu_model: "RTX 2060", vram_gb: 6, count_total: 1, count_available: 0 },
  { gpu_model: "RTX 3060", vram_gb: 12, count_total: 1, count_available: 1 },
  { gpu_model: "B200", vram_gb: 192, count_total: 1, count_available: 0 },
  { gpu_model: "H100", vram_gb: 80, count_total: 1, count_available: 0 },
];

function backend({ cardOk = true, spot = {} as Record<string, number>, fleet = inventory as unknown[] } = {}) {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    if (url.includes("/api/pricing/reference")) {
      if (!cardOk) throw new Error("offline");
      return { ok: true, json: async () => card };
    }
    if (url.includes("/spot-prices")) return { ok: true, json: async () => ({ prices: spot }) };
    return { ok: true, json: async () => ({ inventory: fleet, gpus: [] }) };
  }));
}

afterEach(() => vi.unstubAllGlobals());

describe("pricing rates", () => {
  it("uses the rate card as published, with no invented spot or reserved rate", async () => {
    backend();
    const { rows } = await loadPricing();
    const r3060 = rows.find((row) => row.model === "RTX 3060");
    expect(r3060).toEqual({ model: "RTX 3060", vram: 12, onDemand: 0.12, spot: null, reserved1m: null, reserved1y: null });
    // A row without its memory on the card is left out, not guessed at 24 GB.
    expect(rows.some((row) => row.model === "Mystery")).toBe(false);
  });

  it("prefers the live spot market and computes savings from the rows", async () => {
    backend({ spot: { H100: 1.56 } });
    const { rows, maxSpotSaving, maxReservedSaving } = await loadPricing();
    expect(rows.find((row) => row.model === "H100")?.spot).toBe(1.56);
    expect(maxSpotSaving).toBe(60); // B200 and RTX 2060 card spot
    expect(maxReservedSaving).toBe(45);
  });

  it("always lists the GPUs in the live fleet", async () => {
    backend({ fleet: [{ gpu_model: "RTX 3060", vram_gb: 12, count_total: 1, count_available: 1 }] });
    const { rows, fleet } = await loadPricing(2);
    expect(rows.map((row) => row.model)).toEqual(["RTX 3060"]);
    expect(fleet).toEqual({ "RTX 3060": 1 });
  });

  it("keeps busy inventory and never turns the rate card into an empty fleet's inventory", async () => {
    backend({ fleet: [inventory[0]] });
    const busy = await loadPricing();
    expect(busy.rows.map((row) => row.model)).toEqual(["RTX 2060"]);
    expect(busy.fleet).toEqual({ "RTX 2060": 0 });
    backend({ fleet: [] });
    expect((await loadPricing()).rows).toEqual([]);
  });

  it("shows nothing rather than a stand-in list when the card cannot be read", async () => {
    backend({ cardOk: false });
    const result = await loadPricing();
    expect(result.rows).toEqual([]);
    expect(result.maxSpotSaving).toBeNull();
    expect(result.maxReservedSaving).toBeNull();
  });
});

describe("savings calculator", () => {
  it("uses the published reserved rate, not a fixed 45 % discount", () => {
    render(<SavingsCalculator gpus={[{ model: "RTX 2060", onDemand: 0.1, reserved: 0.075 }]} />);
    expect(screen.getAllByText("$17.60").length).toBeGreaterThan(0);
    expect(screen.getByText("$13.20")).toBeInTheDocument();
    fireEvent.change(screen.getAllByRole("spinbutton")[0], { target: { value: "10" } });
    expect(screen.getByText("$16.50")).toBeInTheDocument();
  });

  it("says so when there is no rate to estimate from", () => {
    const { container } = render(<SavingsCalculator gpus={[]} />);
    expect(screen.getByRole("combobox")).toBeDisabled();
    expect(screen.getByRole("option")).toHaveTextContent("pricing.rates_unavailable");
    expect(Array.from(container.querySelectorAll(".site-result-value")).map((el) => el.textContent)).toEqual(["—", "—", "—"]);
  });
});

describe("host GPU picker", () => {
  beforeEach(() => {
    mocks.apiFetch.mockReset();
    Element.prototype.scrollIntoView = vi.fn();
  });

  function hostBackend(fleet = [{ gpu_model: "RTX 3060", vram_gb: 8, count_total: 1, count_available: 0 }]) {
    mocks.apiFetch.mockImplementation(async (path: string) => path === "/api/pricing/reference"
      ? card : { gpus: [], inventory: fleet });
  }

  it("offers only live inventory, even when busy, using its measured memory", async () => {
    const { GpuModelSelector } = await import("@/components/ui/gpu-model-selector");
    hostBackend();
    const selected = vi.fn();
    render(<GpuModelSelector value="" onChange={selected} />);
    fireEvent.focus(screen.getByRole("combobox", { name: "GPU model" }));
    await waitFor(() => expect(mocks.apiFetch).toHaveBeenCalled());
    fireEvent.click(await screen.findByRole("option", { name: /RTX 3060/ }));
    expect(selected).toHaveBeenCalledWith({ model: "RTX 3060", vram_gb: 8 });
    fireEvent.focus(screen.getByRole("combobox"));
    expect(screen.queryByRole("option", { name: /B200/ })).toBeNull();
    expect(screen.queryByRole("option", { name: /Mystery/ })).toBeNull();
    expect(mocks.apiFetch).toHaveBeenCalledWith("/api/pricing/reference");
    expect(mocks.apiFetch).toHaveBeenCalledWith("/api/v2/gpu/available");
  });

  it("reports an empty fleet without listing priced models", async () => {
    const { GpuModelSelector } = await import("@/components/ui/gpu-model-selector");
    hostBackend([]);
    render(<GpuModelSelector value="" onChange={vi.fn()} />);
    fireEvent.focus(screen.getByRole("combobox"));
    expect(await screen.findByText("No GPUs are listed in the live inventory.")).toBeInTheDocument();
    expect(screen.queryByRole("option")).toBeNull();
  });

  it("supports keyboard selection and gives each picker's options valid distinct IDs", async () => {
    const { GpuModelSelector } = await import("@/components/ui/gpu-model-selector");
    hostBackend();
    const selected = vi.fn();
    render(<><GpuModelSelector value="" onChange={selected} /><GpuModelSelector value="" onChange={vi.fn()} /></>);
    const [first, second] = screen.getAllByRole("combobox");
    expect(first.getAttribute("aria-controls")).not.toBe(second.getAttribute("aria-controls"));
    fireEvent.focus(first);
    const option = await screen.findByRole("option", { name: /RTX 3060/ });
    expect(first.getAttribute("aria-activedescendant")).toBe(option.id);
    expect(option.id).not.toMatch(/\s/);
    fireEvent.keyDown(first, { key: "Escape" });
    expect(first).toHaveAttribute("aria-expanded", "false");
    fireEvent.keyDown(first, { key: "ArrowDown" });
    fireEvent.keyDown(first, { key: "Enter" });
    expect(selected).toHaveBeenCalledWith({ model: "RTX 3060", vram_gb: 8 });
  });

  it("reports a failed load and retries", async () => {
    const { GpuModelSelector } = await import("@/components/ui/gpu-model-selector");
    hostBackend();
    mocks.apiFetch.mockRejectedValueOnce(new Error("offline"));
    render(<GpuModelSelector value="" onChange={vi.fn()} />);
    fireEvent.focus(screen.getByRole("combobox"));
    fireEvent.click(await screen.findByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("option", { name: /RTX 3060/ })).toBeInTheDocument();
    expect(mocks.apiFetch).toHaveBeenCalledTimes(4);
  });
});
