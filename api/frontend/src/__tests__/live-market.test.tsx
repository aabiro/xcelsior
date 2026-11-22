import { render as rtlRender, screen, waitFor } from "@testing-library/react";
import type { ReactElement } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LocaleProvider } from "@/lib/locale";
import { featuredGpu, liveGpus } from "@/lib/live-market";
import HomePage from "@/app/(marketing)/page";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => "/",
}));

function render(ui: ReactElement) {
  return rtlRender(ui, { wrapper: LocaleProvider });
}

const ROWS = [
  { gpu_model: "RTX 3060", vram_gb: 12, count_available: 1, price_per_hour_cad: 0.12, evidence: { hosts: 1, verified: 1, latest_benchmark_at: 1_763_800_000 } },
  { gpu_model: "RTX 2060", vram_gb: 6, count_available: 0, price_per_hour_cad: 0.08, evidence: { hosts: 1, verified: 0 } },
  { gpu_model: "Mystery", vram_gb: 8, count_available: 2, price_per_hour_cad: 0, evidence: { hosts: 1, verified: 0 } },
];

afterEach(() => vi.unstubAllGlobals());

describe("liveGpus", () => {
  it("keeps only priced GPUs that are free now, cheapest first", () => {
    const gpus = liveGpus(ROWS);
    expect(gpus.map((g) => g.gpuModel)).toEqual(["RTX 3060"]);
    expect(gpus[0]).toMatchObject({ pricePerHour: 0.12, verifiedHosts: 1, hosts: 1 });
  });
});

describe("featuredGpu", () => {
  it("prefers the cheapest verified GPU, and falls back to the cheapest", () => {
    const rows = [
      { ...ROWS[1], count_available: 1 },
      ROWS[0],
    ];
    const gpus = liveGpus(rows);
    expect(gpus.map((g) => g.gpuModel)).toEqual(["RTX 2060", "RTX 3060"]);
    expect(featuredGpu(gpus)?.gpuModel).toBe("RTX 3060");
    expect(featuredGpu(liveGpus([{ ...ROWS[1], count_available: 1 }]))?.gpuModel).toBe("RTX 2060");
    expect(featuredGpu([])).toBeUndefined();
  });
});

describe("home hero", () => {
  it("features a verified GPU but quotes the cheapest from-price", async () => {
    const rows = [{ ...ROWS[1], count_available: 1 }, ROWS[0]];
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => ({ gpus: rows }) })));
    render(<HomePage />);
    await waitFor(() => expect(screen.getByTestId("live-gpu-model").textContent).toBe("RTX 3060"));
    expect(screen.getByTestId("live-gpu-price").textContent).toBe("$0.12/hr");
    expect(screen.getByText("From $0.08/hr")).toBeTruthy();
  });

  it("shows the cheapest live GPU, not a fixed RTX 4090 price", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => ({ gpus: ROWS }) })));
    render(<HomePage />);
    await waitFor(() => expect(screen.getByTestId("live-gpu-model").textContent).toBe("RTX 3060"));
    expect(screen.getByTestId("live-gpu-price").textContent).toBe("$0.12/hr");
    expect(screen.getByText("From $0.12/hr")).toBeTruthy();
    expect(screen.queryByText(/RTX 4090/)).toBeNull();
    expect(screen.queryByText(/\$0\.30/)).toBeNull();
    expect(screen.queryByText(/CAD/)).toBeNull();
  });

  it("names no price when nothing is free", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, json: async () => ({}) })));
    render(<HomePage />);
    await waitFor(() => expect(screen.getByText("No GPU is free this minute")).toBeTruthy());
    expect(screen.getByText("Billed by the second")).toBeTruthy();
    expect(screen.queryByTestId("live-gpu-price")).toBeNull();
  });
});
