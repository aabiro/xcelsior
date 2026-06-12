import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { GearOnboarding } from "@/components/onboarding/gear-onboarding";

const t = (key: string, vars?: Record<string, string | number>) => {
  if (vars) {
    return `${key}:${JSON.stringify(vars)}`;
  }
  return key;
};

describe("GearOnboarding", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function mockDetection({
    name = "Ada",
    canadaOnly = true,
    onboarding = {},
    keys = [{ id: "k1" }],
    instances = [],
  } = {}) {
    fetchMock.mockImplementation((url: string) => {
      if (url === "/api/users/me/preferences") {
        return Promise.resolve({
          ok: true,
          json: () =>
            Promise.resolve({
              canada_only_routing: canadaOnly,
              preferences: { onboarding },
            }),
        });
      }
      if (url === "/api/keys") {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ keys }),
        });
      }
      if (url === "/instances") {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ instances }),
        });
      }
      return Promise.resolve({ ok: false, json: () => Promise.resolve(null) });
    });

    return { user: { name, email: "ada@example.com" } };
  }

  it("shows loading then progress with jurisdiction from canada_only_routing", async () => {
    const { user } = mockDetection({ canadaOnly: true, keys: [] });
    render(
      <GearOnboarding t={t} user={user} pathname="/dashboard" />,
    );

    expect(screen.getByText("gear.loading")).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "40");
    });

    expect(screen.queryByText("gear.loading")).not.toBeInTheDocument();
  });

  it("shows error state with retry on preferences failure", async () => {
    fetchMock.mockRejectedValueOnce(new Error("network"));
    render(
      <GearOnboarding t={t} user={{ email: "a@b.com" }} pathname="/dashboard" />,
    );

    await waitFor(() => {
      expect(screen.getByText("gear.load_error")).toBeInTheDocument();
    });

    mockDetection();
    fireEvent.click(screen.getByText("gear.retry"));

    await waitFor(() => {
      expect(screen.getByRole("progressbar")).toBeInTheDocument();
    });
  });

  it("exposes accessible progressbar", async () => {
    const { user } = mockDetection();
    render(
      <GearOnboarding t={t} user={user} pathname="/dashboard/marketplace" />,
    );

    await waitFor(() => {
      const bar = screen.getByRole("progressbar");
      expect(bar).toHaveAttribute("aria-labelledby", "gear-onboarding-progress-label");
      expect(bar.getAttribute("aria-valuenow")).not.toBe("0");
    });
  });
});