import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

vi.mock("@/lib/locale", () => ({ useLocale: () => ({ t: (key: string) => key }) }));
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));
const fetchEmailConsents = vi.fn();
const grantEmailConsent = vi.fn();
const withdrawEmailConsent = vi.fn();
vi.mock("@/lib/api", () => ({
  fetchEmailConsents: (...a: unknown[]) => fetchEmailConsents(...a),
  grantEmailConsent: (...a: unknown[]) => grantEmailConsent(...a),
  withdrawEmailConsent: (...a: unknown[]) => withdrawEmailConsent(...a),
}));

import { ProductUpdatesPreference } from "@/components/settings/ProductUpdatesPreference";

const consent = (over = {}) => ({
  purpose: "product_updates", consent_type: "express", granted_at: "2026-11-01T00:00:00Z",
  expires_at: null, withdrawn_at: null, active: true, ...over,
});

describe("product updates preference", () => {
  beforeEach(() => {
    fetchEmailConsents.mockReset().mockResolvedValue({ ok: true, consents: [] });
    grantEmailConsent.mockReset().mockResolvedValue({ ok: true });
    withdrawEmailConsent.mockReset().mockResolvedValue({ ok: true });
  });

  it("is off until the person opts in, and opting in records express consent", async () => {
    render(<ProductUpdatesPreference />);
    const toggle = screen.getByRole("switch");
    await waitFor(() => expect(toggle).toHaveProperty("disabled", false));
    expect(toggle.getAttribute("aria-checked")).toBe("false");
    fireEvent.click(toggle);
    await waitFor(() => expect(grantEmailConsent).toHaveBeenCalledWith("product_updates"));
    await waitFor(() => expect(toggle.getAttribute("aria-checked")).toBe("true"));
  });

  it("shows an existing opt-in and withdraws it at once", async () => {
    fetchEmailConsents.mockResolvedValue({ ok: true, consents: [consent()] });
    render(<ProductUpdatesPreference />);
    const toggle = screen.getByRole("switch");
    await waitFor(() => expect(toggle.getAttribute("aria-checked")).toBe("true"));
    fireEvent.click(toggle);
    await waitFor(() => expect(withdrawEmailConsent).toHaveBeenCalledWith("product_updates"));
    await waitFor(() => expect(toggle.getAttribute("aria-checked")).toBe("false"));
  });

  it("treats withdrawn or expired consent as off", async () => {
    fetchEmailConsents.mockResolvedValue({ ok: true, consents: [consent({ expires_at: "2020-01-01T00:00:00Z" })] });
    render(<ProductUpdatesPreference />);
    const toggle = screen.getByRole("switch");
    await waitFor(() => expect(toggle).toHaveProperty("disabled", false));
    expect(toggle.getAttribute("aria-checked")).toBe("false");
  });

  it("stays locked when the preference cannot be read, and keeps its state when a save fails", async () => {
    fetchEmailConsents.mockRejectedValue(new Error("offline"));
    const { unmount } = render(<ProductUpdatesPreference />);
    await waitFor(() => expect(fetchEmailConsents).toHaveBeenCalled());
    expect(screen.getByRole("switch")).toHaveProperty("disabled", true);
    unmount();

    fetchEmailConsents.mockResolvedValue({ ok: true, consents: [] });
    grantEmailConsent.mockRejectedValue(new Error("500"));
    render(<ProductUpdatesPreference />);
    const toggle = screen.getByRole("switch");
    await waitFor(() => expect(toggle).toHaveProperty("disabled", false));
    fireEvent.click(toggle);
    await waitFor(() => expect(grantEmailConsent).toHaveBeenCalled());
    await waitFor(() => expect(toggle).toHaveProperty("disabled", false));
    expect(toggle.getAttribute("aria-checked")).toBe("false");
  });
});
