import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

vi.mock("@/lib/locale", () => ({ useLocale: () => ({ t: (key: string) => key }) }));
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));
const apiFetch = vi.fn();
vi.mock("@/lib/api", () => ({ apiFetch: (...a: unknown[]) => apiFetch(...a) }));

import { ReceiptEmailsPreference } from "@/components/settings/ReceiptEmailsPreference";

describe("receipt emails preference", () => {
  beforeEach(() => {
    apiFetch.mockReset().mockImplementation((_url: string, init?: RequestInit) =>
      Promise.resolve(init?.method === "PUT" ? { ok: true } : { notifications: true }),
    );
  });

  it("is on when the account has never set it, and turning it off saves at once", async () => {
    render(<ReceiptEmailsPreference />);
    const toggle = screen.getByRole("switch");
    await waitFor(() => expect(toggle).toHaveProperty("disabled", false));
    expect(toggle.getAttribute("aria-checked")).toBe("true");
    fireEvent.click(toggle);
    await waitFor(() =>
      expect(apiFetch).toHaveBeenCalledWith("/api/users/me/preferences", {
        method: "PUT",
        body: JSON.stringify({ receipt_emails: false }),
      }),
    );
    await waitFor(() => expect(toggle.getAttribute("aria-checked")).toBe("false"));
  });

  it("shows a saved opt-out", async () => {
    apiFetch.mockResolvedValue({ receipt_emails: false });
    render(<ReceiptEmailsPreference />);
    const toggle = screen.getByRole("switch");
    await waitFor(() => expect(toggle).toHaveProperty("disabled", false));
    expect(toggle.getAttribute("aria-checked")).toBe("false");
  });

  it("reads as off, and cannot be changed, while email notifications are off", async () => {
    render(<ReceiptEmailsPreference emailOn={false} />);
    const toggle = screen.getByRole("switch");
    expect(toggle.getAttribute("aria-checked")).toBe("false");
    expect(toggle).toHaveProperty("disabled", true);
    expect(screen.getByText("dash.settings.receipt_emails_needs_email")).toBeTruthy();
  });

  it("stays disabled when the setting could not be read", async () => {
    apiFetch.mockRejectedValue(new Error("offline"));
    render(<ReceiptEmailsPreference />);
    await new Promise((r) => setTimeout(r, 0));
    expect(screen.getByRole("switch")).toHaveProperty("disabled", true);
  });
});
