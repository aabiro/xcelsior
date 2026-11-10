import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

// The page a product-update email's Unsubscribe link opens. Opening it must
// not unsubscribe anyone (mail scanners follow every link); pressing the
// button must, without a sign-in.

let token: string | null = "signed-token";
vi.mock("next/navigation", () => ({ useSearchParams: () => ({ get: () => token }) }));
vi.mock("@/lib/locale", () => ({ useLocale: () => ({ t: (key: string) => key }) }));
vi.mock("@/components/marketing/SiteAuthShell", () => ({
  SiteAuthCard: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SiteAuthHeader: ({ title, subtitle }: { title: string; subtitle?: string }) => (
    <div><h1>{title}</h1><p>{subtitle}</p></div>
  ),
}));
const unsubscribeWithToken = vi.fn();
vi.mock("@/lib/api", () => ({ unsubscribeWithToken: (...a: unknown[]) => unsubscribeWithToken(...a) }));

import UnsubscribePage from "@/app/(marketing)/unsubscribe/page";

describe("unsubscribe page", () => {
  beforeEach(() => {
    token = "signed-token";
    unsubscribeWithToken.mockReset().mockResolvedValue({ ok: true, purpose: "product_updates" });
  });

  it("asks first, then unsubscribes with the link's token", async () => {
    render(<UnsubscribePage />);
    expect(screen.getByText("auth.unsub_title")).toBeTruthy();
    expect(unsubscribeWithToken).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "auth.unsub_confirm" }));
    await waitFor(() => expect(unsubscribeWithToken).toHaveBeenCalledWith("signed-token"));
    await screen.findByText("auth.unsub_done");
  });

  it("says why it failed and lets the person try again", async () => {
    unsubscribeWithToken.mockRejectedValueOnce(new Error("This unsubscribe link is not valid."));
    render(<UnsubscribePage />);
    fireEvent.click(screen.getByRole("button", { name: "auth.unsub_confirm" }));
    await screen.findByText("This unsubscribe link is not valid.");
    fireEvent.click(screen.getByRole("button", { name: "auth.unsub_confirm" }));
    await screen.findByText("auth.unsub_done");
  });

  it("explains a link with no token and offers nothing to press", () => {
    token = null;
    render(<UnsubscribePage />);
    expect(screen.getByText("auth.unsub_missing")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "auth.unsub_confirm" })).toBeNull();
  });
});
