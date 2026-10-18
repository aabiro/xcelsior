import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

// The CLI wizard's browser sign-in used to land on a developer placeholder
// ("submit your device code with POST /api/auth/verify"), so nobody could
// approve a terminal from a browser. This is the page that replaced it.

const state = vi.hoisted(() => ({
  user: null as { email: string } | null,
  params: new URLSearchParams(),
}));
const api = vi.hoisted(() => {
  class ApiError extends Error {
    constructor(public status: number, message: string) {
      super(message);
    }
  }
  return { ApiError, approveDeviceCode: vi.fn() };
});

vi.mock("@/lib/api", () => api);
vi.mock("@/lib/auth", () => ({ useAuth: () => ({ user: state.user, loading: false }) }));
vi.mock("next/navigation", () => ({ useSearchParams: () => state.params }));
vi.mock("next/link", () => ({
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}));
vi.mock("@/components/marketing/SiteAuthShell", () => ({
  SiteAuthCard: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SiteAuthAlert: ({ children }: { children: React.ReactNode }) => <div role="alert">{children}</div>,
  SiteAuthHeader: ({ title, subtitle }: { title: React.ReactNode; subtitle?: React.ReactNode }) => (
    <header><h1>{title}</h1>{subtitle ? <p>{subtitle}</p> : null}</header>
  ),
}));

import DevicePage from "@/app/(marketing)/device/page";

describe("device sign-in page", () => {
  beforeEach(() => {
    api.approveDeviceCode.mockReset();
    state.user = null;
    state.params = new URLSearchParams();
  });

  it("sends a signed-out visitor to sign in and back, keeping the code", () => {
    state.params = new URLSearchParams("user_code=ab12cd34");
    render(<DevicePage />);
    expect(screen.getByText("AB12-CD34")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Sign in" })).toHaveAttribute(
      "href",
      `/login?redirect=${encodeURIComponent("/device?user_code=AB12-CD34")}`,
    );
    expect(api.approveDeviceCode).not.toHaveBeenCalled();
  });

  it("never approves on its own, and approves the shown code on Approve", async () => {
    state.user = { email: "person@example.test" };
    state.params = new URLSearchParams("user_code=AB12-CD34");
    api.approveDeviceCode.mockResolvedValue({ ok: true, user_code: "AB12-CD34" });
    render(<DevicePage />);
    expect(screen.getByText("person@example.test")).toBeInTheDocument();
    expect(api.approveDeviceCode).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /approve/i }));
    await waitFor(() => expect(screen.getByText("Your terminal is signed in")).toBeInTheDocument());
    expect(api.approveDeviceCode).toHaveBeenCalledWith("AB12-CD34");
  });

  it("explains an expired code in words", async () => {
    state.user = { email: "person@example.test" };
    state.params = new URLSearchParams("user_code=AB12-CD34");
    api.approveDeviceCode.mockRejectedValue(new api.ApiError(404, "Invalid or expired device user code"));
    render(<DevicePage />);
    fireEvent.click(screen.getByRole("button", { name: /approve/i }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/expired or doesn't exist/));
  });

  it("formats a typed code and only continues once it is complete", () => {
    render(<DevicePage />);
    const input = screen.getByLabelText("Code");
    fireEvent.change(input, { target: { value: "ab12c" } });
    expect(input).toHaveValue("AB12-C");
    expect(screen.getByRole("button", { name: "Continue" })).toBeDisabled();
    fireEvent.change(input, { target: { value: "ab12 cd34" } });
    expect(input).toHaveValue("AB12-CD34");
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(screen.getByText("AB12-CD34")).toBeInTheDocument();
  });
});
