import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

const apiMocks = vi.hoisted(() => ({
  register: vi.fn(),
  oauthInitiate: vi.fn(),
  resendVerification: vi.fn(),
}));

const authMocks = vi.hoisted(() => ({
  login: vi.fn(),
}));

const navigationMocks = vi.hoisted(() => ({
  push: vi.fn(),
}));

const translations: Record<string, string> = {
  "auth.verify_check_email": "Check your email",
  "auth.verify_sent_to": "We sent a verification link to",
  "auth.verify_instructions": "Click the link in your email to verify your address and activate your account.",
  "auth.verify_resending": "Sending...",
  "auth.verify_resend": "Resend verification email",
  "auth.verify_resent": "Verification email sent!",
  "auth.register_signin": "Already have an account?",
  "auth.register_signin_link": "Sign in",
  "auth.register_title": "Create your account",
  "auth.register_subtitle": "Start computing on Canada's sovereign GPU cloud",
  "auth.github": "Continue with GitHub",
  "auth.google": "Continue with Google",
  "auth.huggingface": "Continue with Hugging Face",
  "auth.or": "or",
  "auth.name": "Name (optional)",
  "auth.name_placeholder": "Your name",
  "auth.email": "Email",
  "auth.email_placeholder": "you@example.com",
  "auth.password": "Password",
  "auth.confirm_password": "Confirm Password",
  "auth.pw_min": "Use 8 to 64 characters",
  "auth.pw_rule_letter": "Include at least one letter",
  "auth.pw_rule_number": "Include at least one number",
  "auth.pw_rule_symbol": "Include at least one symbol. Only these symbols are supported: !@#$%^*-_+=",
  "auth.pw_match": "Passwords match",
  "auth.register_loading": "Creating account...",
  "auth.register_button": "Create Account",
  "auth.reset_mismatch": "Passwords do not match",
  "auth.pw_policy_error": "Please satisfy all password requirements.",
};

vi.mock("@/lib/api", () => apiMocks);

vi.mock("@/lib/auth", () => ({
  useAuth: () => authMocks,
}));

vi.mock("@/lib/locale", () => ({
  useLocale: () => ({
    t: (key: string) => translations[key] ?? key,
    locale: "en",
  }),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => navigationMocks,
}));

vi.mock("next/link", () => ({
  default: ({
    children,
    href,
    ...rest
  }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) => (
    <a href={href} {...rest}>{children}</a>
  ),
}));

vi.mock("next/image", () => ({
  default: ({
    alt,
    ...rest
  }: React.ImgHTMLAttributes<HTMLImageElement>) => <img alt={alt} {...rest} />,
}));

import RegisterPage from "@/app/(marketing)/register/page";

describe("RegisterPage password requirements", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiMocks.register.mockResolvedValue({ email_verification_required: true });
  });

  it("updates the live checklist and only enables submit when the password is valid and confirmed", async () => {
    const user = userEvent.setup();

    render(<RegisterPage />);

    const emailInput = screen.getByLabelText("Email");
    const passwordInput = screen.getByLabelText("Password");
    const confirmInput = screen.getByLabelText("Confirm Password");
    const submitButton = screen.getByRole("button", { name: "Create Account" });
    const symbolRequirement = screen.getByText("Include at least one symbol. Only these symbols are supported: !@#$%^*-_+=").closest("li");
    const matchRequirement = screen.getByText("Passwords match").closest("li");

    expect(submitButton).toBeDisabled();
    expect(symbolRequirement).toHaveAttribute("data-satisfied", "false");
    expect(matchRequirement).toHaveAttribute("data-satisfied", "false");

    await user.type(emailInput, "user@example.com");
    await user.type(passwordInput, "StrongPass123?");

    expect(symbolRequirement).toHaveAttribute("data-satisfied", "false");
    expect(submitButton).toBeDisabled();

    await user.clear(passwordInput);
    await user.type(passwordInput, "StrongPass123!");

    expect(screen.getByText("Use 8 to 64 characters").closest("li")).toHaveAttribute("data-satisfied", "true");
    expect(screen.getByText("Include at least one letter").closest("li")).toHaveAttribute("data-satisfied", "true");
    expect(screen.getByText("Include at least one number").closest("li")).toHaveAttribute("data-satisfied", "true");
    expect(symbolRequirement).toHaveAttribute("data-satisfied", "true");
    expect(matchRequirement).toHaveAttribute("data-satisfied", "false");

    await user.type(confirmInput, "StrongPass123!");

    expect(matchRequirement).toHaveAttribute("data-satisfied", "true");
    expect(submitButton).toBeEnabled();
  });
});
