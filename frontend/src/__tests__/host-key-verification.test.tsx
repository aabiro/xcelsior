import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

/**
 * Gate P2's parity clause, from the human's side.
 *
 * `open_instance_access` returns a fingerprint and a command to check it
 * against. The instance page dropped the field at the type boundary and showed
 * `ssh root@host -p port` with nothing to verify it with, so an agent could
 * check the host key and the person looking at the same instance could not.
 *
 * `tests/test_host_key_verification_parity.py` pins the command against the
 * tool. What is left for this file is the behaviour that pin cannot see: that
 * the fingerprint rendered is the API's own value, that the command names the
 * host actually being connected to, and — the one that matters most — that a
 * missing fingerprint produces a visible "cannot be verified" state instead of
 * an empty space.
 *
 * The empty-space case is the dangerous one. A row that disappears reads as
 * "nothing to check here", and the user accepts whatever key ssh offers.
 */

vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

import {
  HostKeyVerification,
  hostKeyVerifyCommand,
} from "@/components/instances/host-key-verification";

/** Shape-valid: `SHA256:` + exactly 43 unpadded-base64 characters. */
const FINGERPRINT = "SHA256:" + "a".repeat(43);

beforeEach(() => {
  vi.clearAllMocks();
});

describe("with a fingerprint the platform observed", () => {
  it("shows the API's value verbatim, never a truncation", () => {
    render(<HostKeyVerification host="connect.xcelsior.ca" port={12345} fingerprint={FINGERPRINT} />);
    // Truncation is a CSS concern; the DOM must carry the whole value, because
    // a user compares all 43 characters and `select-all` copies what is there.
    expect(screen.getByTestId("host-key-fingerprint")).toHaveTextContent(FINGERPRINT);
  });

  it("names the same host and port the ssh command uses", () => {
    render(<HostKeyVerification host="connect.xcelsior.ca" port={12345} fingerprint={FINGERPRINT} />);
    const cmd = screen.getByTestId("host-key-verify-command").textContent ?? "";
    // A check run against a different endpoint than the one being connected to
    // returns "verified" without having verified anything that matters.
    expect(cmd).toContain("connect.xcelsior.ca");
    expect(cmd).toContain("12345");
    expect(cmd).toBe(hostKeyVerifyCommand("connect.xcelsior.ca", 12345));
  });

  it("tells the user what a mismatch means", () => {
    render(<HostKeyVerification host="connect.xcelsior.ca" port={12345} fingerprint={FINGERPRINT} />);
    expect(screen.getByText(/do not connect/i)).toBeInTheDocument();
  });
});

describe("with no fingerprint — the permanent null state", () => {
  it.each([
    ["null", null],
    ["undefined", undefined],
    ["empty string", ""],
  ])("renders a visible unverifiable state for %s", (_label, value) => {
    render(
      <HostKeyVerification
        host="connect.xcelsior.ca"
        port={12345}
        fingerprint={value as string | null | undefined}
      />,
    );
    expect(screen.getByTestId("host-key-unverifiable")).toBeInTheDocument();
    expect(screen.getByText(/cannot be checked against one/i)).toBeInTheDocument();
  });

  it("offers no command, because there is nothing to compare against", () => {
    render(<HostKeyVerification host="connect.xcelsior.ca" port={12345} fingerprint={null} />);
    // A verify command with no expected value is theatre: it prints a
    // fingerprint the user has no way to judge, and running it feels like
    // verification.
    expect(screen.queryByTestId("host-key-verify-command")).toBeNull();
    expect(screen.queryByTestId("host-key-fingerprint")).toBeNull();
  });

  it("does not suggest trusting the key regardless", () => {
    const { container } = render(
      <HostKeyVerification host="connect.xcelsior.ca" port={12345} fingerprint={null} />,
    );
    expect(container.textContent?.toLowerCase()).not.toMatch(/safe to accept|accept it anyway/);
  });
});
