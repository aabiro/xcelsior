import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const DASHBOARD_THEME = resolve(__dirname, "../components/marketing/dashboard-theme.css");
const PORTALED_MODALS = [
  "../components/ui/dialog.tsx",
  "../components/ui/confirm-dialog.tsx",
  "../components/instances/launch-instance-modal.tsx",
];

const INLINE_MODALS = [
  "../components/billing/deposit-modal.tsx",
  "../components/billing/payment-method-modal.tsx",
  "../components/instances/save-as-template-dialog.tsx",
];

function expectModalOverlayClass(source: string) {
  expect(source).toMatch(/dashboard-site-modal-overlay/);
  expect(source).not.toMatch(/bg-black\/70/);
}

describe("portaled modal overlay dimming", () => {
  const dashboardCss = readFileSync(DASHBOARD_THEME, "utf8");

  it("styles dashboard-site-modal-overlay globally (not scoped under .dashboard-shell)", () => {
    expect(dashboardCss).toMatch(/\.dashboard-site-modal-overlay\s*\{[\s\S]*?background:\s*rgba\(0,\s*0,\s*0,\s*0\.58\)/);
    expect(dashboardCss).toMatch(/html\.light \.dashboard-site-modal-overlay\s*\{[\s\S]*?background:\s*rgba\(0,\s*0,\s*0,\s*0\.42\)/);
    expect(dashboardCss).not.toMatch(/\.dashboard-shell \.dashboard-site-modal-overlay/);
  });

  it.each(PORTALED_MODALS)("%s portals overlay with dashboard-site-modal-overlay class", (relPath) => {
    const source = readFileSync(resolve(__dirname, relPath), "utf8");
    expect(source).toContain("createPortal");
    expectModalOverlayClass(source);
  });

  it.each(INLINE_MODALS)("%s uses dashboard-site-modal-overlay without inline bg-black/70", (relPath) => {
    const source = readFileSync(resolve(__dirname, relPath), "utf8");
    expectModalOverlayClass(source);
  });
});