import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

describe("shared gradient primary button tokens", () => {
  const marketingCss = readFileSync(
    resolve(__dirname, "../components/marketing/marketing-theme.css"),
    "utf8",
  );

  it("site-button-primary and btn-gradient-primary share the same gradient angle and text-shadow", () => {
    expect(marketingCss).toMatch(
      /\.site-button-primary,\s*\n\.btn-gradient-primary\s*\{[\s\S]*?background:\s*linear-gradient\(100deg/,
    );
    expect(marketingCss).toMatch(
      /\.site-button-primary,\s*\n\.btn-gradient-primary\s*\{[\s\S]*?text-shadow:\s*0 1px 2px rgba\(0,\s*0,\s*0,\s*0\.28\)/,
    );
  });

  it("overview Launch Instance button uses site-button-primary", () => {
    const overviewPage = readFileSync(
      resolve(__dirname, "../app/(dashboard)/dashboard/page.tsx"),
      "utf8",
    );
    expect(overviewPage).toContain("site-button site-button-primary dashboard-overview-hero-cta");
    expect(overviewPage).toContain("Launch Instance");
  });

  it("Button default variant uses btn-gradient-primary", () => {
    const buttonTsx = readFileSync(resolve(__dirname, "../components/ui/button.tsx"), "utf8");
    expect(buttonTsx).toContain('default: "btn-gradient-primary');
  });

  it("gradient primary tokens do not set font-family or font-size (typography unchanged)", () => {
    const block = marketingCss.match(
      /\.site-button-primary,\s*\n\.btn-gradient-primary\s*\{([^}]+)\}/,
    )?.[1];
    expect(block).toBeTruthy();
    expect(block).not.toMatch(/font-family/i);
    expect(block).not.toMatch(/font-size/i);
    expect(block).not.toMatch(/letter-spacing/i);
  });
});