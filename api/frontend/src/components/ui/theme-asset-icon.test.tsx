import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { render } from "@testing-library/react";
import { ThemeAssetIcon } from "./theme-asset-icon";

describe("ThemeAssetIcon", () => {
  it("keeps site-theme-dark and site-theme-light when className is provided", () => {
    const { container } = render(<ThemeAssetIcon name="overview" className="h-6 w-6" />);
    const imgs = container.querySelectorAll("img");
    expect(imgs).toHaveLength(2);
    expect(imgs[0]?.className).toContain("site-theme-dark");
    expect(imgs[0]?.className).toContain("h-6");
    expect(imgs[1]?.className).toContain("site-theme-light");
    expect(imgs[1]?.className).toContain("h-6");
  });

  it("defaults to theme pairing classes without caller className", () => {
    const { container } = render(<ThemeAssetIcon name="overview" />);
    const imgs = container.querySelectorAll("img");
    expect(imgs[0]?.className).toBe("site-theme-dark");
    expect(imgs[1]?.className).toBe("site-theme-light");
  });

  it("merges theme classes via cn in source (not className ?? fallback)", () => {
    const source = readFileSync(resolve(__dirname, "./theme-asset-icon.tsx"), "utf8");
    expect(source).toMatch(/cn\("site-theme-dark", className\)/);
    expect(source).toMatch(/cn\("site-theme-light", className\)/);
    expect(source).not.toMatch(/className \?\? "site-theme-/);
  });
});