import { describe, expect, it } from "vitest";
import {
  isVolumeReadyForAttach,
  isVolumeTransient,
  provisioningAgeMinutes,
  shouldSurfaceVolumeStatus,
} from "./volume-ui";

describe("volume UI helpers", () => {
  const now = new Date("2026-07-14T12:00:00Z").getTime();
  const fiveMinutesAgo = Math.floor(now / 1000) - 5 * 60;
  const elevenMinutesAgo = Math.floor(now / 1000) - 11 * 60;

  it("shouldSurfaceVolumeStatus hides routine provisioning", () => {
    expect(shouldSurfaceVolumeStatus("provisioning", fiveMinutesAgo, now)).toBe(false);
    expect(shouldSurfaceVolumeStatus("provisioning", elevenMinutesAgo, now)).toBe(true);
    expect(shouldSurfaceVolumeStatus("error", elevenMinutesAgo, now)).toBe(true);
    expect(shouldSurfaceVolumeStatus("available", elevenMinutesAgo, now)).toBe(false);
  });

  it("isVolumeTransient tracks provisioning lifecycle only", () => {
    expect(isVolumeTransient("provisioning")).toBe(true);
    expect(isVolumeTransient("deleting")).toBe(true);
    expect(isVolumeTransient("creating")).toBe(false);
    expect(isVolumeTransient("available")).toBe(false);
  });

  it("isVolumeReadyForAttach gates attach affordances", () => {
    expect(isVolumeReadyForAttach("available")).toBe(true);
    expect(isVolumeReadyForAttach("attached")).toBe(true);
    expect(isVolumeReadyForAttach("provisioning")).toBe(false);
  });

  it("provisioningAgeMinutes computes elapsed minutes", () => {
    const created = Math.floor(now / 1000) - 5 * 60;
    expect(provisioningAgeMinutes(created, now)).toBe(5);
  });
});