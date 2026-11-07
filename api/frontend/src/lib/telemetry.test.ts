import { describe, expect, it } from "vitest";
import { readTelemetry } from "./telemetry";

describe("reported host telemetry", () => {
  const read = (metrics: Record<string, unknown>) => readTelemetry({metrics, timestamp: 100, received_at: 101, stale: false});

  it("reads real nested sensor values and preserves their units", () => {
    expect(read({utilization: 37, temp: 56, memory_used_gb: 3.5, memory_total_gb: 24, power_draw_w: 108})).toMatchObject({utilization:37, temp:56, memory_used_gb:3.5, memory_total_gb:24, power_draw_w:108});
  });

  it("distinguishes a reported zero from an absent sensor", () => {
    expect(read({utilization:0}).utilization).toBe(0);
    expect(read({}).utilization).toBeNull();
    expect(read({}).temp).toBeNull();
  });

  it("does not turn malformed readings into plausible numbers", () => {
    for (const value of ["42", true, NaN, Infinity, -1, 101]) {
      expect(read({utilization:value}).utilization).toBeNull();
    }
  });

  it("a machine reporting no GPU has no GPU sensor readings", () => {
    expect(read({gpu_count:0, utilization:0, memory_used_gb:0, memory_total_gb:0, active_jobs:2})).toMatchObject({utilization:null, memory_used_gb:null, memory_total_gb:null, active_jobs:2});
  });
});
