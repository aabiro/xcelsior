import type { TelemetryData, TelemetrySnapshot } from "@/lib/api";

/** Missing or invalid sensor data is unavailable, never an observed zero. */
export function readTelemetry(snapshot: TelemetrySnapshot): TelemetryData {
  const metrics = snapshot.metrics;
  const metric = (key: string, minimum = 0, maximum = Infinity): number | null => {
    const value = metrics[key];
    return typeof value === "number" && Number.isFinite(value) && value >= minimum && value <= maximum ? value : null;
  };
  const noGpu = metrics.gpu_count === 0;
  return {
    utilization: noGpu ? null : metric("utilization", 0, 100),
    temp: noGpu ? null : metric("temp", -273.15),
    memory_used_gb: noGpu ? null : metric("memory_used_gb"),
    memory_total_gb: noGpu ? null : metric("memory_total_gb"),
    power_draw_w: noGpu ? null : metric("power_draw_w"),
    active_jobs: metric("active_jobs"),
    net_rx_mbps: metric("net_rx_mbps"), net_tx_mbps: metric("net_tx_mbps"),
    disk_read_mb_s: metric("disk_read_mb_s"), disk_write_mb_s: metric("disk_write_mb_s"),
    stale: snapshot.stale,
  };
}
