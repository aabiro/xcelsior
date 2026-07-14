/** Pure UI helpers for volumes page — exported for unit tests. */

export function provisioningAgeMinutes(createdAt?: number, nowMs = Date.now()): number {
  if (!createdAt) return 0;
  return Math.max(0, Math.floor((nowMs / 1000 - createdAt) / 60));
}

export function shouldSurfaceVolumeStatus(
  status: string,
  createdAt?: number,
  nowMs = Date.now(),
): boolean {
  if (status === "error") return true;
  if (status === "provisioning" && provisioningAgeMinutes(createdAt, nowMs) >= 10) {
    return true;
  }
  return false;
}

export function isVolumeReadyForAttach(status: string): boolean {
  return status === "available" || status === "attached";
}

export function isVolumeTransient(status: string): boolean {
  return status === "provisioning" || status === "deleting";
}