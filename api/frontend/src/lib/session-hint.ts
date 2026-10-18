/**
 * Lightweight, client-only "is signed in" hint with a TTL.
 *
 * The real session lives in an httpOnly cookie that JS can't read, so on a hard
 * page load the marketing navbar can't know the auth state until `/api/auth/me`
 * resolves, which causes a "Sign In" → "Dashboard" flash. This hint is written
 * whenever a session is confirmed and read synchronously so the navbar can show
 * the signed-in state immediately. It's only a hint: the server probe is still
 * the source of truth and corrects a stale hint on the next load.
 */
const KEY = "xcelsior.session_hint";
/** Default hint lifetime, refreshed on every confirmed session, so it only
 *  matters for users who never return; the probe corrects it regardless. */
const DEFAULT_TTL_MS = 30 * 24 * 60 * 60 * 1000;

export function setSessionHint(ttlMs: number = DEFAULT_TTL_MS): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(KEY, String(Date.now() + ttlMs));
  } catch {
    /* storage unavailable (private mode / quota), hint is best-effort */
  }
}

export function clearSessionHint(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(KEY);
  } catch {
    /* noop */
  }
}

/** True when a non-expired session hint exists. Expired hints are cleared. */
export function hasSessionHint(): boolean {
  if (typeof window === "undefined") return false;
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return false;
    const expiresAt = Number(raw);
    if (!Number.isFinite(expiresAt) || Date.now() > expiresAt) {
      window.localStorage.removeItem(KEY);
      return false;
    }
    return true;
  } catch {
    return false;
  }
}
